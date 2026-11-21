// Unit cases of the receiver dispatch core (Sprint 2.4, selector replay_pusch).
//
// The target links dispatch_core and replay_core and nothing else: no OCUDU library, no test
// framework. The grids are the storage model of the channel application core, and the decoders
// are doubles written in this file: one that answers at once with scripted bytes, and one that
// keeps every request until the case completes it.
//
//     dispatch_core_test            run every case
//     dispatch_core_test NAME       run one case (CTest registers each)
//     dispatch_core_test --list     print the case names
//     dispatch_core_test --classify COMPLETED ERROR CRC_OK BYTES_EQUAL GUARD_INTACT INPUT_INTACT
//                                   print the outcome of the rule for six flags (0 or 1), for a
//                                   reference in another language to reproduce
//
// ASCII only.

#include "dispatch_core.h"
#include "replay_core.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace isac_receiver;
using isac_replay::cf32;
using isac_replay::grid_shape;
using isac_replay::model_grid;
using isac_replay::o1_descriptor;
using isac_replay::o1_pool;
using isac_replay::o1_view;
using isac_replay::ticket;

namespace {

int failures = 0;

void failed(const char* file, int line, const std::string& what)
{
  ++failures;
  std::cerr << file << ":" << line << ": " << what << std::endl;
}

#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      failed(__FILE__, __LINE__, "expected " #condition);                                                              \
    }                                                                                                                  \
  } while (false)

// The statement must throw the given type with a message that holds the given text.
#define CHECK_THROWS(statement, type, text)                                                                            \
  do {                                                                                                                 \
    bool thrown_ = false;                                                                                              \
    try {                                                                                                              \
      statement;                                                                                                       \
    } catch (const type& error_) {                                                                                     \
      thrown_ = true;                                                                                                  \
      if (std::string(error_.what()).find(text) == std::string::npos) {                                                \
        failed(__FILE__, __LINE__, std::string("message '") + error_.what() + "' lacks '" + (text) + "'");             \
      }                                                                                                                \
    } catch (const std::exception& error_) {                                                                           \
      thrown_ = true;                                                                                                  \
      failed(__FILE__, __LINE__, std::string("another exception: ") + error_.what());                                  \
    }                                                                                                                  \
    if (!thrown_) {                                                                                                    \
      failed(__FILE__, __LINE__, "no exception from " #statement);                                                     \
    }                                                                                                                  \
  } while (false)

const std::chrono::milliseconds LONG_WAIT(5000);
const std::chrono::milliseconds SHORT_WAIT(30);

grid_shape small_shape()
{
  grid_shape shape;
  shape.rx_ports    = 2;
  shape.tx_ports    = 1;
  shape.symbols     = 2;
  shape.subcarriers = 6;
  return shape;
}

// Fills a free grid of the pool with values that depend on the seed and makes it ready for a grant record.
ticket make_ready(o1_pool& pool, const std::string& record, unsigned seed, const std::string& opportunity = "op")
{
  const grid_shape shape = small_shape();
  ticket           key;
  if (!pool.try_acquire(key)) {
    throw std::runtime_error("the case has no free grid");
  }
  isac_replay::grid_sink& sink = pool.writer(key);
  sink.clear();
  std::vector<cf32> row(shape.subcarriers);
  for (unsigned rx = 0; rx != shape.rx_ports; ++rx) {
    for (unsigned symbol = 0; symbol != shape.symbols; ++symbol) {
      for (unsigned k = 0; k != shape.subcarriers; ++k) {
        row[k] = cf32(static_cast<float>(seed) + 0.25F * static_cast<float>(rx), 0.5F * static_cast<float>(symbol * 6 + k));
      }
      sink.put(rx, symbol, row.data(), shape.subcarriers);
    }
  }
  o1_descriptor descriptor;
  descriptor.run_id              = "unit";
  descriptor.opportunity_id      = opportunity;
  descriptor.grant_id            = "grant-" + record;
  descriptor.grant_record_sha256 = record;
  descriptor.shape               = shape;
  descriptor.grid_gain           = 1.0;
  descriptor.storage             = sink.storage();
  pool.commit(key, descriptor);
  return key;
}

payload block(const std::string& id, std::size_t bytes, unsigned seed)
{
  payload result;
  result.id = id;
  result.bytes.resize(bytes);
  for (std::size_t index = 0; index != bytes; ++index) {
    result.bytes[index] = static_cast<std::uint8_t>((index * 31U + seed * 7U + 3U) & 0xffU);
  }
  result.sha256 = isac_replay::sha256_hex(result.bytes.data(), result.bytes.size());
  return result;
}

job_key key_of(const std::string& opportunity, const std::string& backend)
{
  job_key key;
  key.opportunity_id = opportunity;
  key.backend        = backend;
  return key;
}

// A decoder that answers during start: it writes the bytes it was given and reports its flags.
struct scripted_decoder final : decoder {
  std::vector<std::uint8_t> bytes;
  bool                      crc_ok      = true;
  std::string               error;
  unsigned                  completions = 1;
  std::size_t               overrun     = 0; ///< Bytes written behind the transport block.
  bool                      raise       = false;
  unsigned                  starts      = 0;

  void start(const request& job, completion_sink& sink) override
  {
    ++starts;
    if (raise) {
      throw std::runtime_error("no device");
    }
    std::copy_n(bytes.begin(), std::min(bytes.size(), job.output_bytes), job.output);
    std::fill_n(job.output + job.output_bytes, overrun, static_cast<std::uint8_t>(0x11));
    for (unsigned n = 0; n != completions; ++n) {
      sink.complete(job.token, crc_ok, error);
    }
  }
};

// A decoder that keeps every request. The case completes them in the order and with the bytes it chooses.
struct deferred_decoder final : decoder {
  std::vector<request> pending;
  completion_sink*     sink = nullptr;

  void start(const request& job, completion_sink& sink_) override
  {
    pending.push_back(job);
    sink = &sink_;
  }

  void finish(std::size_t index, const std::vector<std::uint8_t>& bytes, bool crc_ok)
  {
    const request& job = pending.at(index);
    std::copy_n(bytes.begin(), std::min(bytes.size(), job.output_bytes), job.output);
    sink->complete(job.token, crc_ok, std::string());
  }
};

bool has_anomaly(const ledger& book, const std::string& kind, const std::string& key)
{
  for (const anomaly& item : book.anomalies()) {
    if (item.kind == kind && item.key == key) {
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------------------------- cases

void case_compare()
{
  const payload      original = block("tb", 48, 1);
  std::vector<std::uint8_t> decoded  = original.bytes;
  comparison         same     = compare_bytes(decoded.data(), decoded.size(), original.bytes.data(), original.bytes.size());
  CHECK(same.equal && same.first_differing_byte == -1 && same.bytes_compared == 48);
  decoded[17] ^= 0x01;
  comparison one = compare_bytes(decoded.data(), decoded.size(), original.bytes.data(), original.bytes.size());
  CHECK(!one.equal && one.first_differing_byte == 17 && one.bytes_compared == 48);
  decoded[3] ^= 0x80;
  CHECK(compare_bytes(decoded.data(), decoded.size(), original.bytes.data(), original.bytes.size()).first_differing_byte == 3);
  // The first and the last byte are compared like every other.
  std::vector<std::uint8_t> edge = original.bytes;
  edge[0] ^= 0x01;
  comparison first = compare_bytes(edge.data(), edge.size(), original.bytes.data(), original.bytes.size());
  CHECK(!first.equal && first.first_differing_byte == 0);
  edge = original.bytes;
  edge[47] ^= 0x01;
  comparison last = compare_bytes(edge.data(), edge.size(), original.bytes.data(), original.bytes.size());
  CHECK(!last.equal && last.first_differing_byte == 47);
  // A shorter or a longer block is never equal, even when every compared byte is.
  comparison shorter = compare_bytes(original.bytes.data(), 47, original.bytes.data(), 48);
  CHECK(!shorter.equal && shorter.first_differing_byte == -1 && shorter.bytes_compared == 47);
  std::vector<std::uint8_t> longer = original.bytes;
  longer.push_back(0x00);
  comparison extended = compare_bytes(longer.data(), longer.size(), original.bytes.data(), original.bytes.size());
  CHECK(!extended.equal && extended.first_differing_byte == -1 && extended.bytes_compared == 48);
  CHECK(!compare_bytes(original.bytes.data(), 0, original.bytes.data(), 0).equal);

  // The rule, one line per row of its table.
  CHECK(classify(false, false, true, true, true, true) == outcome::timeout);
  CHECK(classify(true, true, true, true, true, true) == outcome::decoder_error);
  CHECK(classify(true, false, true, true, false, true) == outcome::decoder_error);
  CHECK(classify(true, false, true, true, true, false) == outcome::decoder_error);
  CHECK(classify(true, false, false, true, true, true) == outcome::crc_fail);
  CHECK(classify(true, false, true, false, true, true) == outcome::payload_mismatch);
  CHECK(classify(true, false, true, true, true, true) == outcome::decoded);
  CHECK(std::string(to_string(outcome::payload_mismatch)) == "payload_mismatch");
  CHECK(std::string(to_string(outcome::not_dispatched)) == "not_dispatched");
  CHECK(std::string(to_string(admission::duplicate_delivery)) == "duplicate_delivery");
}

void case_decoded()
{
  model_grid    grid(2, 2, 6);
  o1_pool       pool({&grid});
  ledger        book(pool);
  const ticket  ready    = make_ready(pool, "record-a", 1, "op-0");
  const payload original = block("grant-a", 48, 1);
  scripted_decoder cpu;
  cpu.bytes = original.bytes;

  const dispatched done = dispatch(book, cpu, key_of("op-0", "cpu"), ready, "record-a", original, LONG_WAIT);
  CHECK(done.admitted == admission::admitted && done.result == outcome::decoded && done.token == 1);
  const job record = book.snapshot(done.token);
  CHECK(record.key.text() == "op-0/cpu" && record.grant_id == "grant-record-a");
  CHECK(record.completed && record.crc_ok && record.error.empty() && record.completions == 1);
  CHECK(record.compared.equal && record.compared.bytes_compared == 48 && record.compared.first_differing_byte == -1);
  CHECK(record.guard_intact && record.input_intact);
  CHECK(record.decoded_sha256 == original.sha256 && record.payload_sha256 == original.sha256);
  CHECK(record.ingress_sha256 == pool.view(ready, "record-a").descriptor().readback_sha256);
  CHECK(record.ingress_sha256.size() == 64 && record.payload_bytes == 48 && record.payload_id == "grant-a");
  CHECK(book.decoded(done.token) == original.bytes);
  CHECK(book.anomalies().empty() && book.count(outcome::decoded) == 1 && book.count(outcome::decoded, "cuda") == 0);
  // The same grid goes to the second backend under its own key, and is released once both are done.
  const dispatched second = dispatch(book, cpu, key_of("op-0", "cuda"), ready, "record-a", original, LONG_WAIT);
  CHECK(second.result == outcome::decoded && second.token == 2 && cpu.starts == 2);
  CHECK(book.snapshot(second.token).ingress_sha256 == record.ingress_sha256);
  CHECK(book.holds(ready) == 0 && book.release_grid(ready));
  CHECK(pool.state(0) == isac_replay::grid_state::free);
}

void case_valid_crc_wrong_byte()
{
  model_grid    grid(2, 2, 6);
  o1_pool       pool({&grid});
  ledger        book(pool);
  const ticket  ready    = make_ready(pool, "record-a", 2);
  const payload original = block("grant-a", 1217, 2);
  // A notifier that reports a valid CRC for a block with one wrong byte.
  scripted_decoder wrong;
  wrong.bytes = original.bytes;
  wrong.bytes[1000] ^= 0x04;
  const dispatched done = dispatch(book, wrong, key_of("op-0", "cuda"), ready, "record-a", original, LONG_WAIT);
  CHECK(done.admitted == admission::admitted);
  CHECK(done.result == outcome::payload_mismatch);
  const job record = book.snapshot(done.token);
  CHECK(record.crc_ok && !record.compared.equal && record.compared.first_differing_byte == 1000);
  CHECK(record.compared.bytes_compared == 1217 && record.decoded_sha256 != original.sha256);
  CHECK(book.count(outcome::decoded) == 0 && book.count(outcome::payload_mismatch, "cuda") == 1);
  // The last byte alone is enough.
  wrong.bytes = original.bytes;
  wrong.bytes[1216] ^= 0x80;
  const dispatched last = dispatch(book, wrong, key_of("op-0", "cpu"), ready, "record-a", original, LONG_WAIT);
  CHECK(last.result == outcome::payload_mismatch && book.snapshot(last.token).compared.first_differing_byte == 1216);
}

void case_crc_fail()
{
  model_grid    grid(2, 2, 6);
  o1_pool       pool({&grid});
  ledger        book(pool);
  const ticket  ready    = make_ready(pool, "record-a", 3);
  const payload original = block("grant-a", 48, 3);
  // The CRC flag is the decoder's: even with the right bytes in the buffer a failure stays a failure.
  scripted_decoder failing;
  failing.bytes  = original.bytes;
  failing.crc_ok = false;
  const dispatched done = dispatch(book, failing, key_of("op-0", "cpu"), ready, "record-a", original, LONG_WAIT);
  CHECK(done.result == outcome::crc_fail);
  const job record = book.snapshot(done.token);
  CHECK(record.completed && !record.crc_ok && record.error.empty());
  CHECK(!record.compared.equal && record.compared.bytes_compared == 0 && record.decoded_sha256.empty());
  // It stays in the denominator: one job, one terminal outcome.
  CHECK(book.jobs().size() == 1 && book.count(outcome::crc_fail) == 1 && book.count(outcome::decoded) == 0);
  CHECK(book.release_grid(ready));
}

void case_swapped_completions()
{
  model_grid    first(2, 2, 6);
  model_grid    second(2, 2, 6);
  o1_pool       pool({&first, &second});
  ledger        book(pool);
  const ticket  grid_a = make_ready(pool, "record-a", 4, "op-a");
  const ticket  grid_b = make_ready(pool, "record-b", 5, "op-b");
  const payload tb_a   = block("grant-a", 48, 4);
  const payload tb_b   = block("grant-b", 48, 5);
  deferred_decoder decoder;
  std::uint64_t    token_a = 0;
  std::uint64_t    token_b = 0;
  CHECK(book.admit(key_of("op-a", "cuda"), grid_a, "record-a", tb_a, token_a) == admission::admitted);
  CHECK(book.admit(key_of("op-b", "cuda"), grid_b, "record-b", tb_b, token_b) == admission::admitted);
  book.started(token_a);
  book.started(token_b);
  decoder.start(book.describe(token_a), book);
  decoder.start(book.describe(token_b), book);
  CHECK(!book.wait(token_a, SHORT_WAIT));
  // The second decode completes first, from another thread, then the first one.
  std::thread worker([&]() {
    decoder.finish(1, tb_b.bytes, true);
    decoder.finish(0, tb_a.bytes, true);
  });
  CHECK(book.wait(token_a, LONG_WAIT) && book.wait(token_b, LONG_WAIT));
  worker.join();
  CHECK(book.resolve(token_a) == outcome::decoded && book.resolve(token_b) == outcome::decoded);
  CHECK(book.snapshot(token_a).key.opportunity_id == "op-a" && book.snapshot(token_a).decoded_sha256 == tb_a.sha256);
  CHECK(book.snapshot(token_b).key.opportunity_id == "op-b" && book.snapshot(token_b).decoded_sha256 == tb_b.sha256);
  CHECK(book.anomalies().empty());
  // The service time runs from the start of a decode to its callback: both waited out the short wait.
  const double waited_us = 1000.0 * static_cast<double>(SHORT_WAIT.count());
  CHECK(book.snapshot(token_a).service_us >= waited_us && book.snapshot(token_b).service_us >= waited_us);
  CHECK(book.snapshot(token_a).completions == 1 && book.snapshot(token_b).completions == 1);
  CHECK(book.release_grid(grid_a) && book.release_grid(grid_b));

  // A receiver that hands each decode the bytes of the other one reports two valid CRCs. With a
  // transport block per opportunity both comparisons fail. With one block for all they would pass.
  const ticket     again_a = make_ready(pool, "record-a", 4, "op-c");
  const ticket     again_b = make_ready(pool, "record-b", 5, "op-d");
  deferred_decoder crossed;
  CHECK(book.admit(key_of("op-c", "cuda"), again_a, "record-a", tb_a, token_a) == admission::admitted);
  CHECK(book.admit(key_of("op-d", "cuda"), again_b, "record-b", tb_b, token_b) == admission::admitted);
  crossed.start(book.describe(token_a), book);
  crossed.start(book.describe(token_b), book);
  crossed.finish(0, tb_b.bytes, true);
  crossed.finish(1, tb_a.bytes, true);
  CHECK(book.resolve(token_a) == outcome::payload_mismatch && book.resolve(token_b) == outcome::payload_mismatch);
  CHECK(book.snapshot(token_a).decoded_sha256 == tb_b.sha256);
  CHECK(book.count(outcome::decoded) == 2 && book.count(outcome::payload_mismatch) == 2 && book.jobs().size() == 4);
  // A decode whose start was never marked has no service time.
  CHECK(book.snapshot(token_a).service_us == 0.0);
}

void case_decoder_error()
{
  model_grid    grid(2, 2, 6);
  o1_pool       pool({&grid});
  ledger        book(pool);
  const ticket  ready    = make_ready(pool, "record-a", 6);
  const payload original = block("grant-a", 48, 6);
  // A decoder that raises when it is started.
  scripted_decoder raising;
  raising.raise = true;
  const dispatched raised = dispatch(book, raising, key_of("op-0", "cuda"), ready, "record-a", original, LONG_WAIT);
  CHECK(raised.admitted == admission::admitted && raised.result == outcome::decoder_error);
  CHECK(book.snapshot(raised.token).error == "the decoder raised: no device" && !book.snapshot(raised.token).crc_ok);
  // A decoder that reports an error in its callback, even beside a CRC pass and the right bytes.
  scripted_decoder reporting;
  reporting.bytes = original.bytes;
  reporting.error = "no soft buffer";
  const dispatched reported = dispatch(book, reporting, key_of("op-0", "cpu"), ready, "record-a", original, LONG_WAIT);
  CHECK(reported.result == outcome::decoder_error && book.snapshot(reported.token).error == "no soft buffer");
  CHECK(book.count(outcome::decoder_error) == 2 && book.count(outcome::decoded) == 0);
  // Both are terminal and neither holds the grid.
  CHECK(book.held_grids() == 0 && book.release_grid(ready));
}

void case_duplicate_delivery()
{
  model_grid    grid(2, 2, 6);
  o1_pool       pool({&grid});
  ledger        book(pool);
  const ticket  ready    = make_ready(pool, "record-a", 7);
  const payload original = block("grant-a", 48, 7);
  scripted_decoder cpu;
  cpu.bytes = original.bytes;
  CHECK(dispatch(book, cpu, key_of("op-0", "cpu"), ready, "record-a", original, LONG_WAIT).result == outcome::decoded);
  // The same grid for the same key a second time: refused, recorded, and the decoder is not started.
  const dispatched again = dispatch(book, cpu, key_of("op-0", "cpu"), ready, "record-a", original, LONG_WAIT);
  CHECK(again.admitted == admission::duplicate_delivery && again.token == 0 && again.result == outcome::pending);
  CHECK(again.reason.find("duplicate delivery") != std::string::npos);
  CHECK(cpu.starts == 1 && book.jobs().size() == 1 && book.count(outcome::decoded) == 1);
  CHECK(has_anomaly(book, "duplicate_delivery", "op-0/cpu"));

  // A decoder that calls back twice: the first result stands and the second is recorded.
  scripted_decoder twice;
  twice.bytes       = original.bytes;
  twice.completions = 2;
  const dispatched doubled = dispatch(book, twice, key_of("op-0", "cuda"), ready, "record-a", original, LONG_WAIT);
  CHECK(doubled.result == outcome::decoded && book.snapshot(doubled.token).completions == 2);
  CHECK(has_anomaly(book, "duplicate_completion", "op-0/cuda"));
  // A callback for a job that does not exist.
  book.complete(99, true, std::string());
  CHECK(book.anomalies().size() == 3 && book.anomalies().back().kind == "unknown_completion");
  CHECK(book.jobs().size() == 2 && book.holds(ready) == 0);
  // A decode that was planned and never started has a key and an outcome too, once.
  book.skip(key_of("op-1", "cuda"), "grant-record-a", "record-a", "the processor did not complete op-0");
  book.skip(key_of("op-1", "cuda"), "grant-record-a", "record-a", "again");
  CHECK(book.jobs().size() == 3 && book.count(outcome::not_dispatched, "cuda") == 1);
  CHECK(book.jobs().back().error == "the processor did not complete op-0" && has_anomaly(book, "duplicate_delivery", "op-1/cuda"));
  CHECK_THROWS(book.describe(3), state_error, "was never dispatched");
  std::uint64_t unused = 0;
  CHECK_THROWS(book.admit(key_of("", "cpu"), ready, "record-a", original, unused), input_error, "needs the identifier");
}

void case_silent_timeout()
{
  model_grid    grid(2, 2, 6);
  o1_pool       pool({&grid});
  ledger        book(pool);
  const ticket  ready    = make_ready(pool, "record-a", 8);
  const payload original = block("grant-a", 48, 8);
  deferred_decoder silent;
  const auto       before = std::chrono::steady_clock::now();
  const dispatched done   = dispatch(book, silent, key_of("op-0", "cuda"), ready, "record-a", original, SHORT_WAIT);
  CHECK(std::chrono::steady_clock::now() - before >= SHORT_WAIT);
  CHECK(done.admitted == admission::admitted && done.result == outcome::timeout);
  const job record = book.snapshot(done.token);
  CHECK(!record.completed && !record.crc_ok && record.completions == 0);
  CHECK(record.error == "no completion within the declared time");
  CHECK(book.count(outcome::timeout, "cuda") == 1);
  // The processor may still read the grid: it stays held, cannot be released and cannot be filled.
  ticket other;
  CHECK(book.holds(ready) == 1 && book.held_grids() == 1 && !book.release_grid(ready));
  CHECK(pool.state(0) == isac_replay::grid_state::ready && !pool.try_acquire(other));
  CHECK_THROWS(book.decoded(done.token), state_error, "has no completed result");
  // The completion arrives late: it is recorded, the outcome stays, and the grid can go.
  silent.finish(0, original.bytes, true);
  CHECK(book.resolve(done.token) == outcome::timeout && book.snapshot(done.token).completed);
  CHECK(has_anomaly(book, "late_completion", "op-0/cuda") && book.count(outcome::decoded) == 0);
  CHECK(book.holds(ready) == 0 && book.release_grid(ready) && pool.try_acquire(other));
}

void case_owned_input()
{
  model_grid       grid(2, 2, 6);
  o1_pool          pool({&grid});
  ledger           book(pool);
  const grid_shape shape    = small_shape();
  const ticket     ready    = make_ready(pool, "record-a", 9);
  const payload    original = block("grant-a", 48, 9);
  deferred_decoder decoder;
  std::uint64_t    token = 0;
  CHECK(book.admit(key_of("op-0", "cuda"), ready, "record-a", original, token) == admission::admitted);
  std::vector<cf32> at_admission(shape.subcarriers);
  book.view(token).get(1, 1, at_admission.data(), shape.subcarriers);
  book.started(token);
  decoder.start(book.describe(token), book);

  // While the decode runs the grid is neither released nor written: it is ready, held and of its generation.
  ticket other;
  CHECK(book.holds(ready) == 1 && !book.release_grid(ready) && !book.release_grid(ready));
  CHECK(!pool.try_acquire(other) && pool.generation(0) == 1);
  CHECK_THROWS(pool.writer(ready), state_error, "is ready, not filling");
  CHECK(pool.view(ready, "record-a").digest() == book.snapshot(token).ingress_sha256);

  // At its completion the decoder still reads the values it was admitted with.
  std::vector<cf32> at_completion(shape.subcarriers);
  book.view(token).get(1, 1, at_completion.data(), shape.subcarriers);
  CHECK(std::memcmp(at_admission.data(), at_completion.data(), shape.subcarriers * sizeof(cf32)) == 0);
  decoder.finish(0, original.bytes, true);
  CHECK(book.resolve(token) == outcome::decoded && book.snapshot(token).input_intact);

  // Only now the grid goes back, exactly once, and every earlier access is stale.
  CHECK(book.release_grid(ready));
  CHECK_THROWS(book.release_grid(ready), state_error, "is free, not ready");
  CHECK_THROWS(book.view(token), state_error, "is free, not ready");
  std::string         why;
  std::uint64_t       unused = 0;
  CHECK(book.admit(key_of("op-1", "cuda"), ready, "record-a", original, unused, &why) == admission::stale_grid);
  CHECK(why.find("is free, not ready") != std::string::npos);
  const ticket refilled = make_ready(pool, "record-b", 10);
  CHECK(refilled.grid == 0 && refilled.generation == 2);
  CHECK(book.admit(key_of("op-1", "cuda"), ready, "record-a", original, unused, &why) == admission::stale_grid);
  CHECK(why.find("stale generation 1") != std::string::npos);
  // A grid that was filled for another grant record is not decoded with this PDU.
  CHECK(book.admit(key_of("op-1", "cuda"), refilled, "record-a", original, unused, &why) == admission::changed_allocation);
  CHECK(why.find("changed allocation") != std::string::npos);
  payload empty;
  CHECK(book.admit(key_of("op-1", "cuda"), refilled, "record-b", empty, unused, &why) == admission::empty_payload);
  CHECK(book.jobs().size() == 1 && book.anomalies().size() == 4 && book.anomalies().front().kind == "refused_grid");
  CHECK(book.admit(key_of("op-1", "cuda"), refilled, "record-b", original, unused) == admission::admitted);
}

void case_changed_input()
{
  model_grid    grid(2, 2, 6);
  o1_pool       pool({&grid});
  ledger        book(pool);
  const payload original = block("grant-a", 48, 11);
  const cf32    other[6] = {{9, 9}, {9, 9}, {9, 9}, {9, 9}, {9, 9}, {9, 9}};

  // A grid that changed after it became ready is not admitted.
  const ticket  ready = make_ready(pool, "record-a", 11);
  grid.put(0, 0, other, 6);
  std::uint64_t token = 0;
  std::string   why;
  CHECK(book.admit(key_of("op-0", "cpu"), ready, "record-a", original, token, &why) == admission::changed_grid);
  CHECK(why.find("changed after it became ready") != std::string::npos && book.jobs().empty());
  pool.release(ready);

  // A grid that changes while it is decoded: the result cannot be attributed to the admitted input.
  const ticket     second = make_ready(pool, "record-a", 12);
  deferred_decoder decoder;
  CHECK(book.admit(key_of("op-1", "cpu"), second, "record-a", original, token) == admission::admitted);
  decoder.start(book.describe(token), book);
  grid.put(1, 1, other, 6);
  decoder.finish(0, original.bytes, true);
  CHECK(book.resolve(token) == outcome::decoder_error);
  const job record = book.snapshot(token);
  CHECK(record.crc_ok && record.compared.equal && !record.input_intact && record.guard_intact);
  CHECK(record.error == "the grid changed while it was decoded");
}

void case_guard()
{
  model_grid    grid(2, 2, 6);
  o1_pool       pool({&grid});
  ledger        book(pool);
  const ticket  ready    = make_ready(pool, "record-a", 13);
  const payload original = block("grant-a", 48, 13);
  const request job      = [&]() {
    std::uint64_t token = 0;
    book.admit(key_of("op-0", "probe"), ready, "record-a", original, token);
    return book.describe(token);
  }();
  // The buffer is the transport block, filled with one value, then the guard.
  CHECK(job.output_bytes == 48 && job.output[0] == FILL_VALUE && job.output[47] == FILL_VALUE);
  CHECK(job.output[48] == GUARD_VALUE && job.output[48 + GUARD_BYTES - 1] == GUARD_VALUE);
  book.complete(job.token, false, std::string());
  CHECK(book.resolve(job.token) == outcome::crc_fail);

  // A decoder that writes one byte behind the transport block, with a CRC pass and the right bytes.
  scripted_decoder overrunning;
  overrunning.bytes   = original.bytes;
  overrunning.overrun = 1;
  const dispatched done = dispatch(book, overrunning, key_of("op-0", "cuda"), ready, "record-a", original, LONG_WAIT);
  CHECK(done.result == outcome::decoder_error);
  CHECK(!book.snapshot(done.token).guard_intact && book.snapshot(done.token).compared.equal);
  CHECK(book.snapshot(done.token).error == "the decoder wrote past the transport block");
}

void case_accounting()
{
  model_grid first(2, 2, 6);
  model_grid second(2, 2, 6);
  o1_pool    pool({&first, &second});
  ledger     book(pool);
  scripted_decoder cpu;
  scripted_decoder cuda;
  std::vector<std::string> names;
  const unsigned           opportunities = 6;
  for (unsigned index = 0; index != opportunities; ++index) {
    const std::string opportunity = "op-" + std::to_string(index);
    const std::string record      = "record-" + std::to_string(index % 3);
    const payload     original    = block("grant-" + std::to_string(index), 48, index);
    const ticket      ready       = make_ready(pool, record, index, opportunity);
    cpu.bytes   = original.bytes;
    cuda.bytes  = original.bytes;
    cuda.crc_ok = index != 4; // one CRC failure on one backend
    CHECK(dispatch(book, cpu, key_of(opportunity, "cpu"), ready, record, original, LONG_WAIT).result == outcome::decoded);
    const outcome wanted = index != 4 ? outcome::decoded : outcome::crc_fail;
    CHECK(dispatch(book, cuda, key_of(opportunity, "cuda"), ready, record, original, LONG_WAIT).result == wanted);
    CHECK(book.release_grid(ready));
  }
  // Every admitted grid and every terminal result has one key, and the outcomes add up.
  const std::vector<job> jobs = book.jobs();
  CHECK(jobs.size() == 2 * opportunities);
  for (std::size_t index = 0; index != jobs.size(); ++index) {
    CHECK(jobs[index].token == index + 1 && jobs[index].sequence == index && jobs[index].result != outcome::pending);
    names.push_back(jobs[index].key.text());
  }
  std::sort(names.begin(), names.end());
  CHECK(std::adjacent_find(names.begin(), names.end()) == names.end());
  CHECK(book.count(outcome::decoded, "cpu") == opportunities && book.count(outcome::decoded, "cuda") == opportunities - 1);
  CHECK(book.count(outcome::crc_fail, "cuda") == 1 && book.count(outcome::crc_fail, "cpu") == 0);
  CHECK(book.count(outcome::decoded) + book.count(outcome::crc_fail) == jobs.size());
  CHECK(book.anomalies().empty() && book.held_grids() == 0);
  // Two grids served twelve decodes: each was filled three times and both are free again.
  CHECK(pool.generation(0) + pool.generation(1) == opportunities);
  CHECK(pool.state(0) == isac_replay::grid_state::free && pool.state(1) == isac_replay::grid_state::free);
  CHECK(jobs[8].key.text() == "op-4/cpu" && jobs[9].key.text() == "op-4/cuda" && jobs[9].ingress_sha256 == jobs[8].ingress_sha256);
  CHECK(jobs[8].ingress_sha256 != jobs[6].ingress_sha256);

  // A decode of the plan that was never started has a key and an outcome too, and stays in the count.
  book.skip(key_of("op-6", "cuda"), "grant-6", "record-0", "the run stopped before this decode was started");
  const std::vector<job> all = book.jobs();
  CHECK(all.size() == 2 * opportunities + 1 && all.back().result == outcome::not_dispatched && !all.back().completed);
  CHECK(all.back().key.text() == "op-6/cuda" && all.back().error == "the run stopped before this decode was started");
  CHECK(book.count(outcome::not_dispatched, "cuda") == 1 && book.count(outcome::not_dispatched, "cpu") == 0);
  CHECK(book.count(outcome::decoded) + book.count(outcome::crc_fail) + book.count(outcome::not_dispatched) == all.size());
  // A completion that names it is an anomaly. It does not become a decode.
  book.complete(all.back().token, true, std::string());
  CHECK(has_anomaly(book, "unknown_completion", "op-6/cuda"));
  CHECK(book.snapshot(all.back().token).result == outcome::not_dispatched && !book.snapshot(all.back().token).completed);
  CHECK(book.count(outcome::decoded) == 2 * opportunities - 1 && book.held_grids() == 0);
}

int print_classification(int argc, char** argv)
{
  if (argc != 8) {
    std::cerr << "dispatch_core_test --classify COMPLETED ERROR CRC_OK BYTES_EQUAL GUARD_INTACT INPUT_INTACT" << std::endl;
    return 2;
  }
  bool flags[6];
  for (int index = 0; index != 6; ++index) {
    const std::string word = argv[index + 2];
    if (word != "0" && word != "1") {
      std::cerr << "dispatch_core_test: a flag is 0 or 1" << std::endl;
      return 2;
    }
    flags[index] = word == "1";
  }
  std::cout << to_string(classify(flags[0], flags[1], flags[2], flags[3], flags[4], flags[5])) << std::endl;
  return 0;
}

struct named_case {
  const char*           name;
  std::function<void()> body;
};

const named_case CASES[] = {
    {"compare", case_compare},
    {"decoded", case_decoded},
    {"valid_crc_wrong_byte", case_valid_crc_wrong_byte},
    {"crc_fail", case_crc_fail},
    {"swapped_completions", case_swapped_completions},
    {"decoder_error", case_decoder_error},
    {"duplicate_delivery", case_duplicate_delivery},
    {"silent_timeout", case_silent_timeout},
    {"owned_input", case_owned_input},
    {"changed_input", case_changed_input},
    {"guard", case_guard},
    {"accounting", case_accounting},
};

} // namespace

int main(int argc, char** argv)
{
  const std::string selected = argc > 1 ? argv[1] : "";
  if (selected == "--classify") {
    return print_classification(argc, argv);
  }
  if (selected == "--list") {
    for (const named_case& item : CASES) {
      std::cout << item.name << "\n";
    }
    return 0;
  }
  unsigned ran = 0;
  for (const named_case& item : CASES) {
    if (!selected.empty() && selected != item.name) {
      continue;
    }
    const int before = failures;
    try {
      item.body();
    } catch (const std::exception& error) {
      failed(__FILE__, __LINE__, std::string(item.name) + " threw: " + error.what());
    }
    std::cout << (failures == before ? "PASS  " : "FAIL  ") << item.name << std::endl;
    ++ran;
  }
  if (ran == 0) {
    std::cerr << "dispatch_core_test: no case named '" << selected << "'" << std::endl;
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
