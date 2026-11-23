// Unit cases of the pacing core (Sprint 4.1, selector mps_pacing).
//
// The target links pacing_core, dispatch_core and replay_core and nothing else: no OCUDU library,
// no test framework. The pacer is driven by a clock of the case. The grids are the storage model
// of the channel application core, and the decoders are doubles written in this file: one that
// answers at once with scripted bytes, one that keeps every request until the case completes it,
// and one that answers from a thread of its own after a given time.
//
//     pacing_core_test            run every case
//     pacing_core_test NAME       run one case (CTest registers each)
//     pacing_core_test --list     print the case names
//     pacing_core_test --trace PERIOD_NS WORK_NS JOBS ALLOWANCE_NS LATEST_START_NS
//                                 print the outcome of every release for a worker that takes
//                                 WORK_NS on each job it runs, for a reference in another
//                                 language to reproduce
//
// ASCII only.

#include "dispatch_core.h"
#include "pacing_core.h"
#include "replay_core.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace isac_pacing;
using isac_receiver::admission;
using isac_receiver::anomaly;
using isac_receiver::completion_sink;
using isac_receiver::decoder;
using isac_receiver::job;
using isac_receiver::job_key;
using isac_receiver::ledger;
using isac_receiver::outcome;
using isac_receiver::payload;
using isac_receiver::request;
using isac_replay::cf32;
using isac_replay::grid_shape;
using isac_replay::model_grid;
using isac_replay::o1_descriptor;
using isac_replay::o1_pool;
using isac_replay::ticket;

std::uint64_t allocation_count();

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

const std::int64_t MS        = 1000000;
const std::int64_t US        = 1000;
const std::int64_t LONG_WAIT = 5000 * MS;

grid_shape small_shape()
{
  grid_shape shape;
  shape.rx_ports    = 2;
  shape.tx_ports    = 1;
  shape.symbols     = 2;
  shape.subcarriers = 6;
  return shape;
}

// Fills a free grid of the pool and makes it ready without a readback digest, as a timed run does.
ticket make_ready(o1_pool& pool, const std::string& record, unsigned seed)
{
  const grid_shape shape = small_shape();
  ticket           key;
  if (!pool.try_acquire(key)) {
    throw std::runtime_error("the case has no free grid");
  }
  isac_replay::grid_sink& sink = pool.writer(key);
  sink.clear();
  std::vector<cf32> row(shape.subcarriers, cf32(static_cast<float>(seed), 0.5F));
  for (unsigned rx = 0; rx != shape.rx_ports; ++rx) {
    for (unsigned symbol = 0; symbol != shape.symbols; ++symbol) {
      sink.put(rx, symbol, row.data(), shape.subcarriers);
    }
  }
  o1_descriptor descriptor;
  descriptor.run_id              = "unit";
  descriptor.opportunity_id      = "op";
  descriptor.grant_id            = "grant-" + record;
  descriptor.grant_record_sha256 = record;
  descriptor.shape               = shape;
  descriptor.grid_gain           = 1.0;
  descriptor.storage             = sink.storage();
  pool.commit(key, descriptor, isac_replay::evidence::timed);
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

  void start(const request& handed, completion_sink& sink) override
  {
    if (raise) {
      throw std::runtime_error("no device");
    }
    std::copy_n(bytes.begin(), std::min(bytes.size(), handed.output_bytes), handed.output);
    std::fill_n(handed.output + handed.output_bytes, overrun, static_cast<std::uint8_t>(0x11));
    for (unsigned n = 0; n != completions; ++n) {
      sink.complete(handed.token, crc_ok, error);
    }
  }
};

// A decoder that keeps every request. The case completes them when it chooses.
struct deferred_decoder final : decoder {
  std::vector<request> pending;
  completion_sink*     sink = nullptr;

  void start(const request& handed, completion_sink& sink_) override
  {
    pending.push_back(handed);
    sink = &sink_;
  }

  void finish(std::size_t index, const std::vector<std::uint8_t>& bytes, bool crc_ok)
  {
    const request& handed = pending.at(index);
    std::copy_n(bytes.begin(), std::min(bytes.size(), handed.output_bytes), handed.output);
    sink->complete(handed.token, crc_ok, std::string());
  }
};

// A decoder with one worker thread, as a processor has: the callback arrives from that thread,
// work_ns after the start. The hand-off is two atomics, so that a case can count allocations.
struct threaded_decoder final : decoder {
  explicit threaded_decoder(std::int64_t work_ns_) : work_ns(work_ns_), worker([this]() { run(); }) {}
  ~threaded_decoder() override
  {
    stop.store(true);
    worker.join();
  }

  void start(const request& handed, completion_sink& sink_) override
  {
    sink    = &sink_;
    current = &handed;
    posted.fetch_add(1, std::memory_order_release);
  }

  void run()
  {
    std::uint64_t seen = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      if (posted.load(std::memory_order_acquire) == seen) {
        continue;
      }
      ++seen;
      const request* handed = current;
      spin_until(monotonic_ns() + work_ns);
      std::copy_n(bytes.begin(), std::min(bytes.size(), handed->output_bytes), handed->output);
      sink->complete(handed->token, true, std::string());
    }
  }

  std::vector<std::uint8_t>  bytes;
  std::int64_t               work_ns;
  std::atomic<std::uint64_t> posted{0};
  std::atomic<bool>          stop{false};
  const request*             current = nullptr;
  completion_sink*           sink    = nullptr;
  std::thread                worker;
};

schedule every(std::int64_t period_ns, std::size_t jobs, std::int64_t allowance_ns, std::int64_t latest_start_ns)
{
  schedule plan;
  for (std::size_t n = 0; n != jobs; ++n) {
    plan.release_ns.push_back(static_cast<std::int64_t>(n) * period_ns);
  }
  plan.allowance_ns    = allowance_ns;
  plan.latest_start_ns = latest_start_ns;
  return plan;
}

// A worker that takes work_ns on each job it runs, on a clock of the case. Its input is ready
// prepare_ns after the worker is free. Returns the time the worker is free at the end.
std::int64_t drive(pacer& pace, std::int64_t start_ns, std::int64_t work_ns, std::int64_t prepare_ns = 0)
{
  std::int64_t now = start_ns;
  pace.begin(start_ns);
  for (std::size_t n = 0; n != pace.size(); ++n) {
    if (!pace.take(n, now)) {
      continue;
    }
    now += prepare_ns;
    pace.ready(n, now);
    now = std::max(now, pace.release_time(n));
    pace.start(n, now);
    now += work_ns;
    pace.done(n, now);
  }
  return now;
}

bool same_job(const job& a, const job& b)
{
  return a.token == b.token && a.sequence == b.sequence && a.key.text() == b.key.text() && a.grid.grid == b.grid.grid &&
         a.grid.generation == b.grid.generation && a.grant_id == b.grant_id &&
         a.grant_record_sha256 == b.grant_record_sha256 && a.ingress_sha256 == b.ingress_sha256 &&
         a.input_intact == b.input_intact && a.payload_id == b.payload_id && a.payload_sha256 == b.payload_sha256 &&
         a.payload_bytes == b.payload_bytes && a.result == b.result && a.started == b.started &&
         a.completed == b.completed && a.crc_ok == b.crc_ok && a.error == b.error && a.guard_intact == b.guard_intact &&
         a.compared.equal == b.compared.equal && a.compared.first_differing_byte == b.compared.first_differing_byte &&
         a.compared.bytes_compared == b.compared.bytes_compared && a.decoded_sha256 == b.decoded_sha256 &&
         a.completions == b.completions;
}

bool has_anomaly(const std::vector<anomaly>& found, const std::string& kind, const std::string& key)
{
  for (const anomaly& item : found) {
    if (item.kind == kind && item.key == key) {
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------------------------- cases

void case_schedule_rules()
{
  schedule plan = every(5 * MS, 4, 2 * MS, 1 * MS);
  check(plan);
  schedule empty = plan;
  empty.release_ns.clear();
  CHECK_THROWS(check(empty), input_error, "at least one release");
  schedule falling = plan;
  falling.release_ns[2] = falling.release_ns[1];
  CHECK_THROWS(check(falling), input_error, "release 2");
  schedule negative = plan;
  negative.release_ns[0] = -1;
  CHECK_THROWS(check(negative), input_error, "release 0");
  schedule none = plan;
  none.allowance_ns = 0;
  CHECK_THROWS(check(none), input_error, "allowance");
  schedule beyond = plan;
  beyond.latest_start_ns = plan.allowance_ns + 1;
  CHECK_THROWS(check(beyond), input_error, "latest start");
  CHECK_THROWS(pacer{beyond}, input_error, "latest start");

  pacer pace(plan);
  CHECK(!pace.begun() && pace.size() == 4);
  CHECK_THROWS(pace.take(0, 0), state_error, "has not begun");
  pace.begin(100);
  CHECK_THROWS(pace.begin(200), state_error, "fixed once");
  CHECK(pace.release_time(0) == 100 && pace.release_time(3) == 100 + 15 * MS);
  CHECK_THROWS(pace.take(1, 100), state_error, "out of order");
  CHECK_THROWS(pace.start(0, 100), state_error, "not prepared");
  CHECK(pace.take(0, 100));
  CHECK_THROWS(pace.done(0, 100), state_error, "not started");
  pace.ready(0, 100);
  pace.start(0, 100);
  CHECK_THROWS(pace.start(0, 100), state_error, "started before");
  pace.done(0, 200);
  CHECK_THROWS(pace.done(0, 300), state_error, "done before");
  CHECK_THROWS(pace.take(4, 0), state_error, "no release 4");
  // A job that was taken and whose input could not be prepared is a skipped release.
  CHECK(pace.take(1, 300));
  pace.give_up(1, 400);
  CHECK(pace.marks()[1].outcome == release_outcome::skipped && pace.marks()[1].free_ns == 400);
  CHECK_THROWS(pace.give_up(1, 500), state_error, "not taken");
  CHECK_THROWS(pace.ready(1, 500), state_error, "not taken");
  CHECK(pace.counts().skipped == 1 && pace.counts().taken == 2 && pace.counts().offered == 4);
}

void case_releases_never_shift()
{
  // The card: a schedule released every 5 ms, a worker that takes 12 ms.
  pacer              pace(every(5 * MS, 7, 6 * MS, 1 * MS));
  const std::int64_t start = 1000 * MS;
  drive(pace, start, 12 * MS);
  const std::vector<release_mark>& marks = pace.marks();
  // No release time moved, and no missed job was deleted: seven marks, each on its offset.
  CHECK(marks.size() == 7);
  for (std::size_t n = 0; n != marks.size(); ++n) {
    CHECK(marks[n].release_ns == start + static_cast<std::int64_t>(n) * 5 * MS);
    CHECK(marks[n].outcome != release_outcome::pending);
  }
  // Job 0 runs from 0 to 12 ms. The releases at 5 and 10 ms were worked through. Job 3 starts on
  // its own release time, 15 ms, and neither at 12 ms nor 12 ms after an earlier start.
  CHECK(marks[0].outcome == release_outcome::on_time && marks[0].start_ns == start && marks[0].overrun);
  CHECK(marks[1].outcome == release_outcome::skipped && marks[1].start_ns == 0 && marks[1].free_ns == start + 12 * MS);
  CHECK(marks[2].outcome == release_outcome::skipped);
  CHECK(marks[3].outcome == release_outcome::on_time && marks[3].start_ns == start + 15 * MS);
  CHECK(marks[3].done_ns == start + 27 * MS && marks[3].overrun);
  CHECK(marks[4].outcome == release_outcome::skipped && marks[5].outcome == release_outcome::skipped);
  CHECK(marks[6].outcome == release_outcome::on_time && marks[6].start_ns == start + 30 * MS);
  const pace_counts counts = pace.counts();
  CHECK(counts.offered == 7 && counts.taken == 7 && counts.on_time == 3 && counts.late == 0 && counts.skipped == 4);
  CHECK(counts.overruns == 3 && counts.max_lateness_ns == 0 && counts.max_excess_ns == 6 * MS);

  // A worker inside its allowance keeps every release.
  pacer fast(every(5 * MS, 7, 6 * MS, 1 * MS));
  drive(fast, start, 3 * MS);
  const pace_counts kept = fast.counts();
  CHECK(kept.on_time == 7 && kept.skipped == 0 && kept.overruns == 0 && kept.max_lateness_ns == 0);
}

void case_late_start()
{
  // Releases every 500 us, 400 us of work and 150 us to prepare the input: the input of every job
  // but the first is ready 50 us after its release. Inside the latest start that is a late start.
  pacer pace(every(500 * US, 4, 600 * US, 100 * US));
  drive(pace, 0, 400 * US, 150 * US);
  const std::vector<release_mark>& marks = pace.marks();
  CHECK(marks[0].outcome == release_outcome::late && marks[0].start_ns == 150 * US); // no time ahead of the first job
  CHECK(marks[1].outcome == release_outcome::late && marks[1].start_ns == 700 * US && marks[1].release_ns == 500 * US);
  CHECK(marks[1].done_ns == 1100 * US && !marks[1].overrun);
  // The lateness accumulates until a release is skipped, and the job after it is on time again.
  CHECK(marks[2].outcome == release_outcome::late && marks[2].start_ns == 1250 * US && marks[2].overrun);
  CHECK(marks[3].outcome == release_outcome::skipped && marks[3].free_ns == 1650 * US);
  const pace_counts counts = pace.counts();
  CHECK(counts.late == 3 && counts.skipped == 1 && counts.on_time == 0 && counts.overruns == 1);
  CHECK(counts.max_lateness_ns == 250 * US && counts.max_excess_ns == 50 * US);

  // The boundary: free exactly at release + latest start runs, one nanosecond later is skipped.
  pacer edge(every(500 * US, 2, 600 * US, 100 * US));
  edge.begin(0);
  CHECK(edge.take(0, 100 * US));
  pacer past(every(500 * US, 2, 600 * US, 100 * US));
  past.begin(0);
  CHECK(!past.take(0, 100 * US + 1));
  CHECK(past.take(1, 100 * US + 1)); // the job after a skipped release is offered on its own time
  // Done exactly at release + allowance is inside the allowance.
  edge.ready(0, 100 * US);
  edge.start(0, 100 * US);
  edge.done(0, 600 * US);
  CHECK(!edge.marks()[0].overrun);
}

void case_book_decoded()
{
  model_grid first(2, 2, 6), second(2, 2, 6);
  o1_pool    pool({&first}), control_pool({&second});
  paced_book book(pool);
  ledger     control(control_pool, isac_replay::evidence::timed);
  const payload original = block("grant-a", 48, 21);
  scripted_decoder decoder;
  decoder.bytes = original.bytes;

  const std::uint64_t token = book.prepare(key_of("op-0", "cuda"), "grant-record-a", "record-a", original);
  CHECK(token == 1 && book.size() == 1);
  CHECK_THROWS(book.describe(token), state_error, "never dispatched");
  const isac_replay::hash_count before = isac_replay::sha256_count();
  const ticket                  ready  = make_ready(pool, "record-a", 21);
  const paced_result            done   = paced_dispatch(book, decoder, token, ready, LONG_WAIT);
  CHECK(done.admitted == admission::admitted && done.result == outcome::decoded);
  CHECK(book.holds(ready) == 0 && book.held_grids() == 0 && book.release_grid(ready));
  const isac_replay::hash_count after = isac_replay::sha256_count();
  CHECK(after.digests == before.digests && after.bytes == before.bytes); // nothing of a decode is hashed
  CHECK(book.decoded(token) == original.bytes && book.count(outcome::decoded) == 1 && book.count(outcome::decoded, "cpu") == 0);

  // The same decode through the ledger of Sprint 2.4 in the mode timed: the same record.
  const ticket                     again   = make_ready(control_pool, "record-a", 21);
  const isac_receiver::dispatched  through = dispatch(control, decoder, key_of("op-0", "cuda"), again, "record-a", original,
                                                      std::chrono::milliseconds(5000));
  CHECK(same_job(book.snapshot(token), control.snapshot(through.token)));
  const job record = book.snapshot(token);
  CHECK(record.input_intact && record.guard_intact && record.compared.equal && record.crc_ok && record.completions == 1);
  CHECK(record.ingress_sha256.empty() && record.decoded_sha256.empty() && record.grant_id == "grant-record-a");
  CHECK(book.started_ns(token) != 0 && book.completed_ns(token) >= book.started_ns(token) && record.service_us >= 0.0);
  CHECK(book.anomalies().empty());
}

void case_book_outcomes()
{
  model_grid first(2, 2, 6), second(2, 2, 6);
  o1_pool    pool({&first}), control_pool({&second});
  paced_book book(pool);
  ledger     control(control_pool, isac_replay::evidence::timed);
  const payload original = block("grant-a", 48, 5);
  const char*   names[]  = {"mismatch", "crc", "error", "raise", "guard", "silent", "skipped", "stopped", "unreached"};
  std::vector<std::uint64_t> tokens;
  for (const char* name : names) {
    tokens.push_back(book.prepare(key_of(name, "cuda"), "grant-record-a", "record-a", original));
  }
  auto both = [&](std::size_t index, scripted_decoder& decoder, outcome wanted) {
    const ticket       ready = make_ready(pool, "record-a", 5);
    const paced_result done  = paced_dispatch(book, decoder, tokens[index], ready, LONG_WAIT);
    CHECK(done.result == wanted && book.release_grid(ready));
    const ticket                    again   = make_ready(control_pool, "record-a", 5);
    const isac_receiver::dispatched through = dispatch(control, decoder, key_of(names[index], "cuda"), again, "record-a",
                                                       original, std::chrono::milliseconds(5000));
    CHECK(through.result == wanted && control.release_grid(again));
    CHECK(same_job(book.snapshot(tokens[index]), control.snapshot(through.token)));
  };
  scripted_decoder wrong;
  wrong.bytes = original.bytes;
  wrong.bytes[5] ^= 0x40;
  both(0, wrong, outcome::payload_mismatch);
  CHECK(book.snapshot(tokens[0]).compared.first_differing_byte == 5);
  scripted_decoder failing;
  failing.bytes  = original.bytes;
  failing.crc_ok = false;
  both(1, failing, outcome::crc_fail);
  scripted_decoder broken;
  broken.bytes = original.bytes;
  broken.error = "the device was lost";
  both(2, broken, outcome::decoder_error);
  scripted_decoder raising;
  raising.raise = true;
  both(3, raising, outcome::decoder_error);
  CHECK(book.snapshot(tokens[3]).error == "the decoder raised: no device");
  scripted_decoder spilling;
  spilling.bytes   = original.bytes;
  spilling.overrun = 3;
  both(4, spilling, outcome::decoder_error);
  CHECK(book.snapshot(tokens[4]).error == "the decoder wrote past the transport block");

  // A decoder that never calls back: a timeout that keeps its grid held, and a late completion
  // that is recorded and frees the grid without changing the outcome.
  deferred_decoder   silent;
  const ticket       held = make_ready(pool, "record-a", 5);
  const paced_result late = paced_dispatch(book, silent, tokens[5], held, 2 * MS);
  CHECK(late.result == outcome::timeout && book.holds(held) == 1 && book.held_grids() == 1 && !book.release_grid(held));
  CHECK_THROWS(book.decoded(tokens[5]), state_error, "no completed result");
  silent.finish(0, original.bytes, true);
  silent.finish(0, original.bytes, true);
  CHECK(book.resolve(tokens[5]) == outcome::timeout && book.holds(held) == 0 && book.release_grid(held));
  const job timed_out = book.snapshot(tokens[5]);
  CHECK(timed_out.result == outcome::timeout && !timed_out.completed && timed_out.completions == 2);
  CHECK(timed_out.error == "no completion within the declared time");
  const std::vector<anomaly> found = book.anomalies();
  CHECK(has_anomaly(found, "late_completion", "silent/cuda") && has_anomaly(found, "duplicate_completion", "silent/cuda"));

  // Decodes that were never started keep their key and their place in every count.
  book.skip(tokens[6], skip_reason::missed_release, 730 * US);
  book.skip(tokens[7], skip_reason::backend_stopped);
  const job missed = book.snapshot(tokens[6]);
  CHECK(missed.result == outcome::not_dispatched && !missed.started && book.skipped(tokens[6]) == skip_reason::missed_release);
  CHECK(missed.error.find("skipped release") != std::string::npos && missed.error.find("730 us") != std::string::npos);
  CHECK(book.snapshot(tokens[7]).error.find("stopped") != std::string::npos);
  CHECK(book.snapshot(tokens[8]).result == outcome::not_dispatched);
  CHECK(book.snapshot(tokens[8]).error == "the run stopped before this decode was started");
  CHECK(book.count(outcome::not_dispatched) == 3 && book.count(outcome::timeout) == 1 &&
        book.count(outcome::decoder_error) == 3 && book.jobs().size() == 9);
  // A completion for a decode that was never started, and one for a job the book does not hold.
  book.complete(tokens[6], true, std::string());
  book.complete(99, true, std::string());
  CHECK(has_anomaly(book.anomalies(), "unknown_completion", "skipped/cuda") &&
        has_anomaly(book.anomalies(), "unknown_completion", ""));
  CHECK(book.snapshot(tokens[6]).result == outcome::not_dispatched);
}

void case_book_refusals()
{
  model_grid grid(2, 2, 6);
  o1_pool    pool({&grid});
  paced_book book(pool);
  const payload original = block("grant-a", 48, 9);
  payload       nothing  = original;
  nothing.bytes.clear();
  CHECK_THROWS(book.prepare(key_of("", "cuda"), "g", "record-a", original), input_error, "identifier");
  CHECK_THROWS(book.prepare(key_of("op-0", "cuda"), "g", "record-a", nothing), input_error, "no original transport block");
  const std::uint64_t first  = book.prepare(key_of("op-0", "cuda"), "g", "record-a", original);
  const std::uint64_t second = book.prepare(key_of("op-1", "cuda"), "g", "record-a", original);
  const std::uint64_t third  = book.prepare(key_of("op-2", "cuda"), "g", "record-a", original);
  CHECK_THROWS(book.prepare(key_of("op-1", "cuda"), "g", "record-a", original), input_error, "duplicate delivery");
  CHECK_THROWS(book.admit(7, ticket()), state_error, "no job 7");
  scripted_decoder decoder;
  decoder.bytes = original.bytes;
  std::string why;

  // Ownership by generation: a stale ticket and another grant record are refused.
  const ticket ready = make_ready(pool, "record-a", 9);
  CHECK(paced_dispatch(book, decoder, first, ready, LONG_WAIT).result == outcome::decoded);
  CHECK(book.admit(first, ready, &why) == admission::duplicate_delivery && why.find("admitted or skipped before") != std::string::npos);
  CHECK(book.release_grid(ready));
  CHECK(book.admit(second, ready, &why) == admission::stale_grid);
  const ticket other = make_ready(pool, "record-b", 10);
  CHECK(other.generation == 2);
  CHECK(book.admit(second, ready, &why) == admission::stale_grid && why.find("stale generation 1") != std::string::npos);
  CHECK(book.admit(second, other, &why) == admission::changed_allocation);
  // A refused decode can still be admitted on the grid of its own grant record, or be skipped.
  pool.release(other);
  const ticket right = make_ready(pool, "record-a", 11);
  CHECK(paced_dispatch(book, decoder, second, right, LONG_WAIT).result == outcome::decoded && book.release_grid(right));
  book.skip(third, skip_reason::no_free_grid);
  book.skip(third, skip_reason::no_free_grid);
  CHECK(book.admit(third, right, &why) == admission::duplicate_delivery);
  const std::vector<anomaly> found = book.anomalies();
  CHECK(has_anomaly(found, "refused_grid", "op-1/cuda") && has_anomaly(found, "duplicate_delivery", "op-0/cuda") &&
        has_anomaly(found, "duplicate_delivery", "op-2/cuda"));
  CHECK(book.snapshot(third).error.find("no free grid") != std::string::npos);
  // A grid that was given back and filled again while its decode ran is not the admitted input.
  model_grid one(2, 2, 6);
  o1_pool    small({&one});
  paced_book racing(small);
  const std::uint64_t token = racing.prepare(key_of("op-0", "cuda"), "g", "record-a", original);
  deferred_decoder    slow;
  const ticket        input = make_ready(small, "record-a", 12);
  CHECK(racing.admit(token, input) == admission::admitted);
  racing.started(token);
  slow.start(racing.describe(token), racing);
  small.release(input); // behind the back of the book
  make_ready(small, "record-a", 13);
  slow.finish(0, original.bytes, true);
  CHECK(racing.wait(token, LONG_WAIT) && racing.resolve(token) == outcome::decoder_error);
  CHECK(!racing.snapshot(token).input_intact && racing.snapshot(token).error == "the grid changed while it was decoded");
}

void case_book_no_allocation()
{
  model_grid first(2, 2, 6), second(2, 2, 6);
  o1_pool    pool({&first, &second});
  paced_book book(pool);
  const payload original = block("grant-a", 1217, 3);
  const std::size_t jobs = 64;
  std::vector<std::uint64_t> tokens;
  for (std::size_t n = 0; n != jobs; ++n) {
    tokens.push_back(book.prepare(key_of("op-" + std::to_string(n), "cuda"), "grant-record-a", "record-a", original));
  }
  threaded_decoder decoder(20 * US);
  decoder.bytes = original.bytes;
  pacer pace(every(200 * US, jobs, 100 * MS, 100 * MS));
  std::vector<ticket> keys(jobs);
  // Every grid is filled outside the counted window: the fill is stimulus. From the admission of
  // a ready grid to its release the paced path allocates nothing.
  std::uint64_t allocated = 0;
  std::size_t   decoded   = 0;
  pace.begin(monotonic_ns() + 1 * MS);
  for (std::size_t n = 0; n != jobs; ++n) {
    if (!pace.take(n, monotonic_ns())) {
      book.skip(tokens[n], skip_reason::missed_release);
      continue;
    }
    keys[n] = make_ready(pool, "record-a", static_cast<unsigned>(n));
    pace.ready(n, monotonic_ns());
    const std::uint64_t memory = allocation_count();
    pace.start(n, spin_until(pace.release_time(n)));
    const paced_result done = paced_dispatch(book, decoder, tokens[n], keys[n], LONG_WAIT);
    pace.done(n, monotonic_ns());
    const bool released = book.release_grid(keys[n]);
    allocated += allocation_count() - memory;
    decoded += done.result == outcome::decoded && released ? 1 : 0;
  }
  const pace_counts counts = pace.counts();
  CHECK(allocated == 0);
  CHECK(decoded + counts.skipped == jobs && counts.taken == jobs && decoded >= 1);
  CHECK(book.count(outcome::decoded) == decoded && book.anomalies().empty() && book.held_grids() == 0);
  // The callbacks came from the worker, after the declared work, and no job started before its release.
  for (std::size_t n = 0; n != jobs; ++n) {
    const release_mark& mark = pace.marks()[n];
    if (mark.outcome == release_outcome::skipped) {
      continue;
    }
    CHECK(mark.start_ns >= mark.release_ns && mark.done_ns >= mark.start_ns);
    CHECK(book.started_ns(tokens[n]) >= mark.start_ns && book.completed_ns(tokens[n]) <= mark.done_ns);
    CHECK(book.snapshot(tokens[n]).service_us >= 20.0);
  }
}

void case_spin_until()
{
  const std::int64_t before = monotonic_ns();
  const std::int64_t seen   = spin_until(before + 300 * US);
  CHECK(seen >= before + 300 * US && monotonic_ns() >= seen);
  CHECK(spin_until(before) >= before); // a time that has passed returns at once
}

int print_trace(int argc, char** argv)
{
  if (argc != 7) {
    std::cerr << "pacing_core_test --trace PERIOD_NS WORK_NS JOBS ALLOWANCE_NS LATEST_START_NS" << std::endl;
    return 2;
  }
  const std::int64_t period = std::atoll(argv[2]), work = std::atoll(argv[3]);
  pacer              pace(every(period, static_cast<std::size_t>(std::atoll(argv[4])), std::atoll(argv[5]), std::atoll(argv[6])));
  drive(pace, 0, work);
  for (const release_mark& mark : pace.marks()) {
    std::cout << to_string(mark.outcome) << " " << mark.release_ns << " " << mark.start_ns << " " << mark.done_ns << " "
              << (mark.overrun ? 1 : 0) << "\n";
  }
  return 0;
}

struct named_case {
  const char*           name;
  std::function<void()> body;
};

const named_case CASES[] = {
    {"schedule_rules", case_schedule_rules},
    {"releases_never_shift", case_releases_never_shift},
    {"late_start", case_late_start},
    {"book_decoded", case_book_decoded},
    {"book_outcomes", case_book_outcomes},
    {"book_refusals", case_book_refusals},
    {"book_no_allocation", case_book_no_allocation},
    {"spin_until", case_spin_until},
};

} // namespace

int main(int argc, char** argv)
{
  const std::string selected = argc > 1 ? argv[1] : "";
  if (selected == "--trace") {
    return print_trace(argc, argv);
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
    std::cerr << "pacing_core_test: no case named '" << selected << "'" << std::endl;
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
