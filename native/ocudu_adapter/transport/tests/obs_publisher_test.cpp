// Unit cases of the producer glue (Sprint 3.2, selector snapshot_ring).
//
// The target links obs_publisher, observer_core, replay_core and obs_ring and nothing else: no
// OCUDU library, no test framework. The grids are the storage model of the channel application
// core, and the consumer is the one of the ring library, in this process.
//
//     obs_publisher_test         run every case
//     obs_publisher_test NAME    run one case (CTest registers each)
//     obs_publisher_test --list  print the case names
//
// alloc_count.cpp replaces the global allocation functions, so that a case can count allocations.
// ASCII only.

#include "obs_publisher.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Allocations of this process so far, counted by the replaced allocation functions of alloc_count.cpp.
std::uint64_t allocation_count();

using namespace isac_transport;
using isac_observer::cf32;
using isac_observer::observation;
using isac_observer::observer;
using isac_observer::pilot;
using isac_observer::slot_context;
using isac_observer::source_spec;
using isac_replay::hash_count;
using isac_replay::model_grid;
using isac_replay::o1_descriptor;
using isac_replay::o1_pool;
using isac_replay::o1_view;
using isac_replay::sha256_count;
using isac_replay::sha256_hex;
using isac_replay::state_error;
using isac_replay::ticket;
using std::chrono::nanoseconds;

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

constexpr unsigned SYMBOLS     = 14;
constexpr unsigned SUBCARRIERS = 24;
constexpr unsigned SLOT        = 7;
constexpr unsigned PLANES      = 2;
const char* const  RECORD      = "record-a";

// A source on symbols 2 and 11 with a comb of step two from subcarrier 4: eight pilots, two planes.
source_spec comb_source()
{
  source_spec spec;
  spec.grant_id               = "grant-a";
  spec.grant_record_sha256    = RECORD;
  spec.source                 = "pusch_dmrs";
  spec.channel                = "PUSCH";
  spec.link_direction         = "uplink";
  spec.processor              = "pusch";
  spec.ports                  = {0};
  spec.grid_symbols           = SYMBOLS;
  spec.grid_subcarriers       = SUBCARRIERS;
  spec.scs_khz                = 30;
  spec.slot                   = SLOT;
  spec.slot_symbols           = {2, 11};
  spec.symbol_useful_start_tc = {145920, 777216};
  spec.tc_per_second          = isac_observer::TC_PER_SECOND;
  spec.amplitude              = 0.70710678118654757;
  for (unsigned symbol : {2U, 11U}) {
    for (unsigned k = 4; k != 12; k += 2) {
      pilot item;
      item.symbol     = symbol;
      item.subcarrier = k;
      spec.index.push_back(item);
    }
  }
  for (std::size_t n = 0; n != spec.index.size(); ++n) {
    spec.reference.push_back(cf32((n % 2) != 0 ? -0.5F : 0.5F, (n % 3) != 0 ? 0.5F : -0.5F));
  }
  spec.antenna_map = {5, 3}; // plane 0 is array element 5
  return spec;
}

ticket make_ready(o1_pool& pool, float seed, evidence mode, unsigned slot = SLOT)
{
  ticket key;
  if (!pool.try_acquire(key)) {
    throw std::runtime_error("the case has no free grid");
  }
  isac_replay::grid_sink& sink = pool.writer(key);
  sink.clear();
  std::vector<cf32> row(SUBCARRIERS);
  for (unsigned plane = 0; plane != PLANES; ++plane) {
    for (unsigned symbol = 0; symbol != SYMBOLS; ++symbol) {
      for (unsigned k = 0; k != SUBCARRIERS; ++k) {
        row[k] = cf32(seed + 0.125F * static_cast<float>(k) + static_cast<float>(plane), seed - 0.25F * static_cast<float>(symbol));
      }
      sink.put(plane, symbol, row.data(), SUBCARRIERS);
    }
  }
  o1_descriptor descriptor;
  descriptor.run_id              = "unit";
  descriptor.opportunity_id      = "op";
  descriptor.grant_id            = "grant-a";
  descriptor.grant_record_sha256 = RECORD;
  descriptor.slot                = slot;
  descriptor.shape.rx_ports      = PLANES;
  descriptor.shape.tx_ports      = 1;
  descriptor.shape.symbols       = SYMBOLS;
  descriptor.shape.subcarriers   = SUBCARRIERS;
  descriptor.grid_gain           = 1.0;
  descriptor.storage             = sink.storage();
  pool.commit(key, descriptor, mode);
  return key;
}

slot_context context(const std::string& opportunity, unsigned sfn = 0)
{
  slot_context item;
  item.opportunity_id = opportunity;
  item.episode_id     = "rz-unit";
  item.sfn            = sfn;
  item.slot           = SLOT;
  item.cell           = 1;
  item.ue             = 0x4601;
  return item;
}

std::string scratch_directory()
{
  const char* base    = std::getenv("TMPDIR");
  std::string pattern = std::string(base != nullptr && *base != '\0' ? base : "/tmp") + "/obs-publisher-test-XXXXXX";
  if (::mkdtemp(&pattern[0]) == nullptr) {
    throw std::runtime_error("cannot create a scratch directory");
  }
  return pattern;
}

/// One ring sized for the comb source, with a consumer attached.
struct ring_fixture {
  std::string                    directory = scratch_directory();
  std::unique_ptr<obs::producer> writer;
  std::unique_ptr<obs::consumer> reader;

  explicit ring_fixture(std::uint32_t slots, std::uint32_t payload_bytes = 0)
  {
    const isac_observer::pilot_source compiled(comb_source());
    ring_needs                        needs;
    needs.include(compiled);
    obs::ring_config config;
    config.socket_path      = directory + "/obs.sock";
    config.slot_count       = slots;
    config.host_capacity    = needs.host_bytes;
    config.mask_capacity    = needs.mask_bytes;
    config.payload_capacity = payload_bytes != 0 ? payload_bytes : needs.payload_bytes;
    writer.reset(new obs::producer(config));
    std::atomic<bool> finished{false};
    std::thread       worker([&]() {
      obs::consumer_config attach;
      attach.socket_path = config.socket_path;
      reader.reset(new obs::consumer(attach));
      finished = true;
    });
    while (!finished) {
      writer->service();
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    worker.join();
  }
  ~ring_fixture()
  {
    reader.reset();
    writer.reset();
    ::rmdir(directory.c_str());
  }
};

// ---------------------------------------------------------------------------------------------- cases

void case_needs()
{
  const isac_observer::pilot_source compiled(comb_source());
  ring_needs                        needs;
  needs.include(compiled);
  // Two planes, two reference symbols, four pilots: 16 complex64 values and 16 validity bits.
  CHECK(needs.payload_bytes == 128 && needs.mask_bytes == 2 && needs.host_bytes == 40 && sizeof(staged_host) == 40);
  ring_fixture ring(4);
  CHECK(ring.writer->header().payload_capacity == 128 && ring.writer->header().slot_bytes % 64 == 0);
}

void case_timed()
{
  ring_fixture ring(4);
  model_grid   first(PLANES, SYMBOLS, SUBCARRIERS), second(PLANES, SYMBOLS, SUBCARRIERS);
  o1_pool      pool({&first, &second});

  // The audited observation of the same values is the reference of the timed one.
  observer audited("unit", 1);
  audited.bind(comb_source());
  const ticket      a    = make_ready(pool, 3.0F, evidence::audit);
  const observation kept = audited.observe(pool.view(a, RECORD), context("op.0000"));
  pool.release(a);

  observer tap("unit", 1, evidence::timed);
  tap.bind(comb_source());
  publisher out(*ring.writer, tap, evidence::timed);
  const std::size_t handle = out.prepare(context("op.0000"), RECORD, 17);
  out.prepare(context("op.0001"), RECORD, 18);
  out.reserve(8);
  const ticket  b    = make_ready(pool, 3.0F, evidence::timed);
  const o1_view view = pool.view(b, RECORD);

  // One observation and one publish: no hash, no allocation and no system call of the ring.
  const hash_count    hashed = sha256_count();
  const std::uint64_t memory = allocation_count();
  const std::uint64_t calls  = obs::sys::calls();
  const published_record& line = out.publish(handle, view);
  CHECK(sha256_count().digests == hashed.digests && sha256_count().updates == hashed.updates &&
        sha256_count().bytes == hashed.bytes);
  CHECK(allocation_count() == memory);
  CHECK(obs::sys::calls() == calls && line.window_calls == 0 && ring.writer->counts().window_calls == 0);
  CHECK(line.outcome == publication::published && line.epoch == 1 && line.sequence == 0 && line.depth == 0);
  CHECK(line.valid == 16 && line.holes == 0 && line.bytes == 256 + 40 + 2 + 128);
  CHECK(line.grid.grid == b.grid && line.grid.generation == b.generation);
  CHECK(line.t_read_ns > 0 && line.t_read_ns <= line.t_ready_ns && line.t_ready_ns <= line.t_publish_ns &&
        line.t_publish_ns <= line.t_done_ns);
  CHECK(line.estimate_us >= 0.0 && line.publish_us >= 0.0 && line.service_us >= line.estimate_us);

  // The consumer reads the record in place: the header of obs/v1 and the bytes of the audited payload.
  obs::record_view got;
  CHECK(ring.reader->next(got, nanoseconds(0)) == obs::read_result::record && got.sequence == 0);
  const obs::record_header& header = *got.header;
  CHECK(header.schema == obs::RECORD_SCHEMA && header.header_bytes == 256 && header.epoch == 1 && header.sequence == 0);
  CHECK(header.episode == kept.episode && header.opportunity == kept.opportunity && header.grant == kept.grant);
  CHECK(header.absolute_slot == kept.absolute_slot && header.cell == 1 && header.ue == 0x4601);
  CHECK(header.stage == obs::STAGE_O2 && header.source == obs::SOURCE_PUSCH_DMRS && header.port == 0 &&
        header.layout == obs::LAYOUT_OCUDU_O2_COMB_C64 && header.dtype == obs::DTYPE_C64 &&
        header.producer_steps == obs::STEP_PILOT_LS && header.scale == kept.scale);
  CHECK(header.rank == 3 && header.shape[0] == 2 && header.shape[1] == 2 && header.shape[2] == 4 && header.shape[3] == 0);
  CHECK(header.strides[0] == 8 && header.strides[1] == 4 && header.strides[2] == 1);
  CHECK(header.first_subcarrier == 4 && header.subcarrier_step == 2 && header.subcarrier_count == 4);
  CHECK(header.dmrs_symbols == ((1U << 2) | (1U << 11)) && header.rx_count == 2 && header.layer_count == 1);
  CHECK(header.antenna_map[0] == 5 && header.antenna_map[1] == 3 && header.antenna_map[2] == 0);
  CHECK(header.mask_bytes == 2 && header.payload_bytes == 128 && header.host_bytes == 40 && header.valid == 16);
  CHECK(header.time_unit == obs::TIME_TC && header.t_first_sample == kept.t_first_sample &&
        header.t_last_sample == kept.t_last_sample);
  CHECK(header.t_ready_ns == line.t_ready_ns && header.t_publish_ns == line.t_publish_ns && header.reserved == 0);
  // Timed: no digest in the record. The evaluator digests the archived payload after the run.
  CHECK(header.evidence == obs::EVIDENCE_TIMED && header.digest_kind == obs::DIGEST_NONE);
  bool zero = true;
  for (std::uint8_t byte : header.digest) {
    zero = zero && byte == 0;
  }
  CHECK(zero);
  CHECK(std::memcmp(got.payload, kept.values.data(), 128) == 0 && std::memcmp(got.mask, kept.mask.data(), 2) == 0);
  CHECK(sha256_hex(got.payload, 128) == kept.payload_sha256); // hashed offline, equal to the audit run
  staged_host host;
  std::memcpy(&host, got.host, sizeof(host));
  CHECK(host.kind == HOST_STAGED_HARNESS && host.opportunity_sequence == 17 && host.grid_index == b.grid &&
        host.grid_generation == b.generation && host.sfn == 0 && host.slot == SLOT && host.t_read_ns >= line.t_read_ns);
  CHECK(ring.reader->intact(got) && reinterpret_cast<std::uintptr_t>(got.payload) % 64 == 0);
  // The record outlives the grid: the grid is released and filled again, the slot is unchanged.
  pool.release(b);
  make_ready(pool, 9.0F, evidence::timed);
  CHECK(std::memcmp(got.payload, kept.values.data(), 128) == 0 && ring.reader->intact(got));
  ring.reader->release(1);
}

void case_audit()
{
  ring_fixture ring(4);
  model_grid   first(PLANES, SYMBOLS, SUBCARRIERS);
  o1_pool      pool({&first});
  observer     tap("unit", 1);
  tap.bind(comb_source());
  publisher out(*ring.writer, tap, evidence::audit);
  const std::size_t handle = out.prepare(context("op.0000"), RECORD, 0);
  const ticket      a      = make_ready(pool, 3.0F, evidence::audit);
  const hash_count  before = sha256_count();
  const observation& made  = tap.observe(pool.view(a, RECORD), context("op.0000"));
  CHECK(sha256_count().digests - before.digests == 6); // the grid, the mask, the payload and three keys
  const published_record& line = out.publish(handle, made);
  CHECK(line.outcome == publication::published && line.sequence == 0 && line.valid == 16);
  CHECK(line.service_us >= made.service_us && line.estimate_us == made.estimate_us && line.publish_us > 0.0);
  obs::record_view got;
  CHECK(ring.reader->next(got, nanoseconds(0)) == obs::read_result::record);
  CHECK(got.header->evidence == obs::EVIDENCE_AUDIT && got.header->digest_kind == obs::DIGEST_SHA256);
  // The digest in the record is the SHA-256 of the payload that lies in the slot.
  const std::string digest = sha256_hex(got.payload, got.header->payload_bytes);
  char              text[65];
  for (unsigned n = 0; n != 32; ++n) {
    std::snprintf(text + 2 * n, 3, "%02x", got.header->digest[n]);
  }
  CHECK(digest == text && digest == made.payload_sha256);
  CHECK(std::memcmp(got.payload, made.values.data(), 128) == 0 && got.header->opportunity == made.opportunity);
  // An observation of another slot than the prepared one is refused, and nothing is acquired.
  const std::size_t other = out.prepare(context("op.0001"), RECORD, 1);
  CHECK_THROWS(out.publish(other, made), input_error, "not the one of its prepared slot");
  CHECK(out.account().size() == 1 && ring.writer->counts().admitted == 1);
  // The two modes are not mixed.
  observer timed("unit", 1, evidence::timed);
  CHECK_THROWS(publisher(*ring.writer, timed, evidence::audit), input_error, "different evidence modes");
}

void case_drop()
{
  ring_fixture ring(2);
  model_grid   first(PLANES, SYMBOLS, SUBCARRIERS), second(PLANES, SYMBOLS, SUBCARRIERS);
  o1_pool      pool({&first, &second});
  observer     tap("unit", 1, evidence::timed);
  tap.bind(comb_source());
  publisher out(*ring.writer, tap, evidence::timed);
  out.reserve(8);
  std::vector<std::size_t> handles;
  for (unsigned n = 0; n != 5; ++n) {
    handles.push_back(out.prepare(context("op.000" + std::to_string(n), n), RECORD, n));
  }
  const ticket  a    = make_ready(pool, 1.0F, evidence::timed);
  const o1_view view = pool.view(a, RECORD);
  CHECK(out.publish(handles[0], view).outcome == publication::published);
  CHECK(out.publish(handles[1], view).outcome == publication::published);
  // No credit: the observation is dropped at once and counted. No estimate is computed, nothing
  // waits, and the grid is as ready for its decode as before.
  const std::int64_t      before  = obs::monotonic_ns();
  const published_record& dropped = out.publish(handles[2], view);
  CHECK(obs::monotonic_ns() - before < 1000000);
  CHECK(dropped.outcome == publication::dropped && std::string(dropped.reason) == "no_credit");
  CHECK(dropped.valid == 0 && dropped.estimate_us == 0.0 && dropped.t_publish_ns == 0);
  CHECK(pool.state(a.grid) == isac_replay::grid_state::ready);
  obs::record_view got;
  CHECK(ring.reader->next(got, nanoseconds(0)) == obs::read_result::record);
  ring.reader->release(1); // exactly one credit comes back
  CHECK(out.publish(handles[3], view).outcome == publication::published);
  CHECK(out.publish(handles[4], view).outcome == publication::dropped);
  // Every eligible observation is in the account once: published or dropped.
  const obs::producer_counts counts = ring.writer->counts();
  CHECK(out.account().size() == 5 && counts.admitted == 3 && counts.published == 3 && counts.dropped == 2);
  CHECK(counts.released == 1 && counts.pending == 2 && counts.max_depth == 1);
  std::size_t published = 0;
  for (const published_record& line : out.account()) {
    published += line.outcome == publication::published ? 1 : 0;
  }
  CHECK(published == 3 && out.account()[3].sequence == 2 && out.account()[3].depth == 1);
}

void case_stale_grid()
{
  ring_fixture ring(4);
  model_grid   first(PLANES, SYMBOLS, SUBCARRIERS);
  o1_pool      pool({&first});
  observer     tap("unit", 1, evidence::timed);
  tap.bind(comb_source());
  publisher out(*ring.writer, tap, evidence::timed);
  out.reserve(4);
  const std::size_t handle = out.prepare(context("op.0000"), RECORD, 0);
  const ticket      a      = make_ready(pool, 1.0F, evidence::timed);
  const o1_view     view   = pool.view(a, RECORD);
  pool.release(a);
  // The grid was given back before the tap: nothing is read, the slot is given back and no
  // record of it exists.
  CHECK_THROWS(out.publish(handle, view), state_error, "is free, not ready");
  CHECK(out.account().empty() && ring.writer->counts().cancelled == 1 && ring.writer->counts().published == 0);
  obs::record_view got;
  CHECK(ring.reader->next(got, nanoseconds(0)) == obs::read_result::none);
  const ticket b = make_ready(pool, 1.0F, evidence::timed);
  CHECK(out.publish(handle, pool.view(b, RECORD)).sequence == 0);
  CHECK(ring.reader->next(got, nanoseconds(0)) == obs::read_result::record && got.sequence == 0);
}

void case_refusals()
{
  ring_fixture ring(2, 64); // a slot that holds half of the payload
  observer     tap("unit", 1, evidence::timed);
  tap.bind(comb_source());
  publisher out(*ring.writer, tap, evidence::timed);
  CHECK_THROWS(out.prepare(context("op.0000"), RECORD, 0), input_error, "cannot carry the observation");
  CHECK_THROWS(out.prepare(context("op.0000"), "record-b", 0), input_error, "changed allocation");
  ring_fixture wide(2);
  observer     named("unit", 1, evidence::timed);
  source_spec  other = comb_source();
  other.source       = "srs";
  named.bind(other);
  publisher unknown(*wide.writer, named, evidence::timed);
  CHECK_THROWS(unknown.prepare(context("op.0000"), RECORD, 0), input_error, "has no number for the sensing source srs");
}

struct named_case {
  const char*           name;
  std::function<void()> body;
};

const named_case CASES[] = {
    {"needs", case_needs},   {"timed", case_timed},           {"audit", case_audit},
    {"drop", case_drop},     {"stale_grid", case_stale_grid}, {"refusals", case_refusals},
};

} // namespace

int main(int argc, char** argv)
{
  const std::string selected = argc > 1 ? argv[1] : "";
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
    std::cerr << "obs_publisher_test: no case named '" << selected << "'" << std::endl;
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
