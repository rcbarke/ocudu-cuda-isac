// Unit cases of the observation ring (Sprint 3.2, selector snapshot_ring).
//
// The target links obs_ring and the thread library and nothing else: no OCUDU, no CUDA, no test
// framework. Producer and consumer run in one process here, on two threads where a case needs it.
//
//     obs_ring_test             run every case
//     obs_ring_test NAME        run one case (CTest registers each)
//     obs_ring_test --list      print the case names
//     obs_ring_test --definition
//                               print the hash and the text of the record and ring definition,
//                               for the reference reader in another language
//
// ASCII only.

#include "obs/ring.h"
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

using namespace obs;
using std::chrono::milliseconds;
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

constexpr std::uint32_t PAYLOAD = 256;
constexpr std::uint32_t MASK    = 32;
constexpr std::uint32_t HOST    = 16;

std::string scratch_directory()
{
  const char* base = std::getenv("TMPDIR");
  std::string pattern = std::string(base != nullptr && *base != '\0' ? base : "/tmp") + "/obs-ring-test-XXXXXX";
  const char* made    = ::mkdtemp(&pattern[0]);
  if (made == nullptr) {
    throw std::runtime_error("cannot create a scratch directory");
  }
  return made;
}

struct ring_fixture {
  std::string               directory = scratch_directory();
  std::string               path      = directory + "/obs.sock";
  std::unique_ptr<producer> writer;

  std::string nested;

  explicit ring_fixture(std::uint32_t slots, const std::string& subdirectory = std::string())
  {
    if (!subdirectory.empty()) {
      nested = directory + "/" + subdirectory;
      if (::mkdir(nested.c_str(), 0700) != 0) {
        throw std::runtime_error("cannot create a nested scratch directory");
      }
      path = nested + "/obs.sock";
    }
    ring_config config;
    config.socket_path      = path;
    config.slot_count       = slots;
    config.host_capacity    = HOST;
    config.mask_capacity    = MASK;
    config.payload_capacity = PAYLOAD;
    writer.reset(new producer(config));
  }
  ~ring_fixture()
  {
    writer.reset();
    if (!nested.empty()) {
      ::rmdir(nested.c_str());
    }
    ::rmdir(directory.c_str());
  }

  /// Attaches a consumer while this thread serves the control channel.
  std::unique_ptr<consumer> attach(wait_mode mode = wait_mode::spin, std::uint64_t definition = DEFINITION,
                                   std::string* problem = nullptr, int limit_ms = 2000)
  {
    std::unique_ptr<consumer> reader;
    std::atomic<bool>         finished{false};
    std::thread               worker([&]() {
      consumer_config config;
      config.socket_path   = path;
      config.mode          = mode;
      config.connect_limit = milliseconds(limit_ms);
      try {
        reader.reset(new consumer(config, definition));
      } catch (const ring_error& error) {
        if (problem != nullptr) {
          *problem = error.what();
        }
      }
      finished = true;
    });
    while (!finished) {
      writer->service();
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    worker.join();
    writer->service();
    return reader;
  }
};

/// Writes one record with a payload that names its tag, in place, and publishes it.
bool put(producer& writer, std::uint8_t tag, std::uint64_t* sequence = nullptr)
{
  slot acquired;
  if (!writer.try_acquire(acquired)) {
    return false;
  }
  std::memset(acquired.header, 0, sizeof(record_header));
  acquired.header->schema        = RECORD_SCHEMA;
  acquired.header->header_bytes  = RECORD_HEADER_BYTES;
  acquired.header->stage         = STAGE_O2;
  acquired.header->layout        = LAYOUT_OCUDU_O2_COMB_C64;
  acquired.header->dtype         = DTYPE_C64;
  acquired.header->payload_bytes = PAYLOAD;
  acquired.header->mask_bytes    = MASK;
  acquired.header->host_bytes    = HOST;
  acquired.header->opportunity   = tag;
  acquired.header->t_ready_ns    = monotonic_ns();
  std::memset(acquired.host, tag ^ 0x55, HOST);
  std::memset(acquired.mask, 0xff, MASK);
  std::memset(acquired.payload, tag, PAYLOAD);
  if (sequence != nullptr) {
    *sequence = acquired.sequence;
  }
  writer.publish(acquired);
  return true;
}

bool holds(const record_view& view, std::uint8_t tag)
{
  for (std::uint32_t n = 0; n != PAYLOAD; ++n) {
    if (view.payload[n] != tag) {
      return false;
    }
  }
  return view.header->opportunity == tag && view.header->payload_bytes == PAYLOAD && view.host[0] == (tag ^ 0x55) &&
         view.mask[MASK - 1] == 0xff;
}

// ---------------------------------------------------------------------------------------------- cases

void case_definition()
{
  CHECK(sizeof(record_header) == 256 && sizeof(segment_header) == 4096 && sizeof(cursor_page) == 4096);
  CHECK(sizeof(slot_prefix) == 64 && sizeof(control_message) == 64);
  CHECK(DEFINITION == fnv1a(RING_DEFINITION, fnv1a(RECORD_DEFINITION)));
  CHECK(fnv1a("") == 14695981039346656037ULL && fnv1a("a") == 0xaf63dc4c8601ec8cULL);
  CHECK(slots_for(3, 3) == 8 && slots_for(1, 1) == 4 && slots_for(0, 0) == 1 && slots_for(3, 4) == 8 &&
        slots_for(4, 4) == 16);
  ring_fixture ring(8);
  const segment_header& header = ring.writer->header();
  CHECK(std::memcmp(header.magic, RING_MAGIC, 8) == 0 && header.layout == RING_LAYOUT && header.definition == DEFINITION);
  CHECK(header.slot_count == 8 && header.host_offset == 320 && header.mask_offset == 336 && header.payload_offset == 384);
  CHECK(header.slot_bytes == 640 && header.slot_bytes % 64 == 0 && header.payload_offset % 64 == 0);
  CHECK(ring.writer->segment_bytes() == 4096 + 8 * 640 && header.producer_pid == static_cast<std::uint32_t>(::getpid()));
  // A slot count that is not a power of two and a ring without a payload are refused.
  ring_config bad;
  bad.slot_count       = 6;
  bad.payload_capacity = 8;
  bool refused         = false;
  try {
    producer writer(bad);
  } catch (const ring_error&) {
    refused = true;
  }
  CHECK(refused);
}

void case_capacity_two()
{
  // No consumer at all: A and B are admitted, C is dropped at once and counted.
  {
    ring_fixture ring(2);
    CHECK(put(*ring.writer, 'A') && put(*ring.writer, 'B'));
    const std::int64_t before = monotonic_ns();
    CHECK(!put(*ring.writer, 'C'));
    CHECK(monotonic_ns() - before < 1000000); // the producer returned without waiting
    const producer_counts counts = ring.writer->counts();
    CHECK(counts.admitted == 2 && counts.published == 2 && counts.dropped == 1 && counts.pending == 2);
    CHECK(ring.writer->header().dropped.load() == 1 && ring.writer->header().published.load() == 2);
  }
  // A consumer that has released nothing yet: the same, then release A and admit D.
  ring_fixture              ring(2);
  std::unique_ptr<consumer> reader = ring.attach();
  CHECK(reader != nullptr && ring.writer->attached() && ring.writer->epoch() == 1);
  CHECK(put(*ring.writer, 'A') && put(*ring.writer, 'B') && !put(*ring.writer, 'C'));
  record_view view;
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record && holds(view, 'A') && view.sequence == 0);
  CHECK(!put(*ring.writer, 'C')); // read is not released: the credit is still held
  CHECK(reader->intact(view));
  reader->release(1);
  CHECK(put(*ring.writer, 'D'));
  const producer_counts counts = ring.writer->counts();
  CHECK(counts.admitted == 3 && counts.dropped == 2 && counts.published == 3 && counts.released == 1);
  CHECK(counts.pending == 2 && counts.max_depth <= 1); // at most two retained
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record && holds(view, 'B') && view.sequence == 1);
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record && holds(view, 'D') && view.sequence == 2);
  CHECK(reader->next(view, nanoseconds(0)) == read_result::none);
  reader->release(3);
  CHECK(ring.writer->drain(milliseconds(100)) == 0);
}

void case_scratch_overwrite()
{
  ring_fixture              ring(2);
  std::unique_ptr<consumer> reader = ring.attach();
  std::vector<std::uint8_t> scratch(PAYLOAD, 'B');
  CHECK(put(*ring.writer, 'A'));
  slot acquired;
  CHECK(ring.writer->try_acquire(acquired));
  std::memset(acquired.header, 0, sizeof(record_header));
  acquired.header->header_bytes  = RECORD_HEADER_BYTES;
  acquired.header->payload_bytes = PAYLOAD;
  acquired.header->opportunity   = 'B';
  std::memcpy(acquired.payload, scratch.data(), PAYLOAD);
  ring.writer->publish(acquired);
  std::fill(scratch.begin(), scratch.end(), 'X'); // the producer reuses its scratch buffer
  record_view view;
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record);
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record && view.sequence == 1);
  bool unchanged = true;
  for (std::uint32_t n = 0; n != PAYLOAD; ++n) {
    unchanged = unchanged && view.payload[n] == 'B';
  }
  CHECK(unchanged && reader->intact(view));
  // The stamps of the ring: the depth the record met, its bytes and both completion points.
  CHECK(view.depth == 1 && view.prefix->record_bytes == RECORD_HEADER_BYTES + PAYLOAD);
  CHECK(view.header->epoch == 1 && view.header->sequence == 1 && view.header->t_publish_ns > 0);
}

void case_no_early_read()
{
  ring_fixture              ring(2);
  std::unique_ptr<consumer> reader = ring.attach();
  slot                      acquired;
  CHECK(ring.writer->try_acquire(acquired));
  record_view view;
  // Acquired and being written: nothing is published, so nothing can be read.
  CHECK(reader->next(view, nanoseconds(2000000)) == read_result::none && reader->counts().read == 0);
  CHECK((acquired.prefix->generation.load() & 1U) == 1U);
  std::memset(acquired.header, 0, sizeof(record_header));
  acquired.header->header_bytes = RECORD_HEADER_BYTES;
  acquired.header->opportunity  = 'R';
  acquired.header->t_ready_ns   = monotonic_ns(); // the completion event of the copy, delayed
  ring.writer->publish(acquired);
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record && view.header->opportunity == 'R');
  CHECK(view.header->t_publish_ns >= view.header->t_ready_ns && (view.generation & 1U) == 0);
  // A cancelled slot holds no record and costs no credit.
  CHECK(ring.writer->try_acquire(acquired));
  ring.writer->cancel(acquired);
  CHECK(reader->next(view, nanoseconds(0)) == read_result::none && ring.writer->counts().cancelled == 1);
  CHECK(put(*ring.writer, 'S') && reader->next(view, nanoseconds(0)) == read_result::record && view.sequence == 1 &&
        holds(view, 'S'));
}

void case_stale_read()
{
  ring_fixture              ring(1);
  std::unique_ptr<consumer> reader = ring.attach();
  CHECK(put(*ring.writer, 'A'));
  record_view first;
  CHECK(reader->next(first, nanoseconds(0)) == read_result::record && holds(first, 'A'));
  reader->release(1); // released before the work on it was complete: a faulty consumer
  CHECK(put(*ring.writer, 'B')); // the one slot is rewritten
  CHECK(!reader->intact(first) && reader->counts().stale == 1); // the read straddled the rewrite
  // A slot whose generation is odd, or that holds another sequence, is not delivered.
  record_view second;
  CHECK(reader->next(second, nanoseconds(0)) == read_result::record && holds(second, 'B') && reader->intact(second));
  slot acquired;
  reader->release(2);
  CHECK(ring.writer->try_acquire(acquired));
  CHECK(!reader->intact(second) && reader->counts().stale == 2);
  ring.writer->cancel(acquired);
}

void case_double_release()
{
  ring_fixture              ring(2);
  std::unique_ptr<consumer> reader = ring.attach();
  CHECK(put(*ring.writer, 'A') && put(*ring.writer, 'B'));
  record_view view;
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record);
  reader->release(1);
  reader->release(1);                   // the same count once more
  reader->page()->released.store(1);    // and stored a third time, by hand
  CHECK(put(*ring.writer, 'C'));        // one credit came back
  CHECK(!put(*ring.writer, 'D'));       // and only one
  CHECK(ring.writer->counts().released == 1 && reader->counts().released == 1 && ring.writer->counts().faults == 0);
  // A release beyond what was read is not stored by the library.
  reader->release(40);
  CHECK(reader->page()->released.load() == 1);
}

void case_released_fault()
{
  // The count passes the published count.
  {
    ring_fixture              ring(2);
    std::unique_ptr<consumer> reader = ring.attach();
    CHECK(put(*ring.writer, 'A') && put(*ring.writer, 'B'));
    reader->page()->released.store(7);
    CHECK(!put(*ring.writer, 'C')); // no slot is reused on the strength of that count
    CHECK(ring.writer->counts().faults == 1 && ring.writer->counts().dropped == 1);
    ring.writer->service(); // the epoch is closed outside the window
    CHECK(!ring.writer->attached() && ring.writer->header().faults.load() == 1);
    record_view view;
    CHECK(reader->next(view, nanoseconds(0)) == read_result::record); // what was published stays readable
    CHECK(reader->producer_done());
  }
  // The count moves backwards.
  ring_fixture              ring(4);
  std::unique_ptr<consumer> reader = ring.attach();
  CHECK(put(*ring.writer, 'A') && put(*ring.writer, 'B') && put(*ring.writer, 'C'));
  record_view view;
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record);
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record);
  reader->release(2);
  CHECK(put(*ring.writer, 'D'));
  reader->page()->released.store(1);
  CHECK(put(*ring.writer, 'E')); // the last accepted count stands: 2 released, 4 slots
  CHECK(ring.writer->counts().faults == 1);
  ring.writer->service();
  CHECK(!ring.writer->attached() && ring.writer->counts().released == 2 && ring.writer->counts().pending == 3);
}

void case_fifo()
{
  ring_fixture              ring(8);
  std::unique_ptr<consumer> reader = ring.attach();
  for (std::uint8_t tag = 0; tag != 5; ++tag) {
    CHECK(put(*ring.writer, tag));
  }
  record_view view;
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record);
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record);
  reader->release(2);
  std::uint64_t first = 0;
  CHECK(put(*ring.writer, 100, &first) && put(*ring.writer, 101) && put(*ring.writer, 102) && first == 5);
  const std::uint8_t* base = nullptr;
  for (std::uint64_t n = 2; n != 8; ++n) {
    CHECK(reader->next(view, nanoseconds(0)) == read_result::record && view.sequence == n && view.header->sequence == n);
    CHECK(holds(view, n < 5 ? static_cast<std::uint8_t>(n) : static_cast<std::uint8_t>(100 + n - 5)));
    if (n == 5) {
      base = reinterpret_cast<const std::uint8_t*>(view.prefix);
    }
    if (n > 5) {
      // Consecutive sequences lie in consecutive slots.
      CHECK(reinterpret_cast<const std::uint8_t*>(view.prefix) == base + (n - 5) * reader->header().slot_bytes);
    }
  }
  // Sequence 8 wraps to slot 0.
  reader->release(8);
  CHECK(put(*ring.writer, 9) && reader->next(view, nanoseconds(0)) == read_result::record && view.sequence == 8);
  CHECK(reinterpret_cast<const std::uint8_t*>(view.prefix) ==
        reinterpret_cast<const std::uint8_t*>(&reader->header()) + SEGMENT_HEADER_BYTES);
}

void case_spin_no_syscalls()
{
  ring_fixture              ring(8);
  std::unique_ptr<consumer> reader = ring.attach(wait_mode::spin);
  constexpr unsigned        RECORDS = 1000;
  std::atomic<unsigned>     seen{0};
  std::atomic<bool>         ordered{true};
  std::uint64_t             reader_calls = 0;
  std::thread               worker([&]() {
    const std::uint64_t before = sys::calls();
    record_view         view;
    for (unsigned n = 0; n != RECORDS; ++n) {
      const read_result got = reader->next(view, nanoseconds(2000000000LL));
      if (got != read_result::record || view.sequence != n || !holds(view, static_cast<std::uint8_t>(n)) ||
          !reader->intact(view)) {
        ordered = false;
        break;
      }
      reader->release(n + 1);
      ++seen;
    }
    reader_calls = sys::calls() - before;
  });
  const std::uint64_t before = sys::calls();
  unsigned            dropped = 0;
  for (unsigned n = 0; n != RECORDS;) {
    if (put(*ring.writer, static_cast<std::uint8_t>(n))) {
      ++n;
    } else {
      ++dropped; // the consumer is behind: the producer does not wait, the test offers the record again
      std::this_thread::yield();
    }
  }
  const std::uint64_t producer_calls = sys::calls() - before;
  worker.join();
  const producer_counts counts = ring.writer->counts();
  CHECK(ordered && seen == RECORDS && counts.published == RECORDS && counts.dropped == dropped);
  // No system call of the library between any try_acquire and its publication store, none at
  // all in the producer's loop, and none in the spinning consumer.
  CHECK(counts.window_calls == 0 && counts.wakes == 0 && producer_calls == 0 && reader_calls == 0);
  CHECK(ring.writer->header().window_calls.load() == 0 && ring.writer->consumer_mode() == wait_mode::spin);
}

void case_wait_flag()
{
  ring_fixture              ring(8);
  std::unique_ptr<consumer> reader = ring.attach(wait_mode::wait);
  CHECK(ring.writer->consumer_mode() == wait_mode::wait);
  CHECK(put(*ring.writer, 1) && ring.writer->counts().wakes == 0); // the flag is clear: no eventfd write
  reader->page()->waiting.store(1);
  const std::uint64_t before = sys::calls();
  CHECK(put(*ring.writer, 2) && ring.writer->counts().wakes == 1); // the publication that found the flag set
  CHECK(sys::calls() - before == 1 && ring.writer->counts().window_calls == 0); // one write, after the store
  reader->page()->waiting.store(0);
  CHECK(put(*ring.writer, 3) && put(*ring.writer, 4) && ring.writer->counts().wakes == 1); // none afterwards
  record_view view;
  for (std::uint64_t n = 0; n != 4; ++n) {
    CHECK(reader->next(view, nanoseconds(0)) == read_result::record && view.sequence == n);
  }
  // A sleeping consumer is woken by the next publication.
  std::atomic<int> got{-1};
  std::thread      worker([&]() {
    record_view late;
    got = reader->next(late, nanoseconds(5000000000LL)) == read_result::record && late.sequence == 4 ? 1 : 0;
  });
  while (reader->page()->waiting.load() == 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  std::this_thread::sleep_for(milliseconds(5));
  CHECK(put(*ring.writer, 5));
  worker.join();
  CHECK(got == 1 && ring.writer->counts().wakes == 2 && reader->counts().sleeps >= 1);
}

void case_wait_no_loss()
{
  // The consumer sets its flag while the producer publishes, 4,000 times with every interleaving
  // the scheduler gives: no publication is slept through.
  ring_fixture              ring(8);
  std::unique_ptr<consumer> reader = ring.attach(wait_mode::wait);
  constexpr unsigned        RECORDS = 4000;
  std::atomic<unsigned>     seen{0};
  std::atomic<bool>         ordered{true};
  std::thread               worker([&]() {
    record_view view;
    for (unsigned n = 0; n != RECORDS; ++n) {
      if (reader->next(view, nanoseconds(5000000000LL)) != read_result::record || view.sequence != n) {
        ordered = false;
        break;
      }
      reader->release(n + 1);
      ++seen;
    }
  });
  for (unsigned n = 0; n != RECORDS;) {
    if (put(*ring.writer, static_cast<std::uint8_t>(n))) {
      ++n;
      if (n % 7 == 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(30)); // lets the consumer fall asleep
      }
    } else {
      std::this_thread::yield();
    }
  }
  worker.join();
  CHECK(ordered && seen == RECORDS && ring.writer->counts().published == RECORDS);
  CHECK(ring.writer->counts().window_calls == 0 && ring.writer->counts().wakes <= RECORDS);
}

void case_consumer_exit()
{
  ring_fixture              ring(4);
  std::unique_ptr<consumer> first = ring.attach();
  CHECK(put(*ring.writer, 'A') && put(*ring.writer, 'B') && put(*ring.writer, 'C'));
  record_view view;
  CHECK(first->next(view, nanoseconds(0)) == read_result::record);
  CHECK(first->next(view, nanoseconds(0)) == read_result::record);
  first->release(1); // it holds B, and C is unread
  first->abandon();  // killed: no BYE
  ring.writer->service();
  CHECK(!ring.writer->attached() && ring.writer->epoch() == 1);
  // Without a consumer the ring fills and then drops. Nothing waits.
  CHECK(put(*ring.writer, 'D') && put(*ring.writer, 'E') && !put(*ring.writer, 'F'));
  CHECK(ring.writer->counts().reclaimed == 0); // nothing is reclaimed before a new owner exists
  std::unique_ptr<consumer> second = ring.attach();
  CHECK(second != nullptr && ring.writer->attached() && ring.writer->epoch() == 2 && second->epoch() == 2);
  producer_counts counts = ring.writer->counts();
  CHECK(counts.epochs == 2 && counts.released == 1 && counts.reclaimed == 4 && counts.pending == 0);
  CHECK(ring.writer->header().reclaimed.load() == 4 && ring.writer->header().published.load() == 0);
  // No credit of the old epoch is accepted: the old page is not read any more.
  first->page()->released.store(3);
  for (std::uint8_t tag = 'G'; tag != 'K'; ++tag) {
    CHECK(put(*ring.writer, tag));
  }
  CHECK(!put(*ring.writer, 'K') && ring.writer->counts().faults == 0);
  // The new epoch starts at sequence 0, and the old consumer reads nothing of it.
  CHECK(second->next(view, nanoseconds(0)) == read_result::record && view.sequence == 0 && holds(view, 'G') &&
        view.header->epoch == 2);
  CHECK(first->next(view, nanoseconds(0)) == read_result::closed);
  second->release(1);
  CHECK(put(*ring.writer, 'K'));
  counts = ring.writer->counts();
  CHECK(counts.admitted == 10 && counts.published == 10 && counts.dropped == 2);
  CHECK(counts.published == counts.released + counts.reclaimed + counts.pending); // every record is accounted for
}

void case_definition_refused()
{
  ring_fixture              ring(2);
  std::string               problem;
  std::unique_ptr<consumer> reader = ring.attach(wait_mode::spin, DEFINITION ^ 1U, &problem);
  CHECK(reader == nullptr && !ring.writer->attached() && ring.writer->counts().refused == 1);
  char mine[17], theirs[17];
  std::snprintf(mine, sizeof(mine), "%016llx", static_cast<unsigned long long>(DEFINITION ^ 1U));
  std::snprintf(theirs, sizeof(theirs), "%016llx", static_cast<unsigned long long>(DEFINITION));
  CHECK(problem.find(mine) != std::string::npos && problem.find(theirs) != std::string::npos); // both hashes
  CHECK(problem.find("record definition") != std::string::npos);
  // The ring is still usable by a consumer of its own definition.
  reader = ring.attach();
  CHECK(reader != nullptr && ring.writer->epoch() == 1);
}

void case_read_only()
{
  ring_fixture              ring(2);
  std::unique_ptr<consumer> reader = ring.attach();
  CHECK(put(*ring.writer, 'A'));
  // Each side holds the other's memory through a read-only descriptor: the kernel refuses to
  // make either mapping writable.
  CHECK(!reader->segment_writable());
  CHECK(!ring.writer->cursor_writable());
  record_view view;
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record && holds(view, 'A'));
}

void case_busy_and_drain()
{
  ring_fixture              ring(4);
  std::unique_ptr<consumer> first = ring.attach();
  std::string               problem;
  std::unique_ptr<consumer> second = ring.attach(wait_mode::spin, DEFINITION, &problem, 30);
  CHECK(second == nullptr && problem.find("within 30 ms") != std::string::npos && ring.writer->counts().refused >= 1);
  CHECK(put(*ring.writer, 'A') && put(*ring.writer, 'B'));
  CHECK(ring.writer->drain(milliseconds(5)) == 2); // pending at the cutoff
  record_view view;
  CHECK(first->next(view, nanoseconds(0)) == read_result::record);
  CHECK(first->next(view, nanoseconds(0)) == read_result::record);
  first->release(2);
  CHECK(ring.writer->drain(milliseconds(100)) == 0);
  ring.writer->finish(); // no record follows
  CHECK(first->next(view, nanoseconds(1000000)) == read_result::closed && !ring.writer->attached());
  const producer_counts counts = ring.writer->counts();
  CHECK(counts.published == 2 && counts.released == 2 && counts.pending == 0 && counts.reclaimed == 0);
  // The ring takes the next consumer: a new epoch, because the old one had a consumer.
  first.reset();
  second = ring.attach();
  CHECK(second != nullptr && ring.writer->epoch() == 2);
}

void case_long_path()
{
  // A run directory is longer than the 107 characters a socket address holds.
  ring_fixture ring(2, std::string(150, 'n'));
  CHECK(ring.path.size() > 160);
  std::unique_ptr<consumer> reader = ring.attach(wait_mode::wait);
  CHECK(reader != nullptr && put(*ring.writer, 'A'));
  record_view view;
  CHECK(reader->next(view, nanoseconds(0)) == read_result::record && holds(view, 'A'));
  reader.reset();
  ring.writer.reset();
  CHECK(::access(ring.path.c_str(), F_OK) != 0); // the producer removed its socket
}

struct named_case {
  const char*           name;
  std::function<void()> body;
};

const named_case CASES[] = {
    {"definition", case_definition},
    {"capacity_two", case_capacity_two},
    {"scratch_overwrite", case_scratch_overwrite},
    {"no_early_read", case_no_early_read},
    {"stale_read", case_stale_read},
    {"double_release", case_double_release},
    {"released_fault", case_released_fault},
    {"fifo", case_fifo},
    {"spin_no_syscalls", case_spin_no_syscalls},
    {"wait_flag", case_wait_flag},
    {"wait_no_loss", case_wait_no_loss},
    {"consumer_exit", case_consumer_exit},
    {"definition_refused", case_definition_refused},
    {"read_only", case_read_only},
    {"busy_and_drain", case_busy_and_drain},
    {"long_path", case_long_path},
};

} // namespace

int main(int argc, char** argv)
{
  const std::string selected = argc > 1 ? argv[1] : "";
  if (selected == "--definition") {
    std::printf("%016llx\n%s\n%s\n", static_cast<unsigned long long>(DEFINITION), RECORD_DEFINITION, RING_DEFINITION);
    return 0;
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
    std::cerr << "obs_ring_test: no case named '" << selected << "'" << std::endl;
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
