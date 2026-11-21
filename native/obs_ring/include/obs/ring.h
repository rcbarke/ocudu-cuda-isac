// The observation ring obs-ring/v1 (Sprint 3.2, selector snapshot_ring).
//
// One producer and one consumer share one ring in POSIX shared memory. It is the only queue
// between a receiver and a sensor, and it is first in, first out: the record with sequence n of
// an epoch lies in slot n modulo the slot count, the consumer reads in that order and releases in
// that order.
//
//   segment (created and written by the producer, mapped read-only by the consumer)
//     [ segment_header, 4096 bytes ][ slot 0 ][ slot 1 ] ... [ slot count - 1 ]
//   slot (slot_bytes each, a multiple of 64)
//     [ slot_prefix, 64 ][ record_header, 256 ][ host block ][ validity bits ][ payload, 64-aligned ]
//   cursor page (created and written by the consumer, mapped read-only by the producer)
//     [ released, one cache line ][ waiting flag and identity, one cache line ]
//
// Counts. published, in the segment header, is the number of records the producer completed in
// this epoch. released, in the cursor page, is the number the consumer has finished with. Each is
// a 64-bit count on its own cache line, stored by one side only. published - released records
// are held, and a credit is a free slot.
//
// What the producer pays for one record:
//   try_acquire   two loads and a compare. No credit: a counted drop, at once.
//   (the caller fills the slot in place)
//   publish       the generation of the slot, and one store of published. That store is the
//                 publication.
// Between the two the producer makes no system call, takes no lock and allocates nothing. It
// never waits. Only when the consumer's waiting flag is set does it write the consumer's eventfd,
// after the publication store.
//
// The control channel is a sequenced-packet socket with three fixed-size messages, HELLO, EPOCH
// and BYE. It passes the descriptors of the cursor page, of the eventfd and of the segment, and
// it shows that a peer has gone. It carries no message per observation. The producer reads it in
// service(), which the owner calls outside the window between try_acquire and publish.
//
// Every system call of this library goes through obs::sys, which counts it. The producer adds up
// the calls it made between each try_acquire and the publication store. A timed run needs zero.
//
// C++17 on the CPU, the standard library and POSIX only: no OCUDU, no CUDA. ASCII only.

#pragma once

#include "obs/record.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace obs {

constexpr std::uint32_t RING_LAYOUT          = 1; ///< obs-ring/v1.
constexpr std::size_t   CACHE_LINE           = 64;
constexpr std::size_t   SEGMENT_HEADER_BYTES = 4096;
constexpr std::size_t   CURSOR_BYTES         = 4096;
constexpr std::size_t   SLOT_PREFIX_BYTES    = 64;
constexpr std::uint32_t MAX_SLOTS            = 1U << 16;
constexpr char          RING_MAGIC[8]        = {'O', 'B', 'S', 'R', 'I', 'N', 'G', '1'};
constexpr char          CURSOR_MAGIC[8]      = {'O', 'B', 'S', 'C', 'U', 'R', 'S', '1'};
constexpr std::uint32_t CONTROL_MAGIC        = 0x4353424fU; ///< "OBSC".

static_assert(std::atomic<std::uint64_t>::is_always_lock_free && std::atomic<std::uint32_t>::is_always_lock_free,
              "the counts are single aligned stores");

/// The segment, the slot and the cursor page, as RECORD_DEFINITION names the record.
constexpr const char RING_DEFINITION[] =
    "obs-ring/v1 header 4096;"
    "magic:0:8;layout:8:4;header_bytes:12:4;definition:16:8;slot_count:24:4;slot_bytes:28:4;"
    "record_header_bytes:32:4;host_capacity:36:4;mask_capacity:40:4;payload_capacity:44:4;host_offset:48:4;"
    "mask_offset:52:4;payload_offset:56:4;producer_pid:60:4;epoch:64:8;published:128:8;admitted:192:8;"
    "dropped:200:8;reclaimed:208:8;faults:216:8;wakes:224:8;window_calls:232:8;"
    "slot prefix 64;generation:0:8;sequence:8:8;epoch:16:8;depth:24:4;record_bytes:28:4;"
    "cursor 4096;released:0:8;waiting:64:4;wait_mode:68:4;magic:72:8;definition:80:8;consumer_pid:88:4;layout:92:4;";

/// Names the build of both definitions. A consumer with another value is refused at attach.
constexpr std::uint64_t DEFINITION = fnv1a(RING_DEFINITION, fnv1a(RECORD_DEFINITION));

struct segment_header {
  // Written once, before a consumer can attach.
  char          magic[8];
  std::uint32_t layout;
  std::uint32_t header_bytes;
  std::uint64_t definition;
  std::uint32_t slot_count; ///< A power of two.
  std::uint32_t slot_bytes;
  std::uint32_t record_header_bytes;
  std::uint32_t host_capacity;
  std::uint32_t mask_capacity;
  std::uint32_t payload_capacity;
  std::uint32_t host_offset; ///< Offsets of the parts of a record inside a slot.
  std::uint32_t mask_offset;
  std::uint32_t payload_offset;
  std::uint32_t producer_pid;
  // The epoch, stored by the producer when a consumer attaches.
  alignas(CACHE_LINE) std::atomic<std::uint64_t> epoch;
  // The publication count of this epoch, on its own cache line.
  alignas(CACHE_LINE) std::atomic<std::uint64_t> published;
  // Counts for a monitor that maps the segment read-only. The producer is their only writer.
  alignas(CACHE_LINE) std::atomic<std::uint64_t> admitted;
  std::atomic<std::uint64_t> dropped;      ///< No credit: the observation was not written.
  std::atomic<std::uint64_t> reclaimed;    ///< Published in an epoch that closed before they were released.
  std::atomic<std::uint64_t> faults;       ///< A released count that moved backwards or passed published.
  std::atomic<std::uint64_t> wakes;        ///< Writes of the consumer's eventfd.
  std::atomic<std::uint64_t> window_calls; ///< System calls between a try_acquire and its publication store.
  char pad[SEGMENT_HEADER_BYTES - 4 * CACHE_LINE];
};
static_assert(sizeof(segment_header) == SEGMENT_HEADER_BYTES, "the segment header is one page");
static_assert(offsetof(segment_header, epoch) == 64 && offsetof(segment_header, published) == 128 &&
                  offsetof(segment_header, admitted) == 192 && offsetof(segment_header, window_calls) == 232,
              "the offsets are those of RING_DEFINITION");

struct slot_prefix {
  /// Odd while the producer writes the slot, even when it is complete. It never decreases.
  std::atomic<std::uint64_t> generation;
  std::uint64_t sequence;     ///< Of the record the slot holds.
  std::uint64_t epoch;
  std::uint32_t depth;        ///< Records held when this one was admitted: the queue depth it met.
  std::uint32_t record_bytes; ///< Header, host block, validity bits and payload of this record.
  char          pad[SLOT_PREFIX_BYTES - 32];
};
static_assert(sizeof(slot_prefix) == SLOT_PREFIX_BYTES, "the slot prefix is one cache line");

enum class wait_mode : std::uint32_t { spin = 1, wait = 2 };
const char* to_string(wait_mode mode);

struct cursor_page {
  alignas(CACHE_LINE) std::atomic<std::uint64_t> released;
  alignas(CACHE_LINE) std::atomic<std::uint32_t> waiting; ///< Set while the consumer sleeps on its eventfd.
  std::uint32_t mode; ///< wait_mode of the consumer.
  char          magic[8];
  std::uint64_t definition;
  std::uint32_t consumer_pid;
  std::uint32_t layout;
  char          pad[CURSOR_BYTES - CACHE_LINE - 32];
};
static_assert(sizeof(cursor_page) == CURSOR_BYTES, "the cursor page is one page, of which two cache lines are used");
static_assert(offsetof(cursor_page, waiting) == 64 && offsetof(cursor_page, definition) == 80, "RING_DEFINITION");

enum class message_type : std::uint32_t { hello = 1, epoch = 2, bye = 3 };
enum class bye_reason : std::uint32_t { done = 0, definition = 1, busy = 2, cursor = 3, fault = 4, closed = 5 };
const char* to_string(bye_reason reason);

/// One message of the control channel. HELLO carries the cursor page and the eventfd, EPOCH the segment.
struct control_message {
  std::uint32_t magic;
  std::uint32_t type;
  std::uint64_t definition; ///< Of the sender.
  std::uint64_t epoch;
  std::uint32_t pid;
  std::uint32_t mode;   ///< HELLO: wait_mode.
  std::uint32_t reason; ///< BYE: bye_reason.
  std::uint32_t slot_count;
  std::uint32_t slot_bytes;
  std::uint32_t reserved;
  std::uint64_t segment_bytes;
  std::uint64_t peer_definition; ///< BYE for another definition: the one the receiver of HELLO was built from.
};
static_assert(sizeof(control_message) == 64, "a control message has one fixed size");

/// The ring could not be created or attached, or a peer broke the contract. Never raised between
/// try_acquire and publish.
struct ring_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// Slot count for a stream: the records a window holds in place, plus one uplink burst, plus one,
/// rounded up to a power of two. Depth follows the window, never a backlog.
std::uint32_t slots_for(std::uint32_t window_records, std::uint32_t burst_records);

std::int64_t monotonic_ns();

// ---------------------------------------------------------------------------------------------- system calls

namespace sys {
/// System calls this thread made through the library.
std::uint64_t calls();
/// System calls of every thread of the process.
std::uint64_t total();
} // namespace sys

// ---------------------------------------------------------------------------------------------- producer

struct ring_config {
  std::string   socket_path;          ///< Where the control channel listens. Empty: no consumer can attach.
  std::uint32_t slot_count       = 8; ///< A power of two.
  std::uint32_t host_capacity    = 0;
  std::uint32_t mask_capacity    = 0;
  std::uint32_t payload_capacity = 0;
  std::uint64_t first_epoch      = 1;
};

/// One acquired slot: where the producer writes one record in place.
struct slot {
  record_header* header  = nullptr;
  std::uint8_t*  host    = nullptr;
  std::uint8_t*  mask    = nullptr;
  std::uint8_t*  payload = nullptr;
  std::uint64_t  sequence = 0;
  std::uint32_t  depth    = 0;
  // Of the ring.
  slot_prefix*  prefix = nullptr;
  std::uint64_t calls  = 0;
};

struct producer_counts {
  std::uint64_t admitted     = 0; ///< try_acquire that yielded a slot.
  std::uint64_t published    = 0; ///< Over every epoch.
  std::uint64_t dropped      = 0; ///< No credit.
  std::uint64_t cancelled    = 0; ///< Acquired and not published: the fill failed.
  std::uint64_t reclaimed    = 0; ///< Published and never released, in an epoch that was closed.
  std::uint64_t released     = 0; ///< Credits that came back, over every epoch.
  std::uint64_t pending      = 0; ///< Published and not released in the open epoch.
  std::uint64_t faults       = 0;
  std::uint64_t epochs       = 0; ///< Consumers that attached.
  std::uint64_t refused      = 0; ///< HELLO messages that were refused.
  std::uint64_t wakes        = 0;
  std::uint64_t window_calls = 0;
  std::uint32_t max_depth    = 0;
};

class producer
{
public:
  /// Creates the segment, touches every page of it and opens the control channel. Throws ring_error.
  explicit producer(const ring_config& config);
  ~producer();
  producer(const producer&)            = delete;
  producer& operator=(const producer&) = delete;

  /// One credit or a counted drop. Two loads and a compare, and the generation of the slot.
  bool try_acquire(slot& out) noexcept;
  /// Completes the record of an acquired slot: sets its epoch, its sequence and its publication
  /// time, and stores published. The caller filled the header and the parts before.
  void publish(slot& acquired) noexcept;
  /// Gives an acquired slot back without a record. Nothing is published.
  void cancel(slot& acquired) noexcept;

  /// The control channel: a consumer that attaches, one that left, one that broke the contract.
  /// It makes system calls. Call it outside the window between try_acquire and publish.
  void service();
  /// service() until a consumer is attached. Returns false when none attached within limit.
  bool wait_attached(std::chrono::milliseconds limit);
  /// service() until every published record of the open epoch is released, the consumer is gone
  /// or limit has passed. Returns the records still pending.
  std::uint64_t drain(std::chrono::milliseconds limit);
  /// Tells the consumer that no record follows and closes the channel. The segment stays mapped.
  void finish();

  bool                  attached() const noexcept { return has_consumer; }
  std::uint64_t         epoch() const noexcept { return epoch_now; }
  std::uint64_t         published() const noexcept { return published_now; }
  producer_counts       counts() const noexcept;
  const segment_header& header() const noexcept { return *segment; }
  std::size_t           segment_bytes() const noexcept { return mapped_bytes; }
  wait_mode             consumer_mode() const noexcept { return mode_now; }
  /// Probe for the tests: whether this process could make the consumer's page writable. It must not.
  bool cursor_writable() const;

private:
  slot_prefix* prefix_of(std::uint64_t sequence) const noexcept;
  void         detach(bye_reason why, bool tell);
  void         open_epoch();
  void         accept_one();
  void         read_hello();

  segment_header*       segment      = nullptr;
  std::uint8_t*         slots        = nullptr;
  std::size_t           mapped_bytes = 0;
  std::uint32_t         slot_count   = 0;
  std::uint32_t         slot_bytes   = 0;
  const cursor_page*    cursor       = nullptr; ///< The consumer's page, or own_cursor without one.
  const cursor_page*    peer_page    = nullptr; ///< The mapping of the consumer's page, read-only.
  cursor_page*          own_cursor   = nullptr; ///< Holds the last released count while no consumer is attached.
  std::uint64_t         published_now = 0;
  std::uint64_t         floor         = 0; ///< The last released count that was accepted.
  std::uint64_t         epoch_now     = 0;
  bool                  has_consumer  = false;
  bool                  had_consumer  = false; ///< In the open epoch.
  bool                  faulted       = false;
  wait_mode             mode_now      = wait_mode::spin;
  producer_counts       total;
  int                   segment_fd = -1; ///< Read-only descriptor, passed to a consumer.
  int                   listen_fd  = -1;
  int                   peer_fd    = -1;
  int                   event_fd   = -1;
  int                   pending_fd = -1; ///< A peer that connected and whose HELLO has not been read.
  std::int64_t          pending_since = 0;
  std::string           socket_path;
};

// ---------------------------------------------------------------------------------------------- consumer

struct consumer_config {
  std::string               socket_path;
  wait_mode                 mode = wait_mode::spin;
  std::chrono::milliseconds connect_limit{5000}; ///< A socket that does not exist yet and a busy ring are retried.
};

enum class read_result {
  record, ///< out names the next record of the epoch, in place.
  none,   ///< None arrived within the limit.
  stale,  ///< The slot was rewritten. Counted. No record is delivered.
  closed  ///< The producer said that no record follows, or it is gone, and every record was read.
};

/// One record, read in place in the read-only mapping.
struct record_view {
  const record_header* header  = nullptr;
  const std::uint8_t*  host    = nullptr;
  const std::uint8_t*  mask    = nullptr;
  const std::uint8_t*  payload = nullptr;
  std::uint64_t        sequence = 0;
  std::uint32_t        depth    = 0;
  // Of the ring.
  const slot_prefix* prefix     = nullptr;
  std::uint64_t      generation = 0;
};

struct consumer_counts {
  std::uint64_t read     = 0;
  std::uint64_t stale    = 0;
  std::uint64_t released = 0;
  std::uint64_t skipped  = 0;
  std::uint64_t sleeps   = 0; ///< Times the consumer slept on its eventfd.
};

class consumer
{
public:
  /// Creates the cursor page, attaches and maps the segment read-only. Throws ring_error with the
  /// reason of a refusal. For another definition the message holds both hashes.
  explicit consumer(const consumer_config& config, std::uint64_t definition = DEFINITION);
  ~consumer();
  consumer(const consumer&)            = delete;
  consumer& operator=(const consumer&) = delete;

  /// The next record in sequence order. In the mode spin it polls published and makes no system
  /// call. In the mode wait it sets its flag, checks once more and sleeps on its eventfd.
  read_result next(record_view& out, std::chrono::nanoseconds limit);
  /// Whether the slot still holds the record that next() returned. Call it after the last use of
  /// the view. A false is a counted stale read, and no decision may use that record.
  bool intact(const record_view& view) noexcept;
  /// The work on every record below sequence is complete: one store. A window that is read in
  /// place holds its oldest record, so releases follow the window.
  void release(std::uint64_t sequence) noexcept;
  /// Passes records without reading them, up to sequence, and releases them. They are counted.
  void skip_to(std::uint64_t sequence) noexcept;
  /// Reads the control channel without blocking. True once the producer said BYE or is gone.
  bool producer_done();
  /// Leaves without BYE and without unmapping in order, as a consumer that was killed does.
  void abandon() noexcept;

  std::uint64_t         epoch() const noexcept { return epoch_now; }
  std::uint64_t         position() const noexcept { return read_next; }
  consumer_counts       counts() const noexcept { return total; }
  const segment_header& header() const noexcept { return *segment; }
  wait_mode             mode() const noexcept { return wait; }
  /// Probe for the tests: whether this process could make the segment writable. It must not.
  bool segment_writable() const;
  /// Test access: the page the consumer alone may write.
  cursor_page* page() noexcept { return cursor; }

private:
  const segment_header* segment      = nullptr;
  const std::uint8_t*   slots        = nullptr;
  std::size_t           mapped_bytes = 0;
  cursor_page*          cursor       = nullptr;
  std::uint64_t         epoch_now    = 0;
  std::uint64_t         read_next    = 0;
  std::uint32_t         mask         = 0;
  wait_mode             wait         = wait_mode::spin;
  bool                  done         = false;
  consumer_counts       total;
  int                   peer_fd  = -1;
  int                   event_fd = -1;
};

} // namespace obs
