// The observation ring obs-ring/v1 (Sprint 3.2, selector snapshot_ring). See obs/ring.h.
// ASCII only.

#include "obs/ring.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

namespace obs {

// ---------------------------------------------------------------------------------------------- system calls

namespace sys {
namespace {

thread_local std::uint64_t thread_calls = 0;
std::atomic<std::uint64_t> process_calls{0};

/// Every system call of the library passes here, once.
inline void count() noexcept
{
  ++thread_calls;
  process_calls.fetch_add(1, std::memory_order_relaxed);
}

int shm_create(const char* name)
{
  count();
  return ::shm_open(name, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
}
int shm_readonly(const char* name)
{
  count();
  return ::shm_open(name, O_RDONLY | O_CLOEXEC, 0);
}
int shm_remove(const char* name)
{
  count();
  return ::shm_unlink(name);
}
int resize(int fd, std::size_t bytes)
{
  count();
  return ::ftruncate(fd, static_cast<off_t>(bytes));
}
void* map(std::size_t bytes, int protection, int flags, int fd)
{
  count();
  return ::mmap(nullptr, bytes, protection, flags, fd, 0);
}
int unmap(void* address, std::size_t bytes)
{
  count();
  return ::munmap(address, bytes);
}
int protect(void* address, std::size_t bytes, int protection)
{
  count();
  return ::mprotect(address, bytes, protection);
}
int close_fd(int fd)
{
  count();
  return ::close(fd);
}
long long size_of(int fd)
{
  count();
  struct stat status;
  return ::fstat(fd, &status) == 0 ? static_cast<long long>(status.st_size) : -1;
}
int open_directory(const char* path)
{
  count();
  return ::open(path, O_PATH | O_DIRECTORY | O_CLOEXEC);
}
int unlink_file(const char* path)
{
  count();
  return ::unlink(path);
}
int packet_socket(bool blocking)
{
  count();
  return ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | (blocking ? 0 : SOCK_NONBLOCK), 0);
}
int bind_to(int fd, const sockaddr_un& address)
{
  count();
  return ::bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
}
int listen_on(int fd)
{
  count();
  return ::listen(fd, 4);
}
int connect_to(int fd, const sockaddr_un& address)
{
  count();
  return ::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
}
int accept_from(int fd)
{
  count();
  return ::accept4(fd, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
}
int new_eventfd()
{
  count();
  return ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
}
long write_fd(int fd, const void* data, std::size_t bytes)
{
  count();
  return static_cast<long>(::write(fd, data, bytes));
}
long read_fd(int fd, void* data, std::size_t bytes)
{
  count();
  return static_cast<long>(::read(fd, data, bytes));
}
int wait_for(pollfd* fds, unsigned count_of, int milliseconds)
{
  count();
  return ::poll(fds, count_of, milliseconds);
}
void sleep_for(long nanoseconds)
{
  count();
  timespec time = {nanoseconds / 1000000000L, nanoseconds % 1000000000L};
  ::nanosleep(&time, nullptr);
}
long send_packet(int fd, msghdr* message)
{
  count();
  return static_cast<long>(::sendmsg(fd, message, MSG_NOSIGNAL));
}
long receive_packet(int fd, msghdr* message, int flags)
{
  count();
  return static_cast<long>(::recvmsg(fd, message, flags | MSG_CMSG_CLOEXEC));
}

} // namespace

std::uint64_t calls()
{
  return thread_calls;
}

std::uint64_t total()
{
  return process_calls.load(std::memory_order_relaxed);
}

} // namespace sys

// ---------------------------------------------------------------------------------------------- helpers

/// The monotonic clock is read in user space on Linux (vDSO), so it does not enter the kernel
/// and is not counted. The clock source of the host is recorded with a run.
std::int64_t monotonic_ns()
{
  timespec time;
  ::clock_gettime(CLOCK_MONOTONIC, &time);
  return static_cast<std::int64_t>(time.tv_sec) * 1000000000LL + time.tv_nsec;
}

const char* to_string(wait_mode mode)
{
  return mode == wait_mode::spin ? "spin" : mode == wait_mode::wait ? "wait" : "unknown";
}

const char* to_string(bye_reason reason)
{
  switch (reason) {
    case bye_reason::done:
      return "done";
    case bye_reason::definition:
      return "another record definition";
    case bye_reason::busy:
      return "the ring has a consumer";
    case bye_reason::cursor:
      return "the cursor page is not a new obs-ring/v1 cursor page";
    case bye_reason::fault:
      return "the released count moved backwards or passed the published count";
    case bye_reason::closed:
      return "closed";
  }
  return "unknown";
}

std::uint32_t slots_for(std::uint32_t window_records, std::uint32_t burst_records)
{
  const std::uint64_t needed = static_cast<std::uint64_t>(window_records) + burst_records + 1;
  std::uint32_t       slots  = 1;
  while (slots < needed && slots < MAX_SLOTS) {
    slots <<= 1;
  }
  if (slots < needed) {
    throw ring_error("a ring holds at most " + std::to_string(MAX_SLOTS) + " slots");
  }
  return slots;
}

namespace {

std::string failed(const std::string& what)
{
  return what + ": " + std::strerror(errno);
}

std::size_t round_up(std::size_t value, std::size_t step)
{
  return (value + step - 1) / step * step;
}

std::string hex(std::uint64_t value)
{
  char text[17];
  std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
  return text;
}

inline void relax() noexcept
{
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

/// A shared page or segment that only this process can write: the returned descriptor is
/// read-only, so the peer that receives it cannot map it for writing. The name is removed at once.
void* create_shared(std::size_t bytes, const char* kind, int& readonly_fd)
{
  static std::atomic<unsigned> serial{0};
  char                         name[96];
  std::snprintf(name, sizeof(name), "/obs-%s-%ld-%u-%llx", kind, static_cast<long>(::getpid()), serial.fetch_add(1),
                static_cast<unsigned long long>(monotonic_ns()));
  const int fd = sys::shm_create(name);
  if (fd < 0) {
    throw ring_error(failed(std::string("cannot create the shared memory object ") + name));
  }
  readonly_fd = sys::shm_readonly(name);
  sys::shm_remove(name);
  void* address = MAP_FAILED;
  if (readonly_fd >= 0 && sys::resize(fd, bytes) == 0) {
    address = sys::map(bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd);
  }
  const int error = errno;
  sys::close_fd(fd);
  if (address == MAP_FAILED) {
    if (readonly_fd >= 0) {
      sys::close_fd(readonly_fd);
      readonly_fd = -1;
    }
    errno = error;
    throw ring_error(failed("cannot size or map " + std::to_string(bytes) + " bytes of shared memory"));
  }
  std::memset(address, 0, bytes); // every page is touched before the first record
  return address;
}

/// The address of a control socket. A path that does not fit sun_path is reached through a
/// descriptor of its directory, which the caller closes once the address was used.
sockaddr_un address_of(const std::string& path, int& directory_fd)
{
  sockaddr_un address;
  std::memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  directory_fd       = -1;
  std::string usable = path;
  if (path.size() >= sizeof(address.sun_path)) {
    const std::size_t cut       = path.find_last_of('/');
    const std::string directory = cut == std::string::npos ? "." : path.substr(0, cut);
    directory_fd                = sys::open_directory(directory.c_str());
    if (directory_fd < 0) {
      throw ring_error(failed("cannot open the directory of the control socket " + path));
    }
    usable = "/proc/self/fd/" + std::to_string(directory_fd) + "/" + path.substr(cut + 1);
  }
  if (usable.empty() || usable.size() >= sizeof(address.sun_path)) {
    throw ring_error("the name of the control socket is empty or too long: " + path);
  }
  std::memcpy(address.sun_path, usable.data(), usable.size());
  return address;
}

control_message make_message(message_type type)
{
  control_message message;
  std::memset(&message, 0, sizeof(message));
  message.magic      = CONTROL_MAGIC;
  message.type       = static_cast<std::uint32_t>(type);
  message.definition = DEFINITION;
  message.pid        = static_cast<std::uint32_t>(::getpid());
  return message;
}

bool send_message(int fd, const control_message& message, const int* fds, unsigned count_of)
{
  iovec part = {const_cast<control_message*>(&message), sizeof(message)};
  alignas(cmsghdr) char control[CMSG_SPACE(2 * sizeof(int))];
  msghdr                packet;
  std::memset(&packet, 0, sizeof(packet));
  std::memset(control, 0, sizeof(control));
  packet.msg_iov    = &part;
  packet.msg_iovlen = 1;
  if (count_of != 0) {
    packet.msg_control    = control;
    packet.msg_controllen = CMSG_SPACE(count_of * sizeof(int));
    cmsghdr* head         = CMSG_FIRSTHDR(&packet);
    head->cmsg_level      = SOL_SOCKET;
    head->cmsg_type       = SCM_RIGHTS;
    head->cmsg_len        = CMSG_LEN(count_of * sizeof(int));
    std::memcpy(CMSG_DATA(head), fds, count_of * sizeof(int));
  }
  return sys::send_packet(fd, &packet) == static_cast<long>(sizeof(message));
}

/// Returns the bytes received: sizeof(control_message) for a message, 0 for a closed channel,
/// -1 when nothing is there. Descriptors that arrived are in fds.
long receive_message(int fd, control_message& message, int* fds, unsigned& count_of, bool blocking)
{
  iovec part = {&message, sizeof(message)};
  alignas(cmsghdr) char control[CMSG_SPACE(2 * sizeof(int))];
  msghdr                packet;
  std::memset(&packet, 0, sizeof(packet));
  packet.msg_iov        = &part;
  packet.msg_iovlen     = 1;
  packet.msg_control    = control;
  packet.msg_controllen = sizeof(control);
  count_of              = 0;
  const long got        = sys::receive_packet(fd, &packet, blocking ? 0 : MSG_DONTWAIT);
  if (got < 0) {
    return (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) ? -1 : 0;
  }
  for (cmsghdr* head = CMSG_FIRSTHDR(&packet); head != nullptr; head = CMSG_NXTHDR(&packet, head)) {
    if (head->cmsg_level == SOL_SOCKET && head->cmsg_type == SCM_RIGHTS) {
      const unsigned arrived = static_cast<unsigned>((head->cmsg_len - CMSG_LEN(0)) / sizeof(int));
      int            all[2]  = {-1, -1};
      std::memcpy(all, CMSG_DATA(head), (arrived < 2 ? arrived : 2) * sizeof(int));
      for (unsigned n = 0; n != arrived && n != 2; ++n) {
        fds[count_of++] = all[n];
      }
    }
  }
  if (got != static_cast<long>(sizeof(message)) || message.magic != CONTROL_MAGIC) {
    for (unsigned n = 0; n != count_of; ++n) {
      sys::close_fd(fds[n]);
    }
    count_of = 0;
    return 0; // not a message of this channel: the peer is treated as gone
  }
  return got;
}

} // namespace

// ---------------------------------------------------------------------------------------------- producer

producer::producer(const ring_config& config) : socket_path(config.socket_path)
{
  if (config.slot_count == 0 || config.slot_count > MAX_SLOTS || (config.slot_count & (config.slot_count - 1)) != 0) {
    throw ring_error("the slot count of a ring is a power of two from 1 to " + std::to_string(MAX_SLOTS));
  }
  if (config.payload_capacity == 0) {
    throw ring_error("a ring needs the payload capacity of a slot");
  }
  if (config.first_epoch == 0) {
    throw ring_error("an epoch is at least 1");
  }
  const std::size_t host_offset    = SLOT_PREFIX_BYTES + RECORD_HEADER_BYTES;
  const std::size_t mask_offset    = round_up(host_offset + config.host_capacity, 8);
  const std::size_t payload_offset = round_up(mask_offset + config.mask_capacity, CACHE_LINE);
  const std::size_t one_slot       = round_up(payload_offset + config.payload_capacity, CACHE_LINE);
  if (one_slot > (1U << 30)) {
    throw ring_error("a slot of a ring holds at most 1 GiB");
  }
  slot_count   = config.slot_count;
  slot_bytes   = static_cast<std::uint32_t>(one_slot);
  mapped_bytes = SEGMENT_HEADER_BYTES + static_cast<std::size_t>(slot_count) * slot_bytes;
  own_cursor   = new cursor_page();
  own_cursor->released.store(0);
  cursor = own_cursor;
  try {
    segment = static_cast<segment_header*>(create_shared(mapped_bytes, "ring", segment_fd));
  } catch (...) {
    delete own_cursor;
    throw;
  }
  slots = reinterpret_cast<std::uint8_t*>(segment) + SEGMENT_HEADER_BYTES;
  std::memcpy(segment->magic, RING_MAGIC, sizeof(RING_MAGIC));
  segment->layout              = RING_LAYOUT;
  segment->header_bytes        = SEGMENT_HEADER_BYTES;
  segment->definition          = DEFINITION;
  segment->slot_count          = slot_count;
  segment->slot_bytes          = slot_bytes;
  segment->record_header_bytes = RECORD_HEADER_BYTES;
  segment->host_capacity       = config.host_capacity;
  segment->mask_capacity       = config.mask_capacity;
  segment->payload_capacity    = config.payload_capacity;
  segment->host_offset         = static_cast<std::uint32_t>(host_offset);
  segment->mask_offset         = static_cast<std::uint32_t>(mask_offset);
  segment->payload_offset      = static_cast<std::uint32_t>(payload_offset);
  segment->producer_pid        = static_cast<std::uint32_t>(::getpid());
  epoch_now                    = config.first_epoch;
  segment->epoch.store(epoch_now);
  segment->published.store(0);

  if (!socket_path.empty()) {
    int               directory_fd = -1;
    const sockaddr_un address      = address_of(socket_path, directory_fd);
    listen_fd                      = sys::packet_socket(false);
    const bool bound = listen_fd >= 0 && sys::bind_to(listen_fd, address) == 0 && sys::listen_on(listen_fd) == 0;
    const int  error = errno;
    if (directory_fd >= 0) {
      sys::close_fd(directory_fd);
    }
    if (!bound) {
      if (listen_fd >= 0) {
        sys::close_fd(listen_fd);
      }
      sys::unmap(segment, mapped_bytes);
      sys::close_fd(segment_fd);
      delete own_cursor;
      errno = error;
      throw ring_error(failed("cannot open the control channel " + socket_path));
    }
  }
}

producer::~producer()
{
  if (peer_fd >= 0) {
    detach(bye_reason::done, true);
  }
  if (pending_fd >= 0) {
    sys::close_fd(pending_fd);
  }
  if (listen_fd >= 0) {
    sys::close_fd(listen_fd);
    int directory_fd = -1;
    try {
      const sockaddr_un address = address_of(socket_path, directory_fd);
      sys::unlink_file(address.sun_path);
    } catch (const ring_error&) {
    }
    if (directory_fd >= 0) {
      sys::close_fd(directory_fd);
    }
  }
  // The consumer has its own mapping of the segment, which outlives this one.
  sys::unmap(segment, mapped_bytes);
  sys::close_fd(segment_fd);
  delete own_cursor;
}

slot_prefix* producer::prefix_of(std::uint64_t sequence) const noexcept
{
  return reinterpret_cast<slot_prefix*>(slots + (sequence & (slot_count - 1)) * static_cast<std::size_t>(slot_bytes));
}

bool producer::try_acquire(slot& out) noexcept
{
  // Admission: the released count, the published count and one compare.
  const std::uint64_t released = cursor->released.load(std::memory_order_acquire);
  if (released - floor > published_now - floor) {
    // The count moved backwards or passed the published count. Nothing is reused on its
    // strength: the last accepted count stands, and service() closes the consumer's channel.
    ++total.faults;
    segment->faults.store(total.faults, std::memory_order_relaxed);
    own_cursor->released.store(floor, std::memory_order_relaxed);
    cursor  = own_cursor;
    faulted = true;
  } else {
    floor = released;
  }
  const std::uint64_t held = published_now - floor;
  if (held >= slot_count) {
    ++total.dropped;
    segment->dropped.store(total.dropped, std::memory_order_relaxed);
    return false;
  }
  slot_prefix*        prefix = prefix_of(published_now);
  const std::uint64_t before = prefix->generation.load(std::memory_order_relaxed);
  prefix->generation.store(before + 1, std::memory_order_release); // odd: the slot is being written
  std::uint8_t* base = reinterpret_cast<std::uint8_t*>(prefix);
  out.prefix         = prefix;
  out.header         = reinterpret_cast<record_header*>(base + SLOT_PREFIX_BYTES);
  out.host           = base + segment->host_offset;
  out.mask           = base + segment->mask_offset;
  out.payload        = base + segment->payload_offset;
  out.sequence       = published_now;
  out.depth          = static_cast<std::uint32_t>(held);
  out.calls          = sys::calls();
  ++total.admitted;
  if (out.depth > total.max_depth) {
    total.max_depth = out.depth;
  }
  return true;
}

void producer::publish(slot& acquired) noexcept
{
  slot_prefix*   prefix = acquired.prefix;
  record_header* header = acquired.header;
  header->epoch         = epoch_now;
  header->sequence      = acquired.sequence;
  prefix->sequence      = acquired.sequence;
  prefix->epoch         = epoch_now;
  prefix->depth         = acquired.depth;
  prefix->record_bytes  = header->header_bytes + header->host_bytes + header->mask_bytes + header->payload_bytes;
  header->t_publish_ns  = monotonic_ns();
  prefix->generation.store(prefix->generation.load(std::memory_order_relaxed) + 1, std::memory_order_release); // even
  published_now = acquired.sequence + 1;
  // The publication. It is ordered before the load of the waiting flag below, and the consumer
  // orders the store of its flag before its second load of published, so no record is slept through.
  segment->published.store(published_now, std::memory_order_seq_cst);
  ++total.published;
  total.window_calls += sys::calls() - acquired.calls;
  if (cursor->waiting.load(std::memory_order_seq_cst) != 0 && event_fd >= 0) {
    const std::uint64_t one = 1;
    sys::write_fd(event_fd, &one, sizeof(one)); // the descriptor does not block
    ++total.wakes;
    segment->wakes.store(total.wakes, std::memory_order_relaxed);
  }
  segment->admitted.store(total.admitted, std::memory_order_relaxed);
  segment->window_calls.store(total.window_calls, std::memory_order_relaxed);
  acquired.prefix = nullptr;
}

void producer::cancel(slot& acquired) noexcept
{
  // The slot holds no record of this sequence. Its generation is even again, and a reader that
  // saw the odd value read nothing.
  acquired.prefix->generation.store(acquired.prefix->generation.load(std::memory_order_relaxed) + 1,
                                    std::memory_order_release);
  ++total.cancelled;
  acquired.prefix = nullptr;
}

producer_counts producer::counts() const noexcept
{
  producer_counts now = total;
  now.released += floor;
  now.pending = published_now - floor;
  return now;
}

bool producer::cursor_writable() const
{
  // The page was mapped from a read-only descriptor, so the kernel refuses write access to it.
  return peer_page != nullptr &&
         sys::protect(const_cast<cursor_page*>(peer_page), CURSOR_BYTES, PROT_READ | PROT_WRITE) == 0;
}

void producer::open_epoch()
{
  // The channel of the old consumer is closed, so nothing can still be released. What it held is
  // reclaimed here, once.
  total.reclaimed += published_now - floor;
  total.released += floor;
  segment->reclaimed.store(total.reclaimed, std::memory_order_relaxed);
  ++epoch_now;
  published_now = 0;
  floor         = 0;
  own_cursor->released.store(0, std::memory_order_relaxed);
  segment->epoch.store(epoch_now, std::memory_order_seq_cst);
  segment->published.store(0, std::memory_order_seq_cst);
  had_consumer = false;
}

void producer::detach(bye_reason why, bool tell)
{
  if (tell && peer_fd >= 0) {
    control_message message = make_message(message_type::bye);
    message.reason          = static_cast<std::uint32_t>(why);
    message.epoch           = epoch_now;
    send_message(peer_fd, message, nullptr, 0);
  }
  // The last accepted count stands for the rest of the epoch.
  own_cursor->released.store(floor, std::memory_order_relaxed);
  cursor = own_cursor;
  if (peer_page != nullptr) {
    sys::unmap(const_cast<cursor_page*>(peer_page), CURSOR_BYTES);
    peer_page = nullptr;
  }
  if (event_fd >= 0) {
    sys::close_fd(event_fd);
    event_fd = -1;
  }
  if (peer_fd >= 0) {
    sys::close_fd(peer_fd);
    peer_fd = -1;
  }
  has_consumer = false;
  faulted      = false;
}

void producer::accept_one()
{
  // One peer at a time is heard. Others stay in the backlog of the listening socket.
  pollfd incoming = {listen_fd, POLLIN, 0};
  if (sys::wait_for(&incoming, 1, 0) <= 0 || (incoming.revents & POLLIN) == 0) {
    return;
  }
  pending_fd    = sys::accept_from(listen_fd);
  pending_since = monotonic_ns();
}

void producer::read_hello()
{
  // A consumer sends HELLO right after it connects. Nothing here waits for it: a peer that stays
  // silent for 200 ms is dropped at a later call.
  const int       fd = pending_fd;
  control_message hello;
  int             fds[2]   = {-1, -1};
  unsigned        count_of = 0;
  const long      got      = receive_message(fd, hello, fds, count_of, false);
  if (got < 0 && monotonic_ns() - pending_since < 200000000LL) {
    return;
  }
  pending_fd = -1;
  bye_reason      refusal  = bye_reason::done;
  const cursor_page* page  = nullptr;
  if (got != static_cast<long>(sizeof(hello)) || hello.type != static_cast<std::uint32_t>(message_type::hello)) {
    refusal = bye_reason::closed;
  } else if (hello.definition != DEFINITION) {
    refusal = bye_reason::definition;
  } else if (has_consumer) {
    refusal = bye_reason::busy;
  } else if (count_of != 2 || sys::size_of(fds[0]) != static_cast<long long>(CURSOR_BYTES) ||
             (hello.mode != static_cast<std::uint32_t>(wait_mode::spin) &&
              hello.mode != static_cast<std::uint32_t>(wait_mode::wait))) {
    refusal = bye_reason::cursor;
  } else {
    // Read-only: this process cannot write the consumer's page.
    void* address = sys::map(CURSOR_BYTES, PROT_READ, MAP_SHARED | MAP_POPULATE, fds[0]);
    if (address == MAP_FAILED) {
      refusal = bye_reason::cursor;
    } else {
      page = static_cast<const cursor_page*>(address);
      if (std::memcmp(page->magic, CURSOR_MAGIC, sizeof(CURSOR_MAGIC)) != 0 || page->definition != DEFINITION ||
          page->layout != RING_LAYOUT || page->released.load() != 0) {
        sys::unmap(address, CURSOR_BYTES);
        page    = nullptr;
        refusal = bye_reason::cursor;
      }
    }
  }
  if (page == nullptr) {
    ++total.refused;
    control_message bye = make_message(message_type::bye);
    bye.reason          = static_cast<std::uint32_t>(refusal);
    bye.peer_definition = DEFINITION;
    bye.epoch           = epoch_now;
    send_message(fd, bye, nullptr, 0);
    for (unsigned n = 0; n != count_of; ++n) {
      sys::close_fd(fds[n]);
    }
    sys::close_fd(fd);
    return;
  }
  // A consumer that attaches starts a new epoch, unless the open one has neither a record nor a
  // consumer before it.
  if (had_consumer || published_now != 0) {
    open_epoch();
  }
  sys::close_fd(fds[0]); // the mapping keeps the page
  peer_page    = page;
  cursor       = page;
  floor        = 0;
  peer_fd      = fd;
  event_fd     = fds[1];
  mode_now     = static_cast<wait_mode>(hello.mode);
  has_consumer = true;
  had_consumer = true;
  ++total.epochs;
  control_message reply = make_message(message_type::epoch);
  reply.epoch           = epoch_now;
  reply.slot_count      = slot_count;
  reply.slot_bytes      = slot_bytes;
  reply.segment_bytes   = mapped_bytes;
  if (!send_message(fd, reply, &segment_fd, 1)) {
    detach(bye_reason::closed, false);
  }
}

void producer::service()
{
  if (listen_fd < 0) {
    return;
  }
  if (peer_fd >= 0) {
    if (faulted) {
      detach(bye_reason::fault, true);
    } else {
      control_message message;
      int             fds[2]   = {-1, -1};
      unsigned        count_of = 0;
      const long      got      = receive_message(peer_fd, message, fds, count_of, false);
      for (unsigned n = 0; n != count_of; ++n) {
        sys::close_fd(fds[n]);
      }
      if (got == 0 || (got > 0 && message.type == static_cast<std::uint32_t>(message_type::bye))) {
        // Take the last count the consumer stored before it left, if it is a valid one.
        const std::uint64_t released = cursor->released.load(std::memory_order_acquire);
        if (released - floor <= published_now - floor) {
          floor = released;
        }
        detach(bye_reason::closed, false);
      }
    }
  }
  for (unsigned turn = 0; turn != 4; ++turn) {
    if (pending_fd < 0) {
      accept_one();
    }
    if (pending_fd < 0) {
      break;
    }
    read_hello();
    if (pending_fd >= 0) {
      break; // its HELLO is not there yet
    }
  }
}

bool producer::wait_attached(std::chrono::milliseconds limit)
{
  const std::int64_t end = monotonic_ns() + limit.count() * 1000000LL;
  for (;;) {
    service();
    if (has_consumer) {
      return true;
    }
    if (listen_fd < 0 || monotonic_ns() >= end) {
      return false;
    }
    pollfd incoming = {listen_fd, POLLIN, 0};
    sys::wait_for(&incoming, 1, 1);
  }
}

std::uint64_t producer::drain(std::chrono::milliseconds limit)
{
  const std::int64_t end = monotonic_ns() + limit.count() * 1000000LL;
  for (;;) {
    const std::uint64_t released = cursor->released.load(std::memory_order_acquire);
    if (released - floor <= published_now - floor) {
      floor = released;
    } else if (!faulted) {
      ++total.faults;
      segment->faults.store(total.faults, std::memory_order_relaxed);
      faulted = true;
    }
    service();
    if (floor == published_now || !has_consumer || monotonic_ns() >= end) {
      return published_now - floor;
    }
    sys::sleep_for(200000);
  }
}

void producer::finish()
{
  if (peer_fd >= 0) {
    const std::uint64_t released = cursor->released.load(std::memory_order_acquire);
    if (released - floor <= published_now - floor) {
      floor = released;
    }
    detach(bye_reason::done, true);
  }
}

// ---------------------------------------------------------------------------------------------- consumer

consumer::consumer(const consumer_config& config, std::uint64_t definition) : wait(config.mode)
{
  int cursor_fd = -1;
  cursor        = static_cast<cursor_page*>(create_shared(CURSOR_BYTES, "cursor", cursor_fd));
  std::memcpy(cursor->magic, CURSOR_MAGIC, sizeof(CURSOR_MAGIC));
  cursor->definition   = definition;
  cursor->layout       = RING_LAYOUT;
  cursor->mode         = static_cast<std::uint32_t>(wait);
  cursor->consumer_pid = static_cast<std::uint32_t>(::getpid());
  event_fd             = sys::new_eventfd();
  std::string problem  = event_fd < 0 ? failed("cannot create the eventfd") : std::string();
  const std::int64_t end = monotonic_ns() + config.connect_limit.count() * 1000000LL;
  int                ring_fd = -1;
  control_message    reply;
  std::memset(&reply, 0, sizeof(reply));
  while (problem.empty() && ring_fd < 0) {
    int         directory_fd = -1;
    sockaddr_un address;
    try {
      address = address_of(config.socket_path, directory_fd);
    } catch (const ring_error& error) {
      if (errno == ENOENT && monotonic_ns() < end) {
        sys::sleep_for(1000000); // the producer has not created the directory of its channel yet
        continue;
      }
      problem = error.what();
      break;
    }
    const int  fd        = sys::packet_socket(true);
    const bool connected = fd >= 0 && sys::connect_to(fd, address) == 0;
    const int  error     = errno;
    if (directory_fd >= 0) {
      sys::close_fd(directory_fd);
    }
    bool retry = false;
    if (!connected) {
      // The producer may not have opened its channel yet.
      retry = fd >= 0 && (error == ENOENT || error == ECONNREFUSED || error == EAGAIN);
      if (!retry) {
        errno   = error;
        problem = failed("cannot connect to the control channel " + config.socket_path);
      }
    } else {
      control_message hello = make_message(message_type::hello);
      hello.definition      = definition;
      hello.mode            = static_cast<std::uint32_t>(wait);
      const int passed[2]   = {cursor_fd, event_fd};
      int       fds[2]      = {-1, -1};
      unsigned  count_of    = 0;
      pollfd    answer      = {fd, POLLIN, 0};
      const std::int64_t left = (end - monotonic_ns()) / 1000000LL;
      if (!send_message(fd, hello, passed, 2) || sys::wait_for(&answer, 1, static_cast<int>(left > 0 ? left : 0)) <= 0 ||
          receive_message(fd, reply, fds, count_of, false) != static_cast<long>(sizeof(reply))) {
        retry = true; // the producer did not answer yet, or it left
      } else if (reply.type == static_cast<std::uint32_t>(message_type::epoch) && count_of == 1) {
        ring_fd = fds[0];
        peer_fd = fd;
        break;
      } else if (reply.type == static_cast<std::uint32_t>(message_type::bye) &&
                 reply.reason == static_cast<std::uint32_t>(bye_reason::busy)) {
        retry = true;
      } else if (reply.type == static_cast<std::uint32_t>(message_type::bye) &&
                 reply.reason == static_cast<std::uint32_t>(bye_reason::definition)) {
        problem = "the ring refuses this consumer: it was built from the record definition " + hex(definition) +
                  ", and the producer from " + hex(reply.peer_definition);
      } else {
        problem = std::string("the ring refuses this consumer: ") +
                  (reply.type == static_cast<std::uint32_t>(message_type::bye)
                       ? to_string(static_cast<bye_reason>(reply.reason))
                       : "its answer is not an EPOCH message with the segment");
      }
      for (unsigned n = 0; n != count_of && ring_fd < 0; ++n) {
        sys::close_fd(fds[n]);
      }
    }
    if (fd >= 0 && peer_fd != fd) {
      sys::close_fd(fd);
    }
    if (retry) {
      if (monotonic_ns() >= end) {
        problem = "no producer accepted this consumer on " + config.socket_path + " within " +
                  std::to_string(config.connect_limit.count()) + " ms";
      } else {
        sys::sleep_for(1000000);
      }
    }
  }
  sys::close_fd(cursor_fd);
  if (problem.empty()) {
    // Read-only: this process cannot write what the producer published.
    void* address = MAP_FAILED;
    if (sys::size_of(ring_fd) == static_cast<long long>(reply.segment_bytes)) {
      address = sys::map(reply.segment_bytes, PROT_READ, MAP_SHARED | MAP_POPULATE, ring_fd);
    }
    if (address == MAP_FAILED) {
      problem = "the segment of the ring cannot be mapped read-only with the size the producer named";
    } else {
      segment      = static_cast<const segment_header*>(address);
      mapped_bytes = reply.segment_bytes;
      const std::uint32_t count_of = segment->slot_count;
      if (std::memcmp(segment->magic, RING_MAGIC, sizeof(RING_MAGIC)) != 0 || segment->layout != RING_LAYOUT ||
          segment->header_bytes != SEGMENT_HEADER_BYTES || segment->record_header_bytes != RECORD_HEADER_BYTES ||
          count_of == 0 || (count_of & (count_of - 1)) != 0 ||
          mapped_bytes != SEGMENT_HEADER_BYTES + static_cast<std::size_t>(count_of) * segment->slot_bytes) {
        problem = "the segment is not an obs-ring/v1 segment of the size the producer named";
      } else if (segment->definition != definition) {
        problem = "the ring refuses this consumer: it was built from the record definition " + hex(definition) +
                  ", and the segment from " + hex(segment->definition);
      }
    }
    sys::close_fd(ring_fd);
  }
  if (!problem.empty()) {
    if (segment != nullptr) {
      sys::unmap(const_cast<segment_header*>(segment), mapped_bytes);
    }
    if (peer_fd >= 0) {
      sys::close_fd(peer_fd);
    }
    if (event_fd >= 0) {
      sys::close_fd(event_fd);
    }
    sys::unmap(cursor, CURSOR_BYTES);
    throw ring_error(problem);
  }
  slots     = reinterpret_cast<const std::uint8_t*>(segment) + SEGMENT_HEADER_BYTES;
  mask      = segment->slot_count - 1;
  epoch_now = reply.epoch;
}

consumer::~consumer()
{
  if (peer_fd >= 0) {
    control_message bye = make_message(message_type::bye);
    bye.epoch           = epoch_now;
    send_message(peer_fd, bye, nullptr, 0);
    sys::close_fd(peer_fd);
  }
  if (event_fd >= 0) {
    sys::close_fd(event_fd);
  }
  if (segment != nullptr) {
    sys::unmap(const_cast<segment_header*>(segment), mapped_bytes);
  }
  if (cursor != nullptr) {
    sys::unmap(cursor, CURSOR_BYTES);
  }
}

bool consumer::segment_writable() const
{
  return sys::protect(const_cast<segment_header*>(segment), mapped_bytes, PROT_READ | PROT_WRITE) == 0;
}

void consumer::abandon() noexcept
{
  if (peer_fd >= 0) {
    sys::close_fd(peer_fd);
    peer_fd = -1;
  }
}

bool consumer::producer_done()
{
  if (done || peer_fd < 0) {
    return true;
  }
  control_message message;
  int             fds[2]   = {-1, -1};
  unsigned        count_of = 0;
  const long      got      = receive_message(peer_fd, message, fds, count_of, false);
  for (unsigned n = 0; n != count_of; ++n) {
    sys::close_fd(fds[n]);
  }
  if (got == 0 || (got > 0 && message.type == static_cast<std::uint32_t>(message_type::bye))) {
    done = true;
  }
  return done;
}

read_result consumer::next(record_view& out, std::chrono::nanoseconds limit)
{
  std::int64_t end   = 0;
  unsigned     turns = 0;
  for (;;) {
    if (segment->epoch.load(std::memory_order_acquire) != epoch_now) {
      return read_result::closed; // another consumer attached: this epoch is over
    }
    if (read_next < segment->published.load(std::memory_order_acquire)) {
      break;
    }
    if (end == 0) {
      end = monotonic_ns() + limit.count();
    }
    if (wait == wait_mode::spin) {
      relax();
      // The clock is read once in 64 turns, and the kernel is not entered at all.
      if ((++turns & 63U) != 0 || monotonic_ns() < end) {
        continue;
      }
    } else {
      cursor->waiting.store(1, std::memory_order_seq_cst);
      if (read_next < segment->published.load(std::memory_order_seq_cst)) {
        cursor->waiting.store(0, std::memory_order_seq_cst);
        break;
      }
      const std::int64_t left = end - monotonic_ns();
      if (left > 0) {
        pollfd waits[2] = {{event_fd, POLLIN, 0}, {peer_fd, POLLIN, 0}};
        ++total.sleeps;
        sys::wait_for(waits, peer_fd >= 0 && !done ? 2 : 1, static_cast<int>((left + 999999) / 1000000));
      }
      cursor->waiting.store(0, std::memory_order_seq_cst);
      std::uint64_t woken = 0;
      sys::read_fd(event_fd, &woken, sizeof(woken));
      if (read_next < segment->published.load(std::memory_order_acquire)) {
        break;
      }
      if (monotonic_ns() < end && !producer_done()) {
        continue;
      }
    }
    // Nothing arrived within the limit. The control channel is read here, off the data path.
    const bool finished = producer_done();
    if (read_next < segment->published.load(std::memory_order_acquire)) {
      continue; // a record was published before the producer left
    }
    return finished ? read_result::closed : read_result::none;
  }
  const std::uint8_t* base   = slots + (read_next & mask) * static_cast<std::size_t>(segment->slot_bytes);
  const slot_prefix*  prefix = reinterpret_cast<const slot_prefix*>(base);
  const std::uint64_t seen   = prefix->generation.load(std::memory_order_acquire);
  const std::uint64_t which  = read_next++;
  if ((seen & 1U) != 0 || prefix->sequence != which || prefix->epoch != epoch_now) {
    ++total.stale;
    return read_result::stale;
  }
  out.prefix     = prefix;
  out.generation = seen;
  out.header     = reinterpret_cast<const record_header*>(base + SLOT_PREFIX_BYTES);
  out.host       = base + segment->host_offset;
  out.mask       = base + segment->mask_offset;
  out.payload    = base + segment->payload_offset;
  out.sequence   = which;
  out.depth      = prefix->depth;
  ++total.read;
  return read_result::record;
}

bool consumer::intact(const record_view& view) noexcept
{
  if (view.prefix->generation.load(std::memory_order_acquire) == view.generation) {
    return true;
  }
  ++total.stale;
  return false;
}

void consumer::release(std::uint64_t sequence) noexcept
{
  // Never beyond what was read, and never backwards.
  const std::uint64_t upto = sequence < read_next ? sequence : read_next;
  const std::uint64_t now  = cursor->released.load(std::memory_order_relaxed);
  if (upto > now) {
    total.released += upto - now;
    cursor->released.store(upto, std::memory_order_release);
  }
}

void consumer::skip_to(std::uint64_t sequence) noexcept
{
  const std::uint64_t published = segment->published.load(std::memory_order_acquire);
  const std::uint64_t upto      = sequence < published ? sequence : published;
  if (upto > read_next) {
    total.skipped += upto - read_next;
    read_next = upto;
  }
  release(upto);
}

} // namespace obs
