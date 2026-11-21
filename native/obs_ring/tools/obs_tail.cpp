// obs_tail: a consumer of one observation ring that keeps what it read (Sprint 3.2, selector
// snapshot_ring).
//
// It allocates and touches the memory for what it will keep, and only then attaches: a producer
// may publish from the moment a consumer is attached, so a consumer attaches when it is ready to
// read. It then reads every record in sequence order and in place, takes the time of each read,
// copies the record into that memory and releases it. After the producer said that no record follows, it writes one file per record,
// with the bytes of the record as they lay in the slot, and consumer.json with its counts and the
// delay of every read. Nothing is written and nothing is allocated while records arrive.
//
// It is the consumer of the Sprint 3.2 gate, with the switches that gate needs: a consumer that
// is slow, one that holds a window of records in place, and one that ends without a word while
// it holds a credit. The sensor process of Sprint 3.6 links the same library.
//
//     obs_tail --socket PATH [--output-dir DIR] [--wait spin|wait] [--connect-ms N] [--idle-ms N]
//              [--hold-us N] [--window N] [--exit-after N] [--capacity N] [--record-bytes N]
//              [--after-pid PID] [--cpu N]
//
// It links obs_ring and nothing of OCUDU. ASCII only.

#include "obs/ring.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

struct kept {
  std::uint64_t sequence     = 0;
  std::uint64_t epoch        = 0;
  std::uint64_t opportunity  = 0;
  std::uint32_t depth        = 0;
  std::uint32_t bytes        = 0;
  std::int64_t  t_publish_ns = 0;
  std::int64_t  t_read_ns    = 0;
  bool          intact       = false;
};

void usage()
{
  std::puts("Usage: obs_tail --socket PATH [--output-dir DIR] [--wait spin|wait] [--connect-ms N] [--idle-ms N]\n"
            "                [--hold-us N] [--window N] [--exit-after N] [--capacity N] [--record-bytes N]\n"
            "                [--after-pid PID] [--cpu N]\n"
            "  --socket PATH     control channel of the producer's ring\n"
            "  --output-dir DIR  existing, empty directory for the records and consumer.json\n"
            "  --wait MODE       spin (default): poll the published count. wait: sleep on the eventfd\n"
            "  --connect-ms N    how long to try to attach (default 10000)\n"
            "  --idle-ms N       give up when no record arrived for this long (default 30000)\n"
            "  --hold-us N       work on each record for N microseconds before it is released\n"
            "  --window N        hold N records in place: record n is released when n + N was read\n"
            "  --exit-after N    end without BYE after N records were read, holding what is unreleased\n"
            "  --capacity N      records to keep in memory (default 1024). Later ones are counted only\n"
            "  --record-bytes N  the largest record to keep, header and parts (default 32768). The memory is\n"
            "                    allocated before the consumer attaches, so the size is named here\n"
            "  --after-pid PID   attach only once the process PID has ended: the consumer that takes over\n"
            "                    when another one was killed\n"
            "  --cpu N           run on this CPU");
}

bool write_file(const std::string& path, const void* data, std::size_t bytes)
{
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return false;
  }
  const bool ok = std::fwrite(data, 1, bytes, file) == bytes;
  return std::fclose(file) == 0 && ok;
}

} // namespace

int main(int argc, char** argv)
{
  std::string socket_path, output_dir, wait_name = "spin";
  long        connect_ms = 10000, idle_ms = 30000, hold_us = 0, window = 0, exit_after = -1, capacity = 1024, cpu = -1;
  long        record_bytes = 32768, after_pid = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg  = argv[i];
    const bool        more = i + 1 < argc;
    if (arg == "--socket" && more) {
      socket_path = argv[++i];
    } else if (arg == "--output-dir" && more) {
      output_dir = argv[++i];
    } else if (arg == "--wait" && more) {
      wait_name = argv[++i];
    } else if (arg == "--connect-ms" && more) {
      connect_ms = std::atol(argv[++i]);
    } else if (arg == "--idle-ms" && more) {
      idle_ms = std::atol(argv[++i]);
    } else if (arg == "--hold-us" && more) {
      hold_us = std::atol(argv[++i]);
    } else if (arg == "--window" && more) {
      window = std::atol(argv[++i]);
    } else if (arg == "--exit-after" && more) {
      exit_after = std::atol(argv[++i]);
    } else if (arg == "--capacity" && more) {
      capacity = std::atol(argv[++i]);
    } else if (arg == "--after-pid" && more) {
      after_pid = std::atol(argv[++i]);
    } else if (arg == "--record-bytes" && more) {
      record_bytes = std::atol(argv[++i]);
    } else if (arg == "--cpu" && more) {
      cpu = std::atol(argv[++i]);
    } else if (arg == "--help") {
      usage();
      return 0;
    } else {
      usage();
      std::fprintf(stderr, "obs_tail: unknown or incomplete option '%s'\n", arg.c_str());
      return 2;
    }
  }
  if (socket_path.empty() || (wait_name != "spin" && wait_name != "wait") || connect_ms < 0 || idle_ms < 1 ||
      hold_us < 0 || window < 0 || capacity < 1 || record_bytes < static_cast<long>(obs::RECORD_HEADER_BYTES)) {
    usage();
    std::fprintf(stderr, "obs_tail: --socket is required, --wait is spin or wait, and the counts are not negative\n");
    return 2;
  }
  if (!output_dir.empty()) {
    struct stat status;
    if (::stat(output_dir.c_str(), &status) != 0 || !S_ISDIR(status.st_mode) ||
        ::access((output_dir + "/consumer.json").c_str(), F_OK) == 0) {
      std::fprintf(stderr, "obs_tail: the output directory %s does not exist or holds a result already\n",
                   output_dir.c_str());
      return 2;
    }
  }
  if (cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<unsigned>(cpu), &set);
    if (::sched_setaffinity(0, sizeof(set), &set) != 0) {
      std::fprintf(stderr, "obs_tail: cannot run on CPU %ld\n", cpu);
      return 2;
    }
  }

  obs::consumer_config config;
  config.socket_path   = socket_path;
  config.mode          = wait_name == "wait" ? obs::wait_mode::wait : obs::wait_mode::spin;
  config.connect_limit = std::chrono::milliseconds(connect_ms);
  try {
    // Everything is allocated and touched before the consumer attaches, so that it reads from
    // the first record on.
    const std::size_t         record_max = static_cast<std::size_t>(record_bytes);
    std::vector<kept>         records(static_cast<std::size_t>(capacity));
    std::vector<std::uint8_t> arena(output_dir.empty() ? 0 : record_max * static_cast<std::size_t>(capacity), 0);
    if (after_pid > 0) {
      // Ready, and waiting for the consumer before this one to end. Its parent must have reaped it.
      const std::int64_t give_up = obs::monotonic_ns() + connect_ms * 1000000LL;
      while (::kill(static_cast<pid_t>(after_pid), 0) == 0 && obs::monotonic_ns() < give_up) {
        ::usleep(200);
      }
    }
    obs::consumer              reader(config);
    const obs::segment_header& ring = reader.header();
    if (ring.record_header_bytes + ring.host_capacity + ring.mask_capacity + static_cast<std::size_t>(ring.payload_capacity) >
        record_max) {
      std::fprintf(stderr, "obs_tail: a record of this ring holds up to %zu bytes. Name it with --record-bytes\n",
                   ring.record_header_bytes + ring.host_capacity + ring.mask_capacity +
                       static_cast<std::size_t>(ring.payload_capacity));
      return 2;
    }
    std::size_t               count = 0, uncounted = 0;
    std::int64_t              last  = obs::monotonic_ns();
    const std::int64_t        attached_ns = last;
    bool                      closed = false, idle = false;
    obs::record_view          view;
    for (;;) {
      const obs::read_result got = reader.next(view, std::chrono::milliseconds(20));
      if (got == obs::read_result::closed) {
        closed = true;
        break;
      }
      if (got == obs::read_result::none) {
        if (obs::monotonic_ns() - last > idle_ms * 1000000LL) {
          idle = true;
          break;
        }
        continue;
      }
      const std::int64_t now = obs::monotonic_ns();
      last                   = now;
      if (got == obs::read_result::record) {
        const obs::record_header& header = *view.header;
        if (count < static_cast<std::size_t>(capacity)) {
          kept& item        = records[count];
          item.sequence     = view.sequence;
          item.epoch        = header.epoch;
          item.opportunity  = header.opportunity;
          item.depth        = view.depth;
          item.t_publish_ns = header.t_publish_ns;
          item.t_read_ns    = now;
          const std::size_t host = header.host_bytes <= ring.host_capacity ? header.host_bytes : 0;
          const std::size_t bits = header.mask_bytes <= ring.mask_capacity ? header.mask_bytes : 0;
          const std::size_t data = header.payload_bytes <= ring.payload_capacity ? header.payload_bytes : 0;
          item.bytes             = static_cast<std::uint32_t>(ring.record_header_bytes + host + bits + data);
          if (!arena.empty()) {
            std::uint8_t* to = arena.data() + count * record_max;
            std::memcpy(to, view.header, ring.record_header_bytes);
            std::memcpy(to + ring.record_header_bytes, view.host, host);
            std::memcpy(to + ring.record_header_bytes + host, view.mask, bits);
            std::memcpy(to + ring.record_header_bytes + host + bits, view.payload, data);
          }
          // The work of this consumer on the record, in place.
          if (hold_us > 0) {
            const std::int64_t until = obs::monotonic_ns() + hold_us * 1000LL;
            while (obs::monotonic_ns() < until) {
            }
          }
          item.intact = reader.intact(view); // checked after the last use of the view
          ++count;
        } else {
          ++uncounted;
        }
      }
      if (exit_after >= 0 && static_cast<long>(reader.counts().read) >= exit_after) {
        // A consumer that is killed: no BYE, no release of what it holds, no file.
        std::fprintf(stderr, "obs_tail: ending without BYE after %llu records, as asked\n",
                     static_cast<unsigned long long>(reader.counts().read));
        std::_Exit(9);
      }
      const std::uint64_t done = view.sequence + 1;
      reader.release(done > static_cast<std::uint64_t>(window) ? done - static_cast<std::uint64_t>(window) : 0);
    }
    reader.release(reader.position()); // the window is closed: nothing is held any more
    const obs::consumer_counts counts = reader.counts();

    if (!output_dir.empty()) {
      for (std::size_t n = 0; n != count; ++n) {
        char name[64];
        std::snprintf(name, sizeof(name), "/%04llu-%08llu.obs", static_cast<unsigned long long>(records[n].epoch),
                      static_cast<unsigned long long>(records[n].sequence));
        if (records[n].intact && !write_file(output_dir + name, arena.data() + n * record_max, records[n].bytes)) {
          std::fprintf(stderr, "obs_tail: cannot write %s%s\n", output_dir.c_str(), name);
          return 1;
        }
      }
    }
    std::string text = "{\n  \"schema\": \"obs-consumer/v1\",\n  \"reader\": \"native\",\n";
    char        line[512];
    std::snprintf(line, sizeof(line),
                  "  \"definition\": \"%016llx\",\n  \"wait\": \"%s\",\n  \"epoch\": %llu,\n  \"pid\": %ld,\n"
                  "  \"cpu\": %ld,\n  \"hold_us\": %ld,\n  \"window\": %ld,\n  \"slot_count\": %u,\n  \"slot_bytes\": %u,\n"
                  "  \"attached_ns\": %lld,\n  \"ended\": \"%s\",\n",
                  static_cast<unsigned long long>(obs::DEFINITION), wait_name.c_str(),
                  static_cast<unsigned long long>(reader.epoch()), static_cast<long>(::getpid()), cpu, hold_us, window,
                  ring.slot_count, ring.slot_bytes, static_cast<long long>(attached_ns),
                  closed ? "closed" : idle ? "idle" : "unknown");
    text += line;
    std::snprintf(line, sizeof(line),
                  "  \"counts\": {\"read\": %llu, \"stale\": %llu, \"released\": %llu, \"skipped\": %llu, \"sleeps\": %llu, "
                  "\"kept\": %zu, \"not_kept\": %zu, \"system_calls\": %llu},\n  \"records\": [\n",
                  static_cast<unsigned long long>(counts.read), static_cast<unsigned long long>(counts.stale),
                  static_cast<unsigned long long>(counts.released), static_cast<unsigned long long>(counts.skipped),
                  static_cast<unsigned long long>(counts.sleeps), count, uncounted,
                  static_cast<unsigned long long>(obs::sys::total()));
    text += line;
    for (std::size_t n = 0; n != count; ++n) {
      const kept& item = records[n];
      std::snprintf(line, sizeof(line),
                    "    {\"epoch\": %llu, \"sequence\": %llu, \"opportunity\": \"%016llx\", \"depth\": %u, \"bytes\": %u, "
                    "\"t_publish_ns\": %lld, \"t_read_ns\": %lld, \"delay_ns\": %lld, \"intact\": %s, \"file\": %s}%s\n",
                    static_cast<unsigned long long>(item.epoch), static_cast<unsigned long long>(item.sequence),
                    static_cast<unsigned long long>(item.opportunity), item.depth, item.bytes,
                    static_cast<long long>(item.t_publish_ns), static_cast<long long>(item.t_read_ns),
                    static_cast<long long>(item.t_read_ns - item.t_publish_ns), item.intact ? "true" : "false",
                    output_dir.empty() || !item.intact ? "null" : "true", n + 1 == count ? "" : ",");
      text += line;
    }
    text += "  ]\n}\n";
    if (!output_dir.empty() && !write_file(output_dir + "/consumer.json", text.data(), text.size())) {
      std::fprintf(stderr, "obs_tail: cannot write %s/consumer.json\n", output_dir.c_str());
      return 1;
    }
    std::printf("obs_tail: epoch %llu, %s, %llu records read, %llu stale, %llu released, ended %s\n",
                static_cast<unsigned long long>(reader.epoch()), wait_name.c_str(),
                static_cast<unsigned long long>(counts.read), static_cast<unsigned long long>(counts.stale),
                static_cast<unsigned long long>(counts.released), closed ? "closed" : "idle");
    return 0;
  } catch (const obs::ring_error& error) {
    std::fprintf(stderr, "obs_tail: %s\n", error.what());
    return 2;
  }
}
