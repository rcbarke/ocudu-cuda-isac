// sense_feed: a producer that publishes the records of a consumer archive on one observation ring
// (Sprint 3.6, selector sense_core).
//
// It feeds a sensor without a receiver. The records are those a consumer kept of an earlier run:
// their headers, their validity bits and their payloads are published unchanged, in the order of
// the archive, at the pace of their slots. Three fields are the carrier's and are set here:
// epoch, sequence and t_publish_ns. t_ready_ns is set to the time each record was complete in
// its slot, as a producer does, so that a consumer measures its own delay from this ring.
//
// It is the producer of the Sprint 3.6 gate where a sensor runs alone on the device: the offered
// trace is the same for every runtime that is compared. It is no receiver, and nothing it prints
// says what a radio pays for a publication.
//
//     sense_feed --socket PATH --archive DIR [--slots N] [--pace slots|fixed|none] [--slot-us N]
//                [--period-us N] [--period-slots N] [--attach-ms N] [--drain-ms N] [--repeat N]
//                [--cpu N]
//
//   --pace slots   (default) a record follows the one before it by the distance of their slots,
//                  --slot-us each (default 500). A slot that does not follow, as at the start of
//                  another episode, comes one period later (--period-slots, default 10).
//   --pace fixed   one record every --period-us microseconds.
//   --pace none    as fast as the ring admits: for the overload cases.
//
// It links obs_ring and nothing of OCUDU, CUDA or the sensing core. ASCII only.

#include "obs/ring.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sched.h>
#include <string>
#include <vector>

namespace {

void usage()
{
  std::puts("Usage: sense_feed --socket PATH --archive DIR [--slots N] [--pace slots|fixed|none] [--slot-us N]\n"
            "                  [--period-us N] [--period-slots N] [--attach-ms N] [--drain-ms N] [--repeat N] [--cpu N]\n"
            "  --socket PATH     where the control channel of the ring listens\n"
            "  --archive DIR     a consumer archive: every *.obs file, in the order of its name\n"
            "  --slots N         slots of the ring, a power of two (default 8)\n"
            "  --pace MODE       slots (default), fixed or none\n"
            "  --slot-us N       pace slots: the length of one slot (default 500)\n"
            "  --period-us N     pace fixed: the time between two records (default 500)\n"
            "  --period-slots N  pace slots: slots between two episodes (default 10)\n"
            "  --attach-ms N     how long to wait for the consumer (default 10000)\n"
            "  --drain-ms N      how long to wait for the releases at the end (default 5000)\n"
            "  --repeat N        publish the archive N times (default 1)\n"
            "  --cpu N           run on this CPU");
}

struct stored {
  obs::record_header header;
  std::size_t        at = 0; // of the parts behind the header, in the arena
};

} // namespace

int main(int argc, char** argv)
{
  std::string socket_path, archive, pace = "slots";
  long        slots = 8, slot_us = 500, period_us = 500, period_slots = 10, attach_ms = 10000, drain_ms = 5000, repeat = 1;
  long        cpu = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg  = argv[i];
    const bool        more = i + 1 < argc;
    if (arg == "--help") {
      usage();
      return 0;
    } else if (arg == "--socket" && more) {
      socket_path = argv[++i];
    } else if (arg == "--archive" && more) {
      archive = argv[++i];
    } else if (arg == "--slots" && more) {
      slots = std::atol(argv[++i]);
    } else if (arg == "--pace" && more) {
      pace = argv[++i];
    } else if (arg == "--slot-us" && more) {
      slot_us = std::atol(argv[++i]);
    } else if (arg == "--period-us" && more) {
      period_us = std::atol(argv[++i]);
    } else if (arg == "--period-slots" && more) {
      period_slots = std::atol(argv[++i]);
    } else if (arg == "--attach-ms" && more) {
      attach_ms = std::atol(argv[++i]);
    } else if (arg == "--drain-ms" && more) {
      drain_ms = std::atol(argv[++i]);
    } else if (arg == "--repeat" && more) {
      repeat = std::atol(argv[++i]);
    } else if (arg == "--cpu" && more) {
      cpu = std::atol(argv[++i]);
    } else {
      usage();
      std::fprintf(stderr, "sense_feed: unknown or incomplete option '%s'\n", arg.c_str());
      return 2;
    }
  }
  if (socket_path.empty() || archive.empty() || slots < 1 || (slots & (slots - 1)) != 0 || slot_us < 1 || period_us < 1 ||
      period_slots < 1 || attach_ms < 0 || drain_ms < 0 || repeat < 1 ||
      (pace != "slots" && pace != "fixed" && pace != "none")) {
    usage();
    std::fprintf(stderr, "sense_feed: --socket and --archive are required, --slots is a power of two, and --pace is "
                         "slots, fixed or none\n");
    return 2;
  }
  if (cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<unsigned>(cpu), &set);
    if (::sched_setaffinity(0, sizeof(set), &set) != 0) {
      std::fprintf(stderr, "sense_feed: cannot run on CPU %ld\n", cpu);
      return 2;
    }
  }

  // The archive, whole, before the ring exists: nothing is read from a disk while it publishes.
  std::vector<std::string> names;
  DIR*                     listing = ::opendir(archive.c_str());
  if (listing == nullptr) {
    std::fprintf(stderr, "sense_feed: cannot list the archive %s\n", archive.c_str());
    return 2;
  }
  while (const dirent* found = ::readdir(listing)) {
    const std::string name = found->d_name;
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".obs") == 0) {
      names.push_back(name);
    }
  }
  ::closedir(listing);
  std::sort(names.begin(), names.end());
  std::vector<stored>       records;
  std::vector<std::uint8_t> arena;
  std::uint32_t             host_max = 0, mask_max = 0, payload_max = 0;
  for (const std::string& name : names) {
    std::FILE* file = std::fopen((archive + "/" + name).c_str(), "rb");
    stored     item;
    if (file == nullptr || std::fread(&item.header, 1, sizeof(item.header), file) != sizeof(item.header) ||
        item.header.schema != obs::RECORD_SCHEMA || item.header.header_bytes != obs::RECORD_HEADER_BYTES) {
      std::fprintf(stderr, "sense_feed: %s/%s is not an obs/v1 record\n", archive.c_str(), name.c_str());
      if (file != nullptr) {
        std::fclose(file);
      }
      return 2;
    }
    const std::size_t rest = static_cast<std::size_t>(item.header.host_bytes) + item.header.mask_bytes + item.header.payload_bytes;
    item.at                = arena.size();
    arena.resize(arena.size() + rest);
    const bool whole = std::fread(arena.data() + item.at, 1, rest, file) == rest && std::fgetc(file) == EOF;
    std::fclose(file);
    if (!whole) {
      std::fprintf(stderr, "sense_feed: %s/%s does not hold the parts its header names\n", archive.c_str(), name.c_str());
      return 2;
    }
    host_max    = std::max(host_max, item.header.host_bytes);
    mask_max    = std::max(mask_max, item.header.mask_bytes);
    payload_max = std::max(payload_max, item.header.payload_bytes);
    records.push_back(item);
  }
  if (records.empty()) {
    std::fprintf(stderr, "sense_feed: the archive %s holds no record\n", archive.c_str());
    return 2;
  }

  try {
    obs::ring_config config;
    config.socket_path      = socket_path;
    config.slot_count       = static_cast<std::uint32_t>(slots);
    config.host_capacity    = host_max;
    config.mask_capacity    = mask_max;
    config.payload_capacity = payload_max;
    obs::producer writer(config);
    if (attach_ms > 0 && !writer.wait_attached(std::chrono::milliseconds(attach_ms))) {
      std::fprintf(stderr, "sense_feed: no consumer attached within %ld ms\n", attach_ms);
      return 1;
    }
    const std::size_t         total = records.size() * static_cast<std::size_t>(repeat);
    std::vector<std::int64_t> cost(total, 0);
    std::size_t               measured = 0;
    std::int64_t              next     = obs::monotonic_ns();
    const std::int64_t        started  = next;
    std::uint64_t             slot_before = 0;
    bool                      first       = true;
    for (long round = 0; round != repeat; ++round) {
      for (const stored& item : records) {
        if (pace == "slots") {
          const std::uint64_t slot  = item.header.absolute_slot;
          const std::uint64_t steps = !first && slot > slot_before ? slot - slot_before : static_cast<std::uint64_t>(period_slots);
          next += first ? 0 : static_cast<std::int64_t>(steps) * slot_us * 1000LL;
          slot_before = slot;
        } else if (pace == "fixed") {
          next += first ? 0 : period_us * 1000LL;
        }
        first = false;
        if (pace != "none") {
          while (obs::monotonic_ns() < next) {
          }
        }
        const std::int64_t before = obs::monotonic_ns();
        obs::slot          acquired;
        if (writer.try_acquire(acquired)) {
          *acquired.header            = item.header;
          const std::uint8_t* parts   = arena.data() + item.at;
          std::memcpy(acquired.host, parts, item.header.host_bytes);
          std::memcpy(acquired.mask, parts + item.header.host_bytes, item.header.mask_bytes);
          std::memcpy(acquired.payload, parts + item.header.host_bytes + item.header.mask_bytes, item.header.payload_bytes);
          acquired.header->t_ready_ns = obs::monotonic_ns(); // the record is complete in its slot
          writer.publish(acquired);
          cost[measured++] = obs::monotonic_ns() - before;
        }
        writer.service(); // outside the window between try_acquire and publish
      }
    }
    const std::int64_t  ended   = obs::monotonic_ns();
    const std::uint64_t pending = writer.drain(std::chrono::milliseconds(drain_ms));
    writer.finish();
    const obs::producer_counts counts = writer.counts();
    std::sort(cost.begin(), cost.begin() + static_cast<std::ptrdiff_t>(measured));
    const auto at = [&](double share) {
      return measured == 0 ? 0LL : static_cast<long long>(cost[static_cast<std::size_t>(share * static_cast<double>(measured - 1))]);
    };
    std::printf("{\"schema\": \"sense-feed/v1\", \"definition\": \"%016llx\", \"archive_records\": %zu, \"repeat\": %ld, "
                "\"offered\": %zu, \"pace\": \"%s\", \"slot_us\": %ld, \"period_us\": %ld, \"slots\": %ld, \"slot_bytes\": %u, "
                "\"admitted\": %llu, \"published\": %llu, \"dropped\": %llu, \"released\": %llu, \"reclaimed\": %llu, "
                "\"pending\": %llu, \"pending_at_cutoff\": %llu, \"faults\": %llu, \"epochs\": %llu, \"wakes\": %llu, "
                "\"window_calls\": %llu, \"max_depth\": %u, \"duration_ns\": %lld, "
                "\"fill_and_publish_ns\": {\"p50\": %lld, \"p99\": %lld, \"max\": %lld}}\n",
                static_cast<unsigned long long>(obs::DEFINITION), records.size(), repeat, total, pace.c_str(), slot_us,
                period_us, slots, writer.header().slot_bytes, static_cast<unsigned long long>(counts.admitted),
                static_cast<unsigned long long>(counts.published), static_cast<unsigned long long>(counts.dropped),
                static_cast<unsigned long long>(counts.released), static_cast<unsigned long long>(counts.reclaimed),
                static_cast<unsigned long long>(counts.pending), static_cast<unsigned long long>(pending),
                static_cast<unsigned long long>(counts.faults), static_cast<unsigned long long>(counts.epochs),
                static_cast<unsigned long long>(counts.wakes), static_cast<unsigned long long>(counts.window_calls),
                counts.max_depth, static_cast<long long>(ended - started), at(0.5), at(0.99), at(1.0));
    return 0;
  } catch (const obs::ring_error& error) {
    std::fprintf(stderr, "sense_feed: %s\n", error.what());
    return 2;
  }
}
