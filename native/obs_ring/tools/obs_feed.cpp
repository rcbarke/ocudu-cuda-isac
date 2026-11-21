// obs_feed: a producer of known records on one observation ring (Sprint 3.2, selector
// snapshot_ring).
//
// It creates a ring, waits for a consumer, publishes records whose every byte follows from the
// record number, and reports what the ring counted and what one publication cost. It feeds a
// reader without a receiver: the unit vectors that both readers must pass, and a sensor under
// development. It also measures the ring alone on a host.
//
//     obs_feed --socket PATH [--records N] [--slots N] [--payload-bytes N] [--period-us N]
//              [--attach-ms N] [--drain-ms N] [--evidence audit|timed] [--cpu N]
//
// Record n (from 0): opportunity key n + 1, episode key 0x0b5e0b5e0b5e0b5e, grant key 7, cell 1,
// ue 0x4601, absolute slot 20 n + 8, stage O2, layout ocudu_o2_comb_c64, complex64, scale 0.5,
// shape [payload_bytes / 8] (or [4, 2, 306] for 19,584 bytes), every validity bit set, payload
// byte i = (31 n + 7 i) mod 256, host block of 8 bytes holding n as a little-endian integer.
//
// It links obs_ring and nothing of OCUDU. ASCII only.

#include "obs/ring.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
  std::string socket_path, evidence = "timed";
  long        records = 16, slots = 8, payload_bytes = 19584, period_us = 0, attach_ms = 10000, drain_ms = 2000, cpu = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg  = argv[i];
    const bool        more = i + 1 < argc;
    if (arg == "--socket" && more) {
      socket_path = argv[++i];
    } else if (arg == "--records" && more) {
      records = std::atol(argv[++i]);
    } else if (arg == "--slots" && more) {
      slots = std::atol(argv[++i]);
    } else if (arg == "--payload-bytes" && more) {
      payload_bytes = std::atol(argv[++i]);
    } else if (arg == "--period-us" && more) {
      period_us = std::atol(argv[++i]);
    } else if (arg == "--attach-ms" && more) {
      attach_ms = std::atol(argv[++i]);
    } else if (arg == "--drain-ms" && more) {
      drain_ms = std::atol(argv[++i]);
    } else if (arg == "--evidence" && more) {
      evidence = argv[++i];
    } else if (arg == "--cpu" && more) {
      cpu = std::atol(argv[++i]);
    } else if (arg == "--help") {
      std::puts("Usage: obs_feed --socket PATH [--records N] [--slots N] [--payload-bytes N] [--period-us N]\n"
                "                [--attach-ms N] [--drain-ms N] [--evidence audit|timed] [--cpu N]");
      return 0;
    } else {
      std::fprintf(stderr, "obs_feed: unknown or incomplete option '%s'\n", arg.c_str());
      return 2;
    }
  }
  if (socket_path.empty() || records < 0 || slots < 1 || payload_bytes < 8 || payload_bytes % 8 != 0 || period_us < 0 ||
      (evidence != "audit" && evidence != "timed")) {
    std::fprintf(stderr, "obs_feed: --socket is required, and the payload holds whole complex64 values\n");
    return 2;
  }
  if (cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<unsigned>(cpu), &set);
    if (::sched_setaffinity(0, sizeof(set), &set) != 0) {
      std::fprintf(stderr, "obs_feed: cannot run on CPU %ld\n", cpu);
      return 2;
    }
  }
  const std::uint32_t elements   = static_cast<std::uint32_t>(payload_bytes / 8);
  const std::uint32_t mask_bytes = (elements + 7) / 8;
  try {
    obs::ring_config config;
    config.socket_path      = socket_path;
    config.slot_count       = static_cast<std::uint32_t>(slots);
    config.host_capacity    = 8;
    config.mask_capacity    = mask_bytes;
    config.payload_capacity = static_cast<std::uint32_t>(payload_bytes);
    obs::producer writer(config);
    const bool    attached = attach_ms > 0 && writer.wait_attached(std::chrono::milliseconds(attach_ms));
    if (attach_ms > 0 && !attached) {
      std::fprintf(stderr, "obs_feed: no consumer attached within %ld ms\n", attach_ms);
      return 1;
    }
    // The header that every record shares is built once. A record costs one copy of it.
    obs::record_header shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.schema         = obs::RECORD_SCHEMA;
    shared.header_bytes   = obs::RECORD_HEADER_BYTES;
    shared.episode        = 0x0b5e0b5e0b5e0b5eULL;
    shared.grant          = 7;
    shared.cell           = 1;
    shared.ue             = 0x4601;
    shared.stage          = obs::STAGE_O2;
    shared.source         = obs::SOURCE_PUSCH_DMRS;
    shared.layout         = obs::LAYOUT_OCUDU_O2_COMB_C64;
    shared.dtype          = obs::DTYPE_C64;
    shared.producer_steps = obs::STEP_PILOT_LS;
    shared.scale          = 0.5;
    if (payload_bytes == 19584) {
      shared.rank       = 3;
      shared.shape[0]   = 4;
      shared.shape[1]   = 2;
      shared.shape[2]   = 306;
      shared.strides[0] = 612;
      shared.strides[1] = 306;
      shared.strides[2] = 1;
      shared.rx_count   = 4;
    } else {
      shared.rank       = 1;
      shared.shape[0]   = elements;
      shared.strides[0] = 1;
      shared.rx_count   = 1;
    }
    for (unsigned plane = 0; plane != shared.rx_count; ++plane) {
      shared.antenna_map[plane] = static_cast<std::uint16_t>(plane);
    }
    shared.first_subcarrier = 0;
    shared.subcarrier_step  = 2;
    shared.subcarrier_count = payload_bytes == 19584 ? 306 : elements;
    shared.dmrs_symbols     = (1U << 2) | (1U << 11);
    shared.layer_count      = 1;
    shared.mask_bytes       = mask_bytes;
    shared.payload_bytes    = static_cast<std::uint32_t>(payload_bytes);
    shared.host_bytes       = 8;
    shared.time_unit        = obs::TIME_TC;
    shared.digest_kind      = obs::DIGEST_NONE;
    shared.evidence         = evidence == "audit" ? obs::EVIDENCE_AUDIT : obs::EVIDENCE_TIMED;
    shared.valid            = elements;

    std::vector<std::int64_t> cost(static_cast<std::size_t>(records), 0);
    std::size_t               measured = 0;
    std::int64_t              next     = obs::monotonic_ns();
    for (long n = 0; n != records; ++n) {
      if (period_us > 0) {
        next += period_us * 1000LL;
        while (obs::monotonic_ns() < next) {
        }
      }
      const std::int64_t before = obs::monotonic_ns();
      obs::slot          acquired;
      if (writer.try_acquire(acquired)) {
        *acquired.header               = shared;
        acquired.header->opportunity   = static_cast<std::uint64_t>(n) + 1;
        acquired.header->absolute_slot = 20ULL * static_cast<std::uint64_t>(n) + 8;
        acquired.header->t_first_sample = 61440ULL * static_cast<std::uint64_t>(n);
        acquired.header->t_last_sample  = 61440ULL * static_cast<std::uint64_t>(n) + 4096;
        const std::uint64_t number      = static_cast<std::uint64_t>(n);
        std::memcpy(acquired.host, &number, 8);
        std::memset(acquired.mask, 0xff, mask_bytes);
        if (elements % 8 != 0) {
          acquired.mask[mask_bytes - 1] = static_cast<std::uint8_t>((1U << (elements % 8)) - 1U);
        }
        for (long i = 0; i != payload_bytes; ++i) {
          acquired.payload[i] = static_cast<std::uint8_t>(31 * n + 7 * i);
        }
        acquired.header->t_ready_ns = obs::monotonic_ns();
        writer.publish(acquired);
        cost[measured++] = obs::monotonic_ns() - before;
      }
      writer.service(); // outside the window between try_acquire and publish
    }
    const std::uint64_t pending = writer.drain(std::chrono::milliseconds(drain_ms));
    writer.finish();
    const obs::producer_counts counts = writer.counts();
    std::sort(cost.begin(), cost.begin() + static_cast<std::ptrdiff_t>(measured));
    const auto at = [&](double share) {
      return measured == 0 ? 0LL : static_cast<long long>(cost[static_cast<std::size_t>(share * (measured - 1))]);
    };
    std::printf("{\"schema\": \"obs-feed/v1\", \"definition\": \"%016llx\", \"records\": %ld, \"slots\": %ld, "
                "\"payload_bytes\": %ld, \"slot_bytes\": %u, \"segment_bytes\": %zu, \"admitted\": %llu, "
                "\"published\": %llu, \"dropped\": %llu, \"released\": %llu, \"reclaimed\": %llu, \"pending\": %llu, "
                "\"pending_at_cutoff\": %llu, \"faults\": %llu, \"epochs\": %llu, \"wakes\": %llu, \"window_calls\": %llu, "
                "\"max_depth\": %u, \"fill_and_publish_ns\": {\"p50\": %lld, \"p99\": %lld, \"max\": %lld}}\n",
                static_cast<unsigned long long>(obs::DEFINITION), records, slots, payload_bytes,
                writer.header().slot_bytes, writer.segment_bytes(), static_cast<unsigned long long>(counts.admitted),
                static_cast<unsigned long long>(counts.published), static_cast<unsigned long long>(counts.dropped),
                static_cast<unsigned long long>(counts.released), static_cast<unsigned long long>(counts.reclaimed),
                static_cast<unsigned long long>(counts.pending), static_cast<unsigned long long>(pending),
                static_cast<unsigned long long>(counts.faults), static_cast<unsigned long long>(counts.epochs),
                static_cast<unsigned long long>(counts.wakes), static_cast<unsigned long long>(counts.window_calls),
                counts.max_depth, at(0.5), at(0.99), at(1.0));
    return 0;
  } catch (const obs::ring_error& error) {
    std::fprintf(stderr, "obs_feed: %s\n", error.what());
    return 2;
  }
}
