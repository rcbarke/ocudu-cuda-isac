// Producer glue of the observation ring (Sprint 3.2, selector snapshot_ring). Host-free.
//
// A publisher joins the observer of observer/ to one ring of native/obs_ring/. For every
// opportunity of a run it builds, before the run, the 256 bytes of the record header that do not
// change. Per slot it then does one of two things.
//
//   timed   try_acquire a slot. Without a credit the observation is dropped and counted, and no
//           estimate is computed. With one, the header is copied into the slot, the observer
//           writes its estimate and its validity bits straight into the slot, and the ring
//           publishes. No hash, no allocation, no text and no system call from the read of the
//           grid to the publication store.
//   audit   the observation that observer::observe made, with every digest, is copied into a
//           slot and published with the digest of its payload. This mode proves bytes. It is
//           not timed.
//
// Every eligible observation gets one line of the account: published with its sequence, its
// depth and its times, or dropped with its reason.
//
// This file includes the observer core, the channel application core and the ring, and nothing
// of OCUDU. The unit target links them with doubles only. ASCII only.

#pragma once

#include "obs/ring.h"
#include "observer_core.h"
#include "replay_core.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace isac_transport {

using isac_replay::evidence;
using isac_replay::input_error;

constexpr std::uint32_t HOST_STAGED_HARNESS = 1;

/// The host block of a record that the staged harness publishes. A sensing core never reads it.
struct staged_host {
  std::uint32_t kind;                 ///< HOST_STAGED_HARNESS.
  std::uint32_t opportunity_sequence; ///< Position of the opportunity in the plan.
  std::uint32_t grid_index;           ///< The ticket of the fill: the grid and its generation.
  std::uint32_t sfn;
  std::uint64_t grid_generation;
  std::uint32_t slot;
  std::uint32_t reserved;
  std::int64_t  t_read_ns; ///< Host monotonic time before the observer read the first row.
};
static_assert(sizeof(staged_host) == 40, "the host block of the staged harness has one fixed size");

/// What a ring must hold for the sources of a run.
struct ring_needs {
  std::uint32_t host_bytes    = sizeof(staged_host);
  std::uint32_t mask_bytes    = 0;
  std::uint32_t payload_bytes = 0;

  void include(const isac_observer::pilot_source& source);
};

enum class publication { published, dropped };

/// One line of the account: what became of one eligible observation.
struct published_record {
  std::size_t         handle  = 0; ///< The prepared slot.
  publication         outcome = publication::dropped;
  const char*         reason  = ""; ///< Of a drop: "no_credit".
  std::uint64_t       epoch    = 0;
  std::uint64_t       sequence = 0;
  std::uint32_t       depth    = 0; ///< Records held when this one was admitted.
  std::uint32_t       bytes    = 0; ///< Header, host block, validity bits and payload.
  isac_replay::ticket grid;
  std::size_t         valid = 0;
  std::size_t         holes = 0;
  std::int64_t        t_read_ns    = 0; ///< Before the admission.
  std::int64_t        t_ready_ns   = 0; ///< The estimate is complete.
  std::int64_t        t_publish_ns = 0; ///< As the ring stamped it, just before the publication store.
  std::int64_t        t_done_ns    = 0; ///< After the publication store.
  double              service_us  = 0.0; ///< t_done - t_read: the whole tap.
  double              estimate_us = 0.0; ///< The rows read, the least-squares values, the validity bits.
  double              publish_us  = 0.0; ///< service - estimate: the admission, the header and the publication.
  std::uint64_t       window_calls = 0;  ///< System calls of the ring between the admission and the publication store.
};

class publisher
{
public:
  /// The ring and the observer are not owned and must outlive the publisher. mode is the
  /// evidence mode of the run, which the observer was created with.
  publisher(obs::producer& ring_, isac_observer::observer& watcher_, evidence mode_);

  /// Binds one opportunity before the run, in the order of the schedule. Throws input_error for
  /// a source the ring cannot carry: a payload or a mask beyond the capacity of a slot, more
  /// receive planes than a record names, an unknown source.
  std::size_t prepare(const isac_observer::slot_context& slot, const std::string& grant_record_sha256,
                      std::uint32_t opportunity_sequence);
  /// Reserves the account, so that no line of it allocates during the run.
  void reserve(std::size_t observations);

  /// Timed: admission, the estimate in place, publication. Never waits.
  const published_record& publish(std::size_t handle, const isac_replay::o1_view& view);
  /// Audit: the copy of an observation that was made with every digest.
  const published_record& publish(std::size_t handle, const isac_observer::observation& made);

  const std::vector<published_record>& account() const { return lines; }
  const obs::record_header&            header_of(std::size_t handle) const { return headers.at(handle); }
  evidence                             mode() const { return evidence_mode; }

private:
  obs::producer&                  ring;
  isac_observer::observer&        watcher;
  evidence                        evidence_mode;
  std::vector<obs::record_header> headers; ///< One per prepared slot, complete but for what a slot adds.
  std::vector<staged_host>        hosts;
  std::vector<published_record>   lines;
};

} // namespace isac_transport
