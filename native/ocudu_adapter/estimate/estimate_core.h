// Capture of the receiver's own channel estimate (Sprint 5.2, selector native_estimate). Host-free core.
//
// The resident PUSCH path of the pinned receiver estimates the channel on the device and keeps it
// there. After pilot least squares, frequency interpolation and frequency smoothing the estimate
// lies in one buffer of the kernel handle, per DM-RS symbol and before time interpolation: the
// stage O3. The receiver patch of patches/ocudu/ tells an observer where that buffer is, once per
// PDU, after its kernels were enqueued. This core is the observer's side of that call without
// the device: what it expects of the next estimate, which estimate it refuses, where the copy
// lands, when a copy may be published and how every eligible estimate is accounted for.
//
// Payload layout ocudu_o3_fp16, exactly as the receiver holds it:
//
//     [dmrs_symbol][prb][rx_port][subcarrier_in_prb]     complex FP16 pairs {real, imaginary}
//
// with the last axis fastest. Nothing is reordered, scaled or converted in the receive path: the
// bytes of the record are the bytes of the device buffer. Element [1][3][2][5] is DM-RS symbol 1,
// receive port 2 and subcarrier 12 * 3 + 5 = 41 of the allocation. A sensing core widens FP16 to
// its working precision once, in its ingest unit. This core widens only to inspect a payload.
//
// Validity: one bit per element, in payload order, bit (n mod 8) of byte (n / 8). The estimate is
// interpolated, so every element of the allocation is defined. A bit is cleared only where a
// part of the value is not finite. A measured zero is a set bit with the value zero.
//
// One capture is open at a time, in three steps that the owner calls in this order.
//
//   expect      before the decode that will produce the estimate. With a ring this is the
//               admission: one credit or a counted drop.
//   on_estimate from the receiver's tap, on the thread that demodulates. It compares what the
//               receiver reports with what was expected, and submits one asynchronous copy of
//               the device buffer, ordered after the producer. It never waits and never throws.
//   resolve     after the decode. It asks whether the copy is complete, without waiting for the
//               device, sets the validity bits and publishes. Only a completed copy of the
//               expected estimate is published.
//
// The device is behind the interface copier. The tool implements it with the CUDA runtime
// (observer/estimate_tap.h) and the unit target with a producer that completes when the case
// says so.
//
// Two evidence modes, as everywhere (Sprint 3.2).
//   timed  the copy lands in the slot of the ring that was acquired for it, where the payload
//          area is page-locked for the device, and otherwise in page-locked staging that is
//          copied into the slot. No hash, no allocation and no text.
//   audit  the copy lands in the staging, is digested, and a copy of it is published with its
//          digest when the plan names a ring. The owner writes the payload to a file.
//
// This file includes the producer glue of the ring, the observer core for the bound slot and the
// channel application core, and nothing of OCUDU or CUDA. ASCII only.

#pragma once

#include "obs/ring.h"
#include "obs_publisher.h"
#include "observer_core.h"
#include "replay_core.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace isac_estimate {

using isac_replay::cf32;
using isac_replay::evidence;
using isac_replay::input_error;
using isac_replay::state_error;
using isac_replay::ticket;

constexpr const char* TAP_NATIVE_ESTIMATE = "native_estimate";
constexpr const char* STAGE_O3            = "O3";
constexpr const char* PRODUCER            = "receiver";
constexpr const char* LAYOUT_O3_FP16      = "ocudu_o3_fp16";
constexpr const char* DTYPE_CFP16         = "<cfp16";
constexpr const char* RULE =
    "pilot LS, frequency interpolation and frequency smoothing of the resident PUSCH kernels, per DM-RS symbol";
constexpr const char* PRECISION       = "complex FP16, the bytes of the device buffer";
constexpr const char* COPY_PRODUCER   = "producer";
constexpr const char* COPY_SIDE       = "side";
constexpr const char* LANDING_SLOT    = "slot";
constexpr const char* LANDING_STAGING = "staging";
constexpr unsigned    SUBCARRIERS_PER_PRB = 12;
/// Bytes of one element: two half-precision parts.
constexpr std::size_t ELEMENT_BYTES = 4;

// ---------------------------------------------------------------------------------------------- half precision

/// The value of an IEEE 754 binary16 number. Exact: every half is a float.
float half_to_float(std::uint16_t bits) noexcept;
/// The binary16 number nearest to a float, ties to even. A magnitude beyond the range is infinite.
std::uint16_t float_to_half(float value) noexcept;
inline bool half_is_finite(std::uint16_t bits) noexcept
{
  return (bits & 0x7c00U) != 0x7c00U;
}

// ---------------------------------------------------------------------------------------------- layout

/// The rectangle of one payload in the layout ocudu_o3_fp16, for one layer.
struct layout {
  unsigned dmrs_symbols = 0;
  unsigned prbs         = 0;
  unsigned rx_ports     = 0;

  std::size_t elements() const
  {
    return static_cast<std::size_t>(dmrs_symbols) * prbs * rx_ports * SUBCARRIERS_PER_PRB;
  }
  std::size_t payload_bytes() const { return elements() * ELEMENT_BYTES; }
  std::size_t mask_bytes() const { return (elements() + 7) / 8; }
  /// Position of one element in the payload. Throws input_error for an index outside the rectangle.
  std::size_t element(unsigned dmrs_symbol, unsigned prb, unsigned rx_port, unsigned subcarrier_in_prb) const;
};

/// What one element of a payload is.
struct coordinate {
  unsigned dmrs_symbol       = 0; ///< Position among the DM-RS symbols of the slot, in rising order.
  unsigned prb               = 0; ///< Counted from the first resource block of the allocation.
  unsigned rx_port           = 0;
  unsigned subcarrier_in_prb = 0;
  unsigned subcarrier        = 0; ///< 12 * prb + subcarrier_in_prb: counted from the first subcarrier of the allocation.
};

/// Throws input_error for an element outside the rectangle.
coordinate coordinate_of(const layout& shape, std::size_t element);

/// One element of a payload, widened to complex64 without change.
cf32 value_at(const void* payload, std::size_t element) noexcept;

struct payload_counts {
  std::size_t valid   = 0; ///< Elements whose two parts are finite.
  std::size_t holes   = 0; ///< Elements with a part that is not finite.
  std::size_t nonzero = 0; ///< Valid elements that are not zero.
};

/// Sets the validity bit of every element whose two parts are finite, clears every other bit of
/// mask, which holds (elements + 7) / 8 bytes, and counts. It allocates nothing.
payload_counts scan(const void* payload, std::size_t elements, std::uint8_t* mask) noexcept;

// ---------------------------------------------------------------------------------------------- producer

/// What the receiver reports of one estimate. The fields are those of ocudu::pusch_resident_estimate
/// of the receiver patch, so that this core needs no header of the receiver.
struct producer_view {
  const void*   estimates = nullptr; ///< Device pointer. This core never reads through it.
  std::size_t   bytes     = 0;
  void*         stream    = nullptr; ///< The stream every writer of the estimate was enqueued on.
  int           device    = 0;
  std::uint64_t launch    = 0; ///< Counts the PDUs that reached the tap in the receiver's process, from 1.
  unsigned      rnti       = 0;
  unsigned      sfn        = 0;
  unsigned      slot       = 0;
  unsigned      numerology = 0;
  unsigned      start_prb  = 0; ///< First resource block, counted from subcarrier 0 of the grid.
  unsigned      nof_prb    = 0;
  bool          contiguous = true;
  std::uint64_t dmrs_symbol_mask = 0;
  unsigned      nof_dmrs_symbols = 0;
  unsigned      rx_ports         = 0;
  unsigned      layers           = 0;
  unsigned      dmrs_type        = 1;
  float         dmrs_scaling     = 1.0F;
  bool          frequency_interpolation = true;
  bool          frequency_smoothing     = false;
  bool          resident                = false;
};

/// What the owner expects of the next estimate: the PDU of the decode it is about to start.
struct expectation {
  unsigned      rnti      = 0;
  unsigned      sfn       = 0;
  unsigned      slot      = 0;
  unsigned      start_prb = 0;
  unsigned      nof_prb   = 0;
  unsigned      rx_ports  = 0;
  unsigned      dmrs_type = 1;
  std::uint64_t dmrs_symbol_mask = 0;

  layout shape() const;
};

/// Why an estimate is not the one that was expected, or an empty text when it is. The reasons,
/// in the order they are tested: "null_estimate", "layers" (this producer exports one layer),
/// "not_contiguous", "not_interpolated", "other_pdu" (another RNTI, frame or slot),
/// "other_allocation" (other blocks, DM-RS symbols, ports or DM-RS type) and "size".
const char* refusal(const expectation& expected, const producer_view& view) noexcept;

// ---------------------------------------------------------------------------------------------- device

enum class copy_state { pending, complete, failed };

/// The device side of a capture. One copy is in flight at a time.
class copier
{
public:
  virtual ~copier() = default;
  /// Enqueues the copy of view.bytes bytes from view.estimates to destination, ordered after the
  /// producer of the estimate. It never waits for the device. False when nothing was enqueued.
  virtual bool submit(const producer_view& view, void* destination) noexcept = 0;
  /// The state of the copy that was submitted last. It never waits for the device.
  virtual copy_state poll() noexcept = 0;
  /// Waits until the copy that was submitted last has ended. Only the owner calls it, after a
  /// resolve that ran out of time, and never inside a decode.
  virtual copy_state drain() noexcept = 0;
  /// True when the copy is enqueued on the producer's own stream. A later launch of the same
  /// producer then cannot overtake it. A copy on another stream is ordered after the producer by
  /// an event only: the owner must resolve it before it starts the next decode.
  virtual bool on_producer_stream() const noexcept = 0;
  /// "producer" or "side".
  virtual const char* stream_name() const noexcept = 0;
};

// ---------------------------------------------------------------------------------------------- capture

struct capture_config {
  evidence      mode             = evidence::audit;
  /// The payload area of the ring is page-locked for the device: a timed copy lands in its slot.
  bool          slot_landing     = false;
  /// Page-locked memory for one payload, used wherever the copy does not land in a slot.
  std::uint8_t* staging          = nullptr;
  std::size_t   staging_bytes    = 0;
  /// How long resolve asks for the completion of a copy before it gives up, in nanoseconds.
  std::int64_t  resolve_limit_ns = 2000000;
};

enum class outcome {
  published, ///< One obs/v1 record on the ring.
  captured,  ///< Audit without a ring: the payload is complete and the owner archives it.
  dropped,   ///< No credit at the admission.
  absent,    ///< The receiver reported no estimate for the decode: it did not run the resident path.
  refused,   ///< The receiver reported an estimate that is not the expected one.
  failed     ///< The copy was not submitted, ended with an error, did not complete in time or was overtaken.
};

const char* to_string(outcome value);

/// One line of the account: what became of the estimate of one expected decode.
struct capture_line {
  std::size_t   handle = 0;
  outcome       result = outcome::absent;
  const char*   reason = "";    ///< Empty for published and captured. "no_credit", "no_estimate", a refusal, or
                                ///< "submit", "copy", "timeout", "overtaken", "busy" of a failure.
  bool          admitted = false; ///< The ring gave a slot at the admission. Always false without a ring.
  bool          seen   = false; ///< The receiver called the tap for this decode.
  bool          landed = false; ///< A complete payload exists: published, captured, or dropped in the mode audit.
  producer_view producer;       ///< What the receiver reported, when seen.
  const char*   copy_stream = ""; ///< "producer" or "side", when a copy was submitted.
  const char*   landing     = ""; ///< "slot" or "staging", when a copy was submitted.
  std::uint64_t epoch    = 0;
  std::uint64_t sequence = 0;
  std::uint32_t depth    = 0;
  std::uint32_t bytes    = 0; ///< Header, host block, validity bits and payload of a published record.
  ticket        grid;
  payload_counts counts;
  std::int64_t  t_expect_ns   = 0; ///< Before the admission.
  std::int64_t  t_tap_ns      = 0; ///< The tap was entered.
  std::int64_t  t_submit_ns   = 0; ///< The copy is enqueued and the tap returns.
  std::int64_t  t_resolve_ns  = 0; ///< resolve was entered, after the decode.
  std::int64_t  t_complete_ns = 0; ///< The completion of the copy was observed: t_ready_ns of the record.
  std::int64_t  t_publish_ns  = 0; ///< As the ring stamped it, just before the publication store.
  std::int64_t  t_done_ns     = 0; ///< resolve returns.
  std::uint32_t polls         = 0; ///< Times resolve asked for the completion.
  double        submit_us  = 0.0; ///< t_submit - t_tap: what the tap costs inside the decode.
  double        resolve_us = 0.0; ///< t_done - t_resolve: completion, validity bits and publication, after the decode.
  std::uint64_t window_calls = 0; ///< System calls of the ring between the admission and the publication store.
};

class capture
{
public:
  /// The copier and the ring are not owned and must outlive the capture. ring may be null in the
  /// mode audit: the estimates are then captured and not published. Throws input_error for a
  /// timed capture without a ring and for a configuration without staging where one is needed.
  capture(copier& device_, obs::producer* ring_, const capture_config& config_);

  /// Binds one decode before the run, in the order of the schedule: the join keys, the slot and
  /// the sample times of its bound slot, the expected PDU, the antenna map of the receive ports
  /// and the reference-signal port. Returns the handle. Throws input_error for an estimate that
  /// the ring or the staging cannot carry, for more receive ports than a record names and for an
  /// antenna map of another size.
  std::size_t prepare(const isac_observer::bound_slot& slot, const expectation& expected,
                      const std::vector<unsigned>& antenna_map, unsigned port, std::uint32_t opportunity_sequence);
  /// Reserves the account, so that no line of it allocates during the run.
  void reserve(std::size_t captures);

  /// Opens the capture of one prepared decode, before that decode starts. With a ring this is
  /// the admission. Throws state_error while another capture is open. While the copy of an
  /// earlier capture that ran out of time can still write its destination, nothing is admitted
  /// and nothing is copied: the capture ends as failed with the reason "busy". It never waits.
  void expect(std::size_t handle, const ticket& grid);
  /// The receiver's tap. Any thread, while the owner is inside the decode. Never waits, never throws.
  void on_estimate(const producer_view& view) noexcept;
  /// Closes the open capture, after the decode. Returns its line. Throws state_error without an open capture.
  const capture_line& resolve();

  /// True after a resolve that ran out of time, until a later expect finds the copy ended or
  /// settle() waited for it: the copy may still write its destination.
  bool in_flight() const noexcept { return flying; }
  /// Waits for the copy in flight and gives its slot back. After the last decode, never inside one.
  void settle();

  /// The payload and the validity bits of the capture that was resolved last, when its line says
  /// landed. They stay valid until the next expect.
  const std::uint8_t* payload() const noexcept { return last_payload; }
  const std::uint8_t* mask() const noexcept { return last_mask; }

  const std::vector<capture_line>& account() const { return lines; }
  const obs::record_header&        header_of(std::size_t handle) const { return headers.at(handle); }
  const expectation&               expected_of(std::size_t handle) const { return expectations.at(handle); }
  std::size_t                      prepared() const { return headers.size(); }
  evidence                         mode() const { return config.mode; }
  /// Estimates that arrived while no capture was open, and while one was open and already served.
  std::uint64_t unexpected() const noexcept { return strays.load(std::memory_order_relaxed); }
  std::uint64_t overruns() const noexcept { return extra.load(std::memory_order_relaxed); }

private:
  enum phase_id : int { idle = 0, expecting = 1, tapping = 2, tapped = 3 };
  enum class verdict { none, submitted, skipped, refused, failed };

  void finish(capture_line& line, outcome result, const char* reason);
  void reclaim();

  copier&                               device;
  obs::producer*                        ring;
  capture_config                        config;
  std::vector<obs::record_header>       headers;
  std::vector<isac_transport::staged_host> hosts;
  std::vector<expectation>              expectations;
  std::vector<capture_line>             lines;
  std::vector<std::uint8_t>             own_mask; ///< Validity bits of a capture that has no slot.
  // The open capture. on_estimate writes tap_* and the producer fields of the line, then stores
  // phase with release order. The owner reads them after it loaded phase with acquire order.
  std::atomic<int>           phase{idle};
  std::atomic<std::uint64_t> strays{0};
  std::atomic<std::uint64_t> extra{0};
  std::atomic<bool>          overtaken{false};
  verdict                    tap_verdict = verdict::none;
  const char*                tap_reason  = "";
  obs::slot                  acquired;
  bool                       has_slot    = false;
  bool                       wants_copy  = false;
  std::uint8_t*              destination = nullptr;
  bool                       flying      = false; ///< A copy that ran out of time may still write its destination.
  bool                       flying_slot = false; ///< That destination is, or is copied into, the slot in acquired.
  bool                       busy        = false; ///< The open capture found a copy in flight and copies nothing.
  const std::uint8_t*        last_payload = nullptr;
  const std::uint8_t*        last_mask    = nullptr;
};

} // namespace isac_estimate
