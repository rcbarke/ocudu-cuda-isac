// Pilot least squares on the receiver grid (Sprint 3.1, selector pilot_ls). Host-free core.
//
// An observer reads the ready grid that the PUSCH processors read, through the read access of the
// channel application core, and exports the stage O2: for every pilot of a registered sensing
// source and every receive plane
//
//     H_ls = Y * conj(X) / |X|^2
//
// where Y is the stored value of the grid and X is the reference symbol of the source record. The
// reference is never replaced by the stored pilot, so O2 stays comparable with the receiver's own
// estimate, which is configured with the same amplitude.
//
// Order of the arithmetic, so that a reference in another language reproduces every bit:
//   1. Y and X are single-precision values, widened to double precision without change;
//   2. d  = xr * xr + xi * xi
//      nr = yr * xr + yi * xi
//      ni = yi * xr - yr * xi
//      Every product is exact in double precision, so each sum rounds once and the result does
//      not depend on whether a compiler fuses a product into the sum;
//   3. hr = nr / d, hi = ni / d, each one correctly rounded division;
//   4. each part is converted to single precision (round to nearest, ties to even).
// The estimator precision is therefore complex64, computed through double precision.
//
// Payload layout ocudu_o2_comb_c64: [rx_plane][reference_symbol][pilot], last axis fastest. The
// pilot axis is the sorted set of subcarriers that carry a pilot on any reference symbol of the
// source, so the payload is a rectangle. The planes are in the order of the grid. Nothing is
// reordered: antenna_map[i] names the array element behind plane i and travels in the record.
//
// Validity: one bit per payload element, in payload order, bit (n mod 8) of byte (n / 8).
//   set      the element is a pilot of the source and its estimate is finite. A measured zero
//            (Y = 0 on a pilot) is a set bit with the value zero.
//   cleared  the element is a hole: the source lists no pilot at that symbol and subcarrier, or
//            the estimate is not finite. Its value is zero.
//
// The estimate is observer-derived: it is computed by this code from the grid and is not the
// estimate of the PUSCH processor. That one is the stage O3, with its own producer (Sprint 5.2).
//
// An observation owns its payload and its validity bits. It stays valid after the grid was
// released and filled again. The observer only reads the grid.
//
// Two evidence modes, one arithmetic (Sprint 3.2, decided 2026-11-21). Both call estimate_into.
//   audit  observe(): every digest, as Sprint 3.1 was accepted. The observation is tied to its
//          grid by the readback digest, which is taken again after the read.
//   timed  observe_into(): the estimate is written straight into storage of the caller, a slot of
//          the observation ring. No hash, no allocation and no text. The join keys, the slot
//          count and the sample times were bound before the run (prepare), and the observation is
//          tied to its grid by the ticket of the fill: the index and the generation of the grid.
//
// This file includes the channel application core and nothing of OCUDU. The unit target links it
// with doubles only. All text in this directory is ASCII.

#pragma once

#include "replay_core.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace isac_observer {

using isac_replay::cf32;
using isac_replay::evidence;
using isac_replay::input_error;
using isac_replay::state_error;

constexpr const char* STAGE_O2        = "O2";
constexpr const char* PRODUCER        = "observer";
constexpr const char* STEP_PILOT_LS   = "pilot_ls";
constexpr const char* LAYOUT_O2_COMB  = "ocudu_o2_comb_c64";
constexpr const char* DTYPE_C64       = "<c8";
constexpr const char* TIME_UNIT_TC    = "tc";
constexpr const char* PRECISION       = "complex64, computed in double precision from single-precision operands";
constexpr const char* RULE            = "H = Y conj(X) / |X|^2";
/// Basic time units per second: 480 kHz * 4096 (TS 38.211 4.1).
constexpr std::uint64_t TC_PER_SECOND = 1966080000ULL;
/// Frame numbers wrap at this count (TS 38.211 4.3.1).
constexpr unsigned SFN_PERIOD = 1024;

// ---------------------------------------------------------------------------------------------- kernel

/// The least-squares estimate of one resource element, by the order of arithmetic above.
/// reference must have nonzero power. ok is false when a part of the result is not finite.
cf32 pilot_ls(cf32 received, cf32 reference, bool* ok = nullptr);

// ---------------------------------------------------------------------------------------------- source

struct pilot {
  unsigned symbol     = 0; ///< Symbol index in the slot.
  unsigned subcarrier = 0; ///< Carrier coordinates: 12 * CRB + resource element.
};

/// One sensing-source record (sensing-source/v1) bound to one grant, with its arrays loaded.
struct source_spec {
  std::string grant_id;
  std::string grant_record_sha256; ///< Identity of the grant record, hence of the allocation.
  std::string source;              ///< Name of the registered source, "pusch_dmrs".
  std::string channel;             ///< "PUSCH".
  std::string link_direction;      ///< "uplink" or "downlink".
  std::string processor;           ///< The receiver processor that executes the signal.
  std::vector<unsigned> ports;     ///< Reference-signal ports of the record. One row of reference each.
  unsigned    grid_symbols     = 0;
  unsigned    grid_subcarriers = 0;
  unsigned    scs_khz          = 0;
  unsigned    slot             = 0; ///< Slot index in the frame the reference was generated for.
  std::vector<unsigned>      slot_symbols;           ///< The reference symbols of the slot, increasing.
  std::vector<std::uint64_t> symbol_useful_start_tc; ///< Start of the useful part of each, in T_c from the slot start.
  std::uint64_t              tc_per_second = 0;
  double                     amplitude     = 0.0; ///< Reference amplitude relative to unit-power data.
  std::vector<pilot>         index;               ///< Every pilot once, ordered by symbol, then by subcarrier.
  std::vector<cf32>          reference;           ///< [port][pilot], last axis fastest.
  std::vector<unsigned>      antenna_map;         ///< antenna_map[i] is the array element behind receive plane i.
};

/// A validated source, compiled into the rectangle of the payload.
class pilot_source
{
public:
  /// Checks the record. Throws input_error for a source without pilots, a pilot outside the
  /// grid, a duplicate pilot, pilots out of order, a reference of another size, a configured
  /// pilot with zero reference power, a value that is not finite, more than one port (this
  /// observer estimates one port), reference symbols that differ from the symbols of the mask,
  /// an amplitude that is not positive and an antenna map that names an element twice.
  explicit pilot_source(source_spec spec_);

  const source_spec&           spec() const { return record; }
  const std::vector<unsigned>& symbols() const { return rows; }
  const std::vector<unsigned>& subcarriers() const { return columns; }
  unsigned                     rx_planes() const { return static_cast<unsigned>(record.antenna_map.size()); }
  /// Elements of one receive plane: reference symbols times pilot subcarriers.
  std::size_t plane_elements() const { return rows.size() * columns.size(); }
  /// Index into the pilots of the record for one element of the rectangle, or -1 for a hole.
  long long pilot_at(std::size_t symbol_row, std::size_t column) const { return cells[symbol_row * columns.size() + column]; }
  std::size_t holes() const { return nof_holes; }
  /// First subcarrier, step and count of the pilot axis. The step is 0 when the axis is not regular.
  unsigned first_subcarrier() const { return columns.front(); }
  unsigned subcarrier_step() const { return step; }
  /// SHA-256 over the coordinates of the payload axes: the symbols, then the subcarriers, each a
  /// little-endian uint32.
  const std::string& coordinates_sha256() const { return coordinates; }
  /// The receive plane that holds an array element. Throws input_error for an element the map does not name.
  unsigned plane_of(unsigned element) const;

private:
  source_spec            record;
  std::vector<unsigned>  rows;
  std::vector<unsigned>  columns;
  std::vector<long long> cells;
  std::size_t            nof_holes = 0;
  unsigned               step      = 0;
  std::string            coordinates;
};

// ---------------------------------------------------------------------------------------------- time

/// Slots per frame at a subcarrier spacing of 15, 30, 60 or 120 kHz. Throws input_error otherwise.
unsigned slots_per_frame(unsigned scs_khz);
/// Length of the useful part of a symbol in T_c.
std::uint64_t useful_tc(unsigned scs_khz);
/// Start of a slot in T_c, counted from slot 0 of the epoch. The first symbol of every half
/// subframe carries the longer cyclic prefix (TS 38.211 5.3.1), so slots are summed, not multiplied.
std::uint64_t slot_start_tc(unsigned scs_khz, std::uint64_t absolute_slot);

// ---------------------------------------------------------------------------------------------- observation

/// What the harness knows of the slot that an observation belongs to. Every identifier is opaque.
struct slot_context {
  std::string opportunity_id;
  std::string episode_id; ///< Names the run of opportunities that share one coherent time axis.
  unsigned    sfn  = 0;
  unsigned    slot = 0;
  unsigned    cell = 0;
  unsigned    ue   = 0;
};

/// One immutable observation: the descriptor fields of the obs/v1 record and the owned payload.
/// It holds nothing of the evaluator: no channel, no noise, no transport block and no label.
struct observation {
  // Join keys. The three hashed keys are the leading 64 bits of the SHA-256 of the opaque text.
  std::string   run_id;
  std::uint64_t epoch    = 0;
  std::uint64_t sequence = 0; ///< Rises by one per observation of an epoch.
  std::uint64_t episode     = 0;
  std::uint64_t opportunity = 0;
  std::uint64_t grant       = 0;
  // Radio identity.
  unsigned      cell = 0;
  unsigned      ue   = 0;
  unsigned      sfn  = 0;
  unsigned      slot = 0;
  std::uint64_t absolute_slot = 0; ///< Counts from the epoch start and does not wrap with the frame number.
  // What it is.
  std::string           stage;
  std::string           producer;
  std::vector<std::string> producer_steps;
  std::string           source;
  unsigned              port = 0;
  std::string           layout;
  std::string           dtype;
  double                scale = 0.0; ///< The reference amplitude the producer used.
  // Shape, in elements.
  std::vector<std::size_t> shape;
  std::vector<std::size_t> strides;
  // Allocation, in carrier coordinates.
  unsigned              first_subcarrier = 0;
  unsigned              subcarrier_step  = 0;
  unsigned              subcarrier_count = 0;
  std::vector<unsigned> symbols;
  unsigned              rx_count    = 0;
  unsigned              layer_count = 0;
  std::vector<unsigned> antenna_map;
  std::string           coordinates_sha256;
  // Validity.
  std::vector<std::uint8_t> mask;
  std::size_t               valid = 0;
  std::size_t               holes = 0; ///< Cleared bits: elements without a pilot or without a finite estimate.
  std::string               mask_sha256;
  // Time.
  std::uint64_t t_first_sample = 0; ///< Start of the useful part of the first reference symbol, T_c from the epoch start.
  std::uint64_t t_last_sample  = 0; ///< End of the useful part of the last reference symbol.
  std::int64_t  t_ready_ns     = 0; ///< Host monotonic time at which the estimate was complete.
  // Payload.
  std::vector<cf32> values;
  std::string       payload_sha256; ///< Over the little-endian complex64 values in payload order.
  // Host block: fields only the staged harness has. A sensing core never reads them.
  std::string         opportunity_id;
  std::string         grant_id;
  std::string         grant_record_sha256;
  isac_replay::ticket grid;
  std::string         ingress_sha256; ///< Readback digest of the grid the estimate was computed from.
  double              service_us = 0.0;
  // The parts of service_us, in the order they run. Recorded, so that an audit run shows where its time goes.
  double estimate_us       = 0.0; ///< The rows read, the least-squares values, the validity bits.
  double grid_digest_us    = 0.0; ///< The digest of the whole grid after the read.
  double keys_mask_us      = 0.0; ///< The record fields, the three join keys and the digest of the mask.
  double payload_digest_us = 0.0;

  std::size_t payload_bytes() const { return values.size() * sizeof(cf32); }
  bool        is_valid(std::size_t element) const { return ((mask[element / 8] >> (element % 8)) & 1U) != 0; }
};

/// The leading 64 bits of the SHA-256 of a text: an opaque integer for joins.
std::uint64_t opaque_key(const std::string& text);

struct estimate_counts {
  std::size_t valid = 0;
  std::size_t holes = 0;
};

/// Estimates one source on a grid that is read plane by plane, into storage of the caller.
/// read(plane, symbol, row, count) returns subcarriers 0 to count-1 of one symbol. values holds
/// rx_planes * plane_elements values, mask one bit per value, rounded up to bytes, and row holds
/// subcarriers values of scratch. It allocates nothing. Both evidence modes run this code.
template <typename Reader>
estimate_counts estimate_into(const pilot_source& source, unsigned subcarriers, Reader&& read, cf32* values,
                              std::uint8_t* mask, cf32* row);

/// The same into an observation, which it sizes: shape, values, mask, valid and holes.
template <typename Reader>
void estimate(const pilot_source& source, unsigned subcarriers, Reader&& read, observation& out);

/// Everything of one observation that is known before the run. Bound once, read per slot.
struct bound_slot {
  const pilot_source* source = nullptr;
  std::string         opportunity_id;
  std::string         episode_id;
  std::string         grant_record_sha256;
  // The join keys, hashed when the plan is bound and not per slot.
  std::uint64_t episode     = 0;
  std::uint64_t opportunity = 0;
  std::uint64_t grant       = 0;
  unsigned      cell = 0;
  unsigned      ue   = 0;
  unsigned      sfn  = 0;
  unsigned      slot = 0;
  std::uint64_t absolute_slot  = 0;
  std::uint64_t t_first_sample = 0;
  std::uint64_t t_last_sample  = 0;
};

/// What observe_into reports of one estimate.
struct tap_result {
  estimate_counts     counts;
  isac_replay::ticket grid;           ///< The ticket of the fill the estimate was computed from.
  std::int64_t        t_read_ns  = 0; ///< Host monotonic time before the first row was read.
  std::int64_t        t_ready_ns = 0; ///< Host monotonic time at which the estimate was complete.
};

/// The observer of one run: the sources it was bound with and the observations it made.
class observer
{
public:
  /// epoch names the run of the producer. It is at least 1. mode is the evidence mode of the run.
  observer(std::string run_id_, std::uint64_t epoch_, evidence mode_ = evidence::audit);

  /// Binds the source of one grant. A second source for a grant record is refused.
  void bind(const source_spec& spec);
  bool bound(const std::string& grant_record_sha256) const;
  const pilot_source& source_of(const std::string& grant_record_sha256) const;
  std::size_t         sources() const { return bindings.size(); }

  /// Estimates the source that is bound to the grant record of a ready grid.
  ///
  /// Throws input_error when no source is bound to the grant record of the grid, when the grid
  /// has another size or another number of planes than the source names, and when the grid lies
  /// in another slot than the reference was generated for. Throws state_error for a grid that
  /// was released or filled again, and when the grid does not have the digest it became ready
  /// with after it was read. The observation is appended only when every check passed. The
  /// reference stays valid until the next call.
  const observation& observe(const isac_replay::o1_view& view, const slot_context& slot);

  /// Binds one opportunity before the run, in the order of the schedule: its source, its join
  /// keys, its slot count and its sample times. This is where the keys are hashed. Returns the
  /// handle of the slot. Throws input_error as observe() does for a slot the source was not
  /// built for and for a grant record without a source.
  std::size_t       prepare(const slot_context& slot, const std::string& grant_record_sha256);
  const bound_slot& prepared(std::size_t handle) const { return slots.at(handle); }
  std::size_t       prepared_slots() const { return slots.size(); }

  /// The timed path. Estimates the source of a prepared slot on a ready grid, straight into
  /// values and mask, which hold what estimate_into names. It takes no digest, allocates nothing
  /// and formats no text.
  ///
  /// Throws input_error for a grid of another grant record, another size or another slot than
  /// the prepared one, and state_error for a grid that was released or filled again before or
  /// while it was read: the ticket of the fill is checked before the first row and after the
  /// last. A grid whose values were changed under an unchanged ticket is not found here. The
  /// audit mode and the digest that the evaluator takes after the run find it.
  tap_result observe_into(std::size_t handle, const isac_replay::o1_view& view, cf32* values, std::uint8_t* mask);

  const std::vector<observation>& observations() const { return made; }
  std::uint64_t                   epoch() const { return epoch_id; }
  evidence                        mode() const { return evidence_mode; }

private:
  std::string                         run_id;
  std::uint64_t                       epoch_id;
  evidence                            evidence_mode;
  std::vector<bound_slot>             slots;
  std::vector<cf32>                   scratch_row; ///< Of observe_into, sized when a source is bound.
  std::map<std::string, pilot_source> bindings;
  std::vector<observation>            made;
  // Frame wraps seen so far, per episode, so that the absolute slot never decreases inside one.
  struct episode_clock {
    std::uint64_t wraps = 0;
    unsigned      sfn   = 0;
    unsigned      slot  = 0;
    bool          seen  = false;
  };
  std::map<std::string, episode_clock> clocks;
  std::map<std::string, episode_clock> bound_clocks; ///< The same count for prepare(), which runs before the run.
  static std::uint64_t absolute_slot_of(episode_clock& clock, const slot_context& slot, unsigned scs_khz);
};

// ---------------------------------------------------------------------------------------------- template

template <typename Reader>
estimate_counts estimate_into(const pilot_source& source, unsigned subcarriers, Reader&& read, cf32* values,
                              std::uint8_t* mask, cf32* row)
{
  const std::size_t planes  = source.rx_planes();
  const std::size_t rows    = source.symbols().size();
  const std::size_t columns = source.subcarriers().size();
  const std::size_t count   = planes * rows * columns;
  estimate_counts   counts;
  for (std::size_t byte = 0; byte != (count + 7) / 8; ++byte) {
    mask[byte] = 0;
  }
  const cf32* reference = source.spec().reference.data(); // one port
  for (std::size_t plane = 0; plane != planes; ++plane) {
    for (std::size_t symbol_row = 0; symbol_row != rows; ++symbol_row) {
      read(static_cast<unsigned>(plane), source.symbols()[symbol_row], row, subcarriers);
      for (std::size_t column = 0; column != columns; ++column) {
        const std::size_t element = (plane * rows + symbol_row) * columns + column;
        const long long   which   = source.pilot_at(symbol_row, column);
        bool              ok      = false;
        cf32              value(0.0F, 0.0F);
        if (which >= 0) {
          value = pilot_ls(row[source.subcarriers()[column]], reference[which], &ok);
        }
        if (ok) {
          values[element]   = value;
          mask[element / 8] = static_cast<std::uint8_t>(mask[element / 8] | (1U << (element % 8)));
          ++counts.valid;
        } else {
          values[element] = cf32(0.0F, 0.0F);
          ++counts.holes;
        }
      }
    }
  }
  return counts;
}

template <typename Reader>
void estimate(const pilot_source& source, unsigned subcarriers, Reader&& read, observation& out)
{
  const std::size_t planes  = source.rx_planes();
  const std::size_t rows    = source.symbols().size();
  const std::size_t columns = source.subcarriers().size();
  const std::size_t count   = planes * rows * columns;
  out.shape   = {planes, rows, columns};
  out.strides = {rows * columns, columns, 1};
  out.values.assign(count, cf32(0.0F, 0.0F));
  out.mask.assign((count + 7) / 8, 0);
  std::vector<cf32>     row(subcarriers);
  const estimate_counts counts = estimate_into(source, subcarriers, read, out.values.data(), out.mask.data(), row.data());
  out.valid                    = counts.valid;
  out.holes                    = counts.holes;
}

} // namespace isac_observer
