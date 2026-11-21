// Pilot least squares on the receiver grid (Sprint 3.1, selector pilot_ls). See observer_core.h.
// ASCII only.

#include "observer_core.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <set>
#include <utility>

namespace isac_observer {

// ---------------------------------------------------------------------------------------------- kernel

cf32 pilot_ls(cf32 received, cf32 reference, bool* ok)
{
  const double yr = static_cast<double>(received.real());
  const double yi = static_cast<double>(received.imag());
  const double xr = static_cast<double>(reference.real());
  const double xi = static_cast<double>(reference.imag());
  const double d  = xr * xr + xi * xi;
  const double nr = yr * xr + yi * xi;
  const double ni = yi * xr - yr * xi;
  const float  hr = static_cast<float>(nr / d);
  const float  hi = static_cast<float>(ni / d);
  if (ok != nullptr) {
    *ok = std::isfinite(hr) && std::isfinite(hi);
  }
  return cf32(hr, hi);
}

// ---------------------------------------------------------------------------------------------- source

namespace {

void put_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
  for (unsigned shift = 0; shift != 32; shift += 8) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

std::string name_of(const source_spec& spec)
{
  return "the sensing source " + (spec.source.empty() ? std::string("(unnamed)") : spec.source) + " of grant " +
         (spec.grant_id.empty() ? std::string("(unnamed)") : spec.grant_id);
}

} // namespace

pilot_source::pilot_source(source_spec spec_) : record(std::move(spec_))
{
  const std::string who = name_of(record);
  if (record.source.empty() || record.grant_id.empty() || record.grant_record_sha256.empty()) {
    throw input_error(who + " names no source, no grant or no grant record");
  }
  if (record.ports.size() != 1) {
    throw input_error(who + " has " + std::to_string(record.ports.size()) +
                      " ports: this observer estimates one port. Register one source record per port.");
  }
  if (record.grid_symbols == 0 || record.grid_subcarriers == 0) {
    throw input_error(who + " names no grid");
  }
  if (!(record.amplitude > 0.0) || !std::isfinite(record.amplitude)) {
    throw input_error(who + " has a reference amplitude that is not positive and finite");
  }
  if (record.index.empty()) {
    throw input_error(who + " has no pilot");
  }
  if (record.reference.size() != record.index.size() * record.ports.size()) {
    throw input_error(who + " has " + std::to_string(record.reference.size()) + " reference symbols for " +
                      std::to_string(record.index.size()) + " pilots");
  }
  std::set<unsigned> symbol_set;
  std::set<unsigned> subcarrier_set;
  for (std::size_t n = 0; n != record.index.size(); ++n) {
    const pilot& item = record.index[n];
    if (item.symbol >= record.grid_symbols || item.subcarrier >= record.grid_subcarriers) {
      throw input_error(who + " has a pilot outside the grid: symbol " + std::to_string(item.symbol) +
                        ", subcarrier " + std::to_string(item.subcarrier));
    }
    if (n != 0) {
      const pilot& before = record.index[n - 1];
      if (before.symbol == item.symbol && before.subcarrier == item.subcarrier) {
        throw input_error(who + " has a duplicate pilot at symbol " + std::to_string(item.symbol) + ", subcarrier " +
                          std::to_string(item.subcarrier));
      }
      if (before.symbol > item.symbol || (before.symbol == item.symbol && before.subcarrier > item.subcarrier)) {
        throw input_error(who + " lists its pilots out of order: they are ordered by symbol, then by subcarrier");
      }
    }
    const cf32 value = record.reference[n];
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
      throw input_error(who + " has a reference symbol that is not finite at pilot " + std::to_string(n));
    }
    if (value.real() == 0.0F && value.imag() == 0.0F) {
      throw input_error(who + " has a configured pilot with zero reference power: pilot " + std::to_string(n) +
                        " at symbol " + std::to_string(item.symbol) + ", subcarrier " + std::to_string(item.subcarrier));
    }
    symbol_set.insert(item.symbol);
    subcarrier_set.insert(item.subcarrier);
  }
  rows.assign(symbol_set.begin(), symbol_set.end());
  columns.assign(subcarrier_set.begin(), subcarrier_set.end());
  if (record.slot_symbols != rows) {
    throw input_error(who + " names other reference symbols for its slot than its pilots lie on");
  }
  if (record.symbol_useful_start_tc.size() != rows.size() || record.tc_per_second != TC_PER_SECOND) {
    throw input_error(who + " names no start time for each reference symbol in T_c");
  }
  for (std::size_t n = 1; n < rows.size(); ++n) {
    if (record.symbol_useful_start_tc[n] <= record.symbol_useful_start_tc[n - 1]) {
      throw input_error(who + " names symbol times that do not increase");
    }
  }
  slots_per_frame(record.scs_khz);
  if (record.slot >= slots_per_frame(record.scs_khz)) {
    throw input_error(who + " lies in slot " + std::to_string(record.slot) + ", outside the frame");
  }
  if (record.antenna_map.empty()) {
    throw input_error(who + " has no antenna map: every receive plane names its array element");
  }
  const std::set<unsigned> elements(record.antenna_map.begin(), record.antenna_map.end());
  if (elements.size() != record.antenna_map.size()) {
    throw input_error(who + " has an antenna map that names an array element twice");
  }

  // The rectangle of the payload. An element without a pilot of the record is a hole.
  cells.assign(rows.size() * columns.size(), -1);
  for (std::size_t n = 0; n != record.index.size(); ++n) {
    const std::size_t row    = static_cast<std::size_t>(std::lower_bound(rows.begin(), rows.end(), record.index[n].symbol) - rows.begin());
    const std::size_t column = static_cast<std::size_t>(
        std::lower_bound(columns.begin(), columns.end(), record.index[n].subcarrier) - columns.begin());
    cells[row * columns.size() + column] = static_cast<long long>(n);
  }
  nof_holes = static_cast<std::size_t>(std::count(cells.begin(), cells.end(), -1LL));
  step      = columns.size() > 1 ? columns[1] - columns[0] : 0;
  for (std::size_t n = 2; n < columns.size(); ++n) {
    if (columns[n] - columns[n - 1] != step) {
      step = 0;
      break;
    }
  }
  std::vector<std::uint8_t> bytes;
  for (unsigned value : rows) {
    put_u32(bytes, value);
  }
  for (unsigned value : columns) {
    put_u32(bytes, value);
  }
  coordinates = isac_replay::sha256_hex(bytes.data(), bytes.size());
}

unsigned pilot_source::plane_of(unsigned element) const
{
  for (std::size_t plane = 0; plane != record.antenna_map.size(); ++plane) {
    if (record.antenna_map[plane] == element) {
      return static_cast<unsigned>(plane);
    }
  }
  throw input_error("the antenna map names no receive plane for the array element " + std::to_string(element));
}

// ---------------------------------------------------------------------------------------------- time

namespace {

unsigned numerology(unsigned scs_khz)
{
  switch (scs_khz) {
    case 15:
      return 0;
    case 30:
      return 1;
    case 60:
      return 2;
    case 120:
      return 3;
    default:
      throw input_error("the subcarrier spacing must be 15, 30, 60 or 120 kHz");
  }
}

constexpr std::uint64_t KAPPA             = 64;
constexpr unsigned      SYMBOLS_PER_SLOT  = 14;
constexpr std::uint64_t TC_PER_SUBFRAME   = TC_PER_SECOND / 1000;

} // namespace

unsigned slots_per_frame(unsigned scs_khz)
{
  return 10U << numerology(scs_khz);
}

std::uint64_t useful_tc(unsigned scs_khz)
{
  return (2048 * KAPPA) >> numerology(scs_khz);
}

std::uint64_t slot_start_tc(unsigned scs_khz, std::uint64_t absolute_slot)
{
  const unsigned      mu       = numerology(scs_khz);
  const std::uint64_t per      = 1ULL << mu; // slots per subframe
  const std::uint64_t within   = absolute_slot % per;
  std::uint64_t       start    = (absolute_slot / per) * TC_PER_SUBFRAME;
  const std::uint64_t useful   = (2048 * KAPPA) >> mu;
  const std::uint64_t prefix   = (144 * KAPPA) >> mu;
  const std::uint64_t boundary = 7ULL << mu; // symbols per half subframe
  for (std::uint64_t symbol = 0; symbol != within * SYMBOLS_PER_SLOT; ++symbol) {
    start += useful + prefix + (symbol % boundary == 0 ? 16 * KAPPA : 0);
  }
  return start;
}

// ---------------------------------------------------------------------------------------------- observer

std::uint64_t opaque_key(const std::string& text)
{
  const std::string hex   = isac_replay::sha256_hex(text.data(), text.size());
  std::uint64_t     value = 0;
  for (unsigned n = 0; n != 16; ++n) {
    const char          digit  = hex[n];
    const std::uint64_t nibble = static_cast<std::uint64_t>(digit <= '9' ? digit - '0' : digit - 'a' + 10);
    value                      = (value << 4) | nibble;
  }
  return value;
}

observer::observer(std::string run_id_, std::uint64_t epoch_, evidence mode_) :
  run_id(std::move(run_id_)), epoch_id(epoch_), evidence_mode(mode_)
{
  if (run_id.empty()) {
    throw input_error("an observer needs the identifier of its run");
  }
  if (epoch_id == 0) {
    throw input_error("an observer epoch is at least 1");
  }
}

void observer::bind(const source_spec& spec)
{
  pilot_source compiled(spec);
  if (bindings.count(spec.grant_record_sha256) != 0) {
    throw input_error("the grant " + spec.grant_id + " has two sensing sources: one source is bound to a grant record");
  }
  bindings.emplace(spec.grant_record_sha256, std::move(compiled));
  if (scratch_row.size() < spec.grid_subcarriers) {
    scratch_row.resize(spec.grid_subcarriers); // the scratch of the timed path exists before the run
  }
}

bool observer::bound(const std::string& grant_record_sha256) const
{
  return bindings.count(grant_record_sha256) != 0;
}

const pilot_source& observer::source_of(const std::string& grant_record_sha256) const
{
  const auto found = bindings.find(grant_record_sha256);
  if (found == bindings.end()) {
    throw input_error("no sensing source is bound to the grant record " + grant_record_sha256);
  }
  return found->second;
}

std::uint64_t observer::absolute_slot_of(episode_clock& clock, const slot_context& slot, unsigned scs_khz)
{
  // The absolute slot does not wrap with the frame number: a frame or a slot that goes back
  // inside one episode is the next period of frame numbers.
  if (clock.seen && (slot.sfn < clock.sfn || (slot.sfn == clock.sfn && slot.slot < clock.slot))) {
    ++clock.wraps;
  }
  clock.seen = true;
  clock.sfn  = slot.sfn;
  clock.slot = slot.slot;
  return (clock.wraps * SFN_PERIOD + slot.sfn) * slots_per_frame(scs_khz) + slot.slot;
}

std::size_t observer::prepare(const slot_context& slot, const std::string& grant_record_sha256)
{
  const auto found = bindings.find(grant_record_sha256);
  if (found == bindings.end()) {
    throw input_error("changed allocation: no sensing source is bound to the grant record of " + slot.opportunity_id);
  }
  const pilot_source& source = found->second;
  const source_spec&  spec   = source.spec();
  if (slot.slot != spec.slot) {
    throw input_error("the opportunity " + slot.opportunity_id + " lies in slot " + std::to_string(slot.slot) +
                      ", and its reference symbols were generated for slot " + std::to_string(spec.slot) +
                      ": the DM-RS depends on the slot index");
  }
  if (slot.sfn >= SFN_PERIOD) {
    throw input_error("the opportunity " + slot.opportunity_id + " names a frame number outside 0 to 1023");
  }
  bound_slot item;
  item.source              = &source;
  item.opportunity_id      = slot.opportunity_id;
  item.episode_id          = slot.episode_id;
  item.grant_record_sha256 = grant_record_sha256;
  item.episode             = opaque_key(slot.episode_id);
  item.opportunity         = opaque_key(slot.opportunity_id);
  item.grant               = opaque_key(grant_record_sha256);
  item.cell                = slot.cell;
  item.ue                  = slot.ue;
  item.sfn                 = slot.sfn;
  item.slot                = slot.slot;
  item.absolute_slot       = absolute_slot_of(bound_clocks[slot.episode_id], slot, spec.scs_khz);
  const std::uint64_t origin = slot_start_tc(spec.scs_khz, item.absolute_slot);
  item.t_first_sample        = origin + spec.symbol_useful_start_tc.front();
  item.t_last_sample         = origin + spec.symbol_useful_start_tc.back() + useful_tc(spec.scs_khz);
  slots.push_back(std::move(item));
  return slots.size() - 1;
}

namespace {

inline std::int64_t now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

tap_result observer::observe_into(std::size_t handle, const isac_replay::o1_view& view, cf32* values, std::uint8_t* mask)
{
  tap_result result;
  result.t_read_ns = now_ns();
  const bound_slot& bound = slots[handle];
  // A released or refilled grid throws state_error here, before anything is read. Nothing below
  // copies the descriptor, hashes or builds a text unless a rule is broken.
  const isac_replay::o1_descriptor& descriptor = view.descriptor();
  const pilot_source&               source     = *bound.source;
  const source_spec&                spec       = source.spec();
  if (descriptor.grant_record_sha256 != bound.grant_record_sha256) {
    throw input_error("changed allocation: the grid of " + bound.opportunity_id + " was filled for another grant record "
                      "than its prepared slot");
  }
  if (descriptor.shape.symbols != spec.grid_symbols || descriptor.shape.subcarriers != spec.grid_subcarriers ||
      descriptor.shape.rx_ports != source.rx_planes()) {
    throw input_error("the grid of " + bound.opportunity_id + " has another size or another number of receive planes "
                      "than its sensing source was built for");
  }
  if (descriptor.slot != spec.slot) {
    throw input_error("the grid of " + bound.opportunity_id + " lies in slot " + std::to_string(descriptor.slot) +
                      ", and its reference symbols were generated for slot " + std::to_string(spec.slot));
  }
  result.counts = estimate_into(
      source, descriptor.shape.subcarriers,
      [&view](unsigned plane, unsigned symbol, cf32* into, unsigned count) { view.get(plane, symbol, into, count); },
      values, mask, scratch_row.data());
  // Ownership by generation: the grid is still ready under the ticket of its fill after the read.
  view.descriptor();
  result.grid       = view.key();
  result.t_ready_ns = now_ns();
  return result;
}

const observation& observer::observe(const isac_replay::o1_view& view, const slot_context& slot)
{
  if (evidence_mode != evidence::audit) {
    throw state_error("observe() takes the digests of the mode audit: a timed observer uses observe_into()");
  }
  const auto started = std::chrono::steady_clock::now();
  // A released or refilled grid throws state_error here, before anything is read.
  const isac_replay::o1_descriptor descriptor = view.descriptor();
  const auto                       found      = bindings.find(descriptor.grant_record_sha256);
  if (found == bindings.end()) {
    throw input_error("changed allocation: no sensing source is bound to the grant record of the grid of " +
                      slot.opportunity_id + " (" + descriptor.grant_id + ")");
  }
  const pilot_source& source = found->second;
  const source_spec&  spec   = source.spec();
  if (descriptor.shape.symbols != spec.grid_symbols || descriptor.shape.subcarriers != spec.grid_subcarriers) {
    throw input_error("the grid of " + slot.opportunity_id + " has " + std::to_string(descriptor.shape.symbols) +
                      " symbols and " + std::to_string(descriptor.shape.subcarriers) +
                      " subcarriers, and its sensing source was built for " + std::to_string(spec.grid_symbols) +
                      " and " + std::to_string(spec.grid_subcarriers));
  }
  if (descriptor.shape.rx_ports != source.rx_planes()) {
    throw input_error("the grid of " + slot.opportunity_id + " has " + std::to_string(descriptor.shape.rx_ports) +
                      " receive planes, and the antenna map of its sensing source names " +
                      std::to_string(source.rx_planes()));
  }
  if (slot.slot != spec.slot || descriptor.slot != spec.slot) {
    throw input_error("the opportunity " + slot.opportunity_id + " lies in slot " + std::to_string(slot.slot) +
                      ", and its reference symbols were generated for slot " + std::to_string(spec.slot) +
                      ": the DM-RS depends on the slot index");
  }
  if (slot.sfn >= SFN_PERIOD) {
    throw input_error("the opportunity " + slot.opportunity_id + " names a frame number outside 0 to 1023");
  }

  observation made_now;
  estimate(source, descriptor.shape.subcarriers,
           [&view](unsigned plane, unsigned symbol, cf32* row, unsigned count) { view.get(plane, symbol, row, count); },
           made_now);
  const auto estimated = std::chrono::steady_clock::now();
  // The estimate belongs to the grid of this digest only if the grid still has it after the read.
  if (view.digest() != descriptor.readback_sha256) {
    throw state_error("the grid of " + slot.opportunity_id + " changed while the observer read it");
  }
  const auto digested = std::chrono::steady_clock::now();

  made_now.run_id         = run_id;
  made_now.epoch          = epoch_id;
  made_now.sequence       = made.size();
  made_now.episode        = opaque_key(slot.episode_id);
  made_now.opportunity    = opaque_key(slot.opportunity_id);
  made_now.grant          = opaque_key(descriptor.grant_record_sha256);
  made_now.cell           = slot.cell;
  made_now.ue             = slot.ue;
  made_now.sfn            = slot.sfn;
  made_now.slot           = slot.slot;
  made_now.absolute_slot  = absolute_slot_of(clocks[slot.episode_id], slot, spec.scs_khz);
  made_now.stage          = STAGE_O2;
  made_now.producer       = PRODUCER;
  made_now.producer_steps = {STEP_PILOT_LS};
  made_now.source         = spec.source;
  made_now.port           = spec.ports.front();
  made_now.layout         = LAYOUT_O2_COMB;
  made_now.dtype          = DTYPE_C64;
  made_now.scale          = spec.amplitude;
  made_now.first_subcarrier   = source.first_subcarrier();
  made_now.subcarrier_step    = source.subcarrier_step();
  made_now.subcarrier_count   = static_cast<unsigned>(source.subcarriers().size());
  made_now.symbols            = source.symbols();
  made_now.rx_count           = source.rx_planes();
  made_now.layer_count        = 1;
  made_now.antenna_map        = spec.antenna_map;
  made_now.coordinates_sha256 = source.coordinates_sha256();
  made_now.mask_sha256        = isac_replay::sha256_hex(made_now.mask.data(), made_now.mask.size());
  const std::uint64_t origin  = slot_start_tc(spec.scs_khz, made_now.absolute_slot);
  made_now.t_first_sample     = origin + spec.symbol_useful_start_tc.front();
  made_now.t_last_sample      = origin + spec.symbol_useful_start_tc.back() + useful_tc(spec.scs_khz);
  const auto described        = std::chrono::steady_clock::now();
  made_now.payload_sha256     = isac_replay::sha256_hex(made_now.values.data(), made_now.payload_bytes());
  made_now.opportunity_id      = slot.opportunity_id;
  made_now.grant_id            = descriptor.grant_id;
  made_now.grant_record_sha256 = descriptor.grant_record_sha256;
  made_now.grid                = view.key();
  made_now.ingress_sha256      = descriptor.readback_sha256;
  const auto done              = std::chrono::steady_clock::now();
  made_now.t_ready_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(done.time_since_epoch()).count();
  made_now.service_us = std::chrono::duration<double, std::micro>(done - started).count();
  made_now.estimate_us       = std::chrono::duration<double, std::micro>(estimated - started).count();
  made_now.grid_digest_us    = std::chrono::duration<double, std::micro>(digested - estimated).count();
  made_now.keys_mask_us      = std::chrono::duration<double, std::micro>(described - digested).count();
  made_now.payload_digest_us = std::chrono::duration<double, std::micro>(done - described).count();
  made.push_back(std::move(made_now));
  return made.back();
}

} // namespace isac_observer
