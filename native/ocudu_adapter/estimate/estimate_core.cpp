// Capture of the receiver's own channel estimate (Sprint 5.2, selector native_estimate). See estimate_core.h.
// ASCII only.

#include "estimate_core.h"
#include <cstring>

namespace isac_estimate {

// ---------------------------------------------------------------------------------------------- half precision

float half_to_float(std::uint16_t bits) noexcept
{
  const std::uint32_t sign     = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
  const std::uint32_t exponent = (bits >> 10) & 0x1fU;
  std::uint32_t       mantissa = bits & 0x3ffU;
  std::uint32_t       word     = 0;
  if (exponent == 0x1fU) {
    word = sign | 0x7f800000U | (mantissa << 13); // infinite or not a number
  } else if (exponent != 0) {
    word = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  } else if (mantissa != 0) {
    // Subnormal: normalize the 10 bits. The value is mantissa * 2^-24.
    std::uint32_t shift = 0;
    while ((mantissa & 0x400U) == 0) {
      mantissa <<= 1;
      ++shift;
    }
    word = sign | ((113U - shift) << 23) | ((mantissa & 0x3ffU) << 13);
  } else {
    word = sign;
  }
  float value;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

std::uint16_t float_to_half(float value) noexcept
{
  std::uint32_t word;
  std::memcpy(&word, &value, sizeof(word));
  const std::uint32_t sign     = (word >> 16) & 0x8000U;
  const std::uint32_t biased   = (word >> 23) & 0xffU;
  std::uint32_t       mantissa = word & 0x7fffffU;
  if (biased == 0xffU) {
    return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa != 0 ? (0x200U | (mantissa >> 13)) : 0U));
  }
  const int exponent = static_cast<int>(biased) - 127;
  if (exponent > 15) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (exponent >= -14) {
    // A normal half. A carry out of the mantissa raises the exponent, up to infinity.
    std::uint32_t       half = (static_cast<std::uint32_t>(exponent + 15) << 10) | (mantissa >> 13);
    const std::uint32_t rest = mantissa & 0x1fffU;
    if (rest > 0x1000U || (rest == 0x1000U && (half & 1U) != 0)) {
      ++half;
    }
    return static_cast<std::uint16_t>(sign | half);
  }
  if (exponent < -25 || biased == 0) {
    return static_cast<std::uint16_t>(sign); // below half of the smallest subnormal half
  }
  // A subnormal half: the value in units of 2^-24, rounded to nearest, ties to even.
  mantissa |= 0x800000U;
  const unsigned      shift   = static_cast<unsigned>(-exponent - 1); // 14 to 24
  std::uint32_t       half    = mantissa >> shift;
  const std::uint32_t rest    = mantissa & ((1U << shift) - 1U);
  const std::uint32_t halfway = 1U << (shift - 1U);
  if (rest > halfway || (rest == halfway && (half & 1U) != 0)) {
    ++half;
  }
  return static_cast<std::uint16_t>(sign | half);
}

// ---------------------------------------------------------------------------------------------- layout

std::size_t layout::element(unsigned dmrs_symbol, unsigned prb, unsigned rx_port, unsigned subcarrier_in_prb) const
{
  if (dmrs_symbol >= dmrs_symbols || prb >= prbs || rx_port >= rx_ports || subcarrier_in_prb >= SUBCARRIERS_PER_PRB) {
    throw input_error("the index [" + std::to_string(dmrs_symbol) + "][" + std::to_string(prb) + "][" +
                      std::to_string(rx_port) + "][" + std::to_string(subcarrier_in_prb) + "] is outside a payload of " +
                      std::to_string(dmrs_symbols) + " by " + std::to_string(prbs) + " by " + std::to_string(rx_ports) +
                      " by 12 elements");
  }
  return ((static_cast<std::size_t>(dmrs_symbol) * prbs + prb) * rx_ports + rx_port) * SUBCARRIERS_PER_PRB +
         subcarrier_in_prb;
}

coordinate coordinate_of(const layout& shape, std::size_t element)
{
  if (element >= shape.elements()) {
    throw input_error("the element " + std::to_string(element) + " is outside a payload of " +
                      std::to_string(shape.elements()) + " elements");
  }
  coordinate where;
  where.subcarrier_in_prb = static_cast<unsigned>(element % SUBCARRIERS_PER_PRB);
  element /= SUBCARRIERS_PER_PRB;
  where.rx_port = static_cast<unsigned>(element % shape.rx_ports);
  element /= shape.rx_ports;
  where.prb         = static_cast<unsigned>(element % shape.prbs);
  where.dmrs_symbol = static_cast<unsigned>(element / shape.prbs);
  where.subcarrier  = where.prb * SUBCARRIERS_PER_PRB + where.subcarrier_in_prb;
  return where;
}

namespace {

inline std::uint16_t half_at(const std::uint8_t* bytes) noexcept
{
  return static_cast<std::uint16_t>(bytes[0] | (static_cast<std::uint16_t>(bytes[1]) << 8));
}

} // namespace

cf32 value_at(const void* payload, std::size_t element) noexcept
{
  const std::uint8_t* bytes = static_cast<const std::uint8_t*>(payload) + element * ELEMENT_BYTES;
  return cf32(half_to_float(half_at(bytes)), half_to_float(half_at(bytes + 2)));
}

payload_counts scan(const void* payload, std::size_t elements, std::uint8_t* mask) noexcept
{
  const std::uint8_t* bytes = static_cast<const std::uint8_t*>(payload);
  payload_counts      counts;
  for (std::size_t byte = 0; byte != (elements + 7) / 8; ++byte) {
    mask[byte] = 0;
  }
  for (std::size_t element = 0; element != elements; ++element) {
    const std::uint16_t real = half_at(bytes + element * ELEMENT_BYTES);
    const std::uint16_t imag = half_at(bytes + element * ELEMENT_BYTES + 2);
    if (half_is_finite(real) && half_is_finite(imag)) {
      mask[element / 8] = static_cast<std::uint8_t>(mask[element / 8] | (1U << (element % 8)));
      ++counts.valid;
      counts.nonzero += ((real | imag) & 0x7fffU) != 0 ? 1 : 0;
    } else {
      ++counts.holes;
    }
  }
  return counts;
}

// ---------------------------------------------------------------------------------------------- producer

layout expectation::shape() const
{
  layout rectangle;
  for (std::uint64_t bits = dmrs_symbol_mask; bits != 0; bits &= bits - 1) {
    ++rectangle.dmrs_symbols;
  }
  rectangle.prbs     = nof_prb;
  rectangle.rx_ports = rx_ports;
  return rectangle;
}

const char* refusal(const expectation& expected, const producer_view& view) noexcept
{
  if (view.estimates == nullptr) {
    return "null_estimate";
  }
  if (view.layers != 1) {
    return "layers";
  }
  if (!view.contiguous) {
    return "not_contiguous";
  }
  if (!view.frequency_interpolation) {
    return "not_interpolated";
  }
  if (view.rnti != expected.rnti || view.sfn != expected.sfn || view.slot != expected.slot) {
    return "other_pdu";
  }
  const layout rectangle = expected.shape();
  if (view.start_prb != expected.start_prb || view.nof_prb != expected.nof_prb ||
      view.dmrs_symbol_mask != expected.dmrs_symbol_mask || view.nof_dmrs_symbols != rectangle.dmrs_symbols ||
      view.rx_ports != expected.rx_ports || view.dmrs_type != expected.dmrs_type) {
    return "other_allocation";
  }
  if (view.bytes != rectangle.payload_bytes()) {
    return "size";
  }
  return "";
}

// ---------------------------------------------------------------------------------------------- capture

const char* to_string(outcome value)
{
  switch (value) {
    case outcome::published:
      return "published";
    case outcome::captured:
      return "captured";
    case outcome::dropped:
      return "dropped";
    case outcome::absent:
      return "absent";
    case outcome::refused:
      return "refused";
    case outcome::failed:
      return "failed";
  }
  return "unknown";
}

capture::capture(copier& device_, obs::producer* ring_, const capture_config& config_) :
  device(device_), ring(ring_), config(config_)
{
  if (config.mode == evidence::timed && ring == nullptr) {
    throw input_error("a timed estimate exists in a slot of the ring only: the capture has no ring");
  }
  if (config.resolve_limit_ns < 0) {
    throw input_error("the time a capture may take to resolve is not negative");
  }
  // The staging is the destination of every audited copy and of a timed copy that cannot land
  // in its slot.
  const bool needs_staging = config.mode == evidence::audit || !config.slot_landing;
  if (needs_staging && (config.staging == nullptr || config.staging_bytes == 0)) {
    throw input_error("the capture has no staging for the copy of the estimate");
  }
}

void capture::reserve(std::size_t captures)
{
  lines.reserve(captures);
}

std::size_t capture::prepare(const isac_observer::bound_slot& slot, const expectation& expected,
                             const std::vector<unsigned>& antenna_map, unsigned port, std::uint32_t opportunity_sequence)
{
  const layout rectangle = expected.shape();
  if (rectangle.elements() == 0) {
    throw input_error("the estimate of " + slot.opportunity_id + " has no element: it needs a DM-RS symbol, a resource "
                      "block and a receive port");
  }
  if (rectangle.payload_bytes() > (1U << 30)) {
    throw input_error("a payload of an observation holds at most 1 GiB");
  }
  if (expected.rx_ports > obs::MAX_PLANES) {
    throw input_error("a record names at most " + std::to_string(obs::MAX_PLANES) + " receive planes");
  }
  if (antenna_map.size() != expected.rx_ports) {
    throw input_error("the antenna map of " + slot.opportunity_id + " names " + std::to_string(antenna_map.size()) +
                      " array elements for " + std::to_string(expected.rx_ports) + " receive ports");
  }
  if (expected.dmrs_symbol_mask >> 32 != 0) {
    throw input_error("a record names DM-RS symbols 0 to 31");
  }
  const bool needs_staging = config.mode == evidence::audit || !config.slot_landing;
  if (needs_staging && rectangle.payload_bytes() > config.staging_bytes) {
    throw input_error("the staging cannot carry the estimate of " + slot.opportunity_id + ": it needs " +
                      std::to_string(rectangle.payload_bytes()) + " bytes, and the staging holds " +
                      std::to_string(config.staging_bytes));
  }
  if (ring != nullptr) {
    const obs::segment_header& segment = ring->header();
    if (rectangle.payload_bytes() > segment.payload_capacity || rectangle.mask_bytes() > segment.mask_capacity ||
        sizeof(isac_transport::staged_host) > segment.host_capacity) {
      throw input_error("the ring cannot carry the estimate of " + slot.opportunity_id + ": it needs " +
                        std::to_string(rectangle.payload_bytes()) + " payload bytes and " +
                        std::to_string(rectangle.mask_bytes()) + " mask bytes, and a slot holds " +
                        std::to_string(segment.payload_capacity) + " and " + std::to_string(segment.mask_capacity));
    }
  }
  obs::record_header header;
  std::memset(&header, 0, sizeof(header));
  header.schema        = obs::RECORD_SCHEMA;
  header.header_bytes  = obs::RECORD_HEADER_BYTES;
  header.episode       = slot.episode;
  header.opportunity   = slot.opportunity;
  header.grant         = slot.grant;
  header.absolute_slot = slot.absolute_slot;
  header.cell          = slot.cell;
  header.ue            = slot.ue;
  header.stage         = obs::STAGE_O3;
  header.source        = obs::SOURCE_PUSCH_DMRS;
  header.layout        = obs::LAYOUT_OCUDU_O3_FP16;
  header.dtype         = obs::DTYPE_CFP16;
  // The steps and the scale are the receiver's. They are set when it reports them.
  header.rank       = 4;
  header.shape[0]   = rectangle.dmrs_symbols;
  header.shape[1]   = rectangle.prbs;
  header.shape[2]   = rectangle.rx_ports;
  header.shape[3]   = SUBCARRIERS_PER_PRB;
  header.strides[0] = rectangle.prbs * rectangle.rx_ports * SUBCARRIERS_PER_PRB;
  header.strides[1] = rectangle.rx_ports * SUBCARRIERS_PER_PRB;
  header.strides[2] = SUBCARRIERS_PER_PRB;
  header.strides[3] = 1;
  header.first_subcarrier = expected.start_prb * SUBCARRIERS_PER_PRB;
  header.subcarrier_step  = 1;
  header.subcarrier_count = rectangle.prbs * SUBCARRIERS_PER_PRB;
  header.dmrs_symbols     = static_cast<std::uint32_t>(expected.dmrs_symbol_mask);
  header.rx_count         = static_cast<std::uint16_t>(expected.rx_ports);
  header.layer_count      = 1;
  for (std::size_t plane = 0; plane != antenna_map.size(); ++plane) {
    for (std::size_t other = 0; other != plane; ++other) {
      if (antenna_map[other] == antenna_map[plane]) {
        throw input_error("the antenna map of " + slot.opportunity_id + " names an array element twice");
      }
    }
    header.antenna_map[plane] = static_cast<std::uint16_t>(antenna_map[plane]);
  }
  header.mask_bytes     = static_cast<std::uint32_t>(rectangle.mask_bytes());
  header.payload_bytes  = static_cast<std::uint32_t>(rectangle.payload_bytes());
  header.host_bytes     = sizeof(isac_transport::staged_host);
  header.time_unit      = obs::TIME_TC;
  header.digest_kind    = config.mode == evidence::audit ? obs::DIGEST_SHA256 : obs::DIGEST_NONE;
  header.evidence       = config.mode == evidence::audit ? obs::EVIDENCE_AUDIT : obs::EVIDENCE_TIMED;
  header.port           = static_cast<std::uint8_t>(port);
  header.t_first_sample = slot.t_first_sample;
  header.t_last_sample  = slot.t_last_sample;
  isac_transport::staged_host host;
  std::memset(&host, 0, sizeof(host));
  host.kind                 = isac_transport::HOST_STAGED_HARNESS;
  host.opportunity_sequence = opportunity_sequence;
  host.sfn                  = slot.sfn;
  host.slot                 = slot.slot;
  headers.push_back(header);
  hosts.push_back(host);
  expectations.push_back(expected);
  if (own_mask.size() < rectangle.mask_bytes()) {
    own_mask.resize(rectangle.mask_bytes());
  }
  return headers.size() - 1;
}

void capture::expect(std::size_t handle, const ticket& grid)
{
  if (handle >= headers.size()) {
    throw state_error("the capture has no prepared decode " + std::to_string(handle));
  }
  if (phase.load(std::memory_order_acquire) != idle) {
    throw state_error("a capture is open: resolve it before the next decode is expected");
  }
  if (flying && device.poll() != copy_state::pending) {
    reclaim(); // the copy that ran out of time has ended: its destination is free again
  }
  // The account was reserved before the run, so this line allocates nothing.
  lines.emplace_back();
  capture_line& line = lines.back();
  line.handle        = handle;
  line.grid          = grid;
  line.t_expect_ns   = obs::monotonic_ns();
  last_payload       = nullptr;
  last_mask          = nullptr;
  tap_verdict        = verdict::none;
  tap_reason         = "";
  has_slot           = false;
  destination        = nullptr;
  overtaken.store(false, std::memory_order_relaxed);
  // The admission: one credit or a counted drop. An audited copy is made in either case,
  // because its payload is archived.
  // While an earlier copy can still write the staging or its slot, nothing is admitted.
  busy = flying;
  if (!busy && ring != nullptr) {
    has_slot = ring->try_acquire(acquired);
  }
  line.admitted = has_slot;
  wants_copy    = !busy && (has_slot || config.mode == evidence::audit);
  if (wants_copy) {
    const bool into_slot = has_slot && config.mode == evidence::timed && config.slot_landing;
    destination          = into_slot ? acquired.payload : config.staging;
    line.landing         = into_slot ? LANDING_SLOT : LANDING_STAGING;
  }
  phase.store(expecting, std::memory_order_release);
}

void capture::on_estimate(const producer_view& view) noexcept
{
  int wanted = expecting;
  if (!phase.compare_exchange_strong(wanted, tapping, std::memory_order_acq_rel)) {
    if (wanted == idle) {
      strays.fetch_add(1, std::memory_order_relaxed);
    } else {
      // A later launch while a capture is open. A copy on the producer's stream was enqueued
      // before that launch and stays whole. A copy on another stream may have been overtaken.
      extra.fetch_add(1, std::memory_order_relaxed);
      if (!device.on_producer_stream()) {
        overtaken.store(true, std::memory_order_release);
      }
    }
    return;
  }
  capture_line& line = lines.back();
  line.t_tap_ns      = obs::monotonic_ns();
  line.seen          = true;
  line.producer      = view;
  const char* why    = refusal(expectations[line.handle], view);
  if (*why != '\0') {
    tap_verdict = verdict::refused;
    tap_reason  = why;
  } else if (!wants_copy) {
    tap_verdict = verdict::skipped; // no credit or no free destination: nothing is copied
  } else if (device.submit(view, destination)) {
    tap_verdict      = verdict::submitted;
    line.copy_stream = device.stream_name();
  } else {
    tap_verdict = verdict::failed;
    tap_reason  = "submit";
  }
  line.t_submit_ns = obs::monotonic_ns();
  line.submit_us   = static_cast<double>(line.t_submit_ns - line.t_tap_ns) / 1000.0;
  phase.store(tapped, std::memory_order_release);
}

void capture::finish(capture_line& line, outcome result, const char* reason)
{
  if (has_slot && result != outcome::published) {
    ring->cancel(acquired); // the slot holds no record: nothing is published
    has_slot = false;
  }
  line.result     = result;
  line.reason     = reason;
  line.t_done_ns  = obs::monotonic_ns();
  line.resolve_us = static_cast<double>(line.t_done_ns - line.t_resolve_ns) / 1000.0;
  phase.store(idle, std::memory_order_release);
}

const capture_line& capture::resolve()
{
  int now = phase.load(std::memory_order_acquire);
  if (now == idle) {
    throw state_error("no capture is open");
  }
  capture_line& line = lines.back();
  line.t_resolve_ns  = obs::monotonic_ns();
  // A tap that is still inside its call on another thread ends within a few device calls.
  while (now == tapping) {
    now = phase.load(std::memory_order_acquire);
  }
  if (now == expecting) {
    // The decode returned and the receiver reported no estimate. No later call may fill this capture.
    int still = expecting;
    if (!phase.compare_exchange_strong(still, tapped, std::memory_order_acq_rel)) {
      while (phase.load(std::memory_order_acquire) == tapping) {
      }
    }
  }
  if (tap_verdict == verdict::none) {
    finish(line, outcome::absent, "no_estimate");
    return line;
  }
  if (tap_verdict == verdict::refused) {
    finish(line, outcome::refused, tap_reason);
    return line;
  }
  if (tap_verdict == verdict::skipped) {
    finish(line, busy ? outcome::failed : outcome::dropped, busy ? "busy" : "no_credit");
    return line;
  }
  if (tap_verdict == verdict::failed) {
    finish(line, outcome::failed, tap_reason);
    return line;
  }
  // The copy was submitted. Ask for its completion without waiting for the device.
  copy_state state = copy_state::pending;
  for (;;) {
    state = device.poll();
    ++line.polls;
    const std::int64_t asked = obs::monotonic_ns();
    if (state != copy_state::pending) {
      line.t_complete_ns = asked;
      break;
    }
    if (asked - line.t_resolve_ns >= config.resolve_limit_ns) {
      break;
    }
  }
  if (state == copy_state::pending) {
    // The copy may still write its destination: the slot stays acquired until the copy has ended.
    flying          = true;
    flying_slot     = has_slot;
    has_slot        = false;
    line.result     = outcome::failed;
    line.reason     = "timeout";
    line.t_done_ns  = obs::monotonic_ns();
    line.resolve_us = static_cast<double>(line.t_done_ns - line.t_resolve_ns) / 1000.0;
    phase.store(idle, std::memory_order_release);
    return line;
  }
  if (state == copy_state::failed) {
    finish(line, outcome::failed, "copy");
    return line;
  }
  if (overtaken.load(std::memory_order_acquire)) {
    // A later launch of the producer was seen before this completion was: the copy may hold
    // values of two estimates. It is never published.
    finish(line, outcome::failed, "overtaken");
    return line;
  }
  // A complete copy of the expected estimate.
  const obs::record_header& bound     = headers[line.handle];
  const std::uint8_t*       source    = destination;
  std::uint8_t*             mask_area = has_slot ? acquired.mask : own_mask.data();
  if (has_slot && destination != acquired.payload) {
    std::memcpy(acquired.payload, destination, bound.payload_bytes);
    source = acquired.payload;
  }
  const layout rectangle = expectations[line.handle].shape();
  line.counts            = scan(source, rectangle.elements(), mask_area);
  line.landed            = true;
  last_payload           = source;
  last_mask              = mask_area;
  if (!has_slot) {
    finish(line, ring == nullptr ? outcome::captured : outcome::dropped, ring == nullptr ? "" : "no_credit");
    return line;
  }
  std::memcpy(acquired.header, &bound, sizeof(obs::record_header));
  obs::record_header* header = acquired.header;
  header->producer_steps     = obs::STEP_PILOT_LS | (line.producer.frequency_interpolation ? obs::STEP_FREQ_INTERP : 0U) |
                               (line.producer.frequency_smoothing ? obs::STEP_SMOOTHING : 0U);
  header->scale              = static_cast<double>(line.producer.dmrs_scaling);
  header->valid              = static_cast<std::uint32_t>(line.counts.valid);
  header->t_ready_ns         = line.t_complete_ns;
  if (config.mode == evidence::audit) {
    const std::string digest = isac_replay::sha256_hex(source, bound.payload_bytes);
    for (unsigned n = 0; n != 32; ++n) {
      const auto nibble = [](char digit) {
        return static_cast<std::uint8_t>(digit <= '9' ? digit - '0' : digit - 'a' + 10);
      };
      header->digest[n] = static_cast<std::uint8_t>((nibble(digest[2 * n]) << 4) | nibble(digest[2 * n + 1]));
    }
  }
  isac_transport::staged_host host = hosts[line.handle];
  host.grid_index                  = static_cast<std::uint32_t>(line.grid.grid);
  host.grid_generation             = line.grid.generation;
  host.t_read_ns                   = line.t_tap_ns;
  std::memcpy(acquired.host, &host, sizeof(host));
  line.sequence             = acquired.sequence;
  line.depth                = acquired.depth;
  const std::uint64_t calls = ring->header().window_calls.load(std::memory_order_relaxed);
  ring->publish(acquired);
  has_slot          = false;
  line.epoch        = header->epoch;
  line.bytes        = header->header_bytes + header->host_bytes + header->mask_bytes + header->payload_bytes;
  line.t_publish_ns = header->t_publish_ns;
  line.window_calls = ring->header().window_calls.load(std::memory_order_relaxed) - calls;
  finish(line, outcome::published, "");
  return line;
}

void capture::reclaim()
{
  if (flying_slot) {
    ring->cancel(acquired); // the slot holds no record: nothing is published
    flying_slot = false;
  }
  flying = false;
}

void capture::settle()
{
  if (!flying) {
    return;
  }
  device.drain();
  reclaim();
}

} // namespace isac_estimate
