// Producer glue of the observation ring (Sprint 3.2, selector snapshot_ring). See obs_publisher.h.
// ASCII only.

#include "obs_publisher.h"
#include <cstring>

namespace isac_transport {

void ring_needs::include(const isac_observer::pilot_source& source)
{
  const std::size_t elements = source.rx_planes() * source.plane_elements();
  const std::size_t bits     = (elements + 7) / 8;
  const std::size_t bytes    = elements * sizeof(isac_observer::cf32);
  if (bytes > (1U << 30)) {
    throw input_error("a payload of an observation holds at most 1 GiB");
  }
  if (bits > mask_bytes) {
    mask_bytes = static_cast<std::uint32_t>(bits);
  }
  if (bytes > payload_bytes) {
    payload_bytes = static_cast<std::uint32_t>(bytes);
  }
}

publisher::publisher(obs::producer& ring_, isac_observer::observer& watcher_, evidence mode_) :
  ring(ring_), watcher(watcher_), evidence_mode(mode_)
{
  if (watcher.mode() != mode_) {
    throw input_error("the observer and its publisher run in different evidence modes");
  }
}

void publisher::reserve(std::size_t observations)
{
  lines.reserve(observations);
}

std::size_t publisher::prepare(const isac_observer::slot_context& slot, const std::string& grant_record_sha256,
                               std::uint32_t opportunity_sequence)
{
  const std::size_t                  handle = watcher.prepare(slot, grant_record_sha256);
  const isac_observer::bound_slot&   bound  = watcher.prepared(handle);
  const isac_observer::pilot_source& source = *bound.source;
  const isac_observer::source_spec&  spec   = source.spec();
  const obs::segment_header&         segment = ring.header();
  ring_needs                         needs;
  needs.include(source);
  if (needs.payload_bytes > segment.payload_capacity || needs.mask_bytes > segment.mask_capacity ||
      needs.host_bytes > segment.host_capacity) {
    throw input_error("the ring cannot carry the observation of " + slot.opportunity_id + ": it needs " +
                      std::to_string(needs.payload_bytes) + " payload bytes and " + std::to_string(needs.mask_bytes) +
                      " mask bytes, and a slot holds " + std::to_string(segment.payload_capacity) + " and " +
                      std::to_string(segment.mask_capacity));
  }
  if (source.rx_planes() > obs::MAX_PLANES) {
    throw input_error("a record names at most " + std::to_string(obs::MAX_PLANES) + " receive planes");
  }
  if (spec.source != "pusch_dmrs") {
    throw input_error("the record obs/v1 has no number for the sensing source " + spec.source);
  }
  obs::record_header header;
  std::memset(&header, 0, sizeof(header));
  header.schema         = obs::RECORD_SCHEMA;
  header.header_bytes   = obs::RECORD_HEADER_BYTES;
  header.episode        = bound.episode;
  header.opportunity    = bound.opportunity;
  header.grant          = bound.grant;
  header.absolute_slot  = bound.absolute_slot;
  header.cell           = bound.cell;
  header.ue             = bound.ue;
  header.stage          = obs::STAGE_O2;
  header.source         = obs::SOURCE_PUSCH_DMRS;
  header.layout         = obs::LAYOUT_OCUDU_O2_COMB_C64;
  header.dtype          = obs::DTYPE_C64;
  header.producer_steps = obs::STEP_PILOT_LS;
  header.scale          = spec.amplitude;
  header.rank           = 3;
  header.shape[0]       = source.rx_planes();
  header.shape[1]       = static_cast<std::uint32_t>(source.symbols().size());
  header.shape[2]       = static_cast<std::uint32_t>(source.subcarriers().size());
  header.strides[0]     = header.shape[1] * header.shape[2];
  header.strides[1]     = header.shape[2];
  header.strides[2]     = 1;
  header.first_subcarrier = source.first_subcarrier();
  header.subcarrier_step  = source.subcarrier_step();
  header.subcarrier_count = header.shape[2];
  for (unsigned symbol : source.symbols()) {
    header.dmrs_symbols |= 1U << symbol;
  }
  header.rx_count    = static_cast<std::uint16_t>(source.rx_planes());
  header.layer_count = 1;
  for (unsigned plane = 0; plane != source.rx_planes(); ++plane) {
    header.antenna_map[plane] = static_cast<std::uint16_t>(spec.antenna_map[plane]);
  }
  header.mask_bytes     = needs.mask_bytes;
  header.payload_bytes  = needs.payload_bytes;
  header.host_bytes     = sizeof(staged_host);
  header.time_unit      = obs::TIME_TC;
  header.digest_kind    = evidence_mode == evidence::audit ? obs::DIGEST_SHA256 : obs::DIGEST_NONE;
  header.evidence       = evidence_mode == evidence::audit ? obs::EVIDENCE_AUDIT : obs::EVIDENCE_TIMED;
  header.port           = static_cast<std::uint8_t>(spec.ports.front());
  header.t_first_sample = bound.t_first_sample;
  header.t_last_sample  = bound.t_last_sample;
  staged_host host;
  std::memset(&host, 0, sizeof(host));
  host.kind                 = HOST_STAGED_HARNESS;
  host.opportunity_sequence = opportunity_sequence;
  host.sfn                  = bound.sfn;
  host.slot                 = bound.slot;
  if (headers.size() != handle) {
    throw input_error("the observer has slots that this publisher did not prepare");
  }
  headers.push_back(header);
  hosts.push_back(host);
  return handle;
}

const published_record& publisher::publish(std::size_t handle, const isac_replay::o1_view& view)
{
  // The account was reserved before the run, so this line allocates nothing.
  lines.emplace_back();
  published_record& line = lines.back();
  line.handle            = handle;
  line.t_read_ns         = obs::monotonic_ns();
  obs::slot acquired;
  if (!ring.try_acquire(acquired)) {
    line.reason     = "no_credit";
    line.t_done_ns  = obs::monotonic_ns();
    line.service_us = static_cast<double>(line.t_done_ns - line.t_read_ns) / 1000.0;
    line.publish_us = line.service_us;
    return line;
  }
  std::memcpy(acquired.header, &headers[handle], sizeof(obs::record_header));
  isac_observer::tap_result made;
  try {
    // The slot is 64-byte aligned, so the estimate is written where the consumer reads it.
    made = watcher.observe_into(handle, view, reinterpret_cast<isac_observer::cf32*>(acquired.payload), acquired.mask);
  } catch (...) {
    ring.cancel(acquired);
    lines.pop_back();
    throw;
  }
  staged_host host     = hosts[handle];
  host.grid_index      = static_cast<std::uint32_t>(made.grid.grid);
  host.grid_generation = made.grid.generation;
  host.t_read_ns       = made.t_read_ns;
  std::memcpy(acquired.host, &host, sizeof(host));
  obs::record_header* header = acquired.header;
  header->valid              = static_cast<std::uint32_t>(made.counts.valid);
  header->t_ready_ns         = made.t_ready_ns;
  line.sequence              = acquired.sequence;
  line.depth                 = acquired.depth;
  const std::uint64_t calls  = ring.header().window_calls.load(std::memory_order_relaxed);
  ring.publish(acquired);
  line.t_done_ns    = obs::monotonic_ns();
  line.outcome      = publication::published;
  line.epoch        = header->epoch;
  line.bytes        = header->header_bytes + header->host_bytes + header->mask_bytes + header->payload_bytes;
  line.grid         = made.grid;
  line.valid        = made.counts.valid;
  line.holes        = made.counts.holes;
  line.t_ready_ns   = made.t_ready_ns;
  line.t_publish_ns = header->t_publish_ns;
  line.window_calls = ring.header().window_calls.load(std::memory_order_relaxed) - calls;
  line.service_us   = static_cast<double>(line.t_done_ns - line.t_read_ns) / 1000.0;
  line.estimate_us  = static_cast<double>(made.t_ready_ns - made.t_read_ns) / 1000.0;
  line.publish_us   = line.service_us - line.estimate_us;
  return line;
}

namespace {

std::uint8_t nibble(char digit)
{
  return static_cast<std::uint8_t>(digit <= '9' ? digit - '0' : digit - 'a' + 10);
}

} // namespace

const published_record& publisher::publish(std::size_t handle, const isac_observer::observation& made)
{
  lines.emplace_back();
  published_record&  line    = lines.back();
  const std::int64_t started = obs::monotonic_ns();
  line.handle                = handle;
  // The read of the audited observation began one service time before its completion.
  line.t_read_ns = made.t_ready_ns - static_cast<std::int64_t>(made.service_us * 1000.0);
  line.grid      = made.grid;
  line.valid             = made.valid;
  line.holes             = made.holes;
  line.t_ready_ns        = made.t_ready_ns;
  line.estimate_us       = made.estimate_us;
  const obs::record_header& bound = headers.at(handle);
  if (made.payload_bytes() != bound.payload_bytes || made.mask.size() != bound.mask_bytes ||
      made.opportunity != bound.opportunity || made.payload_sha256.size() != 64) {
    lines.pop_back();
    throw input_error("the observation of " + made.opportunity_id + " is not the one of its prepared slot");
  }
  obs::slot acquired;
  if (!ring.try_acquire(acquired)) {
    line.reason     = "no_credit";
    line.t_done_ns  = obs::monotonic_ns();
    line.publish_us = static_cast<double>(line.t_done_ns - started) / 1000.0;
    line.service_us = made.service_us + line.publish_us;
    return line;
  }
  std::memcpy(acquired.header, &bound, sizeof(obs::record_header));
  staged_host host     = hosts[handle];
  host.grid_index      = static_cast<std::uint32_t>(made.grid.grid);
  host.grid_generation = made.grid.generation;
  host.t_read_ns       = line.t_read_ns;
  std::memcpy(acquired.host, &host, sizeof(host));
  std::memcpy(acquired.mask, made.mask.data(), made.mask.size());
  std::memcpy(acquired.payload, made.values.data(), made.payload_bytes());
  obs::record_header* header = acquired.header;
  header->valid              = static_cast<std::uint32_t>(made.valid);
  header->t_ready_ns         = made.t_ready_ns;
  for (unsigned n = 0; n != 32; ++n) {
    header->digest[n] = static_cast<std::uint8_t>((nibble(made.payload_sha256[2 * n]) << 4) | nibble(made.payload_sha256[2 * n + 1]));
  }
  line.sequence             = acquired.sequence;
  line.depth                = acquired.depth;
  const std::uint64_t calls = ring.header().window_calls.load(std::memory_order_relaxed);
  ring.publish(acquired);
  line.t_done_ns    = obs::monotonic_ns();
  line.outcome      = publication::published;
  line.epoch        = header->epoch;
  line.bytes        = header->header_bytes + header->host_bytes + header->mask_bytes + header->payload_bytes;
  line.t_publish_ns = header->t_publish_ns;
  line.window_calls = ring.header().window_calls.load(std::memory_order_relaxed) - calls;
  line.publish_us   = static_cast<double>(line.t_done_ns - started) / 1000.0;
  line.service_us   = made.service_us + line.publish_us;
  return line;
}

} // namespace isac_transport
