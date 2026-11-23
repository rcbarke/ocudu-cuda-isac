// Tapped receiver replay for Sprint 3.1 (pilot_ls).
//
// Reads a pilot-ls-plan/v1 file. The plan names a receiver replay plan of Sprint 2.4 by the hash
// that binds it, and what the observer taps. For every opportunity of the replay the tool
//
//   1. fills an owned pinned resource grid, Y = g (sum over tx of H X + W), through the channel
//      application core, makes it ready and reads its digest back, exactly as replay_pusch does;
//   2. with the tap "pilot_ls", hands a read-only view of that ready grid to the observer, which
//      computes H = Y conj(X) / |X|^2 on the pilots of the sensing source of the grant, on every
//      receive plane, into a buffer it owns: the stage O2. The record names the digest of the grid
//      it was computed from;
//   3. hands the reader of the same grid, the PDU of the opportunity's grant and a buffer for the
//      decoded bytes to the host PUSCH processor and to the CUDA PUSCH processor of the pinned
//      source, waits for the true decoder callback of each and compares the decoded bytes with the
//      original transport block;
//   4. releases the grid once every decode on it has called back.
//
// With the tap "none" step 2 is left out and nothing of the observer is written: that run is the
// control of this executable. The control of Sprint 2.4 is the accepted replay_pusch, which is
// not edited. A tapped run must leave the keys, the grid digests, the CRC flags and the decoded
// bytes of both controls unchanged.
//
// Four places keep the sides apart. ingress/ holds what the receiver side may read of its input.
// observer/ holds what the observer produced from that input: one obs-descriptor/v1 record, one
// payload and one validity mask per opportunity, and nothing of a transport block, a channel or a
// label. radio/ holds what the receiver produced. native.json is the report of the evaluator side.
//
// The estimate, its layout, its validity bits and its record are in the host-free observer core
// (observer_core.h). The outcome rule, the unique keys and the ownership of the grids are in the
// dispatch core of Sprint 2.4 (dispatch_core.h). The processors and the dispatch loop below are
// those of receiver/replay_pusch.cpp, kept as they were accepted.
//
// Sprint 3.2 (snapshot_ring) adds the plan version pilot-ls-plan/v2, with two things.
//
// The observation ring. A plan that names a ring makes this tool the producer of one obs-ring/v1
// ring in DIR/ring/: every observation is published as an obs/v1 record, or dropped and counted
// when the consumer left no credit. The decodes never wait for the ring.
//
// The evidence mode. "audit" is the run of Sprint 3.1: every digest in the loop. "timed" computes
// no SHA-256 between the first fill and the drain of the ring: a grid becomes ready without a
// readback digest and is owned by the ticket of its fill, the observer writes its estimate
// straight into the slot of the ring, a decode is admitted and resolved without a digest, and the
// grids, the decoded bytes and every record are written after the last opportunity. Only there,
// on the evaluator side, the stored grids and the decoded bytes are digested, so that the files of
// a timed run are read by the same readers as those of an audit run. One arithmetic serves both
// modes: the same plan gives the same grids, the same estimates and the same decoded bytes.
//
// Sprint 5.2 (native_estimate) adds the plan version pilot-ls-plan/v3, with one object, estimate.
// It names the capture of the receiver's own channel estimate, the stage O3: what the resident
// CUDA kernels of the pinned receiver leave on the device after pilot least squares, frequency
// interpolation and frequency smoothing, per DM-RS symbol and before time interpolation. The
// receiver patch of patches/ocudu/ tells this tool where that buffer is, once per PDU. For every
// decode of the backend the plan names the tool expects one estimate before the decode starts,
// enqueues one asynchronous copy of the buffer from the receiver's callback, ordered after the
// kernels that wrote it, and after the decode asks the completion event of that copy for its
// state. A completed copy of the expected estimate becomes one obs/v1 record of the stage O3 in
// DIR/estimate/: archived with its digest in the mode audit, published on the ring of the O3
// stream in both modes when the plan names one. Nothing here waits for the device while a decode
// runs. The stage O2 is unchanged and stays a stream of its own. The rules of the capture are in
// the host-free core of estimate/ (estimate_core.h) and its device calls in estimate_tap.h, which
// exists only in the build against the patched receiver (ISAC_O3_ROUTE). Every other build
// refuses a plan that names an estimate.
//
// Sprint 4.1 (mps_pacing) adds the plan version pilot-ls-plan/v4, with one object, pacing. It
// names the offered schedule of a paced run: one absolute release time per opportunity on the
// host's monotonic clock, a processing allowance and a latest start. A paced run is a timed run
// in which the harness behaves as a scripted receiver. The grid of an opportunity is filled
// ahead of its slot, the tap and the decodes start at the release time and never before it, a
// release the harness could not start in time is skipped, recorded under its key and counted,
// and no release time ever moves. From the admission of a ready grid to the decoder's callback
// the harness itself allocates nothing and takes no lock: every decode is named before the first
// fill in the paced book of the host-free pacing core (pacing/pacing_core.h), the notifier of
// every decode exists before the run, and the dispatch thread asks an atomic flag for the
// completion. The ledger of Sprint 2.4 is not used by a paced run and is not edited. The run
// writes DIR/pacing/releases.jsonl, one pace-release/v1 line per opportunity, after its drain.
//
// The lead-in of a paced run. The first decode of a PUSCH processor pays for what the processor
// sets up on its first use, about seven times a decode on the GB10, and the two decodes behind it
// are not yet at their steady time. On an offered schedule that cost two releases and the first
// window of every run. A plan may therefore name a lead-in: the first releases of the schedule
// are decoded once more ahead of it, each a declared time before its own release, with the same
// grids, PDUs and transport blocks. A decode of the lead-in is no release: nothing of it is
// observed, exported, counted or timed with the schedule, it is kept in a paced book of its own,
// and the receiver has no observer registered while it runs. The run writes
// DIR/pacing/lead_in.jsonl, one pace-lead-in/v1 line per decode of the lead-in, after its drain:
// that file is the cold start of the receiver, measured apart from every release.
// Built out of the OCUDU tree against the accepted Sprint 1.1 build, and a second time against
// the patched build of Sprint 5.2. ASCII only.

#include "dispatch_core.h"
#include "estimate_core.h"
#include "grant_pdu.h"
#include "obs/ring.h"
#include "obs_publisher.h"
#include "pacing_core.h"
#include "observer_core.h"
#include "observer_plan.h"
#include "pinned_grid.h"
#include "receiver_plan.h"
#include "replay_core.h"
#include "replay_plan.h"
#include "pxsch_bler_test_factories.h"
#include "nlohmann/json.hpp"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/phy/antenna_ports.h"
#include "ocudu/phy/support/resource_grid.h"
#include "ocudu/phy/support/resource_grid_reader.h"
#include "ocudu/phy/support/resource_grid_writer.h"
#include "ocudu/phy/support/support_factories.h"
#include "ocudu/phy/upper/channel_processors/pdsch/pdsch_processor.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_processor.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_processor_result_notifier.h"
#include "ocudu/phy/upper/rx_buffer_pool.h"
#include "ocudu/phy/upper/trx_buffer_identifier.h"
#include "ocudu/phy/upper/unique_rx_buffer.h"
#include "ocudu/ran/cyclic_prefix.h"
#include "ocudu/ran/precoding/precoding_codebooks.h"
#include "ocudu/ran/resource_block.h"
#include "ocudu/ran/sch/sch_dmrs_power.h"
#include "ocudu/support/executors/task_worker_pool.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>
#ifdef ISAC_O3_ROUTE
// The device side of the capture. Only the build against the patched receiver has it.
#include "estimate_tap.h"
#endif

using namespace isac_receiver;
using isac_replay::allocation;
using isac_replay::cf32;
using isac_replay::grid_shape;
using isac_replay::o1_descriptor;
using isac_replay::o1_pool;
using isac_replay::o1_view;
using isac_replay::opportunity_spec;
using isac_replay::pinned_grid;
using isac_replay::run_spec;
using isac_replay::session;
using isac_replay::sha256_hex;
using isac_replay::ticket;
using isac_replay::verified_file;
using json = nlohmann::json;

namespace {

constexpr const char* NATIVE_SCHEMA  = "pilot-ls-native/v1";
constexpr const char* NATIVE_SCHEMA_2 = "pilot-ls-native/v2"; // of a pilot-ls-plan/v2 run
constexpr const char* NATIVE_SCHEMA_3 = "pilot-ls-native/v3"; // of a pilot-ls-plan/v3 run
constexpr const char* NATIVE_SCHEMA_4 = "pilot-ls-native/v4"; // of a pilot-ls-plan/v4 run
constexpr const char* RELEASE_SCHEMA  = "pace-release/v1";
constexpr const char* PACING_SCHEMA   = "pace-report/v1";
constexpr const char* PACING_DIR      = "pacing";
constexpr const char* RELEASES_FILE   = "releases.jsonl";
constexpr const char* LEAD_IN_SCHEMA  = "pace-lead-in/v1";
constexpr const char* LEAD_IN_FILE    = "lead_in.jsonl";
constexpr const char* ESTIMATE_DIR    = "estimate";
constexpr const char* CAPTURE_SCHEMA  = "o3-capture/v1";
constexpr const char* RING_DIR        = "ring";
constexpr const char* PUBLICATION_SCHEMA = "obs-publication/v1";
constexpr const char* RING_SCHEMA        = "obs-ring-report/v1";
constexpr const char* RING_MEMORY        = "host_shared";
constexpr const char* OBS_SCHEMA     = "obs-descriptor/v1";
constexpr const char* OBSERVER_DIR   = "observer";
constexpr const char* OBSERVER_HOST  = "ocudu-deepsig staged harness";
constexpr const char* RADIO_SCHEMA   = "radio/v1";
constexpr const char* INGRESS_SCHEMA = "o1-grid/v1";
constexpr const char* REPORT_FILE    = "native.json";
constexpr const char* INGRESS_DIR    = "ingress";
constexpr const char* RADIO_DIR      = "radio";
constexpr const char* RULE           = "Y = g (sum over tx of H X + W)";
constexpr const char* GRID_MEMORY    = "host resource grid of the pinned factory, read by both processors";

bool exists(const std::string& path)
{
  struct stat info;
  return ::stat(path.c_str(), &info) == 0;
}

bool is_directory(const std::string& path)
{
  struct stat info;
  return ::stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

void make_directory(const std::string& path)
{
  if (::mkdir(path.c_str(), 0777) != 0) {
    throw std::runtime_error("cannot create " + path);
  }
}

std::vector<std::uint8_t> read_bytes(const std::string& path, const std::string& what)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw input_error("cannot read " + what + " " + path);
  }
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

void write_bytes(const std::string& path, const void* data, std::size_t bytes)
{
  std::ofstream stream(path, std::ios::binary);
  stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  stream.close();
  if (!stream) {
    throw std::runtime_error("cannot write " + path);
  }
}

std::string numbered(std::size_t sequence, const std::string& suffix)
{
  char name[32];
  std::snprintf(name, sizeof(name), "%04zu", sequence);
  return std::string(name) + suffix;
}

std::vector<cf32> read_back(const o1_view& view)
{
  const grid_shape  shape = view.descriptor().shape;
  std::vector<cf32> values(static_cast<std::size_t>(shape.rx_ports) * shape.plane());
  for (unsigned rx = 0; rx != shape.rx_ports; ++rx) {
    for (unsigned symbol = 0; symbol != shape.symbols; ++symbol) {
      view.get(rx, symbol, values.data() + (static_cast<std::size_t>(rx) * shape.symbols + symbol) * shape.subcarriers,
               shape.subcarriers);
    }
  }
  return values;
}

void put_number(json& object, const char* key, const std::optional<float>& value)
{
  if (value.has_value() && std::isfinite(value.value())) {
    object[key] = value.value();
  }
}

bool resident_ports(std::size_t ports)
{
  // supports_resident_accelerated_pusch_demodulation of the pinned processor.
  return ports == 1 || ports == 2 || ports == 4 || ports == 8;
}

std::string hex64(std::uint64_t value)
{
  char text[17];
  std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
  return std::string(text);
}

/// The descriptor of one observation: the fields of the obs/v1 header, as a JSON record. It names
/// no channel, no noise, no transport block and no label.
json observation_record(const isac_observer::observation& made, std::size_t opportunity_sequence,
                        const std::string& payload_file, const std::string& mask_file)
{
  return {{"schema", OBS_SCHEMA},
          {"run_id", made.run_id},
          {"keys",
           {{"epoch", made.epoch},
            {"sequence", made.sequence},
            {"episode", hex64(made.episode)},
            {"opportunity", hex64(made.opportunity)},
            {"grant", hex64(made.grant)}}},
          {"radio",
           {{"cell", made.cell},
            {"ue", made.ue},
            {"sfn", made.sfn},
            {"slot", made.slot},
            {"absolute_slot", made.absolute_slot}}},
          {"what",
           {{"stage", made.stage},
            {"producer", made.producer},
            {"producer_steps", made.producer_steps},
            {"source", made.source},
            {"port", made.port},
            {"layout", made.layout},
            {"dtype", made.dtype},
            {"scale", made.scale},
            {"rule", isac_observer::RULE},
            {"precision", isac_observer::PRECISION}}},
          {"shape",
           {{"rank", made.shape.size()},
            {"axes", {"rx_plane", "reference_symbol", "pilot"}},
            {"shape", made.shape},
            {"strides", made.strides}}},
          {"allocation",
           {{"first_subcarrier", made.first_subcarrier},
            {"subcarrier_step", made.subcarrier_step},
            {"subcarrier_count", made.subcarrier_count},
            {"dmrs_symbols", made.symbols},
            {"rx_count", made.rx_count},
            {"layer_count", made.layer_count},
            {"antenna_map", made.antenna_map},
            {"coordinates_sha256", made.coordinates_sha256}}},
          {"validity",
           {{"mask_bytes", made.mask.size()},
            {"file", mask_file},
            {"sha256", made.mask_sha256},
            {"valid", made.valid},
            {"holes", made.holes}}},
          {"time",
           {{"unit", isac_observer::TIME_UNIT_TC},
            {"tc_per_second", isac_observer::TC_PER_SECOND},
            {"t_first_sample", made.t_first_sample},
            {"t_last_sample", made.t_last_sample},
            {"t_ready_ns", made.t_ready_ns},
            {"t_publish_ns", nullptr}}},
          {"payload",
           {{"bytes", made.payload_bytes()},
            {"file", payload_file},
            {"digest_kind", "sha256"},
            {"digest", made.payload_sha256}}},
          {"host",
           {{"name", OBSERVER_HOST},
            {"opportunity_id", made.opportunity_id},
            {"opportunity_sequence", opportunity_sequence},
            {"grant_id", made.grant_id},
            {"grant_record_sha256", made.grant_record_sha256},
            {"ingress",
             {{"grid_index", made.grid.grid}, {"generation", made.grid.generation}, {"readback_sha256", made.ingress_sha256}}},
            {"completion", "host_synchronous"},
            {"service_us", made.service_us}}}};
}

/// Count, mean, least and largest of a list of times in microseconds.
json time_stats(const std::vector<double>& times, const char* excluded)
{
  json stats = {{"count", times.size()}, {"excluded", excluded}};
  if (!times.empty()) {
    double total = 0.0, least = times[0], most = times[0];
    for (double value : times) {
      total += value;
      least = std::min(least, value);
      most  = std::max(most, value);
    }
    stats["mean"] = total / static_cast<double>(times.size());
    stats["min"]  = least;
    stats["max"]  = most;
  }
  return stats;
}

json hash_json(const isac_replay::hash_count& from, const isac_replay::hash_count& to)
{
  return {{"digests", to.digests - from.digests}, {"updates", to.updates - from.updates}, {"bytes", to.bytes - from.bytes}};
}

/// What the receiver reported of one estimate, for the trace of the producer.
json producer_record(const isac_estimate::producer_view& view)
{
  json symbols = json::array();
  for (unsigned symbol = 0; symbol != 64; ++symbol) {
    if (((view.dmrs_symbol_mask >> symbol) & 1U) != 0) {
      symbols.push_back(symbol);
    }
  }
  json steps = json::array({"pilot_ls"});
  if (view.frequency_interpolation) {
    steps.push_back("frequency_interpolation");
  }
  if (view.frequency_smoothing) {
    steps.push_back("smoothing");
  }
  return {{"launch", view.launch},
          {"device", view.device},
          {"backend", "cuda"},
          {"resident_mode", view.resident},
          {"rnti", view.rnti},
          {"sfn", view.sfn},
          {"slot", view.slot},
          {"numerology", view.numerology},
          {"start_prb", view.start_prb},
          {"prb_count", view.nof_prb},
          {"contiguous", view.contiguous},
          {"dmrs_symbols", symbols},
          {"rx_ports", view.rx_ports},
          {"layers", view.layers},
          {"dmrs_type", view.dmrs_type},
          {"scale", static_cast<double>(view.dmrs_scaling)},
          {"steps", steps},
          {"time_interpolation", false},
          {"bytes", view.bytes}};
}

/// One line of the account of the capture: what became of the estimate of one decode. It is the
/// trace of the producer and of the completion of the copy.
json capture_record(const std::string& run_id, const std::string& mode, const std::string& backend,
                    const isac_observer::bound_slot& slot, const json& opportunity_sequence,
                    const obs::record_header& header, const isac_estimate::capture_line& got)
{
  const bool published = got.result == isac_estimate::outcome::published;
  const bool copied    = *got.copy_stream != '\0';
  return {{"schema", CAPTURE_SCHEMA},
          {"run_id", run_id},
          {"mode", mode},
          {"stage", isac_estimate::STAGE_O3},
          {"backend", backend},
          {"opportunity_id", slot.opportunity_id},
          {"opportunity_sequence", opportunity_sequence},
          {"outcome", isac_estimate::to_string(got.result)},
          {"reason", *got.reason != '\0' ? json(got.reason) : json(nullptr)},
          {"admitted", got.admitted},
          {"seen", got.seen},
          {"landed", got.landed},
          {"producer", got.seen ? producer_record(got.producer) : json(nullptr)},
          {"copy",
           copied ? json({{"stream", got.copy_stream},
                          {"landing", got.landing},
                          {"completion", "cuda_event"},
                          {"polls", got.polls},
                          {"submit_us", got.submit_us}})
                  : json(nullptr)},
          {"ring",
           published ? json({{"epoch", got.epoch},
                             {"sequence", got.sequence},
                             {"depth", got.depth},
                             {"bytes", got.bytes},
                             {"window_calls", got.window_calls}})
                     : json(nullptr)},
          {"keys",
           {{"episode", hex64(slot.episode)}, {"opportunity", hex64(slot.opportunity)}, {"grant", hex64(slot.grant)}}},
          {"absolute_slot", slot.absolute_slot},
          {"grid", {{"grid_index", got.grid.grid}, {"generation", got.grid.generation}}},
          {"payload_bytes", header.payload_bytes},
          {"valid", got.landed ? json(got.counts.valid) : json(nullptr)},
          {"holes", got.landed ? json(got.counts.holes) : json(nullptr)},
          {"nonzero", got.landed ? json(got.counts.nonzero) : json(nullptr)},
          {"t_expect_ns", got.t_expect_ns},
          {"t_tap_ns", got.seen ? json(got.t_tap_ns) : json(nullptr)},
          {"t_submit_ns", got.seen ? json(got.t_submit_ns) : json(nullptr)},
          {"t_resolve_ns", got.t_resolve_ns},
          {"t_complete_ns", got.t_complete_ns != 0 ? json(got.t_complete_ns) : json(nullptr)},
          {"t_publish_ns", published ? json(got.t_publish_ns) : json(nullptr)},
          {"t_done_ns", got.t_done_ns},
          {"resolve_us", got.resolve_us}};
}

/// The descriptor of one captured estimate of the receiver: the fields of its obs/v1 header as a
/// JSON record, as observation_record gives them for the stage O2. It names no channel, no
/// noise, no transport block and no label.
json estimate_record(const std::string& run_id, std::uint64_t epoch, std::size_t number,
                     const isac_observer::bound_slot& slot, const obs::record_header& header,
                     const isac_estimate::capture_line& got, const std::string& source, const std::string& grant_id,
                     std::size_t opportunity_sequence, const std::string& readback_sha256, const std::string& payload_file,
                     const std::string& payload_sha256, const std::string& mask_file, const std::string& mask_sha256)
{
  const json producer  = producer_record(got.producer);
  const bool published = got.result == isac_estimate::outcome::published;
  json       map       = json::array();
  for (unsigned plane = 0; plane != header.rx_count; ++plane) {
    map.push_back(header.antenna_map[plane]);
  }
  return {{"schema", OBS_SCHEMA},
          {"run_id", run_id},
          {"keys",
           {{"epoch", epoch},
            {"sequence", number},
            {"episode", hex64(slot.episode)},
            {"opportunity", hex64(slot.opportunity)},
            {"grant", hex64(slot.grant)}}},
          {"radio",
           {{"cell", slot.cell}, {"ue", slot.ue}, {"sfn", slot.sfn}, {"slot", slot.slot}, {"absolute_slot", slot.absolute_slot}}},
          {"what",
           {{"stage", isac_estimate::STAGE_O3},
            {"producer", isac_estimate::PRODUCER},
            {"producer_steps", producer["steps"]},
            {"source", source},
            {"port", header.port},
            {"layout", isac_estimate::LAYOUT_O3_FP16},
            {"dtype", isac_estimate::DTYPE_CFP16},
            {"scale", producer["scale"]},
            {"rule", isac_estimate::RULE},
            {"precision", isac_estimate::PRECISION}}},
          {"shape",
           {{"rank", 4},
            {"axes", {"dmrs_symbol", "prb", "rx_port", "subcarrier_in_prb"}},
            {"shape", {header.shape[0], header.shape[1], header.shape[2], header.shape[3]}},
            {"strides", {header.strides[0], header.strides[1], header.strides[2], header.strides[3]}}}},
          {"allocation",
           {{"first_subcarrier", header.first_subcarrier},
            {"subcarrier_step", header.subcarrier_step},
            {"subcarrier_count", header.subcarrier_count},
            {"dmrs_symbols", producer["dmrs_symbols"]},
            {"rx_count", header.rx_count},
            {"layer_count", header.layer_count},
            {"antenna_map", map},
            {"coordinates_sha256", nullptr}}},
          {"validity",
           {{"mask_bytes", header.mask_bytes},
            {"file", mask_file},
            {"sha256", mask_sha256},
            {"valid", got.counts.valid},
            {"holes", got.counts.holes}}},
          {"time",
           {{"unit", isac_observer::TIME_UNIT_TC},
            {"tc_per_second", isac_observer::TC_PER_SECOND},
            {"t_first_sample", slot.t_first_sample},
            {"t_last_sample", slot.t_last_sample},
            {"t_ready_ns", got.t_complete_ns},
            {"t_publish_ns", published ? json(got.t_publish_ns) : json(nullptr)}}},
          {"payload",
           {{"bytes", header.payload_bytes}, {"file", payload_file}, {"digest_kind", "sha256"}, {"digest", payload_sha256}}},
          {"host",
           {{"name", OBSERVER_HOST},
            {"opportunity_id", slot.opportunity_id},
            {"opportunity_sequence", opportunity_sequence},
            {"grant_id", grant_id},
            {"grant_record_sha256", slot.grant_record_sha256},
            {"ingress",
             {{"grid_index", got.grid.grid}, {"generation", got.grid.generation}, {"readback_sha256", readback_sha256}}},
            {"completion", "cuda_event"},
            {"producer", producer},
            {"copy", {{"stream", got.copy_stream}, {"landing", got.landing}, {"polls", got.polls}}},
            {"ring_sequence", published ? json(got.sequence) : json(nullptr)},
            {"nonzero", got.counts.nonzero},
            {"service_us", got.submit_us + got.resolve_us}}}};
}

void usage()
{
  std::cout << "Usage: pilot_ls --plan FILE --output-dir DIR [--only OPPORTUNITY_ID]... [--no-grids]\n"
               "                [--phy-debug]\n"
               "       pilot_ls --self-test\n"
               "       pilot_ls --route\n"
               "       pilot_ls --pacing\n"
               "Decodes the receiver replay that a pilot-ls-plan/v1, v2, v3 or v4 file names, as replay_pusch does, and with\n"
               "the tap \"pilot_ls\" exports the stage O2 from every ready grid before it is decoded:\n"
               "" << isac_observer::RULE << " on the pilots of the grant's sensing source. It writes\n"
               "  DIR/ingress/ingress.jsonl  one o1-grid/v1 descriptor per opportunity\n"
               "  DIR/ingress/NNNN.c64       the stored grid, little-endian complex64 [rx_port, symbol, subcarrier]\n"
               "  DIR/observer/o2.jsonl      one obs-descriptor/v1 record per opportunity (tap pilot_ls only)\n"
               "  DIR/observer/NNNN.o2.c64   the estimate, little-endian complex64 [rx_plane, reference_symbol, pilot]\n"
               "  DIR/observer/NNNN.o2.mask  its validity bits, one per element in payload order\n"
               "  DIR/radio/radio.jsonl      one radio/v1 record per planned decode\n"
               "  DIR/radio/NNNN.BACKEND.bin the decoded bytes of a decode whose CRC passed\n"
               "  DIR/native.json            the report, with the comparison against the original blocks\n"
               "A pilot-ls-plan/v2 file names the evidence mode and, with a ring, publishes every observation on the\n"
               "obs-ring/v1 ring DIR/ring/STREAM.sock, where one consumer attaches:\n"
               "  DIR/ring/publications.jsonl one obs-publication/v1 line per observation: published or dropped\n"
               "  DIR/ring/ring.json         the ring, its counts and what was pending at the cutoff\n"
               "In the mode timed nothing is hashed between the first fill and the drain of the ring, observer/ is\n"
               "not written, because the estimate exists only in the slot, and every file is written after the\n"
               "last opportunity.\n"
               "A pilot-ls-plan/v3 file may name the capture of the receiver's own estimate, the stage O3, which the\n"
               "build against the patched receiver exports:\n"
               "  DIR/estimate/captures.jsonl one o3-capture/v1 line per decode of the producing backend\n"
               "  DIR/estimate/o3.jsonl       one obs-descriptor/v1 record per captured estimate (mode audit)\n"
               "  DIR/estimate/NNNN.o3.fp16   the estimate, complex FP16 [dmrs_symbol, prb, rx_port, 12], as the\n"
               "                              device holds it (mode audit)\n"
               "  DIR/estimate/NNNN.o3.mask   its validity bits (mode audit)\n"
               "  DIR/estimate/ring.json      the ring of the O3 stream, DIR/estimate/STREAM.sock, when the plan names one\n"
               "A pilot-ls-plan/v4 file may name the offered schedule of a paced run, in the mode timed: the grid of\n"
               "an opportunity is filled ahead of its slot, the tap and the decodes start at its absolute release\n"
               "time, a release that could not start within its latest start is skipped and counted, and no release\n"
               "time moves:\n"
               "  DIR/pacing/releases.jsonl   one pace-release/v1 line per opportunity: released, started, done\n"
               "Its pacing object may name a lead-in: the first releases are decoded once more ahead of the schedule,\n"
               "so that the first use of the processor is no release. Nothing of it is observed, exported or counted:\n"
               "  DIR/pacing/lead_in.jsonl    one pace-lead-in/v1 line per decode of the lead-in: the cold start\n"
               "DIR must exist and must not hold a result. --only repeats named opportunities of the plan exactly.\n"
               "--phy-debug prints the path decision of the pinned processor for every PDU.\n"
               "--self-test decodes a two-block grant through the host processor with the tap off and on, and once\n"
               "more in the mode timed through a ring.\n"
               "--route prints whether this build has the route to the estimate of the patched receiver.\n"
               "--pacing prints the plan version with the offered schedule that this build reads.\n"
               "Exit status 2 for a refused plan, 1 for a decode without a completion or any other error.\n";
}

// ---------------------------------------------------------------------------------------------- processors

/// What one decode needs beside its grid: the PDU of the opportunity and the code blocks of its grant.
struct decode_setup {
  ocudu::pusch_processor::pdu_t pdu;
  unsigned                      codeblocks = 0;
};

/// Receives the true decoder callback of one decode and passes its CRC flag on.
class job_notifier final : public ocudu::pusch_processor_result_notifier
{
public:
  job_notifier(std::uint64_t token_, completion_sink& sink_) : token(token_), sink(sink_) {}

  /// True for the one start of the decode this notifier belongs to. Dispatch thread only.
  bool claim()
  {
    const bool first = !claimed;
    claimed          = true;
    return first;
  }

  void on_uci(const ocudu::pusch_processor_result_control& /*uci*/) override {}

  void on_sch(const ocudu::pusch_processor_result_data& sch_) override
  {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!reported) {
        sch      = sch_;
        reported = true;
      }
    }
    sink.complete(token, sch_.data.tb_crc_ok, std::string());
  }

  bool result(ocudu::pusch_processor_result_data& out) const
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (reported) {
      out = sch;
    }
    return reported;
  }

private:
  const std::uint64_t                token;
  completion_sink&                   sink;
  mutable std::mutex                 mutex;
  bool                               reported = false;
  bool                               claimed  = false;
  ocudu::pusch_processor_result_data sch;
};

/// One PUSCH processor of the pinned source, created as the pinned test creates it.
class processor_backend final : public decoder
{
public:
  processor_backend(const std::string&        name_,
                    const std::string&        type_,
                    unsigned                  harq_,
                    ocudu::task_executor&     executor,
                    unsigned                  threads,
                    const receiver_settings&  settings,
                    unsigned                  max_codeblocks,
                    std::vector<pinned_grid*> grids_) :
    label(name_), type(type_), harq(harq_), grids(std::move(grids_))
  {
    std::shared_ptr<ocudu::pusch_processor_factory> factory =
        ocudu::create_sw_pusch_processor_factory(executor,
                                                 threads + 1,
                                                 settings.ldpc_iterations,
                                                 settings.early_stop,
                                                 type,
                                                 ocudu::port_channel_estimator_td_interpolation_strategy::average,
                                                 ocudu::channel_equalizer_algorithm_type::mmse);
    if (!factory) {
      throw std::runtime_error("the " + label + " PUSCH processor factory is not available");
    }
    processor = factory->create();
    if (!processor) {
      throw std::runtime_error("the " + label + " PUSCH processor could not be created");
    }
    ocudu::rx_buffer_pool_config config;
    config.max_codeblock_size   = ocudu::ldpc::MAX_CODEBLOCK_SIZE;
    config.nof_buffers          = 2;
    config.nof_codeblocks       = std::max(1U, max_codeblocks);
    config.expire_timeout_slots = 10;
    config.external_soft_bits   = false;
    buffers                     = ocudu::create_rx_buffer_pool(config);
    if (!buffers) {
      throw std::runtime_error("the receive buffer pool of the " + label + " processor could not be created");
    }
  }

  const std::string& name() const { return label; }
  const std::string& factory_type() const { return type; }
  unsigned           harq_id() const { return harq; }

  /// Names the PDU of the next decode of an opportunity. Called by the dispatch thread before the start.
  void expect(const std::string& opportunity_id, const decode_setup& setup) { setups[opportunity_id] = setup; }

  /// Before a paced run: the notifier of a decode that is started later, so that its start
  /// allocates nothing. sink is where every completion of that decode is reported.
  void arrange(std::uint64_t token, completion_sink& sink)
  {
    if (!notifiers.emplace(token, std::make_unique<job_notifier>(token, sink)).second) {
      throw std::runtime_error("the decode " + std::to_string(token) + " of " + label + " was arranged before");
    }
  }

  void start(const request& job, completion_sink& sink) override
  {
    const auto found = setups.find(job.key.opportunity_id);
    if (found == setups.end()) {
      throw std::runtime_error("no PDU was prepared for " + job.key.text());
    }
    const decode_setup&     setup  = found->second;
    ocudu::unique_rx_buffer buffer = buffers->get_pool().reserve(
        setup.pdu.slot, ocudu::trx_buffer_identifier(setup.pdu.rnti, harq), setup.codeblocks, true);
    if (!buffer.is_valid()) {
      throw std::runtime_error("the receive buffer pool has no buffer for " + job.key.text());
    }
    // A paced run arranged the notifier of this decode before its first fill. Every other run makes it here.
    auto known = notifiers.find(job.token);
    if (known == notifiers.end()) {
      known = notifiers.emplace(job.token, std::make_unique<job_notifier>(job.token, sink)).first;
    }
    if (!known->second->claim()) {
      throw std::runtime_error("the decode " + job.key.text() + " was started before");
    }
    // The same owned grid for every backend: the reader of the pinned resource grid of this ticket.
    processor->process(ocudu::span<std::uint8_t>(job.output, job.output_bytes),
                       std::move(buffer),
                       *known->second,
                       grids.at(job.grid.grid)->ocudu_grid().get_reader(),
                       setup.pdu);
  }

  /// The decoder's own result of a job, once its callback arrived.
  bool result(std::uint64_t token, ocudu::pusch_processor_result_data& out) const
  {
    const auto found = notifiers.find(token);
    return found != notifiers.end() && found->second->result(out);
  }

  void stop() { buffers->stop(); }

private:
  std::string                                             label;
  std::string                                             type;
  unsigned                                                harq;
  std::vector<pinned_grid*>                               grids;
  std::unique_ptr<ocudu::pusch_processor>                 processor;
  std::unique_ptr<ocudu::rx_buffer_pool_controller>       buffers;
  std::map<std::string, decode_setup>                     setups;
  std::map<std::uint64_t, std::unique_ptr<job_notifier>>  notifiers;
};

/// The worker pool of the pinned test.
struct workers {
  unsigned                                                                                        threads;
  std::unique_ptr<ocudu::task_worker_pool<ocudu::concurrent_queue_policy::locking_mpmc>>          pool;
  std::unique_ptr<ocudu::task_worker_pool_executor<ocudu::concurrent_queue_policy::locking_mpmc>> executor;

  workers() : threads(std::max(1U, std::min(8U, std::thread::hardware_concurrency())))
  {
    pool = std::make_unique<ocudu::task_worker_pool<ocudu::concurrent_queue_policy::locking_mpmc>>("thread", threads, 1024);
    executor =
        std::make_unique<ocudu::task_worker_pool_executor<ocudu::concurrent_queue_policy::locking_mpmc>>(*pool);
  }
};

/// Joins the workers when it leaves scope. A worker is still inside its task for a moment after the
/// processor's callback: it returns its codeblock decoder to the pool of the processor. A processor
/// that is destroyed in that moment is used after it is freed. Declare this after every processor,
/// so that on each way out of the function the workers are joined first.
struct joined_workers {
  explicit joined_workers(workers& pool_) : pool(pool_) {}
  joined_workers(const joined_workers&)            = delete;
  joined_workers& operator=(const joined_workers&) = delete;
  ~joined_workers() { pool.pool->stop(); }

  workers& pool;
};

/// What the decoder itself said about one decode, for the radio record.
json decoder_report(const ocudu::pusch_processor_result_data& sch)
{
  json report = {{"codeblocks", sch.data.nof_codeblocks_total}};
  if (sch.data.ldpc_decoder_stats.get_nof_observations() > 0) {
    report["ldpc_iterations_mean"] = sch.data.ldpc_decoder_stats.get_mean();
    report["ldpc_codeblocks"]      = sch.data.ldpc_decoder_stats.get_nof_observations();
  }
  put_number(report, "sinr_db", sch.csi.get_sinr_dB());
  put_number(report, "epre_db", sch.csi.get_epre_dB());
  put_number(report, "rsrp_db", sch.csi.get_rsrp_dB());
  put_number(report, "evm", sch.csi.get_total_evm());
  put_number(report, "cfo_hz", sch.csi.get_cfo_Hz());
  const std::optional<ocudu::phy_time_unit> alignment = sch.csi.get_time_alignment();
  if (alignment.has_value()) {
    report["time_alignment_us"] = alignment.value().to_seconds() * 1e6;
  }
  return report;
}

/// The path a decode took. The resident flag is the pinned decoder's: it is set only when the
/// resident CUDA demodulation and decode produced this result.
json path_report(const std::string& requested, bool completed, const ocudu::pusch_processor_result_data& sch,
                 bool built_with_cuda, bool acceleration, std::size_t ports)
{
  const auto& timing = sch.data.acceleration_pipeline_timing;
  json        path   = {{"requested", requested}, {"executed", nullptr}, {"fallback", nullptr}};
  if (!completed) {
    return path;
  }
  path["executed"] = timing.valid ? "resident" : "host";
  if (timing.valid) {
    path["device_grid"]       = timing.device_grid_used;
    path["owned_device_grid"] = timing.owned_device_grid_used;
    path["resident_us"]       = {{"channel_estimate", timing.ch_estimate_us},
                                 {"setup", timing.process_data_setup_us},
                                 {"grid_staging", timing.grid_staging_us},
                                 {"demodulation", timing.demod_call_us},
                                 {"decode", timing.decode_call_us},
                                 {"join_and_notify", timing.join_notify_us}};
  } else if (requested == "cuda") {
    path["fallback"] = !built_with_cuda        ? "the adapter was built without CUDA"
                       : !acceleration         ? "no usable CUDA device at run time"
                       : !resident_ports(ports) ? "the resident path takes 1, 2, 4 or 8 receive ports"
                                                : "the processor chose the host path";
  }
  return path;
}

// ---------------------------------------------------------------------------------------------- self-test

struct self_test {
  unsigned failed = 0;

  void expect(bool condition, const std::string& what)
  {
    std::cout << (condition ? "PASS  " : "FAIL  ") << what << std::endl;
    failed += condition ? 0 : 1;
  }
};

class transmit_done final : public ocudu::pdsch_processor_notifier
{
public:
  void on_finish_processing() override { completed = true; }

  void wait()
  {
    while (!completed.load()) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

private:
  std::atomic<bool> completed = {false};
};

// A two-block grant in a bandwidth part that starts above CRB 0, on two receive ports.
constexpr const char* SELF_TEST_GRANT = R"({
  "schema": "pusch-grant/v1", "grant_id": "self-test",
  "carrier": {"scs_khz": 30, "cyclic_prefix": "normal", "grid_size_rb": 51, "carrier_hz": 3500000000, "n_cell_id": 1},
  "bwp": {"start_rb": 10, "size_rb": 24}, "slot": {"sfn": 0, "slot": 17}, "rnti": 4660, "n_id": 101,
  "allocation": {"mapping_type": "A", "prb_start": 5, "prb_count": 2, "symbol_start": 0, "symbol_count": 14},
  "mcs": {"table": "qam64", "index": 10}, "codeword": {"rv": 0, "new_data": true}, "layers": 1,
  "dmrs": {"config_type": 1, "length": 1, "type_a_position": 2, "additional_position": 1, "symbols": [2, 11],
           "ports": [0], "scrambling_id": 101, "n_scid": 0, "cdm_groups_without_data": 2},
  "tx": {"antenna_ports": 1, "precoding": "identity"}, "rx_ports": [0, 1], "tb": {"hex": ""}})";

// A known grant through the pinned host processor: the true callback, the comparison and the grid ownership.
int run_self_test()
{
  self_test         test;
  const grant_record grant  = parse_grant(SELF_TEST_GRANT, "the self-test grant");
  const grant_coding coding = derive(grant, "the self-test grant");
  payload            original;
  original.id = grant.grant_id;
  original.bytes.resize(coding.tbs_bits / 8);
  for (std::size_t index = 0; index != original.bytes.size(); ++index) {
    original.bytes[index] = static_cast<std::uint8_t>((index * 37U + 11U) & 0xffU);
  }
  original.sha256          = sha256_hex(original.bytes.data(), original.bytes.size());
  const std::string record = record_sha256(grant, original.bytes);
  test.expect(coding.tbs_bits == 384 && coding.codeblocks == 1 && record.size() == 64,
              "the grant carries 384 bits in one code block and its record has a hash");

  workers                                       pool_of_workers;
  std::shared_ptr<ocudu::resource_grid_factory> factory = ocudu::create_resource_grid_factory();
  std::shared_ptr<ocudu::pdsch_processor_factory> transmit_factory =
      ocudu::create_sw_pdsch_processor_factory(*pool_of_workers.executor, pool_of_workers.threads + 1, "", "auto");
  if (!factory || !transmit_factory) {
    pool_of_workers.pool->stop();
    throw std::runtime_error("the pinned factories are not available");
  }

  // The transmit grid, with the chain of the fixture adapter and the DM-RS referenced to CRB 0.
  // The transmitter lives until the workers are joined, as every processor here does.
  const unsigned subcarriers = grant.grid_size_rb * ocudu::NOF_SUBCARRIERS_PER_RB;
  auto           tx_grid     = factory->create(1, ocudu::MAX_NSYMB_PER_SLOT, ocudu::MAX_NOF_SUBCARRIERS);
  auto           transmitter = transmit_factory->create();
  const joined_workers transmit_join(pool_of_workers);
  tx_grid->set_all_zero();
  {
    ocudu::symbol_slot_mask dmrs_mask(ocudu::MAX_NSYMB_PER_SLOT);
    for (unsigned symbol : grant.dmrs_symbols) {
      dmrs_mask.set(symbol);
    }
    ocudu::pdsch_processor::pdu_t pdu;
    pdu.context      = std::nullopt;
    pdu.slot         = ocudu::slot_point(ocudu::to_numerology_value(spacing_of(grant)), grant.sfn, grant.slot);
    pdu.rnti         = ocudu::to_rnti(static_cast<std::uint16_t>(grant.rnti));
    pdu.bwp_size_rb  = grant.bwp_size_rb;
    pdu.bwp_start_rb = grant.bwp_start_rb;
    pdu.cp           = ocudu::cyclic_prefix::NORMAL;
    pdu.n_id         = grant.n_id;
    pdu.ref_point    = ocudu::pdsch_processor::pdu_t::CRB0;
    pdu.dmrs_symbol_mask            = dmrs_mask;
    pdu.dmrs                        = dmrs_type_of(grant);
    pdu.scrambling_id               = grant.scrambling_id;
    pdu.n_scid                      = (grant.n_scid != 0);
    pdu.nof_cdm_groups_without_data = grant.cdm_groups;
    pdu.freq_alloc         = ocudu::rb_allocation::make_type1(grant.prb_start, grant.prb_count, std::nullopt);
    pdu.start_symbol_index = grant.symbol_start;
    pdu.nof_symbols        = grant.symbol_count;
    pdu.tbs_lbrm           = ocudu::tbs_lbrm_default;
    pdu.reserved           = {};
    pdu.ratio_pdsch_data_to_sss_dB = 0.0F;
    pdu.ratio_pdsch_dmrs_to_sss_dB = ocudu::get_sch_to_dmrs_ratio_dB(grant.cdm_groups);
    pdu.precoding = ocudu::precoding_configuration::make_wideband(ocudu::make_identity(grant.layers));
    pdu.codewords.emplace_back(
        ocudu::pdsch_processor::codeword_description{coding.mcs.modulation, grant.rv, coding.base_graph});
    transmit_done done;
    transmitter->process(tx_grid->get_writer(), done, {ocudu::shared_transport_block(original.bytes)}, pdu);
    done.wait();
  }

  grid_shape shape;
  shape.rx_ports    = static_cast<unsigned>(grant.rx_ports.size());
  shape.tx_ports    = 1;
  shape.symbols     = ocudu::MAX_NSYMB_PER_SLOT;
  shape.subcarriers = subcarriers;
  pinned_grid first(*factory, shape.rx_ports, ocudu::MAX_NSYMB_PER_SLOT, ocudu::MAX_NOF_SUBCARRIERS);
  pinned_grid second(*factory, shape.rx_ports, ocudu::MAX_NSYMB_PER_SLOT, ocudu::MAX_NOF_SUBCARRIERS);
  o1_pool     grids({&first, &second});
  ledger      book(grids);

  // Fills a grid with the transmit grid on every receive plane, or leaves it empty, and makes it ready.
  // With rotate, receive plane 1 carries the transmit grid times j.
  auto ready = [&](bool transmitted, const std::string& opportunity, bool rotate = false,
                   isac_replay::evidence how = isac_replay::evidence::audit) {
    ticket key;
    if (!grids.try_acquire(key)) {
      throw std::runtime_error("the self-test has no free grid");
    }
    isac_replay::grid_sink& sink = grids.writer(key);
    sink.clear();
    std::vector<ocudu::cf_t> row(ocudu::MAX_NOF_SUBCARRIERS);
    for (unsigned symbol = 0; transmitted && symbol != shape.symbols; ++symbol) {
      tx_grid->get_reader().get(row, 0, symbol, 0);
      for (unsigned rx = 0; rx != shape.rx_ports; ++rx) {
        if (rotate && rx == 1) {
          std::vector<ocudu::cf_t> turned(row.begin(), row.begin() + subcarriers);
          for (ocudu::cf_t& value : turned) {
            value = ocudu::cf_t(-value.imag(), value.real());
          }
          sink.put(rx, symbol, turned.data(), subcarriers);
        } else {
          sink.put(rx, symbol, row.data(), subcarriers);
        }
      }
    }
    o1_descriptor descriptor;
    descriptor.run_id              = "self-test";
    descriptor.opportunity_id      = opportunity;
    descriptor.grant_id            = grant.grant_id;
    descriptor.grant_record_sha256 = record;
    descriptor.sfn                 = grant.sfn;
    descriptor.slot                = grant.slot;
    descriptor.shape               = shape;
    descriptor.grid_gain           = 1.0;
    descriptor.storage             = sink.storage();
    grids.commit(key, descriptor, how);
    return key;
  };

  receiver_settings settings;
  settings.backends = {"cpu"};
  settings.tap      = "none";
  processor_backend host("cpu", "auto", 0, *pool_of_workers.executor, pool_of_workers.threads, settings,
                         coding.codeblocks, {&first, &second});
  // A second host processor for the ledger of the timed mode, which numbers its jobs from one again.
  processor_backend timed_host("cpu", "auto", 1, *pool_of_workers.executor, pool_of_workers.threads, settings,
                               coding.codeblocks, {&first, &second});
  const joined_workers join(pool_of_workers);
  decode_setup      setup;
  setup.pdu        = make_pdu(grant, coding, grant.sfn, grant.slot);
  setup.codeblocks = coding.codeblocks;
  const std::chrono::milliseconds limit(30000);
  auto key_of = [](const std::string& opportunity) {
    job_key key;
    key.opportunity_id = opportunity;
    key.backend        = "cpu";
    return key;
  };
  for (const char* opportunity : {"sent", "other-block", "empty", "tapped", "rotated"}) {
    host.expect(opportunity, setup);
  }
  timed_host.expect("timed", setup);

  const ticket     sent    = ready(true, "sent");
  const dispatched decoded = dispatch(book, host, key_of("sent"), sent, record, original, limit);
  ocudu::pusch_processor_result_data sch;
  const bool                         reported = decoded.token != 0 && host.result(decoded.token, sch);
  test.expect(decoded.result == outcome::decoded && reported && sch.data.tb_crc_ok &&
                  book.decoded(decoded.token) == original.bytes && book.snapshot(decoded.token).input_intact,
              "the host processor decodes the noiseless grid: true callback, CRC pass, every byte equal to the "
              "original, the grid unchanged");
  const dispatched again = dispatch(book, host, key_of("sent"), sent, record, original, limit);
  test.expect(again.admitted == admission::duplicate_delivery && book.jobs().size() == 1,
              "a second delivery of the same key is refused and not decoded");

  // The receiver returns the bytes that were sent with a valid CRC. Against an original that
  // differs in one byte the comparison must fail.
  payload other = original;
  other.bytes[other.bytes.size() - 1] ^= 0x01;
  other.sha256                        = sha256_hex(other.bytes.data(), other.bytes.size());
  const dispatched mismatch           = dispatch(book, host, key_of("other-block"), sent, record, other, limit);
  const job        mismatched         = mismatch.token != 0 ? book.snapshot(mismatch.token) : job();
  test.expect(mismatch.result == outcome::payload_mismatch && mismatched.crc_ok &&
                  mismatched.compared.first_differing_byte == static_cast<long long>(other.bytes.size() - 1),
              "a valid CRC with one byte that differs from the original fails the comparison");
  test.expect(book.release_grid(sent) && grids.state(sent.grid) == isac_replay::grid_state::free,
              "the grid is released once every decode on it has called back");
  std::uint64_t unused = 0;
  test.expect(book.admit(key_of("stale"), sent, record, original, unused) == admission::stale_grid,
              "a released grid is not decoded again");

  const ticket     empty  = ready(false, "empty");
  const dispatched silent = dispatch(book, host, key_of("empty"), empty, record, original, limit);
  test.expect(silent.result == outcome::crc_fail && !book.snapshot(silent.token).crc_ok,
              "an empty grid ends as a CRC failure of the decoder, not as a decode");
  test.expect(book.release_grid(empty) && book.held_grids() == 0 && book.count(outcome::decoded) == 1 &&
                  book.count(outcome::payload_mismatch) == 1 && book.count(outcome::crc_fail) == 1,
              "three decodes, three keys, three outcomes, no grid held");

  // ---- The tap. The reference symbols of the self-test are read off the noiseless transmit grid:
  // each stored pilot component has magnitude one, and the reference has the amplitude of two CDM
  // groups without data, 10^(3/20), which is 0.998815 per component. The integration gate takes
  // the reference from the sensing-source record of the grant instead.
  isac_observer::source_spec source;
  source.grant_id            = grant.grant_id;
  source.grant_record_sha256 = record;
  source.source              = "pusch_dmrs";
  source.channel             = "PUSCH";
  source.link_direction      = "uplink";
  source.processor           = "pusch";
  source.ports               = {0};
  source.grid_symbols        = shape.symbols;
  source.grid_subcarriers    = shape.subcarriers;
  source.scs_khz             = grant.scs_khz;
  source.slot                = grant.slot;
  source.slot_symbols        = grant.dmrs_symbols;
  // Start of the useful part of symbols 2 and 11 of a slot at 30 kHz, in T_c from the slot start.
  source.symbol_useful_start_tc = {145920, 777216};
  source.tc_per_second          = isac_observer::TC_PER_SECOND;
  source.amplitude              = std::pow(10.0, 3.0 / 20.0);
  source.antenna_map            = {0, 1};
  const float component         = static_cast<float>(source.amplitude / std::sqrt(2.0));
  bool        unit_pilots       = true;
  {
    std::vector<ocudu::cf_t> row(ocudu::MAX_NOF_SUBCARRIERS);
    const unsigned           first_pilot = (grant.bwp_start_rb + grant.prb_start) * ocudu::NOF_SUBCARRIERS_PER_RB;
    for (unsigned symbol : grant.dmrs_symbols) {
      tx_grid->get_reader().get(row, 0, symbol, 0);
      for (unsigned n = 0; n != 6 * grant.prb_count; ++n) {
        isac_observer::pilot item;
        item.symbol     = symbol;
        item.subcarrier = first_pilot + 2 * n;
        const cf32 sent_pilot = row[item.subcarrier];
        unit_pilots = unit_pilots && std::fabs(sent_pilot.real()) == 1.0F && std::fabs(sent_pilot.imag()) == 1.0F;
        source.index.push_back(item);
        source.reference.push_back(cf32(sent_pilot.real() < 0.0F ? -component : component,
                                        sent_pilot.imag() < 0.0F ? -component : component));
      }
    }
  }
  test.expect(unit_pilots && source.index.size() == 24,
              "the transmit grid carries 24 pilots on CRB 15 and 16, each component stored with magnitude one");
  isac_observer::observer tap("self-test", 1);
  tap.bind(source);
  isac_observer::slot_context where;
  where.opportunity_id = "tapped";
  where.episode_id     = "self-test";
  where.sfn            = grant.sfn;
  where.slot           = grant.slot;
  where.cell           = 1;
  where.ue             = grant.rnti;
  const float unity    = static_cast<float>(1.0 / static_cast<double>(component));

  const ticket      tapped_key = ready(true, "tapped");
  const o1_view     tapped_view = grids.view(tapped_key, record);
  const std::string tapped_digest = tapped_view.digest();
  const isac_observer::observation flat = tap.observe(tapped_view, where);
  bool every_unity = flat.values.size() == 48 && flat.valid == 48 && flat.holes == 0;
  for (const cf32 value : flat.values) {
    every_unity = every_unity && value.real() == unity && value.imag() == 0.0F;
  }
  test.expect(every_unity && std::fabs(unity - 1.00119F) < 5e-6F,
              "pilot LS on the pinned grid reads a unity channel as 1.00119 on all 48 pilots of both receive planes");
  test.expect(flat.ingress_sha256 == tapped_digest && tapped_view.digest() == tapped_digest &&
                  flat.first_subcarrier == 180 && flat.subcarrier_step == 2 && flat.subcarrier_count == 12 &&
                  flat.symbols == grant.dmrs_symbols,
              "the observation names the digest of the grid it read, the grid is unchanged after the tap, and the "
              "pilots lie on subcarriers 180 to 202 of symbols 2 and 11");
  const dispatched after_tap = dispatch(book, host, key_of("tapped"), tapped_key, record, original, limit);
  test.expect(after_tap.result == outcome::decoded && book.decoded(after_tap.token) == original.bytes &&
                  book.snapshot(after_tap.token).input_intact &&
                  book.snapshot(after_tap.token).ingress_sha256 == book.snapshot(decoded.token).ingress_sha256,
              "the host processor decodes the tapped grid to the original bytes, from a grid with the digest of the "
              "untapped one");
  test.expect(book.release_grid(tapped_key), "the tapped grid is released after its decode");

  where.opportunity_id          = "rotated";
  const ticket      rotated_key = ready(true, "rotated", true);
  const isac_observer::observation turned = tap.observe(grids.view(rotated_key, record), where);
  bool phases = turned.values.size() == 48;
  for (std::size_t n = 0; phases && n != 24; ++n) {
    phases = turned.values[n].real() == unity && turned.values[n].imag() == 0.0F &&
             turned.values[24 + n].real() == 0.0F && turned.values[24 + n].imag() == unity;
  }
  test.expect(phases && flat.values == tap.observations().front().values,
              "a receive plane turned by j reads as j times 1.00119 on that plane alone, and the first observation "
              "still holds its values");
  const dispatched after_turn = dispatch(book, host, key_of("rotated"), rotated_key, record, original, limit);
  test.expect(after_turn.result == outcome::decoded && book.release_grid(rotated_key) && book.held_grids() == 0,
              "the host processor decodes the rotated grid as well, and no grid stays held");
  // ---- The timed mode on the pinned grid: a ring of two slots without a consumer, the estimate
  // written into the slot, the decode admitted and resolved by the ticket of the fill. Nothing
  // is hashed from the fill to the released grid.
  {
    const isac_observer::pilot_source compiled(source);
    isac_transport::ring_needs        needs;
    needs.include(compiled);
    obs::ring_config config; // no control channel: nothing attaches in the self-test
    config.slot_count       = 2;
    config.host_capacity    = needs.host_bytes;
    config.mask_capacity    = needs.mask_bytes;
    config.payload_capacity = needs.payload_bytes;
    obs::producer             ring(config);
    isac_observer::observer   timed_tap("self-test", 1, isac_replay::evidence::timed);
    timed_tap.bind(source);
    isac_transport::publisher out(ring, timed_tap, isac_replay::evidence::timed);
    where.opportunity_id     = "timed";
    const std::size_t handle = out.prepare(where, record, 0);
    out.reserve(4);
    ledger timed_book(grids, isac_replay::evidence::timed);
    job_key timed_key;
    timed_key.opportunity_id = "timed";
    timed_key.backend        = "cpu";

    const isac_replay::hash_count before = isac_replay::sha256_count();
    const ticket      timed_grid = ready(true, "timed", false, isac_replay::evidence::timed);
    const o1_view     timed_view = grids.view(timed_grid, record);
    const isac_transport::published_record timed_sent = out.publish(handle, timed_view);
    const dispatched  timed_done = dispatch(timed_book, timed_host, timed_key, timed_grid, record, original, limit);
    const bool        given_back = timed_book.release_grid(timed_grid);
    const isac_replay::hash_count after = isac_replay::sha256_count();

    test.expect(after.digests == before.digests && after.updates == before.updates && after.bytes == before.bytes,
                "in the mode timed no SHA-256 is computed from the fill of the grid to its release: " +
                    std::to_string(after.digests - before.digests) + " digests, " +
                    std::to_string(after.bytes - before.bytes) + " bytes");
    // The slot the record lies in: slot 0 of the segment the producer mapped.
    const obs::segment_header& segment = ring.header();
    const std::uint8_t*        slot    = reinterpret_cast<const std::uint8_t*>(&segment) + obs::SEGMENT_HEADER_BYTES;
    const obs::record_header*  header  = reinterpret_cast<const obs::record_header*>(slot + obs::SLOT_PREFIX_BYTES);
    test.expect(timed_sent.outcome == isac_transport::publication::published && timed_sent.sequence == 0 && timed_sent.valid == 48 &&
                    timed_sent.window_calls == 0 && segment.published.load() == 1 && header->evidence == obs::EVIDENCE_TIMED &&
                    header->digest_kind == obs::DIGEST_NONE && header->payload_bytes == 48 * sizeof(cf32) &&
                    header->shape[0] == 2 && header->shape[1] == 2 && header->shape[2] == 12 && header->valid == 48,
                "the timed observation is published as one obs/v1 record of 2 by 2 by 12 values without a digest, "
                "with no system call between its admission and its publication");
    test.expect(std::memcmp(slot + segment.payload_offset, flat.values.data(), 48 * sizeof(cf32)) == 0 &&
                    std::memcmp(slot + segment.mask_offset, flat.mask.data(), flat.mask.size()) == 0,
                "the payload and the validity bits in the slot equal those of the audited observation, bit for bit");
    test.expect(timed_done.result == outcome::decoded && timed_book.decoded(timed_done.token) == original.bytes &&
                    timed_book.snapshot(timed_done.token).input_intact &&
                    timed_book.snapshot(timed_done.token).ingress_sha256.empty() && given_back,
                "the host processor decodes the timed grid to the original bytes, admitted and resolved by the ticket "
                "of its fill");
    obs::slot unused_slot;
    const isac_transport::published_record second_sent = out.publish(handle, grids.view(ready(true, "timed-2", false,
                                                                     isac_replay::evidence::timed), record));
    const isac_transport::published_record third_sent  = out.publish(handle, grids.view(ready(true, "timed-3", false,
                                                                     isac_replay::evidence::timed), record));
    test.expect(second_sent.outcome == isac_transport::publication::published &&
                    third_sent.outcome == isac_transport::publication::dropped && ring.counts().dropped == 1 &&
                    !ring.try_acquire(unused_slot),
                "a ring of two slots without a release admits two observations and drops the third at once");
  }
  host.stop();
  timed_host.stop();
  pool_of_workers.pool->stop();
  std::cout << "pilot_ls self-test: " << (test.failed == 0 ? "all checks passed" : "FAILED") << std::endl;
  return test.failed == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------------------------- run

/// One grant of the run on the receiver side: its record, what the pinned library derives and its original block.
struct grant_entry {
  grant_record record;
  grant_coding coding;
  payload      original;
  std::string  record_file_sha256;
};

/// The slot rule of a ring, as obs::slots_for states it, without its exception.
struct plan_ring_check {
  std::uint32_t window;
  std::uint32_t burst;

  std::uint32_t slots() const
  {
    try {
      return obs::slots_for(window, burst);
    } catch (const obs::ring_error&) {
      return 0;
    }
  }
};

/// The position in the plan of the opportunity behind a prepared slot.
struct staged_sequence {
  const std::vector<std::size_t>& chosen;
  const std::vector<std::size_t>& handle_of;
  std::size_t                     handle;

  json sequence() const
  {
    for (const std::size_t candidate : chosen) {
      if (handle_of[candidate] == handle) {
        return candidate;
      }
    }
    return nullptr;
  }
};

/// What a timed run keeps of one decode until the last opportunity is done.
struct timed_decode {
  std::uint64_t token     = 0;
  bool          first_use = false;
};

/// What a paced run keeps of one decode of its lead-in until the run is done. Nothing of it is text.
struct lead_slot {
  std::size_t                sequence  = 0;     ///< The opportunity it decodes once more.
  bool                       delivered = false; ///< It had a grid.
  ticket                     key;
  double                     fill_us   = 0.0;
  std::int64_t               issue_ns  = 0;     ///< start + offset - lead: the time it is issued at.
  std::int64_t               ready_ns  = 0;
  std::int64_t               start_ns  = 0;
  std::int64_t               done_ns   = 0;
  bool                       released  = false;
  std::vector<std::uint64_t> tokens;            ///< One per backend, of the book of the lead-in.
};

/// What a timed run keeps of one opportunity until the last one is done. Nothing of it is text.
struct timed_slot {
  std::size_t               sequence  = 0;
  bool                      delivered = false;
  ticket                    key;
  double                    fill_us   = 0.0;
  bool                      observed  = false;
  std::size_t               line      = 0; ///< Of the publisher's account.
  bool                      retained  = false;
  bool                      released  = false;
  bool                      captured  = false; ///< The capture of the receiver's estimate was opened for this opportunity.
  std::size_t               capture   = 0;     ///< Its line in the account of the capture.
  std::vector<timed_decode> decodes;
};

json pdu_echo(const grant_record& grant)
{
  return {{"scs_khz", grant.scs_khz},
          {"rnti", grant.rnti},
          {"n_id", grant.n_id},
          {"bwp_start_rb", grant.bwp_start_rb},
          {"bwp_size_rb", grant.bwp_size_rb},
          {"prb_start", grant.prb_start},
          {"prb_count", grant.prb_count},
          {"symbol_start", grant.symbol_start},
          {"symbol_count", grant.symbol_count},
          {"layers", grant.layers},
          {"rv", grant.rv},
          {"dmrs_config_type", grant.dmrs_config_type},
          {"dmrs_symbols", grant.dmrs_symbols},
          {"dmrs_scrambling_id", grant.scrambling_id},
          {"dmrs_n_scid", grant.n_scid},
          {"cdm_groups_without_data", grant.cdm_groups},
          {"dmrs_reference", "crb0"},
          {"rx_ports", grant.rx_ports}};
}

int run(int argc, char** argv)
{
  std::string           plan_path;
  std::string           output_dir;
  std::set<std::string> only;
  bool                  keep_grids = true;
  bool                  self_check = false;
  bool                  phy_debug  = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--plan" && i + 1 < argc) {
      plan_path = argv[++i];
    } else if (arg == "--output-dir" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--only" && i + 1 < argc) {
      only.insert(argv[++i]);
    } else if (arg == "--no-grids") {
      keep_grids = false;
    } else if (arg == "--self-test") {
      self_check = true;
    } else if (arg == "--phy-debug") {
      phy_debug = true;
    } else if (arg == "--pacing") {
      // What a launcher asks before it writes a plan with an offered schedule. No device is touched.
      std::cout << "pilot_ls pacing: " << isac_observer::TAP_PLAN_SCHEMA_V4
                << ", absolute releases on the host monotonic clock, paced book without an allocation per decode"
                << ", lead-in ahead of the schedule" << std::endl;
      return 0;
    } else if (arg == "--route") {
#ifdef ISAC_O3_ROUTE
      // The accessor is defined by the receiver patch: this line links only against the patched objects.
      std::cout << "pilot_ls route: O3 from the patched receiver, tap interface version "
                << ocudu::PUSCH_RESIDENT_ESTIMATE_TAP_VERSION << ", "
                << (ocudu::get_pusch_resident_estimate_tap() == nullptr ? "no observer registered" : "an observer registered")
                << std::endl;
#else
      std::cout << "pilot_ls route: none, linked against the receiver without the patch" << std::endl;
#endif
      return 0;
    } else if (arg == "--help") {
      usage();
      return 0;
    } else {
      usage();
      throw input_error("unknown or incomplete option '" + arg + "'");
    }
  }
  ocudulog::init();
  ocudulog::fetch_basic_logger("ALL").set_level(ocudulog::basic_levels::warning);
  // The pinned processor logs its path decision of every PDU at debug level. --phy-debug keeps it.
  ocudulog::fetch_basic_logger("PHY").set_level(phy_debug ? ocudulog::basic_levels::debug
                                                          : ocudulog::basic_levels::warning);
  if (self_check) {
    return run_self_test();
  }
  if (plan_path.empty() || output_dir.empty()) {
    usage();
    throw input_error("--plan and --output-dir are required");
  }
  if (!is_directory(output_dir)) {
    throw input_error("the output directory " + output_dir + " does not exist");
  }
  const std::string report_path  = output_dir + "/" + REPORT_FILE;
  const std::string ingress_path = output_dir + "/" + INGRESS_DIR;
  const std::string radio_path   = output_dir + "/" + RADIO_DIR;
  const std::string observer_path = output_dir + "/" + OBSERVER_DIR;
  const std::string ring_path     = output_dir + "/" + RING_DIR;
  const std::string estimate_path = output_dir + "/" + ESTIMATE_DIR;
  const std::string pacing_path   = output_dir + "/" + PACING_DIR;
  if (exists(report_path) || exists(ingress_path) || exists(radio_path) || exists(observer_path) || exists(ring_path) ||
      exists(estimate_path) || exists(pacing_path)) {
    throw input_error("the output directory already holds a result. Evidence is never overwritten.");
  }

  // ---- Everything is read and checked before a processor exists or a file is written. ----
  std::string                   plan_sha256;
  const isac_observer::tap_plan tap    = isac_observer::load_tap_plan(plan_path, &plan_sha256);
  const bool                    tapped = tap.tap == isac_observer::TAP_PILOT_LS;
  if (!tapped && tap.tap != isac_observer::TAP_NONE) {
    throw input_error("unknown tap '" + tap.tap + "': observer.tap is \"none\" or \"pilot_ls\"");
  }
  if (tap.epoch < 1) {
    throw input_error("observer.epoch is at least 1");
  }
  if (tapped && (tap.stage != isac_observer::STAGE_O2 || tap.layout != isac_observer::LAYOUT_O2_COMB || tap.source.empty())) {
    throw input_error(std::string("the tap pilot_ls exports the stage ") + isac_observer::STAGE_O2 + " in the layout " +
                      isac_observer::LAYOUT_O2_COMB + " and names its sensing source");
  }
  // The capture of the receiver's estimate, which a plan of version 3 may name. It needs the
  // sensing source of every grant as well: the records carry its keys and its sample times.
  const bool estimating = tap.estimate.present;
  if (!tapped && (!tap.stage.empty() || !tap.layout.empty() || !tap.source.empty() ||
                  (!estimating && !tap.sources.empty()))) {
    throw input_error("a run without a tap names no stage, no layout and no sensing source");
  }
  // ---- The evidence mode and the ring of a plan of version 2. A plan of version 1 has neither. ----
  const bool timed = tap.evidence == isac_observer::EVIDENCE_TIMED;
  if (!timed && tap.evidence != isac_observer::EVIDENCE_AUDIT) {
    throw input_error("unknown evidence mode '" + tap.evidence + "': evidence.mode is \"audit\" or \"timed\"");
  }
  const isac_replay::evidence mode   = timed ? isac_replay::evidence::timed : isac_replay::evidence::audit;
  const bool                  ringed = tap.ring.present;
  if (ringed && !tapped) {
    throw input_error("a run without a tap publishes nothing: the plan names a ring and no tap");
  }
  if (timed && tapped && !ringed) {
    throw input_error("a timed observer writes its estimate into a slot of the ring: the plan names a tap and no ring");
  }
  if (ringed) {
    const plan_ring_check rule = {tap.ring.window_records, tap.ring.burst_records};
    if (tap.ring.slots != rule.slots()) {
      throw input_error("ring.slots is " + std::to_string(tap.ring.slots) + ", and a window of " +
                        std::to_string(tap.ring.window_records) + " records with a burst of " +
                        std::to_string(tap.ring.burst_records) + " needs " + std::to_string(rule.slots()) +
                        ": the records a window holds in place, plus one burst, plus one, rounded up to a power of two");
    }
    if (tap.ring.wait != "spin" && tap.ring.wait != "wait") {
      throw input_error("ring.wait is \"spin\" or \"wait\"");
    }
    if (tap.ring.attach != isac_observer::ATTACH_REQUIRED && tap.ring.attach != isac_observer::ATTACH_NONE) {
      throw input_error("ring.attach is \"required\" or \"none\"");
    }
    if (tap.ring.stream.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789") != std::string::npos ||
        tap.ring.stream.size() > 16) {
      throw input_error("ring.stream is at most 16 lower-case letters and digits");
    }
  }
  if (estimating) {
    const isac_observer::plan_estimate& wanted = tap.estimate;
    if (wanted.tap != isac_estimate::TAP_NATIVE_ESTIMATE || wanted.stage != isac_estimate::STAGE_O3 ||
        wanted.layout != isac_estimate::LAYOUT_O3_FP16) {
      throw input_error(std::string("the estimate of a plan is the tap ") + isac_estimate::TAP_NATIVE_ESTIMATE +
                        ", which exports the stage " + isac_estimate::STAGE_O3 + " in the layout " +
                        isac_estimate::LAYOUT_O3_FP16);
    }
    if (wanted.backend != "cuda") {
      throw input_error("estimate.backend is \"cuda\": only the resident CUDA path of the receiver leaves its estimate "
                        "on the device. The host path has no such buffer and is recorded as unsupported");
    }
    if (wanted.copy_stream != isac_estimate::COPY_PRODUCER && wanted.copy_stream != isac_estimate::COPY_SIDE) {
      throw input_error("estimate.copy_stream is \"producer\" or \"side\"");
    }
    if (wanted.landing != isac_estimate::LANDING_SLOT && wanted.landing != isac_estimate::LANDING_STAGING) {
      throw input_error("estimate.landing is \"slot\" or \"staging\"");
    }
    if (tapped && wanted.source != tap.source) {
      throw input_error("the estimate names the sensing source " + wanted.source + ", and the observer taps " + tap.source);
    }
    if (timed && !wanted.ring.present) {
      throw input_error("a timed estimate exists in a slot of its ring only: the plan names an estimate and no ring for it");
    }
    if (wanted.ring.present) {
      const plan_ring_check rule = {wanted.ring.window_records, wanted.ring.burst_records};
      if (wanted.ring.slots != rule.slots()) {
        throw input_error("estimate.ring.slots is " + std::to_string(wanted.ring.slots) + ", and a window of " +
                          std::to_string(wanted.ring.window_records) + " records with a burst of " +
                          std::to_string(wanted.ring.burst_records) + " needs " + std::to_string(rule.slots()));
      }
      if (wanted.ring.wait != "spin" && wanted.ring.wait != "wait") {
        throw input_error("estimate.ring.wait is \"spin\" or \"wait\"");
      }
      if (wanted.ring.attach != isac_observer::ATTACH_REQUIRED && wanted.ring.attach != isac_observer::ATTACH_NONE) {
        throw input_error("estimate.ring.attach is \"required\" or \"none\"");
      }
      if (wanted.ring.stream.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789") != std::string::npos ||
          wanted.ring.stream.size() > 16) {
        throw input_error("estimate.ring.stream is at most 16 lower-case letters and digits");
      }
    }
#ifndef ISAC_O3_ROUTE
    throw input_error("the plan names the receiver's own estimate, and this build has no route to it: it is linked "
                      "against the receiver without the patch of patches/ocudu/. Run the plan with the pilot_ls of the "
                      "adapter build against the patched receiver (python scripts/native_adapter.py build --o3)");
#endif
  }
  // ---- The offered schedule of a plan of version 4. Its length is checked once the replay is read. ----
  const bool            paced = tap.pacing.present;
  isac_pacing::schedule offered;
  if (paced) {
    if (!timed) {
      throw input_error("a paced run is a timed run: the plan names an offered schedule and the evidence mode audit");
    }
    if (tap.pacing.clock != isac_observer::PACING_CLOCK) {
      throw input_error(std::string("pacing.clock is \"") + isac_observer::PACING_CLOCK + "\"");
    }
    if (!only.empty()) {
      throw input_error("--only selects opportunities, and a paced run offers every release of its schedule");
    }
    offered.allowance_ns    = static_cast<std::int64_t>(tap.pacing.allowance_us) * 1000;
    offered.latest_start_ns = static_cast<std::int64_t>(tap.pacing.latest_start_us) * 1000;
    for (const std::uint64_t offset : tap.pacing.release_us) {
      offered.release_ns.push_back(static_cast<std::int64_t>(offset) * 1000);
    }
    isac_pacing::check(offered);
  }
  // The lead-in of that schedule, or none: a plan that names none runs as the accepted Sprint 4.1 run did.
  isac_pacing::lead_in warm;
  if (paced) {
    warm.releases = static_cast<std::size_t>(tap.pacing.lead_releases);
    warm.lead_ns  = static_cast<std::int64_t>(tap.pacing.lead_us) * 1000;
    isac_pacing::check(offered, warm, static_cast<std::int64_t>(tap.pacing.start_delay_us) * 1000);
  }
  if (tap.replay_plan.sha256.empty()) {
    throw input_error("the replay plan is bound by no hash");
  }
  std::string       replay_sha256;
  const replay_plan plan = load_replay_plan(tap.replay_plan.path, &replay_sha256);
  if (replay_sha256 != tap.replay_plan.sha256) {
    throw input_error("the replay plan does not have the hash its record binds (" + tap.replay_plan.path + ")");
  }
  if (plan.payload.rule != "fixed" && plan.payload.rule != "per_opportunity") {
    throw input_error("payload.rule must be \"fixed\" or \"per_opportunity\"");
  }
  if ((plan.payload.rule == "per_opportunity") != !plan.payload.generator.empty() ||
      plan.payload.generator.empty() != plan.payload.seed.empty()) {
    throw input_error("a payload per opportunity names its generator and its seed, and a fixed payload names neither");
  }
  if (plan.receiver.tap != "none") {
    throw input_error("the replay plan is refused: it must be the control plan of Sprint 2.4, with receiver.tap "
                      "\"none\". The tap is named by the observer plan");
  }
  if (plan.receiver.ldpc_iterations < 1 || plan.receiver.ldpc_iterations > 64 || plan.receiver.timeout_ms < 1 ||
      plan.receiver.timeout_ms > 600000) {
    throw input_error("receiver needs 1 to 64 LDPC iterations and a timeout of 1 to 600000 ms");
  }
  if (plan.receiver.backends.empty()) {
    throw input_error("receiver.backends names no backend");
  }
  for (std::size_t index = 0; index != plan.receiver.backends.size(); ++index) {
    const std::string& name = plan.receiver.backends[index];
    if (name != "cpu" && name != "cuda") {
      throw input_error("unknown backend '" + name + "': a backend is \"cpu\" or \"cuda\"");
    }
    if (std::count(plan.receiver.backends.begin(), plan.receiver.backends.end(), name) != 1) {
      throw input_error("the backend " + name + " is listed twice");
    }
  }
  if (estimating && std::find(plan.receiver.backends.begin(), plan.receiver.backends.end(), tap.estimate.backend) ==
                        plan.receiver.backends.end()) {
    throw input_error("the plan names the estimate of the backend " + tap.estimate.backend +
                      ", and the replay plan does not decode on it");
  }
  if (plan.channel_plan.sha256.empty()) {
    throw input_error("the channel plan is bound by no hash");
  }
  std::string    channel_sha256;
  const run_spec spec = isac_replay::load_plan(plan.channel_plan.path, &channel_sha256);
  if (channel_sha256 != plan.channel_plan.sha256) {
    throw input_error("the channel plan does not have the hash its record binds (" + plan.channel_plan.path + ")");
  }
  if (spec.rx_ports > ocudu::MAX_PORTS || spec.symbols != ocudu::MAX_NSYMB_PER_SLOT ||
      spec.subcarriers > ocudu::MAX_NOF_SUBCARRIERS) {
    throw input_error("the carrier grid of the channel plan is not one the pinned receiver decodes (at most " +
                      std::to_string(ocudu::MAX_PORTS) + " ports, " + std::to_string(ocudu::MAX_NSYMB_PER_SLOT) +
                      " symbols, at most " + std::to_string(ocudu::MAX_NOF_SUBCARRIERS) + " subcarriers)");
  }
  if (paced && offered.release_ns.size() != spec.opportunities.size()) {
    throw input_error("pacing.release_us names " + std::to_string(offered.release_ns.size()) + " releases, and the replay "
                      "holds " + std::to_string(spec.opportunities.size()) + " opportunities: one release each");
  }
  // The channel rules of Sprint 2.3: the gain, every array against its hash, every binding.
  const session     replay(spec);
  const grid_shape  shape = replay.shape();
  std::vector<verified_file> files = replay.files();

  // ---- Grants: the record of the PDU and the original block, each bound to the grid's record hash. ----
  std::map<std::string, grant_entry> grants;
  unsigned                           max_codeblocks = 0;
  for (const isac_replay::grant_spec& bound : spec.grants) {
    const auto found = std::find_if(plan.grants.begin(), plan.grants.end(),
                                    [&bound](const plan_grant& item) { return item.grant_id == bound.grant_id; });
    if (found == plan.grants.end()) {
      throw input_error("the plan names no record and no transport block for the grant " + bound.grant_id);
    }
    if (std::count_if(plan.grants.begin(), plan.grants.end(),
                      [&bound](const plan_grant& item) { return item.grant_id == bound.grant_id; }) != 1) {
      throw input_error("the grant " + bound.grant_id + " is listed twice");
    }
    const std::string where = "the record of grant " + bound.grant_id;
    if (found->record.sha256.empty() || found->block.sha256.empty()) {
      throw input_error("the record or the transport block of grant " + bound.grant_id + " is bound by no hash");
    }
    const std::vector<std::uint8_t> raw   = read_bytes(found->record.path, where);
    const std::vector<std::uint8_t> block = read_bytes(found->block.path, "the transport block of grant " + bound.grant_id);
    grant_entry                     entry;
    entry.record_file_sha256 = sha256_hex(raw.data(), raw.size());
    if (entry.record_file_sha256 != found->record.sha256) {
      throw input_error(where + " does not have the hash its record binds (" + found->record.path + ")");
    }
    entry.original.id     = bound.grant_id;
    entry.original.bytes  = block;
    entry.original.sha256 = sha256_hex(block.data(), block.size());
    if (entry.original.sha256 != found->block.sha256) {
      throw input_error("the transport block of grant " + bound.grant_id + " does not have the hash its record binds (" +
                        found->block.path + ")");
    }
    entry.record = parse_grant(std::string(raw.begin(), raw.end()), where);
    entry.coding = derive(entry.record, where);
    if (entry.record.grant_id != bound.grant_id) {
      throw input_error(where + " names the grant " + entry.record.grant_id);
    }
    if (block.size() != found->block_bytes || block.size() * 8 != entry.coding.tbs_bits) {
      throw input_error("mismatched payload: the transport block of grant " + bound.grant_id + " holds " +
                        std::to_string(block.size() * 8) + " bits, the plan names " +
                        std::to_string(found->block_bytes * 8) + " and the grant carries " +
                        std::to_string(entry.coding.tbs_bits));
    }
    // One hash binds the grid, the PDU and the block: the record with its transport block.
    const std::string canonical = record_sha256(entry.record, block);
    if (canonical != bound.record_sha256 || found->record_sha256 != bound.record_sha256) {
      throw input_error("changed allocation: " + where + " with its transport block has the hash " + canonical +
                        ", and the channel plan binds the grid of this grant to " + bound.record_sha256);
    }
    if (entry.record.rx_ports.size() != shape.rx_ports || entry.record.antenna_ports != shape.tx_ports ||
        entry.record.grid_size_rb * ocudu::NOF_SUBCARRIERS_PER_RB != shape.subcarriers) {
      throw input_error("changed allocation: the grant " + bound.grant_id + " has " +
                        std::to_string(entry.record.rx_ports.size()) + " receive ports on " +
                        std::to_string(entry.record.grid_size_rb * ocudu::NOF_SUBCARRIERS_PER_RB) +
                        " subcarriers, and the carrier grid of the run has " + std::to_string(shape.rx_ports) +
                        " on " + std::to_string(shape.subcarriers));
    }
    max_codeblocks = std::max(max_codeblocks, entry.coding.codeblocks);
    files.push_back({where, found->record.path, entry.record_file_sha256, raw.size()});
    files.push_back({"the transport block of grant " + bound.grant_id, found->block.path, entry.original.sha256,
                     block.size()});
    grants.emplace(bound.grant_id, std::move(entry));
  }
  if (plan.grants.size() != spec.grants.size()) {
    throw input_error("the plan lists a grant that the channel plan does not bind");
  }

  // ---- The observer: one sensing source per grant, bound by the hash that binds the grid. ----
  isac_observer::observer watcher(tap.run_id, tap.epoch, mode);
  isac_transport::ring_needs ring_needs;
  // The slots of the O3 stream are bound by an observer of their own, so that both streams count
  // the slots of an episode from the same start and neither depends on the other being on.
  isac_observer::observer estimate_slots(tap.run_id, tap.epoch, mode);
  const bool              sourced     = tapped || estimating;
  const std::string       source_name = tapped ? tap.source : tap.estimate.source;
  for (const isac_replay::grant_spec& bound : spec.grants) {
    if (!sourced) {
      break;
    }
    const auto named = [&bound](const isac_observer::plan_source& item) { return item.grant_id == bound.grant_id; };
    const auto found = std::find_if(tap.sources.begin(), tap.sources.end(), named);
    if (found == tap.sources.end()) {
      throw input_error("the plan binds no sensing source for the grant " + bound.grant_id);
    }
    if (std::count_if(tap.sources.begin(), tap.sources.end(), named) != 1) {
      throw input_error("the sensing source of grant " + bound.grant_id + " is listed twice");
    }
    if (found->record_sha256 != bound.record_sha256) {
      throw input_error("changed allocation: the sensing source of grant " + bound.grant_id + " is bound to the record " +
                        found->record_sha256 + ", and the channel plan binds the grid of this grant to " +
                        bound.record_sha256);
    }
    const isac_observer::source_spec source = isac_observer::load_source(*found, &files);
    const grant_record&              grant  = grants.at(bound.grant_id).record;
    const std::string                where  = "the sensing source of grant " + bound.grant_id;
    if (source.source != source_name) {
      throw input_error(where + " is the source " + source.source + ", and the plan taps " + source_name);
    }
    if (source.grid_symbols != shape.symbols || source.grid_subcarriers != shape.subcarriers ||
        source.scs_khz != grant.scs_khz) {
      throw input_error(where + " was built for another carrier grid than the run decodes");
    }
    if (source.slot != grant.slot || source.slot_symbols != grant.dmrs_symbols || source.ports != grant.dmrs_ports) {
      throw input_error("changed allocation: " + where + " names another slot, other reference symbols or other ports "
                        "than the record of the grant");
    }
    if (source.antenna_map.size() != shape.rx_ports) {
      throw input_error("the antenna map of grant " + bound.grant_id + " names " +
                        std::to_string(source.antenna_map.size()) + " receive planes, and the carrier grid of the run has " +
                        std::to_string(shape.rx_ports));
    }
    const unsigned first = (grant.bwp_start_rb + grant.prb_start) * ocudu::NOF_SUBCARRIERS_PER_RB;
    const unsigned last  = first + grant.prb_count * ocudu::NOF_SUBCARRIERS_PER_RB;
    for (const isac_observer::pilot& item : source.index) {
      if (item.subcarrier < first || item.subcarrier >= last) {
        throw input_error("changed allocation: " + where + " has a pilot on subcarrier " +
                          std::to_string(item.subcarrier) + ", outside the allocation of the grant");
      }
    }
    if (tapped) {
      watcher.bind(source);
      ring_needs.include(watcher.source_of(bound.record_sha256));
    }
    if (estimating) {
      estimate_slots.bind(source);
    }
  }
  if (sourced && tap.sources.size() != spec.grants.size()) {
    throw input_error("the plan lists a sensing source for a grant that the channel plan does not bind");
  }

  // ---- Opportunities: one PDU each, in the frame and slot of the opportunity. ----
  std::vector<decode_setup> setups(replay.opportunities());
  std::set<std::string>     identifiers;
  std::set<std::string>     blocks;
  for (std::size_t sequence = 0; sequence != replay.opportunities(); ++sequence) {
    const opportunity_spec& item  = spec.opportunities[sequence];
    const grant_entry&      grant = grants.at(replay.scheduled(sequence).grant_id);
    if (!identifiers.insert(item.opportunity_id).second) {
      throw input_error("duplicate delivery: the opportunity " + item.opportunity_id + " is listed twice");
    }
    if (item.slot != grant.record.slot) {
      throw input_error("the opportunity " + item.opportunity_id + " lies in slot " + std::to_string(item.slot) +
                        ", and its grant " + grant.record.grant_id + " was built for slot " +
                        std::to_string(grant.record.slot) + ": the DM-RS depends on the slot index");
    }
    setups[sequence].pdu        = make_pdu(grant.record, grant.coding, item.sfn, item.slot);
    setups[sequence].codeblocks = grant.coding.codeblocks;
    blocks.insert(grant.original.sha256);
  }
  for (const std::string& name : only) {
    if (identifiers.count(name) == 0) {
      throw input_error("--only names the opportunity " + name + ", which the plan does not hold");
    }
  }
  if (only.empty() && blocks.size() != plan.payload.blocks) {
    throw input_error("the plan declares " + std::to_string(plan.payload.blocks) + " transport blocks and schedules " +
                      std::to_string(blocks.size()));
  }

  // ---- Two owned grids, sized as the pinned test sizes its receive grid, and the processors. ----
  std::shared_ptr<ocudu::resource_grid_factory> factory = ocudu::create_resource_grid_factory();
  if (!factory) {
    throw std::runtime_error("the pinned resource grid factory is not available");
  }
  pinned_grid first(*factory, shape.rx_ports, ocudu::MAX_NSYMB_PER_SLOT, ocudu::MAX_NOF_SUBCARRIERS);
  pinned_grid second(*factory, shape.rx_ports, ocudu::MAX_NSYMB_PER_SLOT, ocudu::MAX_NOF_SUBCARRIERS);
  o1_pool     pool({&first, &second});
  ledger      book(pool, mode);
  // A paced run keeps its decodes in the paced book of Sprint 4.1 and never uses the ledger. The
  // evaluator side below reads the decodes of a run through these four, whichever book holds them.
  std::unique_ptr<isac_pacing::paced_book> paced_jobs;
  if (paced) {
    paced_jobs = std::make_unique<isac_pacing::paced_book>(pool);
  }
  // The decodes of a lead-in are kept apart, in a book whose tokens lie behind those of the
  // schedule. Nothing below reads it but the lead-in itself and its own file.
  std::unique_ptr<isac_pacing::paced_book> lead_jobs;
  std::vector<lead_slot>                   led;
  auto job_of   = [&](std::uint64_t token) { return paced_jobs ? paced_jobs->snapshot(token) : book.snapshot(token); };
  auto bytes_of = [&](std::uint64_t token) { return paced_jobs ? paced_jobs->decoded(token) : book.decoded(token); };
  auto count_of = [&](outcome value, const std::string& backend = std::string()) {
    return paced_jobs ? paced_jobs->count(value, backend) : book.count(value, backend);
  };

#ifdef ENABLE_CUDA
  const bool built_with_cuda = true;
#else
  const bool built_with_cuda = false;
#endif
  const bool wants_cuda =
      std::find(plan.receiver.backends.begin(), plan.receiver.backends.end(), "cuda") != plan.receiver.backends.end();
  // The device is asked only when the plan requests it: a host-only run starts no CUDA work.
  const bool acceleration = wants_cuda && ocudu::is_pusch_demodulator_acceleration_available();

  workers                                         pool_of_workers;
  std::vector<std::unique_ptr<processor_backend>> backends;
  for (std::size_t index = 0; index != plan.receiver.backends.size(); ++index) {
    const std::string& name = plan.receiver.backends[index];
    backends.push_back(std::make_unique<processor_backend>(name, name == "cuda" ? "gpu" : "auto",
                                                           static_cast<unsigned>(index), *pool_of_workers.executor,
                                                           pool_of_workers.threads, plan.receiver, max_codeblocks,
                                                           std::vector<pinned_grid*>{&first, &second}));
  }
  const joined_workers join(pool_of_workers);

  make_directory(ingress_path);
  make_directory(radio_path);
  if (tapped && !timed) {
    make_directory(observer_path);
  }
  if (estimating) {
    make_directory(estimate_path);
  }
  if (paced) {
    make_directory(pacing_path);
  }

  // ---- The opportunities of this run, in the order of the plan, and what a slot must not look up. ----
  std::vector<std::size_t> chosen;
  for (std::size_t sequence = 0; sequence != replay.opportunities(); ++sequence) {
    if (only.empty() || only.count(spec.opportunities[sequence].opportunity_id) != 0) {
      chosen.push_back(sequence);
    }
  }
  const std::size_t NO_HANDLE = static_cast<std::size_t>(-1);
  std::vector<std::size_t> handle_of(replay.opportunities(), NO_HANDLE);

  // ---- The ring: created, sized for the sources of the run, every slot of the run bound before
  // the first fill. This is where the join keys are hashed. ----
  std::unique_ptr<obs::producer>             ring;
  std::unique_ptr<isac_transport::publisher> out;
  bool                                       attached_at_start = false;
  std::string                                socket_path;
  if (ringed) {
    make_directory(ring_path);
    socket_path = ring_path + "/" + tap.ring.stream + ".sock";
    obs::ring_config config;
    config.socket_path      = socket_path;
    config.slot_count       = tap.ring.slots;
    config.host_capacity    = ring_needs.host_bytes;
    config.mask_capacity    = ring_needs.mask_bytes;
    config.payload_capacity = ring_needs.payload_bytes;
    config.first_epoch      = tap.epoch;
    try {
      ring = std::make_unique<obs::producer>(config);
    } catch (const obs::ring_error& error) {
      throw std::runtime_error(std::string("the observation ring could not be created: ") + error.what());
    }
    out = std::make_unique<isac_transport::publisher>(*ring, watcher, mode);
    for (const std::size_t sequence : chosen) {
      const opportunity_spec&     item  = spec.opportunities[sequence];
      const allocation&           bound = replay.scheduled(sequence);
      const grant_entry&          grant = grants.at(bound.grant_id);
      isac_observer::slot_context where;
      where.opportunity_id = item.opportunity_id;
      where.episode_id     = item.realization_id;
      where.sfn            = item.sfn;
      where.slot           = item.slot;
      where.cell           = grant.record.record.at("carrier").at("n_cell_id").get<unsigned>();
      where.ue             = grant.record.rnti;
      handle_of[sequence]  = out->prepare(where, bound.record_sha256, static_cast<std::uint32_t>(sequence));
    }
    out->reserve(chosen.size());
    if (tap.ring.attach == isac_observer::ATTACH_REQUIRED) {
      attached_at_start = ring->wait_attached(std::chrono::milliseconds(tap.ring.attach_timeout_ms));
      if (!attached_at_start) {
        throw std::runtime_error("no consumer attached to " + socket_path + " within " +
                                 std::to_string(tap.ring.attach_timeout_ms) + " ms, and the plan requires one");
      }
    } else {
      ring->service();
      attached_at_start = ring->attached();
    }
  }

  // ---- The capture of the receiver's estimate: the ring of the O3 stream, the device objects,
  // one expectation per decode of the producing backend, all before the first fill. The tap is
  // registered with the receiver last. ----
  std::unique_ptr<obs::producer> estimate_ring;
  bool                           estimate_attached_at_start = false;
  std::string                    estimate_socket;
  std::vector<std::size_t>       estimate_handle(replay.opportunities(), NO_HANDLE);
  std::vector<std::size_t>       estimate_slot_of; // capture handle -> slot of estimate_slots
  isac_estimate::capture*        taker = nullptr;
  std::size_t                    estimate_backend = backends.size();
#ifdef ISAC_O3_ROUTE
  std::unique_ptr<isac_estimate::device_route> route;
#endif
  if (estimating) {
    for (std::size_t b = 0; b != backends.size(); ++b) {
      if (backends[b]->name() == tap.estimate.backend) {
        estimate_backend = b;
      }
    }
    std::vector<isac_estimate::expectation> expected(replay.opportunities());
    std::size_t                             largest = 0;
    for (const std::size_t sequence : chosen) {
      const opportunity_spec& item  = spec.opportunities[sequence];
      const grant_record&     grant = grants.at(replay.scheduled(sequence).grant_id).record;
      isac_estimate::expectation& want = expected[sequence];
      want.rnti      = grant.rnti;
      want.sfn       = item.sfn;
      want.slot      = item.slot;
      want.start_prb = grant.bwp_start_rb + grant.prb_start;
      want.nof_prb   = grant.prb_count;
      want.rx_ports  = static_cast<unsigned>(grant.rx_ports.size());
      want.dmrs_type = grant.dmrs_config_type;
      for (unsigned symbol : grant.dmrs_symbols) {
        want.dmrs_symbol_mask |= std::uint64_t(1) << symbol;
      }
      largest = std::max(largest, want.shape().payload_bytes());
    }
    if (tap.estimate.ring.present) {
      estimate_socket = estimate_path + "/" + tap.estimate.ring.stream + ".sock";
      obs::ring_config config;
      config.socket_path      = estimate_socket;
      config.slot_count       = tap.estimate.ring.slots;
      config.host_capacity    = sizeof(isac_transport::staged_host);
      config.mask_capacity    = static_cast<std::uint32_t>((largest / isac_estimate::ELEMENT_BYTES + 7) / 8);
      config.payload_capacity = static_cast<std::uint32_t>(largest);
      config.first_epoch      = tap.epoch;
      try {
        estimate_ring = std::make_unique<obs::producer>(config);
      } catch (const obs::ring_error& error) {
        throw std::runtime_error(std::string("the ring of the O3 stream could not be created: ") + error.what());
      }
    }
#ifdef ISAC_O3_ROUTE
    if (!acceleration) {
      throw std::runtime_error("the plan names the receiver's own estimate, and no CUDA device is usable at run time: "
                               "the estimate exists only on the device");
    }
    isac_estimate::capture_config wanted;
    wanted.mode             = mode;
    wanted.slot_landing     = tap.estimate.landing == isac_estimate::LANDING_SLOT;
    wanted.resolve_limit_ns = static_cast<std::int64_t>(tap.estimate.timeout_us) * 1000;
    route = std::make_unique<isac_estimate::device_route>(tap.estimate.copy_stream == isac_estimate::COPY_SIDE, largest,
                                                          estimate_ring.get(), wanted);
    taker = &route->captures();
#endif
    for (const std::size_t sequence : chosen) {
      if (taker == nullptr) {
        break;
      }
      const opportunity_spec&     item  = spec.opportunities[sequence];
      const allocation&           bound = replay.scheduled(sequence);
      const grant_entry&          grant = grants.at(bound.grant_id);
      isac_observer::slot_context where;
      where.opportunity_id = item.opportunity_id;
      where.episode_id     = item.realization_id;
      where.sfn            = item.sfn;
      where.slot           = item.slot;
      where.cell           = grant.record.record.at("carrier").at("n_cell_id").get<unsigned>();
      where.ue             = grant.record.rnti;
      const std::size_t                  slot   = estimate_slots.prepare(where, bound.record_sha256);
      const isac_observer::pilot_source& source = estimate_slots.source_of(bound.record_sha256);
      estimate_handle[sequence] = taker->prepare(estimate_slots.prepared(slot), expected[sequence], source.spec().antenna_map,
                                                 source.spec().ports.front(), static_cast<std::uint32_t>(sequence));
      estimate_slot_of.push_back(slot);
    }
    if (taker != nullptr) {
      taker->reserve(chosen.size());
    }
    if (estimate_ring) {
      if (tap.estimate.ring.attach == isac_observer::ATTACH_REQUIRED) {
        estimate_attached_at_start =
            estimate_ring->wait_attached(std::chrono::milliseconds(tap.estimate.ring.attach_timeout_ms));
        if (!estimate_attached_at_start) {
          throw std::runtime_error("no consumer attached to " + estimate_socket + " within " +
                                   std::to_string(tap.estimate.ring.attach_timeout_ms) + " ms, and the plan requires one");
        }
      } else {
        estimate_ring->service();
        estimate_attached_at_start = estimate_ring->attached();
      }
    }
#ifdef ISAC_O3_ROUTE
    route->open();
#endif
  }

  const std::chrono::milliseconds limit(plan.receiver.timeout_ms);
  json                            opportunities = json::array();
  std::string                     ingress_lines;
  std::string                     radio_lines;
  std::string                     observer_lines;
  std::vector<double>             tap_times; // service time of the tap in microseconds, the first one excluded
  std::size_t                     tap_valid = 0;
  std::size_t                     tap_holes = 0;
  std::size_t                     tap_bytes = 0;
  std::map<std::string, std::string> stopped; // backend -> why no further decode is started on it
  std::map<std::string, json>        service; // backend -> service times in microseconds
  std::map<std::string, bool>        used;
  bool                               have_previous = false;
  ticket                             previous;
  std::string                        previous_digest;
  std::string                        previous_record;
  std::size_t                        previous_entry = 0;
  bool                               all_retained   = true;
  std::size_t                        radio_sequence = 0;
  std::size_t                        unreleased     = 0;
  // The parts of the audited observations, and the times of the ring, for the report.
  std::vector<double> part_estimate, part_grid_digest, part_keys_mask, part_payload_digest, ring_publish, ring_estimate;
  json                publication_lines = json::array();
  // What the capture of the receiver's estimate archives in the mode audit.
  std::string estimate_lines;
  std::size_t estimate_archived = 0;
  // What became of the estimate of one decode, for the entry of its opportunity. In the mode
  // audit a landed payload is archived here, with its digests, as its opportunity is done.
  auto estimate_entry = [&](std::size_t sequence, const isac_estimate::capture_line& got, bool archive,
                            const std::string& readback) {
    json result = {{"outcome", isac_estimate::to_string(got.result)},
                   {"reason", *got.reason != '\0' ? json(got.reason) : json(nullptr)},
                   {"backend", tap.estimate.backend},
                   {"ring", got.result == isac_estimate::outcome::published ? json(got.sequence) : json(nullptr)},
                   {"launch", got.seen ? json(got.producer.launch) : json(nullptr)},
                   {"valid", got.landed ? json(got.counts.valid) : json(nullptr)},
                   {"holes", got.landed ? json(got.counts.holes) : json(nullptr)},
                   {"nonzero", got.landed ? json(got.counts.nonzero) : json(nullptr)},
                   {"payload_sha256", nullptr},
                   {"mask_sha256", nullptr},
                   {"submit_us", got.seen ? json(got.submit_us) : json(nullptr)},
                   {"resolve_us", got.resolve_us}};
    if (archive && got.landed) {
      const obs::record_header& header       = taker->header_of(got.handle);
      const std::string         payload_file = numbered(sequence, ".o3.fp16");
      const std::string         mask_file    = numbered(sequence, ".o3.mask");
      const std::string         payload_sha  = sha256_hex(taker->payload(), header.payload_bytes);
      const std::string         mask_sha     = sha256_hex(taker->mask(), header.mask_bytes);
      write_bytes(estimate_path + "/" + payload_file, taker->payload(), header.payload_bytes);
      write_bytes(estimate_path + "/" + mask_file, taker->mask(), header.mask_bytes);
      estimate_lines += estimate_record(tap.run_id, tap.epoch, estimate_archived,
                                        estimate_slots.prepared(estimate_slot_of[got.handle]), header, got,
                                        tap.estimate.source, replay.scheduled(sequence).grant_id, sequence, readback,
                                        payload_file, payload_sha, mask_file, mask_sha)
                            .dump() +
                        "\n";
      result["payload_sha256"] = payload_sha;
      result["mask_sha256"]    = mask_sha;
      ++estimate_archived;
    }
    return result;
  };
  // The hash count before the first fill. A timed run must show the same count at the drain.
  const isac_replay::hash_count hash_bound = isac_replay::sha256_count();

  auto retire = [&]() {
    // The grid of the earlier opportunity was retained while the next one was filled and decoded.
    // It must still hold the values it was made ready with. Then the ledger releases it, once,
    // unless a decode on it never called back.
    bool retained = false;
    try {
      retained = pool.view(previous, previous_record).digest() == previous_digest;
    } catch (const state_error&) {
      retained = false;
    }
    const bool released                         = book.release_grid(previous);
    opportunities[previous_entry]["retained"]   = retained;
    opportunities[previous_entry]["released"]   = released;
    all_retained                                = all_retained && retained;
    unreleased += released ? 0 : 1;
  };

  auto radio_record = [&](std::size_t sequence, const opportunity_spec& item, const allocation& bound,
                          const grant_entry& grant, const std::string& backend, const job& record, bool first_use,
                          const json& ingress, json& decode) {
    ocudu::pusch_processor_result_data sch;
    bool                               reported = false;
    for (const std::unique_ptr<processor_backend>& candidate : backends) {
      if (candidate->name() == backend && record.token != 0) {
        reported = candidate->result(record.token, sch);
      }
    }
    const bool        crc_pass = record.completed && record.crc_ok && record.result != outcome::timeout;
    const std::string file     = numbered(sequence, "." + backend + ".bin");
    json line = {{"schema", RADIO_SCHEMA},
                 {"run_id", tap.run_id},
                 {"sequence", radio_sequence++},
                 {"key", record.key.text()},
                 {"opportunity_id", item.opportunity_id},
                 {"opportunity_sequence", sequence},
                 {"backend", backend},
                 {"sfn", item.sfn},
                 {"slot", item.slot},
                 {"grant_id", bound.grant_id},
                 {"grant_record_sha256", bound.record_sha256},
                 {"allocation",
                  {{"crb_start", grant.record.bwp_start_rb + grant.record.prb_start},
                   {"prb_count", grant.record.prb_count},
                   {"symbol_start", grant.record.symbol_start},
                   {"symbol_count", grant.record.symbol_count},
                   {"dmrs_symbols", grant.record.dmrs_symbols},
                   {"rx_ports", grant.record.rx_ports},
                   {"layers", grant.record.layers},
                   {"active_re", bound.active.size()},
                   {"active_sha256", bound.active_sha256}}},
                 {"ingress", ingress},
                 {"tb", {{"id", grant.original.id}, {"bytes", grant.original.bytes.size()}, {"bits", grant.coding.tbs_bits}}},
                 {"outcome", to_string(record.result)},
                 {"crc_ok", crc_pass},
                 {"payload_match", crc_pass && record.compared.equal},
                 {"decoded_sha256", crc_pass ? json(record.decoded_sha256) : json(nullptr)},
                 {"decoded_file", crc_pass ? json(file) : json(nullptr)},
                 {"error", record.error.empty() ? json(nullptr) : json(record.error)},
                 {"path", path_report(backend, reported, sch, built_with_cuda, acceleration, grant.record.rx_ports.size())},
                 {"decoder", reported ? decoder_report(sch) : json(nullptr)},
                 {"timing",
                  {{"service_us", record.completed && record.result != outcome::timeout ? json(record.service_us) : json(nullptr)},
                   {"first_use", first_use},
                   {"timeout_ms", plan.receiver.timeout_ms}}},
                 {"retry", 0},
                 {"tap", tap.tap}};
    radio_lines += line.dump() + "\n";
    if (crc_pass) {
      const std::vector<std::uint8_t> bytes = bytes_of(record.token);
      write_bytes(radio_path + "/" + file, bytes.data(), bytes.size());
    }
    if (line["timing"]["service_us"].is_number() && !first_use) {
      service[backend].push_back(record.service_us);
    }
    // The evaluator's view of the same decode: the comparison with the original block.
    decode = {{"backend", backend},
              {"key", record.key.text()},
              {"outcome", to_string(record.result)},
              {"crc_ok", crc_pass},
              {"payload_match", crc_pass && record.compared.equal},
              {"first_differing_byte", crc_pass ? json(record.compared.first_differing_byte) : json(nullptr)},
              {"bytes_compared", crc_pass ? record.compared.bytes_compared : 0},
              {"decoded_sha256", crc_pass ? json(record.decoded_sha256) : json(nullptr)},
              {"original_sha256", grant.original.sha256},
              {"guard_intact", record.completed ? json(record.guard_intact) : json(nullptr)},
              {"input_intact", record.completed ? json(record.input_intact) : json(nullptr)},
              {"completions", record.completions},
              {"executed", line["path"]["executed"]},
              {"service_us", line["timing"]["service_us"]}};
  };

  // ---- The audited run: every digest, every file as its opportunity is done. As Sprint 3.1 accepted it. ----
  for (std::size_t sequence = 0; !timed && sequence != replay.opportunities(); ++sequence) {
    const opportunity_spec& item = spec.opportunities[sequence];
    if (!only.empty() && only.count(item.opportunity_id) == 0) {
      continue;
    }
    const allocation&  bound = replay.scheduled(sequence);
    const grant_entry& grant = grants.at(bound.grant_id);
    json               entry = {{"sequence", sequence},
                                {"opportunity_id", item.opportunity_id},
                                {"realization_id", item.realization_id},
                                {"index", item.index},
                                {"grant_id", bound.grant_id},
                                {"sfn", item.sfn},
                                {"slot", item.slot},
                                {"decodes", json::array()}};
    if (tap.version >= 3) {
      entry["estimate"] = nullptr; // no decode of the producing backend was started for it
    }

    ticket     key;
    const auto filling   = std::chrono::steady_clock::now();
    const bool delivered = replay.deliver(sequence, pool, key);
    if (!delivered) {
      // Every grid is held by a decode that never called back. The opportunity keeps its place
      // in the denominator of every backend.
      for (const std::unique_ptr<processor_backend>& backend : backends) {
        job_key name;
        name.opportunity_id = item.opportunity_id;
        name.backend        = backend->name();
        book.skip(name, bound.grant_id, bound.record_sha256, "no free grid: every owned grid is held by a decode "
                                                             "without a completion");
        json decode;
        radio_record(sequence, item, bound, grant, backend->name(), book.jobs().back(), false, json(nullptr), decode);
        entry["decodes"].push_back(decode);
      }
      entry["grid_index"] = nullptr;
      entry["observer"]   = nullptr;
      opportunities.push_back(entry);
      continue;
    }
    const double fill_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - filling).count();

    // From here on the grid is read the way a consumer reads it: through a view of a ready grid.
    const o1_view        view       = pool.view(key, bound.record_sha256);
    const o1_descriptor& descriptor = view.descriptor();
    const std::string    digest     = view.digest();
    if (digest != descriptor.readback_sha256) {
      throw std::runtime_error("the grid of opportunity " + item.opportunity_id + " changed after it became ready");
    }
    const std::string grid_file = numbered(sequence, ".c64");
    if (keep_grids) {
      const std::vector<cf32> stored = read_back(view);
      write_bytes(ingress_path + "/" + grid_file, stored.data(), stored.size() * sizeof(cf32));
    }
    // The descriptor of the receiver side. Nothing in it names H, W, a transport block or a label.
    const json line = {{"schema", INGRESS_SCHEMA},
                       {"run_id", descriptor.run_id},
                       {"sequence", sequence},
                       {"opportunity_id", descriptor.opportunity_id},
                       {"sfn", descriptor.sfn},
                       {"slot", descriptor.slot},
                       {"grant_id", descriptor.grant_id},
                       {"grant_record_sha256", descriptor.grant_record_sha256},
                       {"active_re", descriptor.active_re},
                       {"active_sha256", descriptor.active_sha256},
                       {"grid_index", key.grid},
                       {"generation", key.generation},
                       {"state", to_string(pool.state(key.grid))},
                       {"grid",
                        {{"file", keep_grids ? json(grid_file) : json(nullptr)},
                         {"axes", {"rx_port", "symbol", "subcarrier"}},
                         {"shape", {shape.rx_ports, shape.symbols, shape.subcarriers}},
                         {"dtype", "<c8"},
                         {"order", "C"},
                         {"storage", descriptor.storage},
                         {"sha256", descriptor.readback_sha256}}},
                       {"amplitude", {{"grid_gain", descriptor.grid_gain}, {"convention", RULE}}}};
    ingress_lines += line.dump() + "\n";
    entry["grid_index"]      = key.grid;
    entry["generation"]      = key.generation;
    entry["readback_sha256"] = digest;
    entry["fill_us"]         = fill_us;

    // ---- The tap: the observer reads the ready grid before a processor does, and only reads. ----
    entry["observer"] = nullptr;
    if (tapped) {
      isac_observer::slot_context where;
      where.opportunity_id = item.opportunity_id;
      where.episode_id     = item.realization_id;
      where.sfn            = item.sfn;
      where.slot           = item.slot;
      where.cell           = grant.record.record.at("carrier").at("n_cell_id").get<unsigned>();
      where.ue             = grant.record.rnti;
      const isac_observer::observation& made = watcher.observe(view, where);
      const std::string payload_file = numbered(sequence, ".o2.c64");
      const std::string mask_file    = numbered(sequence, ".o2.mask");
      write_bytes(observer_path + "/" + payload_file, made.values.data(), made.payload_bytes());
      write_bytes(observer_path + "/" + mask_file, made.mask.data(), made.mask.size());
      observer_lines += observation_record(made, sequence, payload_file, mask_file).dump() + "\n";
      entry["observer"] = {{"sequence", made.sequence},
                           {"payload_sha256", made.payload_sha256},
                           {"mask_sha256", made.mask_sha256},
                           {"ingress_sha256", made.ingress_sha256},
                           {"valid", made.valid},
                           {"holes", made.holes},
                           {"service_us", made.service_us}};
      if (made.sequence != 0) {
        tap_times.push_back(made.service_us);
        part_estimate.push_back(made.estimate_us);
        part_grid_digest.push_back(made.grid_digest_us);
        part_keys_mask.push_back(made.keys_mask_us);
        part_payload_digest.push_back(made.payload_digest_us);
      }
      tap_valid += made.valid;
      tap_holes += made.holes;
      tap_bytes += made.payload_bytes();
      if (ringed) {
        // The audited observation, with the digest of its payload, is copied into a slot. Without
        // a credit it is dropped and counted, and the decodes below start all the same.
        const isac_transport::published_record& sent = out->publish(handle_of[sequence], made);
        entry["observer"]["ring"] = sent.outcome == isac_transport::publication::published ? "published" : "dropped";
        if (sent.outcome == isac_transport::publication::published && made.sequence != 0) {
          ring_publish.push_back(sent.publish_us);
        }
      }
    }

    // ---- The same grid, the same PDU and the same original block for every backend. ----
    const json ingress = {{"grid_index", key.grid}, {"generation", key.generation}, {"readback_sha256", digest}};
    for (const std::unique_ptr<processor_backend>& backend : backends) {
      job_key name;
      name.opportunity_id = item.opportunity_id;
      name.backend        = backend->name();
      job  record;
      bool first_use = false;
      if (stopped.count(backend->name()) != 0) {
        book.skip(name, bound.grant_id, bound.record_sha256, stopped[backend->name()]);
        record = book.jobs().back();
      } else {
        first_use = !used[backend->name()];
        backend->expect(item.opportunity_id, setups[sequence]);
        // The receiver's estimate of this decode is expected before the decode starts and
        // resolved after it. The decode never waits for the copy.
        const bool captured = taker != nullptr && backend->name() == tap.estimate.backend;
        if (captured) {
          taker->expect(estimate_handle[sequence], key);
        }
        const dispatched done = dispatch(book, *backend, name, key, bound.record_sha256, grant.original, limit);
        if (captured) {
          entry["estimate"] = estimate_entry(sequence, taker->resolve(), true, digest);
        }
        if (done.admitted != admission::admitted) {
          throw std::runtime_error("the grid of opportunity " + item.opportunity_id + " was not admitted for " +
                                   backend->name() + ": " + done.reason);
        }
        used[backend->name()] = true;
        record                = book.snapshot(done.token);
        if (done.result == outcome::timeout) {
          stopped[backend->name()] = "the " + backend->name() + " processor did not complete " + item.opportunity_id +
                                     " within " + std::to_string(plan.receiver.timeout_ms) + " ms";
        }
      }
      json ingress_now = ingress;
      if (record.completed) {
        ingress_now["intact"] = record.input_intact;
      }
      json decode;
      radio_record(sequence, item, bound, grant, backend->name(), record, first_use, ingress_now, decode);
      entry["decodes"].push_back(decode);
    }
    opportunities.push_back(entry);

    if (have_previous) {
      retire();
    }
    previous        = key;
    previous_digest = digest;
    previous_record = bound.record_sha256;
    previous_entry  = opportunities.size() - 1;
    have_previous   = true;
    if (ring) {
      ring->service(); // the control channel, after the decodes of this opportunity
    }
    if (estimate_ring) {
      estimate_ring->service();
    }
  }
  if (have_previous) {
    retire();
  }

  // ---- The timed run. Between a ready grid and the last decoder callback on it this loop takes
  // no digest, reads no grid back, writes no file and builds no record. What it keeps of an
  // opportunity is a few numbers in memory that was allocated above. ----
  std::vector<timed_slot> kept(timed ? chosen.size() : 0);
  const std::size_t       grid_values = static_cast<std::size_t>(shape.rx_ports) * shape.plane();
  std::vector<cf32>       arena(timed && keep_grids ? chosen.size() * grid_values : 0);
  if (timed && !paced) {
    std::vector<job_key> names(chosen.size() * backends.size());
    std::vector<const grant_entry*> grant_of(chosen.size());
    for (std::size_t n = 0; n != chosen.size(); ++n) {
      const opportunity_spec& item = spec.opportunities[chosen[n]];
      kept[n].sequence = chosen[n];
      kept[n].decodes.resize(backends.size());
      grant_of[n] = &grants.at(replay.scheduled(chosen[n]).grant_id);
      for (std::size_t b = 0; b != backends.size(); ++b) {
        names[n * backends.size() + b].opportunity_id = item.opportunity_id;
        names[n * backends.size() + b].backend        = backends[b]->name();
        backends[b]->expect(item.opportunity_id, setups[chosen[n]]); // every PDU is named before the first fill
      }
    }
    std::vector<char> halted(backends.size(), 0), started(backends.size(), 0);
    std::size_t       last = 0;
    bool              any  = false;
    auto              retire_timed = [&](timed_slot& old) {
      // Ownership by generation: the grid is still ready under the ticket of its fill.
      old.retained = pool.state(old.key.grid) == isac_replay::grid_state::ready &&
                     pool.generation(old.key.grid) == old.key.generation;
      old.released = book.release_grid(old.key);
      all_retained = all_retained && old.retained;
      unreleased += old.released ? 0 : 1;
    };
    for (std::size_t n = 0; n != chosen.size(); ++n) {
      const std::size_t       sequence = chosen[n];
      const opportunity_spec& item     = spec.opportunities[sequence];
      const allocation&       bound    = replay.scheduled(sequence);
      timed_slot&             now      = kept[n];
      const auto              filling  = std::chrono::steady_clock::now();
      ticket                  key;
      now.delivered = pool.try_acquire(key);
      if (!now.delivered) {
        for (std::size_t b = 0; b != backends.size(); ++b) {
          book.skip(names[n * backends.size() + b], bound.grant_id, bound.record_sha256,
                    "no free grid: every owned grid is held by a decode without a completion");
          now.decodes[b].token = book.jobs().back().token;
        }
        continue;
      }
      try {
        isac_replay::grid_sink& sink = pool.writer(key);
        sink.clear();
        replay.fill(sequence, sink);
        pool.commit(key, replay.describe(sequence, bound, sink), isac_replay::evidence::timed);
      } catch (...) {
        pool.abort(key);
        throw;
      }
      now.key     = key;
      now.fill_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - filling).count();

      // The grid is ready. The tap first: admission, the estimate in the slot, the publication.
      const o1_view ready = pool.view(key, bound.record_sha256);
      if (tapped) {
        now.line     = out->account().size();
        out->publish(handle_of[sequence], ready);
        now.observed = true;
      }
      for (std::size_t b = 0; b != backends.size(); ++b) {
        processor_backend& backend = *backends[b];
        if (halted[b] != 0) {
          book.skip(names[n * backends.size() + b], bound.grant_id, bound.record_sha256, stopped[backend.name()]);
          now.decodes[b].token = book.jobs().back().token;
          continue;
        }
        now.decodes[b].first_use = started[b] == 0;
        // The receiver's estimate: the admission before the decode, the completion after it.
        const bool captured = taker != nullptr && b == estimate_backend;
        if (captured) {
          now.capture = taker->account().size();
          taker->expect(estimate_handle[sequence], key);
        }
        const dispatched result  = dispatch(book, backend, names[n * backends.size() + b], key, bound.record_sha256,
                                            grant_of[n]->original, limit);
        if (captured) {
          taker->resolve();
          now.captured = true;
        }
        if (result.admitted != admission::admitted) {
          throw std::runtime_error("the grid of opportunity " + item.opportunity_id + " was not admitted for " +
                                   backend.name() + ": " + result.reason);
        }
        started[b]           = 1;
        now.decodes[b].token = result.token;
        if (result.result == outcome::timeout) {
          halted[b]               = 1;
          stopped[backend.name()] = "the " + backend.name() + " processor did not complete " + item.opportunity_id +
                                    " within " + std::to_string(plan.receiver.timeout_ms) + " ms";
        }
      }
      // Every decoder of this grid has called back. Only now is the grid read in full, into
      // memory that exists, for the evaluator.
      if (keep_grids) {
        for (unsigned rx = 0; rx != shape.rx_ports; ++rx) {
          for (unsigned symbol = 0; symbol != shape.symbols; ++symbol) {
            ready.get(rx, symbol,
                      arena.data() + n * grid_values + (static_cast<std::size_t>(rx) * shape.symbols + symbol) * shape.subcarriers,
                      shape.subcarriers);
          }
        }
      }
      if (any) {
        retire_timed(kept[last]);
      }
      last = n;
      any  = true;
      if (ring) {
        ring->service(); // the control channel, after the decodes of this opportunity
      }
      if (estimate_ring) {
        estimate_ring->service();
      }
    }
    if (any) {
      retire_timed(kept[last]);
    }
  }

  // ---- The paced run (Sprint 4.1). The harness is a scripted receiver on the offered schedule:
  // the grid of an opportunity is filled ahead of its slot, the tap and the decodes start at the
  // absolute release time of the slot, and a release that the harness could not start within its
  // latest start is skipped, recorded under its key and counted. No release time moves. Between
  // the release and the last decoder callback this loop takes no digest, reads no grid back,
  // writes no file, builds no record, allocates nothing and takes no lock of its own. A plan with
  // a lead-in decodes the first releases once more ahead of the schedule, under the same rule. ----
  std::unique_ptr<isac_pacing::pacer> pace;
  if (paced) {
    pace = std::make_unique<isac_pacing::pacer>(offered);
    // Every decode is named before the first fill: its key, its original block, its output
    // buffer, the request its decoder is handed and the notifier of its callback.
    for (std::size_t n = 0; n != chosen.size(); ++n) {
      const opportunity_spec& item  = spec.opportunities[chosen[n]];
      const allocation&       bound = replay.scheduled(chosen[n]);
      kept[n].sequence = chosen[n];
      kept[n].decodes.resize(backends.size());
      for (std::size_t b = 0; b != backends.size(); ++b) {
        job_key name;
        name.opportunity_id   = item.opportunity_id;
        name.backend          = backends[b]->name();
        const std::uint64_t t = paced_jobs->prepare(name, bound.grant_id, bound.record_sha256,
                                                    grants.at(bound.grant_id).original);
        kept[n].decodes[b].token = t;
        backends[b]->expect(item.opportunity_id, setups[chosen[n]]);
        backends[b]->arrange(t, *paced_jobs);
      }
    }
    // The lead-in: the first releases once more, named in a book of their own behind every token
    // of the schedule. The PDU of each is the one that was named for its opportunity above.
    if (warm.releases != 0) {
      lead_jobs = std::make_unique<isac_pacing::paced_book>(pool, &isac_pacing::monotonic_ns, paced_jobs->next_token());
      led.resize(warm.releases);
      for (std::size_t k = 0; k != led.size(); ++k) {
        const opportunity_spec& item  = spec.opportunities[chosen[k]];
        const allocation&       bound = replay.scheduled(chosen[k]);
        led[k].sequence = chosen[k];
        for (std::size_t b = 0; b != backends.size(); ++b) {
          job_key name;
          name.opportunity_id   = item.opportunity_id;
          name.backend          = backends[b]->name();
          const std::uint64_t t = lead_jobs->prepare(name, bound.grant_id, bound.record_sha256,
                                                     grants.at(bound.grant_id).original);
          led[k].tokens.push_back(t);
          backends[b]->arrange(t, *lead_jobs);
        }
      }
    }
    std::vector<char>  halted(backends.size(), 0), started(backends.size(), 0);
    const std::int64_t limit_ns = static_cast<std::int64_t>(plan.receiver.timeout_ms) * 1000000;
    std::size_t        last     = 0;
    bool               any      = false;
    auto               retire_paced = [&](timed_slot& old) {
      // Ownership by generation: the grid is still ready under the ticket of its fill.
      old.retained = pool.state(old.key.grid) == isac_replay::grid_state::ready &&
                     pool.generation(old.key.grid) == old.key.generation;
      old.released = paced_jobs->release_grid(old.key);
      all_retained = all_retained && old.retained;
      unreleased += old.released ? 0 : 1;
    };
    // The offered schedule is fixed here, after every consumer attached and everything was
    // prepared, and nothing that happens below moves it.
    const std::int64_t schedule_start =
        isac_pacing::monotonic_ns() + static_cast<std::int64_t>(tap.pacing.start_delay_us) * 1000;
    // Said once, before the schedule exists and outside every timed interval: a launcher that injects
    // a fault at a time of the schedule reads the start here (scripts/run_m1.py).
    std::cout << "pilot_ls: schedule start " << schedule_start << " ns on the host monotonic clock, " << chosen.size()
              << " releases" << std::endl;
    pace->begin(schedule_start);
    // ---- The lead-in. The first use of a processor is no release of the schedule: the first
    // releases are decoded once more, each at its own offset a declared lead ahead, with the grid,
    // the PDU and the block of its opportunity. Nothing of it is tapped or published, and the
    // receiver has no observer registered while it runs. A decode of the lead-in is never
    // skipped: it is issued at its time, or as soon as the one before it is done. ----
    if (!led.empty()) {
#ifdef ISAC_O3_ROUTE
      if (route) {
        route->close();
      }
#endif
      std::size_t retained = led.size(); // the decode whose grid is kept while the next one is filled and decoded
      for (std::size_t k = 0; k != led.size(); ++k) {
        lead_slot&        now      = led[k];
        const std::size_t sequence = now.sequence;
        const allocation& bound    = replay.scheduled(sequence);
        now.issue_ns               = isac_pacing::lead_time(offered, warm, schedule_start, k);
        const auto filling         = std::chrono::steady_clock::now();
        ticket     key;
        now.delivered = pool.try_acquire(key);
        if (!now.delivered) {
          break; // a decode of the lead-in that never called back holds its grid: nothing more is issued
        }
        try {
          isac_replay::grid_sink& sink = pool.writer(key);
          sink.clear();
          replay.fill(sequence, sink);
          pool.commit(key, replay.describe(sequence, bound, sink), isac_replay::evidence::timed);
        } catch (...) {
          pool.abort(key);
          throw;
        }
        now.key      = key;
        now.fill_us  = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - filling).count();
        now.ready_ns = isac_pacing::monotonic_ns();
        now.start_ns = isac_pacing::spin_until(now.issue_ns);
        for (std::size_t b = 0; b != backends.size(); ++b) {
          processor_backend& backend = *backends[b];
          if (halted[b] != 0) {
            lead_jobs->skip(now.tokens[b], isac_pacing::skip_reason::backend_stopped);
            continue;
          }
          std::string                     refusal;
          const isac_pacing::paced_result result =
              isac_pacing::paced_dispatch(*lead_jobs, backend, now.tokens[b], key, limit_ns, &refusal);
          if (result.admitted != admission::admitted) {
            throw std::runtime_error("the grid of the lead-in decode " + std::to_string(k) + " was not admitted for " +
                                     backend.name() + ": " + refusal);
          }
          // The first use of this processor is this decode, and no release of the schedule.
          started[b] = 1;
          if (result.result == outcome::timeout) {
            halted[b]               = 1;
            stopped[backend.name()] = "the " + backend.name() + " processor did not complete the lead-in decode " +
                                      std::to_string(k) + " within " + std::to_string(plan.receiver.timeout_ms) + " ms";
          }
        }
        now.done_ns = isac_pacing::monotonic_ns();
        // As in the schedule: the grid of the decode before is given back once this one is done.
        if (retained != led.size()) {
          led[retained].released = lead_jobs->release_grid(led[retained].key);
        }
        retained = k;
      }
      if (retained != led.size()) {
        led[retained].released = lead_jobs->release_grid(led[retained].key);
      }
#ifdef ISAC_O3_ROUTE
      if (route) {
        route->open();
      }
#endif
    }
    for (std::size_t n = 0; n != chosen.size(); ++n) {
      const std::size_t  sequence = chosen[n];
      const allocation&  bound    = replay.scheduled(sequence);
      timed_slot&        now      = kept[n];
      const std::int64_t free_ns  = isac_pacing::monotonic_ns();
      if (!pace->take(n, free_ns)) {
        // A skipped release: the job keeps its key and its place in every denominator.
        for (std::size_t b = 0; b != backends.size(); ++b) {
          paced_jobs->skip(now.decodes[b].token, isac_pacing::skip_reason::missed_release, free_ns - pace->release_time(n));
        }
        continue;
      }
      // The fill, ahead of the slot. It is stimulus: a receiver is handed a grid, it does not make one.
      const auto filling = std::chrono::steady_clock::now();
      ticket     key;
      now.delivered = pool.try_acquire(key);
      if (!now.delivered) {
        for (std::size_t b = 0; b != backends.size(); ++b) {
          paced_jobs->skip(now.decodes[b].token, isac_pacing::skip_reason::no_free_grid);
        }
        pace->give_up(n, isac_pacing::monotonic_ns());
        continue;
      }
      try {
        isac_replay::grid_sink& sink = pool.writer(key);
        sink.clear();
        replay.fill(sequence, sink);
        pool.commit(key, replay.describe(sequence, bound, sink), isac_replay::evidence::timed);
      } catch (...) {
        pool.abort(key);
        throw;
      }
      now.key     = key;
      now.fill_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - filling).count();
      const o1_view ready = pool.view(key, bound.record_sha256);
      pace->ready(n, isac_pacing::monotonic_ns());

      // The release. The grid arrives at the receiver now, and never before the offered time.
      pace->start(n, isac_pacing::spin_until(pace->release_time(n)));
      if (tapped) {
        now.line = out->account().size();
        out->publish(handle_of[sequence], ready);
        now.observed = true;
      }
      for (std::size_t b = 0; b != backends.size(); ++b) {
        processor_backend& backend = *backends[b];
        if (halted[b] != 0) {
          paced_jobs->skip(now.decodes[b].token, isac_pacing::skip_reason::backend_stopped);
          continue;
        }
        now.decodes[b].first_use = started[b] == 0;
        // The receiver's estimate: the admission before the decode, the completion after it.
        const bool captured = taker != nullptr && b == estimate_backend;
        if (captured) {
          now.capture = taker->account().size();
          taker->expect(estimate_handle[sequence], key);
        }
        std::string                     refusal;
        const isac_pacing::paced_result result =
            isac_pacing::paced_dispatch(*paced_jobs, backend, now.decodes[b].token, key, limit_ns, &refusal);
        if (captured) {
          taker->resolve();
          now.captured = true;
        }
        if (result.admitted != admission::admitted) {
          throw std::runtime_error("the grid of opportunity " + spec.opportunities[sequence].opportunity_id +
                                   " was not admitted for " + backend.name() + ": " + refusal);
        }
        started[b] = 1;
        if (result.result == outcome::timeout) {
          halted[b]               = 1;
          stopped[backend.name()] = "the " + backend.name() + " processor did not complete " +
                                    spec.opportunities[sequence].opportunity_id + " within " +
                                    std::to_string(plan.receiver.timeout_ms) + " ms";
        }
      }
      // Every decoder of this grid has called back: the job of this release is done.
      pace->done(n, isac_pacing::monotonic_ns());
      // Only now is the grid read in full, into memory that exists, for the evaluator. A paced
      // run that keeps its grids pays for this copy out of the time ahead of the next slot.
      if (keep_grids) {
        for (unsigned rx = 0; rx != shape.rx_ports; ++rx) {
          for (unsigned symbol = 0; symbol != shape.symbols; ++symbol) {
            ready.get(rx, symbol,
                      arena.data() + n * grid_values + (static_cast<std::size_t>(rx) * shape.symbols + symbol) * shape.subcarriers,
                      shape.subcarriers);
          }
        }
      }
      if (any) {
        retire_paced(kept[last]);
      }
      last = n;
      any  = true;
      if (ring) {
        ring->service(); // the control channel, after the decodes of this opportunity
      }
      if (estimate_ring) {
        estimate_ring->service();
      }
    }
    if (any) {
      retire_paced(kept[last]);
    }
  }

  // ---- The drain: the consumer gets the declared time to release what it holds, then it is told
  // that no record follows. Every observation is published and released, dropped, reclaimed with
  // a closed epoch, or pending at this cutoff. ----
  std::uint64_t        pending_at_cutoff = 0;
  obs::producer_counts ring_counts;
  std::string          consumer_wait;
  bool                 attached_at_drain = false;
  if (ring) {
    pending_at_cutoff = ring->drain(std::chrono::milliseconds(tap.ring.drain_ms));
    attached_at_drain = ring->attached();
    ring->finish();
    ring_counts   = ring->counts();
    consumer_wait = ring_counts.epochs == 0 ? "" : obs::to_string(ring->consumer_mode());
  }
  // The receiver calls no observer from here on. A copy that ran out of time is waited for now,
  // after the last decode, and the O3 stream is drained as the O2 stream is.
  std::uint64_t        estimate_pending = 0;
  obs::producer_counts estimate_counts;
  std::string          estimate_consumer_wait;
  bool                 estimate_attached_at_drain = false;
#ifdef ISAC_O3_ROUTE
  if (route) {
    route->close();
  }
#endif
  if (taker != nullptr) {
    taker->settle();
  }
  if (estimate_ring) {
    estimate_pending           = estimate_ring->drain(std::chrono::milliseconds(tap.estimate.ring.drain_ms));
    estimate_attached_at_drain = estimate_ring->attached();
    estimate_ring->finish();
    estimate_counts        = estimate_ring->counts();
    estimate_consumer_wait = estimate_counts.epochs == 0 ? "" : obs::to_string(estimate_ring->consumer_mode());
  }
  const isac_replay::hash_count hash_drained = isac_replay::sha256_count();

  // ---- A timed run writes its evidence here, after the drain. The stored grids and the decoded
  // bytes are digested now, on the evaluator side, so that the same readers verify both modes. ----
  for (std::size_t n = 0; n != kept.size(); ++n) {
    const timed_slot&       now      = kept[n];
    const std::size_t       sequence = now.sequence;
    const opportunity_spec& item     = spec.opportunities[sequence];
    const allocation&       bound    = replay.scheduled(sequence);
    const grant_entry&      grant    = grants.at(bound.grant_id);
    json                    entry    = {{"sequence", sequence},
                                        {"opportunity_id", item.opportunity_id},
                                        {"realization_id", item.realization_id},
                                        {"index", item.index},
                                        {"grant_id", bound.grant_id},
                                        {"sfn", item.sfn},
                                        {"slot", item.slot},
                                        {"decodes", json::array()}};
    json ingress = nullptr;
    entry["observer"] = nullptr;
    if (tap.version >= 3) {
      entry["estimate"] = now.captured ? estimate_entry(sequence, taker->account()[now.capture], false, std::string())
                                       : json(nullptr);
    }
    if (!now.delivered) {
      entry["grid_index"] = nullptr;
    } else {
      const std::string grid_file = numbered(sequence, ".c64");
      json              digest    = nullptr;
      if (keep_grids) {
        const cf32* stored = arena.data() + n * grid_values;
        digest             = sha256_hex(stored, grid_values * sizeof(cf32));
        write_bytes(ingress_path + "/" + grid_file, stored, grid_values * sizeof(cf32));
      }
      const json line = {{"schema", INGRESS_SCHEMA},
                         {"run_id", spec.run_id},
                         {"sequence", sequence},
                         {"opportunity_id", item.opportunity_id},
                         {"sfn", item.sfn},
                         {"slot", item.slot},
                         {"grant_id", bound.grant_id},
                         {"grant_record_sha256", bound.record_sha256},
                         {"active_re", bound.active.size()},
                         {"active_sha256", bound.active_sha256},
                         {"grid_index", now.key.grid},
                         {"generation", now.key.generation},
                         {"state", to_string(isac_replay::grid_state::ready)},
                         {"grid",
                          {{"file", keep_grids ? json(grid_file) : json(nullptr)},
                           {"axes", {"rx_port", "symbol", "subcarrier"}},
                           {"shape", {shape.rx_ports, shape.symbols, shape.subcarriers}},
                           {"dtype", "<c8"},
                           {"order", "C"},
                           {"storage", first.storage()},
                           {"sha256", digest}}},
                         {"amplitude", {{"grid_gain", spec.gain.value}, {"convention", RULE}}}};
      ingress_lines += line.dump() + "\n";
      entry["grid_index"]      = now.key.grid;
      entry["generation"]      = now.key.generation;
      entry["readback_sha256"] = digest;
      entry["fill_us"]         = now.fill_us;
      entry["retained"]        = now.retained;
      entry["released"]        = now.released;
      ingress = {{"grid_index", now.key.grid}, {"generation", now.key.generation}, {"readback_sha256", digest}};
      if (now.observed) {
        const isac_transport::published_record& sent = out->account()[now.line];
        const bool published = sent.outcome == isac_transport::publication::published;
        entry["observer"]    = {{"sequence", published ? json(sent.sequence) : json(nullptr)},
                                {"ring", published ? "published" : "dropped"},
                                {"payload_sha256", nullptr},
                                {"mask_sha256", nullptr},
                                {"ingress_sha256", nullptr},
                                {"valid", sent.valid},
                                {"holes", sent.holes},
                                {"service_us", sent.service_us}};
      }
    }
    for (std::size_t b = 0; b != backends.size(); ++b) {
      job record = job_of(now.decodes[b].token);
      if (record.completed && record.crc_ok && record.result != outcome::timeout && record.decoded_sha256.empty()) {
        const std::vector<std::uint8_t> bytes = bytes_of(record.token);
        record.decoded_sha256                 = sha256_hex(bytes.data(), bytes.size());
      }
      json ingress_now = ingress;
      if (!ingress_now.is_null() && record.completed) {
        ingress_now["intact"] = record.input_intact;
      }
      json decode;
      radio_record(sequence, item, bound, grant, backends[b]->name(), record, now.decodes[b].first_use, ingress_now, decode);
      entry["decodes"].push_back(decode);
    }
    opportunities.push_back(entry);
  }

  // ---- The account of the ring: one line per eligible observation. ----
  if (ring) {
    std::size_t counted = 0;
    for (const isac_transport::published_record& sent : out->account()) {
      const isac_observer::bound_slot& bound     = watcher.prepared(sent.handle);
      const obs::record_header&        header    = out->header_of(sent.handle);
      const bool                       published = sent.outcome == isac_transport::publication::published;
      const staged_sequence            where     = {chosen, handle_of, sent.handle};
      publication_lines.push_back(
          {{"schema", PUBLICATION_SCHEMA},
           {"run_id", tap.run_id},
           {"mode", tap.evidence},
           {"opportunity_id", bound.opportunity_id},
           {"opportunity_sequence", where.sequence()},
           {"outcome", published ? "published" : "dropped"},
           {"reason", published ? json(nullptr) : json(sent.reason)},
           {"epoch", published ? json(sent.epoch) : json(nullptr)},
           {"sequence", published ? json(sent.sequence) : json(nullptr)},
           {"depth", published ? json(sent.depth) : json(nullptr)},
           {"bytes", published ? json(sent.bytes) : json(nullptr)},
           {"keys",
            {{"episode", hex64(bound.episode)}, {"opportunity", hex64(bound.opportunity)}, {"grant", hex64(bound.grant)}}},
           {"absolute_slot", bound.absolute_slot},
           {"grid", published ? json({{"grid_index", sent.grid.grid}, {"generation", sent.grid.generation}}) : json(nullptr)},
           {"payload_bytes", header.payload_bytes},
           {"valid", published ? json(sent.valid) : json(nullptr)},
           {"holes", published ? json(sent.holes) : json(nullptr)},
           {"t_read_ns", sent.t_read_ns},
           {"t_ready_ns", published ? json(sent.t_ready_ns) : json(nullptr)},
           {"t_publish_ns", published ? json(sent.t_publish_ns) : json(nullptr)},
           {"t_done_ns", sent.t_done_ns},
           {"service_us", sent.service_us},
           {"estimate_us", published ? json(sent.estimate_us) : json(nullptr)},
           {"publish_us", sent.publish_us},
           {"window_calls", sent.window_calls}});
      if (timed && published) {
        if (counted != 0) {
          tap_times.push_back(sent.service_us);
          ring_estimate.push_back(sent.estimate_us);
          ring_publish.push_back(sent.publish_us);
        }
        ++counted;
        tap_valid += sent.valid;
        tap_holes += sent.holes;
        tap_bytes += header.payload_bytes;
      }
    }
    std::string lines;
    for (const json& line : publication_lines) {
      lines += line.dump() + "\n";
    }
    write_bytes(ring_path + "/publications.jsonl", lines.data(), lines.size());
  }

  // ---- The account of the capture: one line per decode of the producing backend, in both modes.
  // It is the trace of the producer and of the completion of every copy. ----
  json                estimate_outcomes = json::object();
  json                estimate_reasons  = json::object();
  std::vector<double> estimate_submit, estimate_resolve;
  std::size_t         estimate_landed = 0, estimate_valid = 0, estimate_holes = 0, estimate_nonzero = 0, estimate_bytes = 0;
  std::uint32_t       estimate_polls = 0;
  if (taker != nullptr) {
    for (const char* name : {"published", "captured", "dropped", "absent", "refused", "failed"}) {
      estimate_outcomes[name] = 0;
    }
    std::string lines;
    std::size_t copied = 0;
    for (const isac_estimate::capture_line& got : taker->account()) {
      const staged_sequence where = {chosen, estimate_handle, got.handle};
      lines += capture_record(tap.run_id, tap.evidence, tap.estimate.backend,
                              estimate_slots.prepared(estimate_slot_of[got.handle]), where.sequence(),
                              taker->header_of(got.handle), got)
                   .dump() +
               "\n";
      const std::string result = isac_estimate::to_string(got.result);
      estimate_outcomes[result] = estimate_outcomes[result].get<std::size_t>() + 1;
      if (*got.reason != '\0') {
        estimate_reasons[got.reason] = estimate_reasons.value(got.reason, std::size_t(0)) + 1;
      }
      if (*got.copy_stream != '\0') {
        // The first copy of a run includes the set-up of the runtime for it.
        if (copied != 0) {
          estimate_submit.push_back(got.submit_us);
          estimate_resolve.push_back(got.resolve_us);
        }
        ++copied;
        estimate_polls = std::max(estimate_polls, got.polls);
      }
      if (got.landed) {
        ++estimate_landed;
        estimate_valid += got.counts.valid;
        estimate_holes += got.counts.holes;
        estimate_nonzero += got.counts.nonzero;
        estimate_bytes += taker->header_of(got.handle).payload_bytes;
      }
    }
    write_bytes(estimate_path + "/captures.jsonl", lines.data(), lines.size());
    if (!timed) {
      write_bytes(estimate_path + "/o3.jsonl", estimate_lines.data(), estimate_lines.size());
    }
  }

  write_bytes(ingress_path + "/ingress.jsonl", ingress_lines.data(), ingress_lines.size());
  write_bytes(radio_path + "/radio.jsonl", radio_lines.data(), radio_lines.size());
  if (tapped && !timed) {
    write_bytes(observer_path + "/o2.jsonl", observer_lines.data(), observer_lines.size());
  }

  // ---- The report of the evaluator side. ----
  const std::vector<job> jobs = paced_jobs ? paced_jobs->jobs() : book.jobs();
  json                   file_list = json::array();
  for (const verified_file& file : files) {
    file_list.push_back({{"role", file.role}, {"path", file.path}, {"sha256", file.sha256}, {"bytes", file.bytes}});
  }
  json grant_list = json::array();
  for (const isac_replay::grant_spec& item : spec.grants) {
    const allocation&  bound = replay.bound(item.grant_id);
    const grant_entry& grant = grants.at(item.grant_id);
    grant_list.push_back({{"grant_id", bound.grant_id},
                          {"record_sha256", bound.record_sha256},
                          {"record_file_sha256", grant.record_file_sha256},
                          {"x_sha256", bound.x_sha256},
                          {"active_re", bound.active.size()},
                          {"active_sha256", bound.active_sha256},
                          {"slot", grant.record.slot},
                          {"tb_sha256", grant.original.sha256},
                          {"tb_bytes", grant.original.bytes.size()},
                          {"pdu", pdu_echo(grant.record)},
                          {"derived",
                           {{"tbs_bits", grant.coding.tbs_bits},
                            {"modulation", to_string(grant.coding.mcs.modulation)},
                            {"ldpc_base_graph", grant.coding.base_graph == ocudu::ldpc_base_graph_type::BG1 ? "BG1" : "BG2"},
                            {"codeblocks", grant.coding.codeblocks}}}});
  }
  const outcome every[]       = {outcome::decoded,       outcome::payload_mismatch, outcome::crc_fail,
                                 outcome::decoder_error, outcome::timeout,          outcome::not_dispatched};
  json          backend_list  = json::array();
  json          total_outcome = json::object();
  for (outcome value : every) {
    total_outcome[to_string(value)] = count_of(value);
  }
  bool incomplete = false;
  for (const std::unique_ptr<processor_backend>& backend : backends) {
    json        outcomes = json::object();
    std::size_t planned  = 0;
    for (outcome value : every) {
      outcomes[to_string(value)] = count_of(value, backend->name());
      planned += count_of(value, backend->name());
    }
    std::size_t resident = 0;
    std::size_t host     = 0;
    for (const job& record : jobs) {
      ocudu::pusch_processor_result_data sch;
      if (record.key.backend == backend->name() && record.token != 0 && backend->result(record.token, sch)) {
        (sch.data.acceleration_pipeline_timing.valid ? resident : host) += 1;
      }
    }
    json        times = service.count(backend->name()) != 0 ? service[backend->name()] : json::array();
    // A lead-in made the first use of the processor one of its own decodes: no release is left out.
    json        stats = {{"count", times.size()},
                         {"excluded", lead_jobs ? "nothing: the first use of the backend is a decode of the lead-in, in pacing/"
                                                  "lead_in.jsonl"
                                                : "the first decode of the backend, which includes its set-up"}};
    if (!times.empty()) {
      double total = 0.0, least = times[0].get<double>(), most = least;
      for (const json& value : times) {
        total += value.get<double>();
        least = std::min(least, value.get<double>());
        most  = std::max(most, value.get<double>());
      }
      stats["mean"] = total / static_cast<double>(times.size());
      stats["min"]  = least;
      stats["max"]  = most;
    }
    incomplete = incomplete || count_of(outcome::timeout, backend->name()) != 0 ||
                 (lead_jobs && lead_jobs->count(outcome::timeout, backend->name()) != 0);
    backend_list.push_back({{"name", backend->name()},
                            {"factory", backend->factory_type()},
                            {"harq_id", backend->harq_id()},
                            {"acceleration_requested", backend->name() == "cuda"},
                            {"acceleration_available", backend->name() == "cuda" ? json(acceleration) : json(nullptr)},
                            {"planned", planned},
                            {"outcomes", outcomes},
                            {"resident", resident},
                            {"host", host},
                            {"stopped", stopped.count(backend->name()) != 0 ? json(stopped[backend->name()]) : json(nullptr)},
                            {"service_us", stats}});
  }
  json anomaly_list = json::array();
  for (const anomaly& item : paced_jobs ? paced_jobs->anomalies() : book.anomalies()) {
    anomaly_list.push_back({{"kind", item.kind}, {"key", item.key}, {"detail", item.detail}});
  }
  json gain = {{"value", spec.gain.value}, {"source", spec.gain.source}, {"convention", RULE}};
  if (spec.gain.has_reference_power) {
    gain["reference_power"] = spec.gain.reference_power;
  }
  if (!spec.gain.reason.empty()) {
    gain["reason"] = spec.gain.reason;
  }
  json payload_record = {{"rule", plan.payload.rule}, {"blocks", plan.payload.blocks}};
  payload_record["generator"] = plan.payload.generator.empty() ? json(nullptr) : json(plan.payload.generator);
  payload_record["seed"]      = plan.payload.seed.empty() ? json(nullptr) : json(plan.payload.seed);
  json environment = json::object();
  for (const char* name : {"OCUDU_PUSCH_ACCELERATION_TIMING", "CUDA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY"}) {
    const char* value = std::getenv(name);
    environment[name] = value != nullptr ? json(value) : json(nullptr);
  }
  // ---- What the observer did. ----
  json source_list = json::array();
  for (const isac_replay::grant_spec& item : spec.grants) {
    if (!tapped) {
      break;
    }
    const isac_observer::pilot_source& source = watcher.source_of(item.record_sha256);
    source_list.push_back({{"grant_id", item.grant_id},
                           {"record_sha256", item.record_sha256},
                           {"source", source.spec().source},
                           {"channel", source.spec().channel},
                           {"link_direction", source.spec().link_direction},
                           {"processor", source.spec().processor},
                           {"port", source.spec().ports.front()},
                           {"pilots", source.spec().index.size()},
                           {"symbols", source.symbols()},
                           {"first_subcarrier", source.first_subcarrier()},
                           {"subcarrier_step", source.subcarrier_step()},
                           {"subcarrier_count", source.subcarriers().size()},
                           {"holes", source.holes()},
                           {"coordinates_sha256", source.coordinates_sha256()},
                           {"amplitude", source.spec().amplitude},
                           {"antenna_map", source.spec().antenna_map}});
  }
  const char* const first_excluded = "the first observation of the run";
  const json        tap_stats      = time_stats(tap_times, first_excluded);
  const std::size_t observed       = timed ? (out ? out->account().size() : 0) : watcher.observations().size();
  // The parts of the tap. Audit: where the time of an audited observation goes. Timed: the
  // estimate and the publication, apart.
  json parts = nullptr;
  if (tapped && !timed) {
    parts = {{"estimate_us", time_stats(part_estimate, first_excluded)},
             {"grid_digest_us", time_stats(part_grid_digest, first_excluded)},
             {"keys_mask_us", time_stats(part_keys_mask, first_excluded)},
             {"payload_digest_us", time_stats(part_payload_digest, first_excluded)},
             {"publish_us", ringed ? time_stats(ring_publish, first_excluded) : json(nullptr)}};
  } else if (tapped) {
    parts = {{"estimate_us", time_stats(ring_estimate, first_excluded)},
             {"publish_us", time_stats(ring_publish, first_excluded)}};
  }
  const json evidence_report = {
      {"mode", tap.evidence},
      {"plan_version", tap.version},
      {"window", "from before the first fill to the drain of the ring"},
      {"hash_in_window", hash_json(hash_bound, hash_drained)},
      {"hash_after_drain", hash_json(hash_drained, isac_replay::sha256_count())},
      {"hash_before_first_fill",
       {{"digests", hash_bound.digests}, {"updates", hash_bound.updates}, {"bytes", hash_bound.bytes}}},
      {"ownership", timed ? "the ticket of the fill: grid index and generation" : "the readback digest of the grid"},
      {"records_written", timed ? "after the last opportunity" : "as each opportunity is done"}};
  json ring_report = nullptr;
  if (ring) {
    ring_report = {
        {"schema", RING_SCHEMA},
        {"layout", "obs-ring/v1"},
        {"definition", hex64(obs::DEFINITION)},
        {"stream", tap.ring.stream},
        {"socket", std::string(RING_DIR) + "/" + tap.ring.stream + ".sock"},
        {"window_records", tap.ring.window_records},
        {"burst_records", tap.ring.burst_records},
        {"slot_count", ring->header().slot_count},
        {"slot_bytes", ring->header().slot_bytes},
        {"segment_bytes", ring->segment_bytes()},
        {"payload_capacity", ring->header().payload_capacity},
        {"mask_capacity", ring->header().mask_capacity},
        {"host_capacity", ring->header().host_capacity},
        {"memory_path", RING_MEMORY},
        {"clock", "CLOCK_MONOTONIC, read without a system call where the host has a vDSO clock"},
        {"wait", {{"planned", tap.ring.wait}, {"consumer", consumer_wait.empty() ? json(nullptr) : json(consumer_wait)}}},
        {"attach", tap.ring.attach},
        {"attached_at_start", attached_at_start},
        {"attached_at_drain", attached_at_drain},
        {"first_epoch", tap.epoch},
        {"last_epoch", ring->epoch()},
        {"drain_ms", tap.ring.drain_ms},
        {"pending_at_cutoff", pending_at_cutoff},
        {"eligible", out->account().size()},
        {"counts",
         {{"admitted", ring_counts.admitted},
          {"published", ring_counts.published},
          {"dropped", ring_counts.dropped},
          {"cancelled", ring_counts.cancelled},
          {"released", ring_counts.released},
          {"reclaimed", ring_counts.reclaimed},
          {"pending", ring_counts.pending},
          {"faults", ring_counts.faults},
          {"epochs", ring_counts.epochs},
          {"refused", ring_counts.refused},
          {"wakes", ring_counts.wakes},
          {"window_calls", ring_counts.window_calls},
          {"max_depth", ring_counts.max_depth}}}};
    const std::string ring_text = ring_report.dump(2) + "\n";
    write_bytes(ring_path + "/ring.json", ring_text.data(), ring_text.size());
  }
  json estimate_ring_report = nullptr;
  if (estimate_ring) {
    estimate_ring_report = {
        {"schema", RING_SCHEMA},
        {"layout", "obs-ring/v1"},
        {"definition", hex64(obs::DEFINITION)},
        {"stream", tap.estimate.ring.stream},
        {"socket", std::string(ESTIMATE_DIR) + "/" + tap.estimate.ring.stream + ".sock"},
        {"window_records", tap.estimate.ring.window_records},
        {"burst_records", tap.estimate.ring.burst_records},
        {"slot_count", estimate_ring->header().slot_count},
        {"slot_bytes", estimate_ring->header().slot_bytes},
        {"segment_bytes", estimate_ring->segment_bytes()},
        {"payload_capacity", estimate_ring->header().payload_capacity},
        {"mask_capacity", estimate_ring->header().mask_capacity},
        {"host_capacity", estimate_ring->header().host_capacity},
        {"memory_path", RING_MEMORY},
        {"clock", "CLOCK_MONOTONIC, read without a system call where the host has a vDSO clock"},
        {"wait",
         {{"planned", tap.estimate.ring.wait},
          {"consumer", estimate_consumer_wait.empty() ? json(nullptr) : json(estimate_consumer_wait)}}},
        {"attach", tap.estimate.ring.attach},
        {"attached_at_start", estimate_attached_at_start},
        {"attached_at_drain", estimate_attached_at_drain},
        {"first_epoch", tap.epoch},
        {"last_epoch", estimate_ring->epoch()},
        {"drain_ms", tap.estimate.ring.drain_ms},
        {"pending_at_cutoff", estimate_pending},
        {"eligible", taker != nullptr ? taker->account().size() : 0},
        {"counts",
         {{"admitted", estimate_counts.admitted},
          {"published", estimate_counts.published},
          {"dropped", estimate_counts.dropped},
          {"cancelled", estimate_counts.cancelled},
          {"released", estimate_counts.released},
          {"reclaimed", estimate_counts.reclaimed},
          {"pending", estimate_counts.pending},
          {"faults", estimate_counts.faults},
          {"epochs", estimate_counts.epochs},
          {"refused", estimate_counts.refused},
          {"wakes", estimate_counts.wakes},
          {"window_calls", estimate_counts.window_calls},
          {"max_depth", estimate_counts.max_depth}}}};
    const std::string ring_text = estimate_ring_report.dump(2) + "\n";
    write_bytes(estimate_path + "/ring.json", ring_text.data(), ring_text.size());
  }
  // ---- What the capture of the receiver's estimate did. ----
  json estimate_report = nullptr;
  if (estimating) {
    const char* const first_copy = "the first copy of the run";
    json              landing    = {{"planned", tap.estimate.landing}, {"effective", nullptr}, {"refusal", nullptr}};
    json              device     = nullptr;
    json              interface  = nullptr;
#ifdef ISAC_O3_ROUTE
    // In the mode audit every copy lands in the staging, because its payload is archived.
    landing["effective"] = timed && route->slot_landing() ? isac_estimate::LANDING_SLOT : isac_estimate::LANDING_STAGING;
    if (!route->landing_refusal().empty()) {
      landing["refusal"] = route->landing_refusal();
    }
    device    = {{"id", route->copier_of().device_id()},
                 {"copies_submitted", route->copier_of().submitted()},
                 {"last_error", route->copier_of().last_error().empty() ? json(nullptr) : json(route->copier_of().last_error())}};
    interface = {{"header", "ocudu/phy/upper/channel_processors/pusch/pusch_resident_estimate_tap.h"},
                 {"version", ocudu::PUSCH_RESIDENT_ESTIMATE_TAP_VERSION}};
#endif
    estimate_report = {
        {"tap", tap.estimate.tap},
        {"stage", tap.estimate.stage},
        {"source", tap.estimate.source},
        {"layout", tap.estimate.layout},
        {"dtype", isac_estimate::DTYPE_CFP16},
        {"backend", tap.estimate.backend},
        {"producer", isac_estimate::PRODUCER},
        {"rule", isac_estimate::RULE},
        {"precision", isac_estimate::PRECISION},
        {"interface", interface},
        {"copy_stream", tap.estimate.copy_stream},
        {"landing", landing},
        {"timeout_us", tap.estimate.timeout_us},
        {"completion", "one CUDA event recorded behind the copy and queried after the decode, never waited for"},
        {"device", device},
        {"mode", tap.evidence},
        {"eligible", taker != nullptr ? taker->account().size() : 0},
        {"outcomes", estimate_outcomes},
        {"reasons", estimate_reasons},
        {"landed", estimate_landed},
        {"archived", estimate_archived},
        {"valid", estimate_valid},
        {"holes", estimate_holes},
        {"nonzero", estimate_nonzero},
        {"payload_bytes", estimate_bytes},
        {"unexpected", taker != nullptr ? taker->unexpected() : 0},
        {"overruns", taker != nullptr ? taker->overruns() : 0},
        {"max_polls", estimate_polls},
        {"submit_us", time_stats(estimate_submit, first_copy)},
        {"resolve_us", time_stats(estimate_resolve, first_copy)},
        {"ring", estimate_ring_report}};
  }
  const json observer_report = {
      {"tap", tap.tap},
      {"stage", tapped ? json(tap.stage) : json(nullptr)},
      {"source", tapped ? json(tap.source) : json(nullptr)},
      {"layout", tapped ? json(tap.layout) : json(nullptr)},
      {"epoch", tap.epoch},
      {"producer", tapped ? json(isac_observer::PRODUCER) : json(nullptr)},
      {"rule", tapped ? json(isac_observer::RULE) : json(nullptr)},
      {"precision", tapped ? json(isac_observer::PRECISION) : json(nullptr)},
      {"memory", tapped ? json(timed ? "a slot of the observation ring, filled in place before the first decode"
                                     : "host buffer owned by the observer, filled synchronously before the first decode")
                        : json(nullptr)},
      {"records", observed},
      {"mode", tap.evidence},
      {"parts", parts},
      {"valid", tap_valid},
      {"holes", tap_holes},
      {"payload_bytes", tap_bytes},
      {"service_us", tap_stats},
      {"sources", source_list}};

  // ---- The offered schedule of a paced run and what became of every release. Written here,
  // after the drain: nothing of it was text while the run was paced. ----
  json pacing_report = nullptr;
  if (paced) {
    std::string                     lines;
    const isac_pacing::pace_counts  counts = pace->counts();
    std::vector<double>             lateness, busy;
    for (std::size_t n = 0; n != kept.size(); ++n) {
      const isac_pacing::release_mark& mark = pace->marks()[n];
      const opportunity_spec&          item = spec.opportunities[kept[n].sequence];
      const bool ran = mark.outcome == isac_pacing::release_outcome::on_time ||
                       mark.outcome == isac_pacing::release_outcome::late;
      json decodes = json::array();
      for (std::size_t b = 0; b != backends.size(); ++b) {
        const std::uint64_t token  = kept[n].decodes[b].token;
        const job           record = paced_jobs->snapshot(token);
        const bool          called = record.completed && record.result != outcome::timeout;
        decodes.push_back({{"backend", backends[b]->name()},
                           {"key", record.key.text()},
                           {"outcome", to_string(record.result)},
                           {"skip", record.result == outcome::not_dispatched
                                        ? json(isac_pacing::to_string(paced_jobs->skipped(token)))
                                        : json(nullptr)},
                           {"t_start_ns", record.started ? json(paced_jobs->started_ns(token)) : json(nullptr)},
                           {"t_callback_ns", called ? json(paced_jobs->completed_ns(token)) : json(nullptr)},
                           {"service_us", called ? json(record.service_us) : json(nullptr)}});
      }
      const json line = {{"schema", RELEASE_SCHEMA},
                         {"run_id", tap.run_id},
                         {"sequence", kept[n].sequence},
                         {"opportunity_id", item.opportunity_id},
                         {"sfn", item.sfn},
                         {"slot", item.slot},
                         {"offset_us", tap.pacing.release_us[kept[n].sequence]},
                         {"outcome", isac_pacing::to_string(mark.outcome)},
                         {"overrun", mark.overrun},
                         {"t_release_ns", mark.release_ns},
                         {"t_free_ns", mark.free_ns},
                         {"t_ready_ns", ran ? json(mark.ready_ns) : json(nullptr)},
                         {"t_start_ns", ran ? json(mark.start_ns) : json(nullptr)},
                         {"t_done_ns", ran ? json(mark.done_ns) : json(nullptr)},
                         {"fill_us", ran ? json(kept[n].fill_us) : json(nullptr)},
                         {"decodes", decodes}};
      lines += line.dump() + "\n";
      if (ran) {
        lateness.push_back(static_cast<double>(mark.start_ns - mark.release_ns) / 1000.0);
        busy.push_back(static_cast<double>(mark.done_ns - mark.start_ns) / 1000.0);
      }
    }
    write_bytes(pacing_path + "/" + RELEASES_FILE, lines.data(), lines.size());
    // ---- The lead-in, written apart: one line per decode that was issued ahead of the schedule.
    // No line of it is a release, a radio record or a sample of a service-time statistic. ----
    json lead_report = nullptr;
    if (lead_jobs) {
      std::string  lead_lines;
      std::size_t  issued = 0, decoded_ahead = 0;
      std::int64_t first_issue = 0, last_done = 0;
      json         service_ahead = json::object();
      for (const std::unique_ptr<processor_backend>& backend : backends) {
        service_ahead[backend->name()] = json::array();
      }
      for (std::size_t k = 0; k != led.size(); ++k) {
        const lead_slot&        now   = led[k];
        const opportunity_spec& item  = spec.opportunities[now.sequence];
        const allocation&       bound = replay.scheduled(now.sequence);
        const grant_entry&      grant = grants.at(bound.grant_id);
        json                    decodes = json::array();
        for (std::size_t b = 0; b != backends.size(); ++b) {
          const std::uint64_t                token  = now.tokens[b];
          const job                          record = lead_jobs->snapshot(token);
          const bool                         called = record.completed && record.result != outcome::timeout;
          const bool                         passed = called && record.crc_ok;
          ocudu::pusch_processor_result_data sch;
          const bool                         reported = backends[b]->result(token, sch);
          decodes.push_back({{"backend", backends[b]->name()},
                             {"key", record.key.text()},
                             {"outcome", to_string(record.result)},
                             {"crc_ok", passed},
                             {"payload_match", passed && record.compared.equal},
                             {"t_start_ns", record.started ? json(lead_jobs->started_ns(token)) : json(nullptr)},
                             {"t_callback_ns", called ? json(lead_jobs->completed_ns(token)) : json(nullptr)},
                             {"service_us", called ? json(record.service_us) : json(nullptr)},
                             {"path", path_report(backends[b]->name(), reported, sch, built_with_cuda, acceleration,
                                                  grant.record.rx_ports.size())}});
          if (called) {
            service_ahead[backends[b]->name()].push_back(record.service_us);
          }
          decoded_ahead += record.result == outcome::decoded ? 1 : 0;
        }
        const json line = {{"schema", LEAD_IN_SCHEMA},
                           {"run_id", tap.run_id},
                           {"sequence", k},
                           {"opportunity_sequence", now.sequence},
                           {"opportunity_id", item.opportunity_id},
                           {"sfn", item.sfn},
                           {"slot", item.slot},
                           {"grant_id", bound.grant_id},
                           {"offset_us", tap.pacing.release_us[now.sequence]},
                           {"lead_us", tap.pacing.lead_us},
                           {"issued", now.delivered},
                           {"t_issue_ns", now.issue_ns},
                           {"t_ready_ns", now.delivered ? json(now.ready_ns) : json(nullptr)},
                           {"t_start_ns", now.delivered ? json(now.start_ns) : json(nullptr)},
                           {"t_done_ns", now.delivered ? json(now.done_ns) : json(nullptr)},
                           {"fill_us", now.delivered ? json(now.fill_us) : json(nullptr)},
                           {"released", now.released},
                           {"decodes", decodes}};
        lead_lines += line.dump() + "\n";
        if (now.delivered) {
          first_issue = issued == 0 ? now.start_ns : first_issue;
          last_done   = now.done_ns;
          ++issued;
        }
      }
      write_bytes(pacing_path + "/" + LEAD_IN_FILE, lead_lines.data(), lead_lines.size());
      lead_report = {
          {"schema", LEAD_IN_SCHEMA},
          {"file", std::string(PACING_DIR) + "/" + LEAD_IN_FILE},
          {"releases", led.size()},
          {"lead_us", tap.pacing.lead_us},
          {"rule", "lead(k) = start + offset(k) - lead: the first releases of the schedule, decoded once more ahead of "
                   "it with their grids, PDUs and blocks. Never skipped: issued at its time, or as soon as the decode "
                   "before it is done. No release, no radio record, no observation, no sample of a statistic"},
          {"observer", "none registered with the receiver while the lead-in ran, and nothing tapped or published"},
          {"issued", issued},
          {"decodes", issued * backends.size()},
          {"decoded", decoded_ahead},
          {"t_first_start_ns", issued != 0 ? json(first_issue) : json(nullptr)},
          {"t_last_done_ns", issued != 0 ? json(last_done) : json(nullptr)},
          {"before_first_release_us",
           issued != 0 ? json(static_cast<double>(pace->release_time(0) - last_done) / 1000.0) : json(nullptr)},
          {"service_us", service_ahead}};
    }
    pacing_report = {
        {"schema", PACING_SCHEMA},
        {"file", std::string(PACING_DIR) + "/" + RELEASES_FILE},
        {"clock", tap.pacing.clock},
        {"clock_source", "CLOCK_MONOTONIC, the clock of the observation ring, read without a system call where the host "
                         "has a vDSO clock"},
        {"rule", "release(n) = start + offset(n), never shifted; a job runs when the receiver is free within the latest "
                 "start of its release, and is a skipped release otherwise; done later than release + allowance is an "
                 "overrun"},
        {"wait", "spin"},
        {"t_start_ns", pace->release_time(0) - offered.release_ns.front()},
        {"start_delay_us", tap.pacing.start_delay_us},
        {"allowance_us", tap.pacing.allowance_us},
        {"latest_start_us", tap.pacing.latest_start_us},
        {"first_offset_us", tap.pacing.release_us.front()},
        {"last_offset_us", tap.pacing.release_us.back()},
        {"counts",
         {{"offered", counts.offered},
          {"taken", counts.taken},
          {"on_time", counts.on_time},
          {"late", counts.late},
          {"skipped", counts.skipped},
          {"overruns", counts.overruns},
          {"max_lateness_ns", counts.max_lateness_ns},
          {"max_excess_ns", counts.max_excess_ns}}},
        {"lateness_us", time_stats(lateness, "nothing")},
        {"busy_us", time_stats(busy, "nothing")},
        {"book", "paced: every decode named before the first fill, no allocation and no lock of the harness from the "
                 "admission of a ready grid to its release, the completion asked for and never slept on"},
        {"lead_in", lead_report}};
  }

  const std::uint16_t endian = 1;
  json                report = {
      {"schema", tap.version == 4 ? NATIVE_SCHEMA_4 : tap.version == 3 ? NATIVE_SCHEMA_3 : tap.version == 2 ? NATIVE_SCHEMA_2 : NATIVE_SCHEMA},
      {"run_id", tap.run_id},
      {"plan", {{"file", plan_path}, {"sha256", plan_sha256}}},
      {"evidence", evidence_report},
      {"ring", ring_report},
      {"replay_plan", {{"file", tap.replay_plan.path}, {"sha256", replay_sha256}, {"run_id", plan.run_id}}},
      {"observer", observer_report},
      {"channel_plan", {{"file", plan.channel_plan.path}, {"sha256", channel_sha256}, {"run_id", spec.run_id}}},
      {"selected", only.empty() ? json(nullptr) : json(only)},
      {"payload", payload_record},
      {"receiver",
       {{"backends", plan.receiver.backends},
        {"ldpc_iterations", plan.receiver.ldpc_iterations},
        {"early_stop", plan.receiver.early_stop},
        {"timeout_ms", plan.receiver.timeout_ms},
        {"tap", tap.tap},
        {"time_interpolation", "average"},
        {"equalizer", "mmse"},
        {"worker_threads", pool_of_workers.threads},
        {"factories", "create_sw_pusch_processor_factory of the pinned test helpers"},
        {"dmrs_reference", "CRB 0, as the pinned PUSCH estimator generates it"},
        {"grid_memory", GRID_MEMORY}}},
      {"backends", backend_list},
      {"carrier",
       {{"rx_ports", shape.rx_ports},
        {"tx_ports", shape.tx_ports},
        {"symbols", shape.symbols},
        {"subcarriers", shape.subcarriers}}},
      {"gain", gain},
      {"grid",
       {{"factory", "create_resource_grid_factory of the pinned source"},
        {"grids", pool.size()},
        {"ports", first.planes()},
        {"symbols", first.symbols()},
        {"subcarriers", first.subcarriers()},
        {"storage", first.storage()},
        {"stored_grids_kept", keep_grids},
        {"little_endian_host", *reinterpret_cast<const std::uint8_t*>(&endian) == 1}}},
      {"files", file_list},
      {"grants", grant_list},
      {"opportunities", opportunities},
      {"anomalies", anomaly_list},
      {"totals",
       {{"opportunities", opportunities.size()},
        {"backends", backends.size()},
        {"planned", opportunities.size() * backends.size()},
        {"records", jobs.size()},
        {"observations", observed},
        {"outcomes", total_outcome},
        {"anomalies", anomaly_list.size()},
        {"retained", all_retained},
        {"grids_not_released", unreleased},
        {"generations", {pool.generation(0), pool.generation(1)}}}},
      {"environment", environment},
      {"access",
       {{"ingress", "ingress/ holds the stored grids and one descriptor per grid, and nothing else"},
        {"observer", "observer/ holds one obs-descriptor/v1 record, one payload and one validity mask per opportunity, "
                     "computed from the grid of ingress/ and the sensing source of the grant, and nothing else"},
        {"radio", "radio/ holds one radio/v1 record per planned decode and the decoded bytes of every CRC pass"},
        {"transport_block", "read by the evaluator side of this tool for the comparison, never handed to a processor"},
        {"labels", "not read: a plan names none"}}},
#ifdef ENABLE_CUDA
      {"built_with_cuda", true},
#else
      {"built_with_cuda", false},
#endif
  };
  if (tap.version >= 3) {
    // A report of version 3 names the capture, or null, and whether this build has the route.
    report["estimate"] = estimate_report;
#ifdef ISAC_O3_ROUTE
    report["o3_route"] = true;
#else
    report["o3_route"] = false;
#endif
    report["access"]["estimate"] =
        "estimate/ holds one o3-capture/v1 line per decode of the producing backend and, in the mode audit, one "
        "obs-descriptor/v1 record, one payload and one validity mask per captured estimate of the receiver, and "
        "nothing else";
  }
  if (tap.version >= 4) {
    // A report of version 4 names the offered schedule of the run, or null.
    report["pacing"] = pacing_report;
    report["access"]["pacing"] =
        "pacing/ holds one pace-release/v1 line per opportunity of a paced run: the offered release time and the times "
        "the harness took, and nothing of a grid, a transport block or a label; and, for a plan with a lead-in, one "
        "pace-lead-in/v1 line per decode that was issued ahead of the schedule";
  }
  const std::string text = report.dump(2) + "\n";
  write_bytes(report_path, text.data(), text.size());

  std::cout << "pilot_ls: " << tap.run_id << ", tap " << tap.tap << ", evidence " << tap.evidence << ", " << observed
            << " observations, " << opportunities.size() << " opportunities, payload " << plan.payload.rule;
  if (ring) {
    std::cout << ", ring " << ring_counts.published << " published, " << ring_counts.dropped << " dropped, "
              << ring_counts.released << " released, " << ring_counts.reclaimed << " reclaimed, " << pending_at_cutoff
              << " pending at the cutoff";
  }
  if (taker != nullptr) {
    std::cout << ", estimate " << estimate_outcomes["published"] << " published, " << estimate_outcomes["captured"]
              << " captured, " << estimate_outcomes["dropped"] << " dropped, " << estimate_outcomes["absent"] << " absent, "
              << estimate_outcomes["refused"] << " refused, " << estimate_outcomes["failed"] << " failed";
  }
  for (const json& backend : backend_list) {
    std::cout << ", " << backend["name"].get<std::string>() << " " << backend["outcomes"]["decoded"] << " of "
              << backend["planned"] << " decoded (" << backend["resident"] << " resident)";
  }
  if (paced) {
    const isac_pacing::pace_counts counts = pace->counts();
    std::cout << ", paced " << counts.offered << " releases: " << counts.on_time << " on time, " << counts.late
              << " late, " << counts.skipped << " skipped, " << counts.overruns << " overruns";
    if (lead_jobs) {
      std::cout << ", lead-in " << lead_jobs->count(outcome::decoded) << " of " << lead_jobs->size()
                << " decoded ahead of the schedule";
    }
  }
  std::cout << ", report " << report_path << std::endl;
  if (incomplete) {
    // A processor that never called back may still hold its grid and its buffer. Nothing of it is
    // torn down: the evidence is written, and the process ends here.
    std::cerr << "pilot_ls: a decode did not complete within " << plan.receiver.timeout_ms << " ms" << std::endl;
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(1);
  }
  for (const std::unique_ptr<processor_backend>& backend : backends) {
    backend->stop();
  }
  pool_of_workers.pool->stop();
  return 0;
}

} // namespace

int main(int argc, char** argv)
{
  try {
    return run(argc, argv);
  } catch (const input_error& error) {
    std::cerr << "pilot_ls: input error: " << error.what() << std::endl;
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "pilot_ls: " << error.what() << std::endl;
    return 1;
  }
}
