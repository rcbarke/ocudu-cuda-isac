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
// Built out of the OCUDU tree against the accepted Sprint 1.1 build. ASCII only.

#include "dispatch_core.h"
#include "grant_pdu.h"
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

void usage()
{
  std::cout << "Usage: pilot_ls --plan FILE --output-dir DIR [--only OPPORTUNITY_ID]... [--no-grids]\n"
               "                [--phy-debug]\n"
               "       pilot_ls --self-test\n"
               "Decodes the receiver replay that a pilot-ls-plan/v1 file names, as replay_pusch does, and with\n"
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
               "DIR must exist and must not hold a result. --only repeats named opportunities of the plan exactly.\n"
               "--phy-debug prints the path decision of the pinned processor for every PDU.\n"
               "--self-test decodes a two-block grant through the host processor with the tap off and on.\n"
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
    const auto inserted = notifiers.emplace(job.token, std::make_unique<job_notifier>(job.token, sink));
    if (!inserted.second) {
      throw std::runtime_error("the decode " + job.key.text() + " was started before");
    }
    // The same owned grid for every backend: the reader of the pinned resource grid of this ticket.
    processor->process(ocudu::span<std::uint8_t>(job.output, job.output_bytes),
                       std::move(buffer),
                       *inserted.first->second,
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
  auto ready = [&](bool transmitted, const std::string& opportunity, bool rotate = false) {
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
    grids.commit(key, descriptor);
    return key;
  };

  receiver_settings settings;
  settings.backends = {"cpu"};
  settings.tap      = "none";
  processor_backend host("cpu", "auto", 0, *pool_of_workers.executor, pool_of_workers.threads, settings,
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
  host.stop();
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
  if (exists(report_path) || exists(ingress_path) || exists(radio_path) || exists(observer_path)) {
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
  if (!tapped && (!tap.stage.empty() || !tap.layout.empty() || !tap.source.empty() || !tap.sources.empty())) {
    throw input_error("a run without a tap names no stage, no layout and no sensing source");
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
  isac_observer::observer watcher(tap.run_id, tap.epoch);
  for (const isac_replay::grant_spec& bound : spec.grants) {
    if (!tapped) {
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
    if (source.source != tap.source) {
      throw input_error(where + " is the source " + source.source + ", and the plan taps " + tap.source);
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
    watcher.bind(source);
  }
  if (tapped && tap.sources.size() != spec.grants.size()) {
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
  ledger      book(pool);

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
  if (tapped) {
    make_directory(observer_path);
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
      const std::vector<std::uint8_t> bytes = book.decoded(record.token);
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

  for (std::size_t sequence = 0; sequence != replay.opportunities(); ++sequence) {
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
      }
      tap_valid += made.valid;
      tap_holes += made.holes;
      tap_bytes += made.payload_bytes();
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
        const dispatched done = dispatch(book, *backend, name, key, bound.record_sha256, grant.original, limit);
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
  }
  if (have_previous) {
    retire();
  }

  write_bytes(ingress_path + "/ingress.jsonl", ingress_lines.data(), ingress_lines.size());
  write_bytes(radio_path + "/radio.jsonl", radio_lines.data(), radio_lines.size());
  if (tapped) {
    write_bytes(observer_path + "/o2.jsonl", observer_lines.data(), observer_lines.size());
  }

  // ---- The report of the evaluator side. ----
  const std::vector<job> jobs = book.jobs();
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
    total_outcome[to_string(value)] = book.count(value);
  }
  bool incomplete = false;
  for (const std::unique_ptr<processor_backend>& backend : backends) {
    json        outcomes = json::object();
    std::size_t planned  = 0;
    for (outcome value : every) {
      outcomes[to_string(value)] = book.count(value, backend->name());
      planned += book.count(value, backend->name());
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
    json        stats = {{"count", times.size()}, {"excluded", "the first decode of the backend, which includes its set-up"}};
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
    incomplete = incomplete || book.count(outcome::timeout, backend->name()) != 0;
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
  for (const anomaly& item : book.anomalies()) {
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
  json tap_stats = {{"count", tap_times.size()}, {"excluded", "the first observation of the run"}};
  if (!tap_times.empty()) {
    double total = 0.0, least = tap_times[0], most = tap_times[0];
    for (double value : tap_times) {
      total += value;
      least = std::min(least, value);
      most  = std::max(most, value);
    }
    tap_stats["mean"] = total / static_cast<double>(tap_times.size());
    tap_stats["min"]  = least;
    tap_stats["max"]  = most;
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
      {"memory", tapped ? json("host buffer owned by the observer, filled synchronously before the first decode") : json(nullptr)},
      {"records", watcher.observations().size()},
      {"valid", tap_valid},
      {"holes", tap_holes},
      {"payload_bytes", tap_bytes},
      {"service_us", tap_stats},
      {"sources", source_list}};

  const std::uint16_t endian = 1;
  const json          report = {
      {"schema", NATIVE_SCHEMA},
      {"run_id", tap.run_id},
      {"plan", {{"file", plan_path}, {"sha256", plan_sha256}}},
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
        {"observations", watcher.observations().size()},
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
  const std::string text = report.dump(2) + "\n";
  write_bytes(report_path, text.data(), text.size());

  std::cout << "pilot_ls: " << tap.run_id << ", tap " << tap.tap << ", " << watcher.observations().size()
            << " observations, " << opportunities.size() << " opportunities, payload " << plan.payload.rule;
  for (const json& backend : backend_list) {
    std::cout << ", " << backend["name"].get<std::string>() << " " << backend["outcomes"]["decoded"] << " of "
              << backend["planned"] << " decoded (" << backend["resident"] << " resident)";
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
