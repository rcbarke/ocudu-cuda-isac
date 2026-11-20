// Native fixture adapter for Sprint 2.1 (grant_fixture).
//
// Reads one pusch-grant/v1 record and its transport block, builds the transmit resource grid with
// the same host PDSCH chain that run_e2e_pipeline_test of the pinned OCUDU source uses for its
// ordinary (not transform-precoded) path, and writes that grid in carrier coordinates. The grid is
// what the pinned PUSCH receiver expects for the same grant.
//
// Two things differ from the pinned test on purpose:
//   1. The DM-RS reference point is CRB 0. The pinned test sets PRB0, which equals CRB 0 only for
//      a bandwidth part that starts at CRB 0. PUSCH DM-RS is always referenced to CRB 0
//      (TS 38.211 6.4.1.1.3), and the pinned PUSCH estimator generates it that way. The option
//      --dmrs-reference prb0 reproduces the pinned test's setting so the difference can be archived.
//   2. The transport block comes from the record, not from a random generator.
//
// The optional loopback feeds the transmit grid, unchanged and without noise, to the host PUSCH
// processor on every receive port of the grant and compares the decoded bytes with the record's
// transport block. It checks the grant against the receiver. It is not a replay result (S2.4).
//
// This file is built out of the OCUDU tree against the accepted Sprint 1.1 build. It uses only
// ASCII characters.

#include "pxsch_bler_test_factories.h"
#include "nlohmann/json.hpp"
#include "ocudu/ocudulog/ocudulog.h"
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
#include "ocudu/ran/precoding/precoding_codebooks.h"
#include "ocudu/ran/pusch/pusch_constants.h"
#include "ocudu/ran/pusch/pusch_mcs.h"
#include "ocudu/ran/sch/ldpc_base_graph.h"
#include "ocudu/ran/sch/sch_dmrs_power.h"
#include "ocudu/ran/sch/sch_segmentation.h"
#include "ocudu/ran/sch/tbs_calculator.h"
#include "ocudu/support/executors/task_worker_pool.h"
#include "ocudu/support/math/math_utils.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

using namespace ocudu;
using json = nlohmann::json;

namespace {

// Values the pinned test fixes at compile time. The grant must name the same ones.
constexpr unsigned     NOF_LDPC_ITERATIONS = 6;
constexpr bool         USE_EARLY_STOP      = true;
constexpr const char*  NATIVE_SCHEMA       = "grant-fixture-native/v1";
constexpr const char*  GRANT_SCHEMA        = "pusch-grant/v1";
constexpr const char*  GRID_FILE           = "x_native.c64";
constexpr const char*  REPORT_FILE         = "native.json";

// A record or command-line error. Reported before any grid is allocated.
struct input_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct grant_t {
  std::string           grant_id;
  unsigned              scs_khz          = 0;
  unsigned              grid_size_rb     = 0;
  unsigned              bwp_start_rb     = 0;
  unsigned              bwp_size_rb      = 0;
  unsigned              sfn              = 0;
  unsigned              slot             = 0;
  unsigned              rnti             = 0;
  unsigned              n_id             = 0;
  unsigned              prb_start        = 0;
  unsigned              prb_count        = 0;
  unsigned              symbol_start     = 0;
  unsigned              symbol_count     = 0;
  std::string           mcs_table;
  unsigned              mcs_index        = 0;
  unsigned              rv               = 0;
  bool                  new_data         = true;
  unsigned              layers           = 0;
  unsigned              dmrs_config_type = 0;
  std::vector<unsigned> dmrs_symbols;
  std::vector<unsigned> dmrs_ports;
  unsigned              scrambling_id    = 0;
  unsigned              n_scid           = 0;
  unsigned              cdm_groups       = 0;
  unsigned              antenna_ports    = 0;
  std::vector<unsigned> rx_ports;
};

class pdsch_done_notifier : public pdsch_processor_notifier
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

class pusch_done_notifier : public pusch_processor_result_notifier
{
public:
  void on_uci(const pusch_processor_result_control& /*uci*/) override {}

  void on_sch(const pusch_processor_result_data& sch_) override
  {
    std::unique_lock<std::mutex> lock(mutex);
    sch       = sch_;
    completed = true;
    cvar.notify_all();
  }

  const pusch_processor_result_data& wait()
  {
    std::unique_lock<std::mutex> lock(mutex);
    while (!completed) {
      cvar.wait(lock);
    }
    return sch;
  }

private:
  bool                        completed = false;
  pusch_processor_result_data sch;
  std::mutex                  mutex;
  std::condition_variable     cvar;
};

const json& member(const json& object, const char* key)
{
  if (!object.is_object() || !object.contains(key)) {
    throw input_error(std::string("the grant has no field '") + key + "'");
  }
  return object.at(key);
}

unsigned number(const json& object, const char* key, unsigned low, unsigned high)
{
  const json& value = member(object, key);
  if (!value.is_number_unsigned() && !(value.is_number_integer() && value.get<long long>() >= 0)) {
    throw input_error(std::string("field '") + key + "' must be a non-negative integer");
  }
  unsigned long long parsed = value.get<unsigned long long>();
  if (parsed < low || parsed > high) {
    throw input_error(std::string("field '") + key + "' is " + std::to_string(parsed) + ", allowed " +
                      std::to_string(low) + " to " + std::to_string(high));
  }
  return static_cast<unsigned>(parsed);
}

std::string text(const json& object, const char* key)
{
  const json& value = member(object, key);
  if (!value.is_string()) {
    throw input_error(std::string("field '") + key + "' must be a string");
  }
  return value.get<std::string>();
}

std::vector<unsigned> number_list(const json& object, const char* key, unsigned high)
{
  const json& value = member(object, key);
  if (!value.is_array() || value.empty()) {
    throw input_error(std::string("field '") + key + "' must be a nonempty array");
  }
  std::vector<unsigned> result;
  for (const json& item : value) {
    if (!item.is_number_integer() || item.get<long long>() < 0 || item.get<long long>() > static_cast<long long>(high)) {
      throw input_error(std::string("field '") + key + "' holds a value outside 0 to " + std::to_string(high));
    }
    result.push_back(static_cast<unsigned>(item.get<long long>()));
  }
  return result;
}

std::vector<uint8_t> read_bytes(const std::string& path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw input_error("cannot read " + path);
  }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

grant_t load_grant(const std::string& path)
{
  std::vector<uint8_t> raw = read_bytes(path);
  json                 record;
  try {
    record = json::parse(raw.begin(), raw.end());
  } catch (const json::exception& error) {
    throw input_error("the grant is not valid JSON: " + std::string(error.what()));
  }
  if (text(record, "schema") != GRANT_SCHEMA) {
    throw input_error(std::string("the grant schema is not ") + GRANT_SCHEMA);
  }

  grant_t grant;
  grant.grant_id = text(record, "grant_id");

  const json& carrier = member(record, "carrier");
  grant.scs_khz       = number(carrier, "scs_khz", 15, 120);
  grant.grid_size_rb  = number(carrier, "grid_size_rb", 1, MAX_NOF_PRBS);
  if (text(carrier, "cyclic_prefix") != "normal") {
    throw input_error("only the normal cyclic prefix is supported");
  }

  const json& bwp    = member(record, "bwp");
  grant.bwp_start_rb = number(bwp, "start_rb", 0, MAX_NOF_PRBS - 1);
  grant.bwp_size_rb  = number(bwp, "size_rb", 1, MAX_NOF_PRBS);

  const json& slot = member(record, "slot");
  grant.sfn        = number(slot, "sfn", 0, 1023);
  grant.slot       = number(slot, "slot", 0, 159);

  grant.rnti = number(record, "rnti", 1, 65535);
  grant.n_id = number(record, "n_id", 0, 1023);

  const json& allocation = member(record, "allocation");
  grant.prb_start        = number(allocation, "prb_start", 0, MAX_NOF_PRBS - 1);
  grant.prb_count        = number(allocation, "prb_count", 1, MAX_NOF_PRBS);
  grant.symbol_start     = number(allocation, "symbol_start", 0, MAX_NSYMB_PER_SLOT - 1);
  grant.symbol_count     = number(allocation, "symbol_count", 1, MAX_NSYMB_PER_SLOT);

  const json& mcs = member(record, "mcs");
  grant.mcs_table = text(mcs, "table");
  grant.mcs_index = number(mcs, "index", 0, 31);

  const json& codeword = member(record, "codeword");
  grant.rv             = number(codeword, "rv", 0, 3);
  const json& new_data = member(codeword, "new_data");
  if (!new_data.is_boolean()) {
    throw input_error("field 'new_data' must be true or false");
  }
  grant.new_data = new_data.get<bool>();

  grant.layers = number(record, "layers", 1, 4);

  const json& dmrs       = member(record, "dmrs");
  grant.dmrs_config_type = number(dmrs, "config_type", 1, 2);
  grant.dmrs_symbols     = number_list(dmrs, "symbols", MAX_NSYMB_PER_SLOT - 1);
  grant.dmrs_ports       = number_list(dmrs, "ports", 11);
  grant.scrambling_id    = number(dmrs, "scrambling_id", 0, 65535);
  grant.n_scid           = number(dmrs, "n_scid", 0, 1);
  grant.cdm_groups       = number(dmrs, "cdm_groups_without_data", 1, 3);

  const json& tx      = member(record, "tx");
  grant.antenna_ports = number(tx, "antenna_ports", 1, 4);
  if (text(tx, "precoding") != "identity") {
    throw input_error("only identity precoding is supported");
  }

  grant.rx_ports = number_list(record, "rx_ports", MAX_PORTS - 1);

  // Limits of this adapter. The Python module applies the full grant rules first. These repeat the
  // ones that would otherwise reach an assertion inside the pinned library.
  if (grant.bwp_start_rb + grant.bwp_size_rb > grant.grid_size_rb) {
    throw input_error("the bandwidth part leaves the carrier grid");
  }
  if (grant.prb_start + grant.prb_count > grant.bwp_size_rb) {
    throw input_error("the allocation leaves the bandwidth part");
  }
  if (grant.symbol_start + grant.symbol_count > MAX_NSYMB_PER_SLOT) {
    throw input_error("the symbol allocation leaves the slot");
  }
  if (grant.layers != 1 || grant.antenna_ports != 1 || grant.dmrs_ports.size() != 1 || grant.dmrs_ports[0] != 0) {
    throw input_error("this adapter maps one layer on DM-RS port 0 and one antenna port");
  }
  if (grant.slot >= 10U * (grant.scs_khz / 15U)) {
    throw input_error("the slot index exceeds the slots of one frame at this subcarrier spacing");
  }
  std::vector<unsigned> sorted = grant.dmrs_symbols;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() || sorted != grant.dmrs_symbols) {
    throw input_error("DM-RS symbols must be listed once each, in increasing order");
  }
  for (unsigned symbol : grant.dmrs_symbols) {
    if (symbol < grant.symbol_start || symbol >= grant.symbol_start + grant.symbol_count) {
      throw input_error("a DM-RS symbol lies outside the symbol allocation");
    }
  }
  sorted = grant.rx_ports;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    throw input_error("a receive port is listed twice");
  }
  for (size_t i = 0; i != grant.rx_ports.size(); ++i) {
    // The pinned host demodulator reads grid port i for the i-th listed port when the data mask is
    // contiguous (pusch_demodulator_impl::get_ch_data_re), so the list must be 0 to N-1 in order.
    if (grant.rx_ports[i] != i) {
      throw input_error("receive ports must be 0 to N-1 in order");
    }
  }
  if (grant.rx_ports.size() > pusch_constants::MAX_NOF_RX_PORTS) {
    throw input_error("the pinned PUSCH processor takes at most " + std::to_string(pusch_constants::MAX_NOF_RX_PORTS) +
                      " receive ports");
  }
  return grant;
}

subcarrier_spacing to_scs(unsigned scs_khz)
{
  switch (scs_khz) {
    case 15:
      return subcarrier_spacing::kHz15;
    case 30:
      return subcarrier_spacing::kHz30;
    case 60:
      return subcarrier_spacing::kHz60;
    case 120:
      return subcarrier_spacing::kHz120;
    default:
      throw input_error("unsupported subcarrier spacing");
  }
}

pusch_mcs_table to_mcs_table(const std::string& name)
{
  if (name == "qam64") {
    return pusch_mcs_table::qam64;
  }
  if (name == "qam256") {
    return pusch_mcs_table::qam256;
  }
  if (name == "qam64LowSe") {
    return pusch_mcs_table::qam64LowSe;
  }
  throw input_error("unknown MCS table '" + name + "'");
}

symbol_slot_mask to_symbol_mask(const std::vector<unsigned>& symbols)
{
  symbol_slot_mask mask(MAX_NSYMB_PER_SLOT);
  for (unsigned symbol : symbols) {
    mask.set(symbol);
  }
  return mask;
}

bool file_exists(const std::string& path)
{
  struct stat info;
  return ::stat(path.c_str(), &info) == 0;
}

void usage()
{
  std::cout << "Usage: grant_fixture --grant FILE --tb FILE --output-dir DIR [--dmrs-reference crb0|prb0]\n"
               "                     [--no-loopback]\n"
               "Writes DIR/" << GRID_FILE << " (little-endian complex64, [tx_port, symbol, subcarrier] in carrier\n"
               "coordinates) and DIR/" << REPORT_FILE << ". DIR must exist and must not hold either file.\n";
}

int run(int argc, char** argv)
{
  std::string grant_path;
  std::string tb_path;
  std::string output_dir;
  std::string dmrs_reference = "crb0";
  bool        loopback       = true;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--grant" && i + 1 < argc) {
      grant_path = argv[++i];
    } else if (arg == "--tb" && i + 1 < argc) {
      tb_path = argv[++i];
    } else if (arg == "--output-dir" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--dmrs-reference" && i + 1 < argc) {
      dmrs_reference = argv[++i];
    } else if (arg == "--no-loopback") {
      loopback = false;
    } else if (arg == "--help") {
      usage();
      return 0;
    } else {
      usage();
      throw input_error("unknown or incomplete option '" + arg + "'");
    }
  }
  if (grant_path.empty() || tb_path.empty() || output_dir.empty()) {
    usage();
    throw input_error("--grant, --tb and --output-dir are required");
  }
  if (dmrs_reference != "crb0" && dmrs_reference != "prb0") {
    throw input_error("--dmrs-reference must be crb0 or prb0");
  }
  const std::string grid_path   = output_dir + "/" + GRID_FILE;
  const std::string report_path = output_dir + "/" + REPORT_FILE;
  if (file_exists(grid_path) || file_exists(report_path)) {
    throw input_error("the output directory already holds a result. Evidence is never overwritten.");
  }

  // ---- Everything derived from the record, before any grid exists. ----
  const grant_t              grant = load_grant(grant_path);
  const std::vector<uint8_t> tx_data = read_bytes(tb_path);

  const subcarrier_spacing  scs        = to_scs(grant.scs_khz);
  const pusch_mcs_table     mcs_table  = to_mcs_table(grant.mcs_table);
  const sch_mcs_index       mcs_index  = static_cast<sch_mcs_index>(grant.mcs_index);
  const sch_mcs_description mcs_descr  = pusch_mcs_get_config(mcs_table, mcs_index, false, false);
  const dmrs_config_type    dmrs_type  = (grant.dmrs_config_type == 1) ? dmrs_config_type::type1 : dmrs_config_type::type2;
  const symbol_slot_mask    dmrs_mask  = to_symbol_mask(grant.dmrs_symbols);
  const cyclic_prefix       cp         = cyclic_prefix::NORMAL;
  if (!(mcs_descr.target_code_rate > 0.0F)) {
    throw input_error("the MCS index is reserved in this table");
  }

  // Same transport block size rule as the pinned test.
  tbs_calculator_configuration tbs_config = {};
  tbs_config.mcs_descr                    = mcs_descr;
  tbs_config.n_prb                        = grant.prb_count;
  tbs_config.nof_layers                   = grant.layers;
  tbs_config.nof_symb_sh                  = grant.symbol_count;
  tbs_config.nof_dmrs_prb                 = get_nof_re_per_prb(dmrs_type) * dmrs_mask.count() * grant.cdm_groups;
  const unsigned tbs                      = tbs_calculator_calculate(tbs_config).to_bits().value();
  if (tx_data.size() * 8 != tbs) {
    throw input_error("the transport block holds " + std::to_string(tx_data.size() * 8) + " bits, the grant carries " +
                      std::to_string(tbs));
  }

  const ldpc_base_graph_type base_graph =
      get_ldpc_base_graph(mcs_descr.get_normalised_target_code_rate(), units::bits(tbs));
  const unsigned      nof_codeblocks = compute_nof_codeblocks(units::bits(tbs), base_graph);
  const rb_allocation freq_alloc     = rb_allocation::make_type1(grant.prb_start, grant.prb_count, std::nullopt);
  const slot_point    slot(to_numerology_value(scs), grant.sfn, grant.slot);
  const unsigned      nof_subc       = grant.grid_size_rb * NOF_SUBCARRIERS_PER_RB;
  const unsigned      nof_rx_ports   = static_cast<unsigned>(grant.rx_ports.size());

  // ---- Processors, as the pinned test creates them. ----
  ocudulog::init();
  ocudulog::fetch_basic_logger("ALL").set_level(ocudulog::basic_levels::warning);

  unsigned max_nof_threads = std::max(1U, std::min(8U, std::thread::hardware_concurrency()));
  auto     worker_pool =
      std::make_unique<task_worker_pool<concurrent_queue_policy::locking_mpmc>>("thread", max_nof_threads, 1024);
  auto executor = std::make_unique<task_worker_pool_executor<concurrent_queue_policy::locking_mpmc>>(*worker_pool);

  std::shared_ptr<resource_grid_factory>   grid_factory  = create_resource_grid_factory();
  std::shared_ptr<pdsch_processor_factory> pdsch_factory =
      create_sw_pdsch_processor_factory(*executor, max_nof_threads + 1, "", "auto");
  if (!grid_factory || !pdsch_factory) {
    worker_pool->stop();
    throw std::runtime_error("the pinned factories are not available");
  }
  auto pdsch_proc = pdsch_factory->create();
  auto tx_grid    = grid_factory->create(grant.layers, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);
  tx_grid->set_all_zero();

  // ---- Transmit reference: the PDU of the pinned test with the grant's values. ----
  pdsch_processor::pdu_t pdsch_pdu;
  pdsch_pdu.context                     = std::nullopt;
  pdsch_pdu.slot                        = slot;
  pdsch_pdu.rnti                        = to_rnti(static_cast<uint16_t>(grant.rnti));
  pdsch_pdu.bwp_size_rb                 = grant.bwp_size_rb;
  pdsch_pdu.bwp_start_rb                = grant.bwp_start_rb;
  pdsch_pdu.cp                          = cp;
  pdsch_pdu.n_id                        = grant.n_id;
  pdsch_pdu.ref_point = (dmrs_reference == "crb0") ? pdsch_processor::pdu_t::CRB0 : pdsch_processor::pdu_t::PRB0;
  pdsch_pdu.dmrs_symbol_mask            = dmrs_mask;
  pdsch_pdu.dmrs                        = dmrs_type;
  pdsch_pdu.scrambling_id               = grant.scrambling_id;
  pdsch_pdu.n_scid                      = (grant.n_scid != 0);
  pdsch_pdu.nof_cdm_groups_without_data = grant.cdm_groups;
  pdsch_pdu.freq_alloc                  = freq_alloc;
  pdsch_pdu.start_symbol_index          = grant.symbol_start;
  pdsch_pdu.nof_symbols                 = grant.symbol_count;
  pdsch_pdu.tbs_lbrm                    = tbs_lbrm_default;
  pdsch_pdu.reserved                    = {};
  pdsch_pdu.ratio_pdsch_data_to_sss_dB  = 0.0F;
  pdsch_pdu.ratio_pdsch_dmrs_to_sss_dB  = get_sch_to_dmrs_ratio_dB(grant.cdm_groups);
  pdsch_pdu.precoding                   = precoding_configuration::make_wideband(make_identity(grant.layers));
  pdsch_pdu.codewords.emplace_back(pdsch_processor::codeword_description{mcs_descr.modulation, grant.rv, base_graph});

  pdsch_done_notifier tx_notifier;
  pdsch_proc->process(tx_grid->get_writer(), tx_notifier, {shared_transport_block(tx_data)}, pdsch_pdu);
  tx_notifier.wait();

  // ---- Read the grid back the way a receiver reads it and write it in carrier coordinates. ----
  const resource_grid_reader& tx_reader = tx_grid->get_reader();
  std::vector<cf_t>           row(MAX_NOF_SUBCARRIERS);
  std::vector<float>          grid_values;
  grid_values.reserve(static_cast<size_t>(grant.layers) * MAX_NSYMB_PER_SLOT * nof_subc * 2);
  uint64_t nonzero_re              = 0;
  uint64_t nonzero_outside_carrier = 0;
  float    max_abs                 = 0.0F;
  for (unsigned port = 0; port != grant.layers; ++port) {
    for (unsigned symbol = 0; symbol != MAX_NSYMB_PER_SLOT; ++symbol) {
      tx_reader.get(row, port, symbol, 0);
      for (unsigned k = 0; k != MAX_NOF_SUBCARRIERS; ++k) {
        const bool nonzero = (row[k].real() != 0.0F) || (row[k].imag() != 0.0F);
        if (k < nof_subc) {
          grid_values.push_back(row[k].real());
          grid_values.push_back(row[k].imag());
          nonzero_re += nonzero ? 1 : 0;
          max_abs = std::max(max_abs, std::abs(row[k]));
        } else {
          nonzero_outside_carrier += nonzero ? 1 : 0;
        }
      }
    }
  }

  // ---- Optional noiseless loopback through the host PUSCH processor. ----
  json loopback_report = {{"executed", false}};
  if (loopback) {
    std::shared_ptr<pusch_processor_factory> cpu_factory =
        create_sw_pusch_processor_factory(*executor,
                                          max_nof_threads + 1,
                                          NOF_LDPC_ITERATIONS,
                                          USE_EARLY_STOP,
                                          "auto",
                                          port_channel_estimator_td_interpolation_strategy::average,
                                          channel_equalizer_algorithm_type::mmse);
    if (!cpu_factory) {
      worker_pool->stop();
      throw std::runtime_error("the host PUSCH processor factory is not available");
    }
    auto cpu_proc = cpu_factory->create();

    // The grid is indexed by port number, so it must reach the highest port the grant names.
    const unsigned highest_port = *std::max_element(grant.rx_ports.begin(), grant.rx_ports.end());
    auto           rx_grid      = grid_factory->create(highest_port + 1, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);
    rx_grid->set_all_zero();
    resource_grid_writer& rx_writer = rx_grid->get_writer();
    std::vector<cf_t>     carrier_row(nof_subc);
    for (unsigned symbol = 0; symbol != MAX_NSYMB_PER_SLOT; ++symbol) {
      tx_reader.get(row, 0, symbol, 0);
      std::copy(row.begin(), row.begin() + nof_subc, carrier_row.begin());
      for (unsigned port : grant.rx_ports) {
        rx_writer.put(port, symbol, 0, carrier_row);
      }
    }

    static_vector<uint8_t, MAX_PORTS> rx_ports(nof_rx_ports);
    for (unsigned i = 0; i != nof_rx_ports; ++i) {
      rx_ports[i] = static_cast<uint8_t>(grant.rx_ports[i]);
    }

    pusch_processor::pdu_t pusch_pdu;
    pusch_pdu.slot             = slot;
    pusch_pdu.rnti             = to_rnti(static_cast<uint16_t>(grant.rnti));
    pusch_pdu.bwp_size_rb      = grant.bwp_size_rb;
    pusch_pdu.bwp_start_rb     = grant.bwp_start_rb;
    pusch_pdu.cp               = cp;
    pusch_pdu.mcs_descr        = mcs_descr;
    pusch_pdu.codeword         = {grant.rv, base_graph, grant.new_data};
    pusch_pdu.uci              = {};
    pusch_pdu.n_id             = grant.n_id;
    pusch_pdu.nof_tx_layers    = grant.layers;
    pusch_pdu.rx_ports         = rx_ports;
    pusch_pdu.dmrs_symbol_mask = dmrs_mask;
    pusch_pdu.dmrs = pusch_processor::dmrs_configuration{.dmrs                        = dmrs_type,
                                                         .scrambling_id               = grant.scrambling_id,
                                                         .n_scid                      = (grant.n_scid != 0),
                                                         .nof_cdm_groups_without_data = grant.cdm_groups};
    pusch_pdu.tbs_lbrm           = tbs_lbrm_default;
    pusch_pdu.freq_alloc         = freq_alloc;
    pusch_pdu.start_symbol_index = grant.symbol_start;
    pusch_pdu.nof_symbols        = grant.symbol_count;
    pusch_pdu.dc_position        = std::nullopt;

    rx_buffer_pool_config pool_config;
    pool_config.max_codeblock_size   = ldpc::MAX_CODEBLOCK_SIZE;
    pool_config.nof_buffers          = 1;
    pool_config.nof_codeblocks       = nof_codeblocks;
    pool_config.expire_timeout_slots = 10;
    pool_config.external_soft_bits   = false;
    auto pool                        = create_rx_buffer_pool(pool_config);

    std::vector<uint8_t> rx_data(tx_data.size());
    unique_rx_buffer     buffer = pool->get_pool().reserve(
        slot, trx_buffer_identifier(to_rnti(static_cast<uint16_t>(grant.rnti)), 0), nof_codeblocks, true);
    pusch_done_notifier rx_notifier;
    cpu_proc->process(rx_data, std::move(buffer), rx_notifier, rx_grid->get_reader(), pusch_pdu);
    const pusch_processor_result_data& result = rx_notifier.wait();

    long long first_differing_byte = -1;
    for (size_t i = 0; i != tx_data.size(); ++i) {
      if (rx_data[i] != tx_data[i]) {
        first_differing_byte = static_cast<long long>(i);
        break;
      }
    }
    loopback_report = {{"executed", true},
                       {"processor", "host"},
                       {"channel", "identity on every receive port, no noise"},
                       {"tb_crc_ok", result.data.tb_crc_ok},
                       {"bytes_equal", first_differing_byte < 0},
                       {"first_differing_byte", first_differing_byte},
                       {"bytes_compared", tx_data.size()}};
    if (result.data.ldpc_decoder_stats.get_nof_observations() > 0) {
      loopback_report["ldpc_iterations_mean"] = result.data.ldpc_decoder_stats.get_mean();
    }
    std::optional<float> sinr = result.csi.get_sinr_dB();
    std::optional<float> epre = result.csi.get_epre_dB();
    if (sinr.has_value() && std::isfinite(sinr.value())) {
      loopback_report["sinr_db"] = sinr.value();
    }
    if (epre.has_value() && std::isfinite(epre.value())) {
      loopback_report["epre_db"] = epre.value();
    }
  }
  worker_pool->stop();

  // ---- Write the grid, then the report that names it. ----
  {
    std::ofstream stream(grid_path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(grid_values.data()),
                 static_cast<std::streamsize>(grid_values.size() * sizeof(float)));
    if (!stream) {
      throw std::runtime_error("cannot write " + grid_path);
    }
  }

  const uint16_t endian_probe = 1;
  json           report       = {
      {"schema", NATIVE_SCHEMA},
      {"grant_id", grant.grant_id},
      {"dmrs_reference", dmrs_reference},
      {"transmit_chain", "create_sw_pdsch_processor_factory of the pinned test helpers, host software"},
      {"pdu",
       {{"scs_khz", grant.scs_khz},
        {"sfn", grant.sfn},
        {"slot", grant.slot},
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
        {"rx_ports", grant.rx_ports}}},
      {"derived",
       {{"tbs_bits", tbs},
        {"modulation", to_string(mcs_descr.modulation)},
        {"bits_per_symbol", get_bits_per_symbol(mcs_descr.modulation)},
        {"target_code_rate_x1024", mcs_descr.target_code_rate},
        {"ldpc_base_graph", base_graph == ldpc_base_graph_type::BG1 ? "BG1" : "BG2"},
        {"codeblocks", nof_codeblocks},
        {"tbs_lbrm_bytes", tbs_lbrm_default.value()},
        {"dmrs_amplitude", convert_dB_to_amplitude(-get_sch_to_dmrs_ratio_dB(grant.cdm_groups))},
        {"data_amplitude", convert_dB_to_amplitude(-0.0F)}}},
      {"grid",
       {{"file", GRID_FILE},
        {"axes", {"tx_port", "symbol", "subcarrier"}},
        {"shape", {grant.layers, static_cast<unsigned>(MAX_NSYMB_PER_SLOT), nof_subc}},
        {"dtype", "<c8"},
        {"order", "C"},
        {"storage", "complex bf16 in the pinned resource grid, widened without change"},
        {"little_endian_host", *reinterpret_cast<const uint8_t*>(&endian_probe) == 1},
        {"nonzero_re", nonzero_re},
        {"nonzero_outside_carrier", nonzero_outside_carrier},
        {"max_abs", max_abs}}},
      {"loopback", loopback_report},
#ifdef ENABLE_CUDA
      {"built_with_cuda", true},
#else
      {"built_with_cuda", false},
#endif
  };
  {
    std::ofstream stream(report_path);
    stream << report.dump(2) << "\n";
    if (!stream) {
      throw std::runtime_error("cannot write " + report_path);
    }
  }
  std::cout << "grant_fixture: " << grant.grant_id << ", " << tbs << " bits, " << nonzero_re
            << " occupied resource elements, report " << report_path << std::endl;
  return 0;
}

} // namespace

int main(int argc, char** argv)
{
  try {
    return run(argc, argv);
  } catch (const input_error& error) {
    std::cerr << "grant_fixture: input error: " << error.what() << std::endl;
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "grant_fixture: " << error.what() << std::endl;
    return 1;
  }
}
