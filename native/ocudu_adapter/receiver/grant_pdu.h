// The PUSCH PDU of one grant record (Sprint 2.4, selector replay_pusch).
//
// A pusch-grant/v1 record, as a grant store keeps it in grant.json, becomes the PDU that the
// pinned PUSCH processors decode. The values are the record's, and the DM-RS is the PUSCH one:
// the pinned estimator references it to CRB 0, so no reference point is set here. The frame and
// the slot are those of the opportunity that is decoded, and the slot index must be the one the
// grant was built for, because its DM-RS sequence depends on it.
//
// The record is bound to the grid it is decoded from by one hash: the SHA-256 of the record in
// canonical form with its transport block in hexadecimal, as schedule.record_sha256 of the Python
// package computes it. The channel plan, the grid descriptor and the radio record carry that hash.
//
// Header only. It needs the headers of the locked OCUDU source and a target that links
// ocudu_pin_fixture. ASCII only.

#pragma once

#include "replay_core.h"
#include "nlohmann/json.hpp"
#include "ocudu/adt/static_vector.h"
#include "ocudu/phy/antenna_ports.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_processor.h"
#include "ocudu/ran/cyclic_prefix.h"
#include "ocudu/ran/dmrs/dmrs.h"
#include "ocudu/ran/pusch/pusch_constants.h"
#include "ocudu/ran/pusch/pusch_mcs.h"
#include "ocudu/ran/resource_block.h"
#include "ocudu/ran/rnti.h"
#include "ocudu/ran/sch/sch_constants.h"
#include "ocudu/ran/sch/ldpc_base_graph.h"
#include "ocudu/ran/sch/sch_segmentation.h"
#include "ocudu/ran/sch/tbs_calculator.h"
#include "ocudu/ran/slot_point.h"
#include "ocudu/ran/subcarrier_spacing.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace isac_receiver {

constexpr const char* GRANT_SCHEMA = "pusch-grant/v1";

/// One grant record: the values of its fields and the record itself.
struct grant_record {
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
  nlohmann::json        record; ///< The record as it was read.
};

/// What the pinned library derives from a grant, with the rule of the pinned test.
struct grant_coding {
  ocudu::sch_mcs_description  mcs;
  unsigned                    tbs_bits   = 0;
  ocudu::ldpc_base_graph_type base_graph = ocudu::ldpc_base_graph_type::BG1;
  unsigned                    codeblocks = 0;
};

namespace grant_detail {

using json = nlohmann::json;
using isac_replay::input_error;

inline const json& member(const json& object, const char* key, const std::string& where)
{
  if (!object.is_object() || !object.contains(key)) {
    throw input_error(where + " has no field '" + key + "'");
  }
  return object.at(key);
}

inline unsigned number(const json& object, const char* key, unsigned low, unsigned high, const std::string& where)
{
  const json& value = member(object, key, where);
  if (!value.is_number_integer() || value.get<long long>() < 0) {
    throw input_error(where + ": field '" + key + "' must be a non-negative integer");
  }
  const unsigned long long parsed = value.get<unsigned long long>();
  if (parsed < low || parsed > high) {
    throw input_error(where + ": field '" + key + "' is " + std::to_string(parsed) + ", allowed " +
                      std::to_string(low) + " to " + std::to_string(high));
  }
  return static_cast<unsigned>(parsed);
}

inline std::string text(const json& object, const char* key, const std::string& where)
{
  const json& value = member(object, key, where);
  if (!value.is_string()) {
    throw input_error(where + ": field '" + key + "' must be a string");
  }
  return value.get<std::string>();
}

inline std::vector<unsigned> number_list(const json& object, const char* key, unsigned high, const std::string& where)
{
  const json& value = member(object, key, where);
  if (!value.is_array() || value.empty()) {
    throw input_error(where + ": field '" + key + "' must be a nonempty array");
  }
  std::vector<unsigned> result;
  for (const json& item : value) {
    if (!item.is_number_integer() || item.get<long long>() < 0 || item.get<long long>() > static_cast<long long>(high)) {
      throw input_error(where + ": field '" + key + "' holds a value outside 0 to " + std::to_string(high));
    }
    result.push_back(static_cast<unsigned>(item.get<long long>()));
  }
  return result;
}

} // namespace grant_detail

/// Reads one grant record from the text of a grant.json file. Throws input_error for a record
/// that the pinned receiver cannot decode. The limits repeat those of the fixture adapter, so that
/// no value reaches an assertion inside the pinned library.
inline grant_record parse_grant(const std::string& raw, const std::string& where)
{
  using namespace grant_detail;
  grant_record grant;
  try {
    grant.record = json::parse(raw);
  } catch (const json::exception& error) {
    throw input_error(where + " is not valid JSON: " + error.what());
  }
  const json& record = grant.record;
  if (text(record, "schema", where) != GRANT_SCHEMA) {
    throw input_error(where + ": the schema is not " + GRANT_SCHEMA);
  }
  grant.grant_id = text(record, "grant_id", where);

  const json& carrier = member(record, "carrier", where);
  grant.scs_khz       = number(carrier, "scs_khz", 15, 120, where);
  grant.grid_size_rb  = number(carrier, "grid_size_rb", 1, ocudu::MAX_NOF_PRBS, where);
  if (grant.scs_khz != 15 && grant.scs_khz != 30 && grant.scs_khz != 60 && grant.scs_khz != 120) {
    throw input_error(where + ": unsupported subcarrier spacing");
  }
  if (text(carrier, "cyclic_prefix", where) != "normal") {
    throw input_error(where + ": only the normal cyclic prefix is supported");
  }

  const json& bwp    = member(record, "bwp", where);
  grant.bwp_start_rb = number(bwp, "start_rb", 0, ocudu::MAX_NOF_PRBS - 1, where);
  grant.bwp_size_rb  = number(bwp, "size_rb", 1, ocudu::MAX_NOF_PRBS, where);

  const json& slot = member(record, "slot", where);
  grant.sfn        = number(slot, "sfn", 0, 1023, where);
  grant.slot       = number(slot, "slot", 0, 159, where);

  grant.rnti = number(record, "rnti", 1, 65535, where);
  grant.n_id = number(record, "n_id", 0, 1023, where);

  const json& allocation = member(record, "allocation", where);
  grant.prb_start        = number(allocation, "prb_start", 0, ocudu::MAX_NOF_PRBS - 1, where);
  grant.prb_count        = number(allocation, "prb_count", 1, ocudu::MAX_NOF_PRBS, where);
  grant.symbol_start     = number(allocation, "symbol_start", 0, ocudu::MAX_NSYMB_PER_SLOT - 1, where);
  grant.symbol_count     = number(allocation, "symbol_count", 1, ocudu::MAX_NSYMB_PER_SLOT, where);

  const json& mcs = member(record, "mcs", where);
  grant.mcs_table = text(mcs, "table", where);
  grant.mcs_index = number(mcs, "index", 0, 31, where);
  if (grant.mcs_table != "qam64" && grant.mcs_table != "qam256" && grant.mcs_table != "qam64LowSe") {
    throw input_error(where + ": unknown MCS table '" + grant.mcs_table + "'");
  }

  const json& codeword = member(record, "codeword", where);
  grant.rv             = number(codeword, "rv", 0, 3, where);
  const json& new_data = member(codeword, "new_data", where);
  if (!new_data.is_boolean()) {
    throw input_error(where + ": field 'new_data' must be true or false");
  }
  grant.new_data = new_data.get<bool>();
  grant.layers   = number(record, "layers", 1, 4, where);

  const json& dmrs       = member(record, "dmrs", where);
  grant.dmrs_config_type = number(dmrs, "config_type", 1, 2, where);
  grant.dmrs_symbols     = number_list(dmrs, "symbols", ocudu::MAX_NSYMB_PER_SLOT - 1, where);
  grant.dmrs_ports       = number_list(dmrs, "ports", 11, where);
  grant.scrambling_id    = number(dmrs, "scrambling_id", 0, 65535, where);
  grant.n_scid           = number(dmrs, "n_scid", 0, 1, where);
  grant.cdm_groups       = number(dmrs, "cdm_groups_without_data", 1, 3, where);

  const json& tx      = member(record, "tx", where);
  grant.antenna_ports = number(tx, "antenna_ports", 1, 4, where);
  if (text(tx, "precoding", where) != "identity") {
    throw input_error(where + ": only identity precoding is supported");
  }
  grant.rx_ports = number_list(record, "rx_ports", ocudu::MAX_PORTS - 1, where);
  member(record, "tb", where);

  if (grant.bwp_start_rb + grant.bwp_size_rb > grant.grid_size_rb) {
    throw input_error(where + ": the bandwidth part leaves the carrier grid");
  }
  if (grant.prb_start + grant.prb_count > grant.bwp_size_rb) {
    throw input_error(where + ": the allocation leaves the bandwidth part");
  }
  if (grant.symbol_start + grant.symbol_count > ocudu::MAX_NSYMB_PER_SLOT) {
    throw input_error(where + ": the symbol allocation leaves the slot");
  }
  if (grant.layers != 1 || grant.antenna_ports != 1 || grant.dmrs_ports.size() != 1 || grant.dmrs_ports[0] != 0) {
    throw input_error(where + ": the receiver harness decodes one layer on DM-RS port 0 and one antenna port");
  }
  if (grant.rv != 0 || !grant.new_data) {
    throw input_error(where + ": the receiver harness decodes a new transmission with redundancy version 0");
  }
  if (grant.slot >= 10U * (grant.scs_khz / 15U)) {
    throw input_error(where + ": the slot index exceeds the slots of one frame at this subcarrier spacing");
  }
  std::vector<unsigned> sorted = grant.dmrs_symbols;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() || sorted != grant.dmrs_symbols) {
    throw input_error(where + ": DM-RS symbols must be listed once each, in increasing order");
  }
  if (grant.dmrs_symbols.size() > 4) {
    throw input_error(where + ": at most four DM-RS symbols are supported");
  }
  for (unsigned symbol : grant.dmrs_symbols) {
    if (symbol < grant.symbol_start || symbol >= grant.symbol_start + grant.symbol_count) {
      throw input_error(where + ": a DM-RS symbol lies outside the symbol allocation");
    }
  }
  for (std::size_t index = 0; index != grant.rx_ports.size(); ++index) {
    // The pinned host demodulator reads grid port i for the i-th listed port when the data mask is
    // contiguous (pusch_demodulator_impl::get_ch_data_re), so the list must be 0 to N-1 in order.
    if (grant.rx_ports[index] != index) {
      throw input_error(where + ": receive ports must be 0 to N-1 in order");
    }
  }
  if (grant.rx_ports.size() > ocudu::pusch_constants::MAX_NOF_RX_PORTS) {
    throw input_error(where + ": the pinned PUSCH processor takes at most " +
                      std::to_string(ocudu::pusch_constants::MAX_NOF_RX_PORTS) + " receive ports");
  }
  return grant;
}

/// The hash that binds a grant record: SHA-256 of the record with its transport block in
/// hexadecimal, keys sorted, no white space, ASCII. It equals schedule.record_sha256(Grant.record()).
inline std::string record_sha256(const grant_record& grant, const std::vector<std::uint8_t>& block)
{
  static const char digits[] = "0123456789abcdef";
  std::string       hex;
  hex.reserve(2 * block.size());
  for (std::uint8_t byte : block) {
    hex.push_back(digits[byte >> 4]);
    hex.push_back(digits[byte & 0x0f]);
  }
  nlohmann::json canonical = grant.record;
  canonical["tb"]          = {{"hex", hex}};
  const std::string dumped = canonical.dump(-1, ' ', true);
  return isac_replay::sha256_hex(dumped.data(), dumped.size());
}

inline ocudu::pusch_mcs_table mcs_table_of(const grant_record& grant)
{
  if (grant.mcs_table == "qam256") {
    return ocudu::pusch_mcs_table::qam256;
  }
  if (grant.mcs_table == "qam64LowSe") {
    return ocudu::pusch_mcs_table::qam64LowSe;
  }
  return ocudu::pusch_mcs_table::qam64;
}

inline ocudu::subcarrier_spacing spacing_of(const grant_record& grant)
{
  switch (grant.scs_khz) {
    case 15:
      return ocudu::subcarrier_spacing::kHz15;
    case 60:
      return ocudu::subcarrier_spacing::kHz60;
    case 120:
      return ocudu::subcarrier_spacing::kHz120;
    default:
      return ocudu::subcarrier_spacing::kHz30;
  }
}

inline ocudu::dmrs_config_type dmrs_type_of(const grant_record& grant)
{
  return grant.dmrs_config_type == 1 ? ocudu::dmrs_config_type::type1 : ocudu::dmrs_config_type::type2;
}

/// The transport block size, the base graph and the code block count, as the pinned test derives them.
inline grant_coding derive(const grant_record& grant, const std::string& where)
{
  grant_coding coding;
  coding.mcs = ocudu::pusch_mcs_get_config(mcs_table_of(grant), static_cast<ocudu::sch_mcs_index>(grant.mcs_index),
                                           false, false);
  if (!(coding.mcs.target_code_rate > 0.0F)) {
    throw isac_replay::input_error(where + ": the MCS index is reserved in this table");
  }
  ocudu::tbs_calculator_configuration config = {};
  config.mcs_descr                           = coding.mcs;
  config.n_prb                               = grant.prb_count;
  config.nof_layers                          = grant.layers;
  config.nof_symb_sh                         = grant.symbol_count;
  config.nof_dmrs_prb = ocudu::get_nof_re_per_prb(dmrs_type_of(grant)) *
                        static_cast<unsigned>(grant.dmrs_symbols.size()) * grant.cdm_groups;
  coding.tbs_bits     = ocudu::tbs_calculator_calculate(config).to_bits().value();
  coding.base_graph =
      ocudu::get_ldpc_base_graph(coding.mcs.get_normalised_target_code_rate(), ocudu::units::bits(coding.tbs_bits));
  coding.codeblocks = ocudu::compute_nof_codeblocks(ocudu::units::bits(coding.tbs_bits), coding.base_graph);
  return coding;
}

/// The PDU of one opportunity: the grant's values in the frame and slot of the opportunity.
inline ocudu::pusch_processor::pdu_t
make_pdu(const grant_record& grant, const grant_coding& coding, unsigned sfn, unsigned slot)
{
  ocudu::symbol_slot_mask dmrs_mask(ocudu::MAX_NSYMB_PER_SLOT);
  for (unsigned symbol : grant.dmrs_symbols) {
    dmrs_mask.set(symbol);
  }
  ocudu::static_vector<std::uint8_t, ocudu::MAX_PORTS> rx_ports(grant.rx_ports.size());
  for (std::size_t index = 0; index != grant.rx_ports.size(); ++index) {
    rx_ports[index] = static_cast<std::uint8_t>(grant.rx_ports[index]);
  }

  ocudu::pusch_processor::pdu_t pdu;
  pdu.slot             = ocudu::slot_point(ocudu::to_numerology_value(spacing_of(grant)), sfn, slot);
  pdu.rnti             = ocudu::to_rnti(static_cast<std::uint16_t>(grant.rnti));
  pdu.bwp_size_rb      = grant.bwp_size_rb;
  pdu.bwp_start_rb     = grant.bwp_start_rb;
  pdu.cp               = ocudu::cyclic_prefix::NORMAL;
  pdu.mcs_descr        = coding.mcs;
  pdu.codeword         = {grant.rv, coding.base_graph, grant.new_data};
  pdu.uci              = {};
  pdu.n_id             = grant.n_id;
  pdu.nof_tx_layers    = grant.layers;
  pdu.rx_ports         = rx_ports;
  pdu.dmrs_symbol_mask = dmrs_mask;
  pdu.dmrs             = ocudu::pusch_processor::dmrs_configuration{.dmrs                        = dmrs_type_of(grant),
                                                                    .scrambling_id               = grant.scrambling_id,
                                                                    .n_scid                      = (grant.n_scid != 0),
                                                                    .nof_cdm_groups_without_data = grant.cdm_groups};
  pdu.tbs_lbrm           = ocudu::tbs_lbrm_default;
  pdu.freq_alloc         = ocudu::rb_allocation::make_type1(grant.prb_start, grant.prb_count, std::nullopt);
  pdu.start_symbol_index = grant.symbol_start;
  pdu.nof_symbols        = grant.symbol_count;
  pdu.dc_position        = std::nullopt;
  return pdu;
}

} // namespace isac_receiver
