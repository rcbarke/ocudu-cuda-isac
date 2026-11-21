// Reader of a pilot-ls-plan/v1 file and of the sensing-source records it binds (Sprint 3.1,
// selector pilot_ls).
//
// The plan is written offline by src/ocudu_cuda_isac/episodes/pilot_ls.py. It names the receiver
// replay plan of Sprint 2.4 that the run decodes, by the hash that binds it, and what the
// observer taps: nothing (tap "none", the control of the same executable) or the pilot least
// squares of one registered sensing source (tap "pilot_ls"). For the tap it lists, per grant of
// the replay, the sensing-source record of the grant's store entry with its two arrays and the
// antenna map of the receive planes. This reader only translates. The rules are applied by the
// tool and by the observer core before a processor is created or a file is written.
// Header only. It uses the JSON library of the locked OCUDU source. ASCII only.

#pragma once

#include "observer_core.h"
#include "replay_core.h"
#include "replay_plan.h"
#include "nlohmann/json.hpp"
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace isac_observer {

constexpr const char* TAP_PLAN_SCHEMA = "pilot-ls-plan/v1";
constexpr const char* TAP_PLAN_ACCESS = "offline_stimulus_and_evaluator";
constexpr const char* SOURCE_SCHEMA   = "sensing-source/v1";
constexpr const char* TAP_NONE        = "none";
constexpr const char* TAP_PILOT_LS    = "pilot_ls";

/// What the plan binds for one grant of the replay.
struct plan_source {
  std::string           grant_id;
  std::string           record_sha256; ///< The hash the channel plan carries for this grant.
  isac_replay::file_ref source;        ///< source.json of the grant's store entry.
  isac_replay::file_ref pilot_index;   ///< int32 [pilot, 2]: symbol and subcarrier.
  isac_replay::file_ref pilot_ref;     ///< complex64 [port, pilot]: the reference symbols.
  std::vector<unsigned> antenna_map;   ///< antenna_map[i] is the array element behind receive plane i.
};

struct tap_plan {
  std::string              run_id;
  isac_replay::file_ref    replay_plan;
  std::string              tap;    ///< "none" or "pilot_ls".
  std::string              stage;  ///< "O2" for the tap. Empty without one.
  std::string              source; ///< Name of the registered sensing source. Empty without a tap.
  std::string              layout; ///< Payload layout. Empty without a tap.
  std::uint64_t            epoch = 0;
  std::vector<plan_source> sources;
};

/// Reads a plan. plan_sha256, when given, receives the hash of the file. Throws input_error for
/// a file that is not a plan.
inline tap_plan load_tap_plan(const std::string& path, std::string* plan_sha256 = nullptr)
{
  using namespace isac_replay::plan_detail;
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw input_error("cannot read the plan " + path);
  }
  const std::string raw((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  if (plan_sha256 != nullptr) {
    *plan_sha256 = isac_replay::sha256_hex(raw.data(), raw.size());
  }
  json plan;
  try {
    plan = json::parse(raw);
  } catch (const json::exception& error) {
    throw input_error("the plan is not valid JSON: " + std::string(error.what()));
  }
  keys(plan, "the plan", {"schema", "run_id", "access", "replay_plan", "observer", "sources"});
  if (text(plan, "schema", "the plan") != TAP_PLAN_SCHEMA) {
    throw input_error(std::string("the plan schema is not ") + TAP_PLAN_SCHEMA);
  }
  if (text(plan, "access", "the plan") != TAP_PLAN_ACCESS) {
    throw input_error(std::string("the plan access must be ") + TAP_PLAN_ACCESS);
  }
  tap_plan spec;
  spec.run_id      = text(plan, "run_id", "the plan");
  spec.replay_plan = file(plan.at("replay_plan"), "the replay plan");

  const json& observer = plan.at("observer");
  keys(observer, "observer", {"tap", "stage", "source", "layout", "epoch"});
  spec.tap    = text(observer, "tap", "observer");
  spec.stage  = text(observer, "stage", "observer", true);
  spec.source = text(observer, "source", "observer", true);
  spec.layout = text(observer, "layout", "observer", true);
  spec.epoch  = count(observer, "epoch", "observer", 1U << 30);

  const json& sources = plan.at("sources");
  if (!sources.is_array()) {
    throw input_error("sources must be an array");
  }
  for (const json& item : sources) {
    keys(item, "a source", {"grant_id", "record_sha256", "source", "pilot_index", "pilot_ref", "antenna_map"});
    plan_source source;
    source.grant_id      = text(item, "grant_id", "a source");
    source.record_sha256 = text(item, "record_sha256", "the source of grant " + source.grant_id);
    source.source        = file(item.at("source"), "the source record of grant " + source.grant_id);
    source.pilot_index   = file(item.at("pilot_index"), "the pilot coordinates of grant " + source.grant_id);
    source.pilot_ref     = file(item.at("pilot_ref"), "the reference symbols of grant " + source.grant_id);
    const json& map      = item.at("antenna_map");
    if (!map.is_array()) {
      throw input_error("the antenna map of grant " + source.grant_id + " must be an array");
    }
    for (const json& element : map) {
      if (!element.is_number_integer() || element.get<long long>() < 0 || element.get<long long>() > 4095) {
        throw input_error("the antenna map of grant " + source.grant_id + " holds array elements from 0 to 4095");
      }
      source.antenna_map.push_back(static_cast<unsigned>(element.get<long long>()));
    }
    spec.sources.push_back(source);
  }
  return spec;
}

/// Reads the sensing-source record of one grant with its two arrays, each against the hash the
/// plan binds. The record names its arrays by hash as well, and those must be the same hashes.
/// files, when given, receives every verified file. Throws input_error for a record that is not
/// a sensing-source/v1 record of a store entry. The rules of the estimate are in pilot_source.
inline source_spec load_source(const plan_source& bound, std::vector<isac_replay::verified_file>* files = nullptr)
{
  using namespace isac_replay::plan_detail;
  const std::string where = "the source record of grant " + bound.grant_id;
  if (bound.source.sha256.empty() || bound.pilot_index.sha256.empty() || bound.pilot_ref.sha256.empty()) {
    throw input_error(where + " or one of its arrays is bound by no hash");
  }
  std::ifstream stream(bound.source.path, std::ios::binary);
  if (!stream) {
    throw input_error("cannot read " + where + " " + bound.source.path);
  }
  const std::string raw((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  if (isac_replay::sha256_hex(raw.data(), raw.size()) != bound.source.sha256) {
    throw input_error(where + " does not have the hash its record binds (" + bound.source.path + ")");
  }
  json record;
  try {
    record = json::parse(raw);
  } catch (const json::exception& error) {
    throw input_error(where + " is not valid JSON: " + std::string(error.what()));
  }
  keys(record, where,
       {"schema", "source", "channel", "link_direction", "reference_signal", "ports", "processor", "slot", "grid", "mask",
        "reference"});
  if (text(record, "schema", where) != SOURCE_SCHEMA) {
    throw input_error(where + " is not a " + std::string(SOURCE_SCHEMA) + " record");
  }
  source_spec spec;
  spec.grant_id            = bound.grant_id;
  spec.grant_record_sha256 = bound.record_sha256;
  spec.source              = text(record, "source", where);
  spec.channel             = text(record, "channel", where);
  spec.link_direction      = text(record, "link_direction", where);
  spec.antenna_map         = bound.antenna_map;
  const json& processor    = record.at("processor");
  keys(processor, where + ".processor", {"name", "backend"});
  spec.processor = text(processor, "name", where + ".processor");
  if (!record.at("ports").is_array()) {
    throw input_error(where + ".ports must be an array");
  }
  for (const json& port : record.at("ports")) {
    if (!port.is_number_integer() || port.get<long long>() < 0 || port.get<long long>() > 4095) {
      throw input_error(where + ".ports holds integers from 0 to 4095");
    }
    spec.ports.push_back(static_cast<unsigned>(port.get<long long>()));
  }
  const json& grid = record.at("grid");
  keys(grid, where + ".grid", {"coordinates", "symbols", "subcarriers"});
  if (text(grid, "coordinates", where + ".grid") != "carrier") {
    throw input_error(where + ": pilot coordinates must be carrier coordinates");
  }
  spec.grid_symbols     = static_cast<unsigned>(count(grid, "symbols", where + ".grid", 14));
  spec.grid_subcarriers = static_cast<unsigned>(count(grid, "subcarriers", where + ".grid", 3300));
  const json& slot      = record.at("slot");
  keys(slot, where + ".slot", {"scs_khz", "sfn", "slot", "symbols", "tc_per_second", "symbol_useful_start_tc"});
  spec.scs_khz       = static_cast<unsigned>(count(slot, "scs_khz", where + ".slot", 960));
  spec.slot          = static_cast<unsigned>(count(slot, "slot", where + ".slot", 639));
  spec.tc_per_second = count(slot, "tc_per_second", where + ".slot", static_cast<std::size_t>(1) << 40);
  if (!slot.at("symbols").is_array() || !slot.at("symbol_useful_start_tc").is_array()) {
    throw input_error(where + ".slot names its symbols and their start times as arrays");
  }
  for (const json& symbol : slot.at("symbols")) {
    if (!symbol.is_number_integer() || symbol.get<long long>() < 0 || symbol.get<long long>() > 13) {
      throw input_error(where + ".slot.symbols holds integers from 0 to 13");
    }
    spec.slot_symbols.push_back(static_cast<unsigned>(symbol.get<long long>()));
  }
  for (const json& start : slot.at("symbol_useful_start_tc")) {
    if (!start.is_number_integer() || start.get<long long>() < 0) {
      throw input_error(where + ".slot.symbol_useful_start_tc holds non-negative integers");
    }
    spec.symbol_useful_start_tc.push_back(static_cast<std::uint64_t>(start.get<long long>()));
  }
  const json& reference = record.at("reference");
  keys(reference, where + ".reference", {"axes", "amplitude", "values"});
  spec.amplitude    = real(reference, "amplitude", where + ".reference");
  const json& mask  = record.at("mask");
  keys(mask, where + ".mask", {"axes", "index"});
  const json& index_ref = mask.at("index");
  const json& value_ref = reference.at("values");
  if (!index_ref.is_object() || !value_ref.is_object()) {
    throw input_error(where + " must name its pilot coordinates and its reference symbols by file, as a store entry does");
  }
  keys(index_ref, where + ".mask.index", {"file", "sha256"});
  keys(value_ref, where + ".reference.values", {"file", "sha256"});
  if (text(index_ref, "sha256", where + ".mask.index") != bound.pilot_index.sha256 ||
      text(value_ref, "sha256", where + ".reference.values") != bound.pilot_ref.sha256) {
    throw input_error(where + " names other arrays than the plan binds");
  }

  const isac_replay::array_file index(bound.pilot_index.path);
  const isac_replay::array_file values(bound.pilot_ref.path);
  if (index.file_sha256() != bound.pilot_index.sha256) {
    throw input_error("the pilot coordinates of grant " + bound.grant_id + " do not have the hash their record binds (" +
                      bound.pilot_index.path + ")");
  }
  if (values.file_sha256() != bound.pilot_ref.sha256) {
    throw input_error("the reference symbols of grant " + bound.grant_id + " do not have the hash their record binds (" +
                      bound.pilot_ref.path + ")");
  }
  if (index.shape().size() != 2 || index.shape()[1] != 2) {
    throw input_error("the pilot coordinates of grant " + bound.grant_id + " must be int32 [pilot, 2]");
  }
  if (values.shape().size() != 2 || values.shape()[0] != spec.ports.size() || values.shape()[1] != index.shape()[0]) {
    throw input_error("the reference symbols of grant " + bound.grant_id + " must be complex64 [port, pilot]");
  }
  const std::int32_t* coordinates = index.int32();
  for (std::size_t n = 0; n != index.shape()[0]; ++n) {
    if (coordinates[2 * n] < 0 || coordinates[2 * n + 1] < 0) {
      throw input_error("the pilot coordinates of grant " + bound.grant_id + " hold a negative index");
    }
    pilot item;
    item.symbol     = static_cast<unsigned>(coordinates[2 * n]);
    item.subcarrier = static_cast<unsigned>(coordinates[2 * n + 1]);
    spec.index.push_back(item);
  }
  spec.reference.assign(values.complex64(), values.complex64() + values.count());
  if (files != nullptr) {
    files->push_back({where, bound.source.path, bound.source.sha256, raw.size()});
    files->push_back({"the pilot coordinates of grant " + bound.grant_id, bound.pilot_index.path, bound.pilot_index.sha256,
                      index.file_bytes()});
    files->push_back({"the reference symbols of grant " + bound.grant_id, bound.pilot_ref.path, bound.pilot_ref.sha256,
                      values.file_bytes()});
  }
  return spec;
}

} // namespace isac_observer
