// Reader of a grid-apply-plan/v1 file (Sprint 2.3, selector grid_apply).
//
// The plan is written offline by src/ocudu_cuda_isac/episodes/grid_apply.py. It names the carrier
// grid, the one gain of the run, every grant with the hash that binds its transmit grid, every
// channel realization and every opportunity. This reader only translates it: the rules are in the
// core (check_gain, session), so a harness that builds a run_spec another way meets the same
// refusals. Header only. It uses the JSON library of the locked OCUDU source. ASCII only.

#pragma once

#include "replay_core.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string>
#include <vector>

namespace isac_replay {

constexpr const char* PLAN_SCHEMA = "grid-apply-plan/v1";

namespace plan_detail {

using json = nlohmann::json;

/// Requires an object with exactly the required keys and any of the optional ones.
inline void keys(const json&                        object,
                 const std::string&                 where,
                 std::initializer_list<const char*> required,
                 std::initializer_list<const char*> optional = {})
{
  if (!object.is_object()) {
    throw input_error(where + " must be an object");
  }
  for (const char* key : required) {
    if (!object.contains(key)) {
      throw input_error(where + " has no field '" + key + "'");
    }
  }
  for (auto item = object.begin(); item != object.end(); ++item) {
    bool known = false;
    for (const char* key : required) {
      known = known || item.key() == key;
    }
    for (const char* key : optional) {
      known = known || item.key() == key;
    }
    if (!known) {
      throw input_error(where + " has the unknown field '" + item.key() + "'");
    }
  }
}

inline std::string text(const json& object, const char* key, const std::string& where, bool nullable = false)
{
  const json& value = object.at(key);
  if (nullable && value.is_null()) {
    return std::string();
  }
  if (!value.is_string() || value.get<std::string>().empty()) {
    throw input_error(where + "." + key + " must be a nonempty string");
  }
  return value.get<std::string>();
}

inline std::size_t count(const json& object, const char* key, const std::string& where, std::size_t limit)
{
  const json& value = object.at(key);
  if (!value.is_number_integer() || value.get<long long>() < 0 ||
      static_cast<unsigned long long>(value.get<long long>()) > limit) {
    throw input_error(where + "." + key + " must be an integer from 0 to " + std::to_string(limit));
  }
  return static_cast<std::size_t>(value.get<long long>());
}

inline double real(const json& object, const char* key, const std::string& where)
{
  const json& value = object.at(key);
  if (!value.is_number()) {
    throw input_error(where + "." + key + " must be a number");
  }
  return value.get<double>();
}

/// A file with the hash that binds it. A null or absent hash leaves the file unbound.
inline file_ref file(const json& value, const std::string& where)
{
  keys(value, where, {"path"}, {"sha256"});
  file_ref ref;
  ref.path = text(value, "path", where);
  if (value.contains("sha256")) {
    ref.sha256 = text(value, "sha256", where, true);
  }
  return ref;
}

} // namespace plan_detail

/// Reads a plan. plan_sha256, when given, receives the hash of the file. Throws input_error for
/// a file that is not a plan. The gain and the binding rules are checked by the session.
inline run_spec load_plan(const std::string& path, std::string* plan_sha256 = nullptr)
{
  using namespace plan_detail;
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw input_error("cannot read the plan " + path);
  }
  const std::string raw((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  if (plan_sha256 != nullptr) {
    *plan_sha256 = sha256_hex(raw.data(), raw.size());
  }
  json plan;
  try {
    plan = json::parse(raw);
  } catch (const json::exception& error) {
    throw input_error("the plan is not valid JSON: " + std::string(error.what()));
  }
  keys(plan, "the plan", {"schema", "run_id", "carrier", "grants", "sources", "opportunities"}, {"gain", "access"});
  if (text(plan, "schema", "the plan") != PLAN_SCHEMA) {
    throw input_error(std::string("the plan schema is not ") + PLAN_SCHEMA);
  }
  run_spec spec;
  spec.run_id = text(plan, "run_id", "the plan");

  const json& carrier = plan.at("carrier");
  keys(carrier, "carrier", {"rx_ports", "tx_ports", "symbols", "subcarriers"});
  spec.rx_ports    = static_cast<unsigned>(count(carrier, "rx_ports", "carrier", 1U << 16));
  spec.tx_ports    = static_cast<unsigned>(count(carrier, "tx_ports", "carrier", 1U << 16));
  spec.symbols     = static_cast<unsigned>(count(carrier, "symbols", "carrier", 1U << 16));
  spec.subcarriers = static_cast<unsigned>(count(carrier, "subcarriers", "carrier", 1U << 20));

  // A plan without a gain is read and then refused by the gain rule, which says why.
  if (plan.contains("gain") && !plan.at("gain").is_null()) {
    const json& gain = plan.at("gain");
    keys(gain, "gain", {"value", "source"}, {"reference_power", "reason", "rule"});
    spec.gain.present = true;
    spec.gain.value   = real(gain, "value", "gain");
    spec.gain.source  = text(gain, "source", "gain");
    if (gain.contains("reference_power") && !gain.at("reference_power").is_null()) {
      spec.gain.has_reference_power = true;
      spec.gain.reference_power     = real(gain, "reference_power", "gain");
    }
    if (gain.contains("reason") && !gain.at("reason").is_null()) {
      spec.gain.reason = text(gain, "reason", "gain");
    }
  }

  const json& grants = plan.at("grants");
  if (!grants.is_array()) {
    throw input_error("grants must be an array");
  }
  for (const json& item : grants) {
    keys(item, "a grant", {"grant_id", "record_sha256", "x", "active"}, {"slot"});
    grant_spec grant;
    grant.grant_id      = text(item, "grant_id", "a grant");
    grant.record_sha256 = text(item, "record_sha256", "grant " + grant.grant_id);
    grant.x             = file(item.at("x"), "the transmit grid of grant " + grant.grant_id);
    if (!item.at("active").is_array()) {
      throw input_error("grant " + grant.grant_id + ".active must be an array of files");
    }
    for (const json& index : item.at("active")) {
      grant.active.push_back(file(index, "the occupied elements of grant " + grant.grant_id));
    }
    spec.grants.push_back(grant);
  }

  const json& sources = plan.at("sources");
  if (!sources.is_array()) {
    throw input_error("sources must be an array");
  }
  for (const json& item : sources) {
    keys(item, "a source", {"realization_id", "h", "w", "entry"});
    source_spec source;
    source.realization_id = text(item, "realization_id", "a source");
    source.h              = file(item.at("h"), "the channel of realization " + source.realization_id);
    source.w              = file(item.at("w"), "the noise of realization " + source.realization_id);
    if (!item.at("entry").is_null()) {
      const json& entry = item.at("entry");
      keys(entry, "the entry of realization " + source.realization_id, {"episode_id", "grid_gain", "reference_power"});
      source.from_entry            = true;
      source.entry_grid_gain       = real(entry, "grid_gain", "entry");
      source.entry_reference_power = real(entry, "reference_power", "entry");
    }
    spec.sources.push_back(source);
  }

  const json& opportunities = plan.at("opportunities");
  if (!opportunities.is_array()) {
    throw input_error("opportunities must be an array");
  }
  for (const json& item : opportunities) {
    keys(item, "an opportunity", {"opportunity_id", "realization_id", "index", "grant_id", "sfn", "slot"});
    opportunity_spec opportunity;
    opportunity.opportunity_id = text(item, "opportunity_id", "an opportunity");
    const std::string where    = "opportunity " + opportunity.opportunity_id;
    opportunity.realization_id = text(item, "realization_id", where);
    opportunity.index          = count(item, "index", where, 1U << 30);
    opportunity.grant_id       = text(item, "grant_id", where, true);
    opportunity.sfn            = static_cast<unsigned>(count(item, "sfn", where, 1023));
    opportunity.slot           = static_cast<unsigned>(count(item, "slot", where, 159));
    spec.opportunities.push_back(opportunity);
  }
  return spec;
}

} // namespace isac_replay
