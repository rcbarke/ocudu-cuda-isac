// Reader of a replay-pusch-plan/v1 file (Sprint 2.4, selector replay_pusch).
//
// The plan is written offline by src/ocudu_cuda_isac/episodes/replay_pusch.py. It names the
// channel plan of Sprint 2.3 that fills every grid, by the hash that binds it, and adds what the
// receiver and the comparison need: per grant the record the PDU is built from and the original
// transport block, the payload rule of the run, and the backends that decode each grid. This
// reader only translates it. The rules are applied by the tool before a file is written.
// Header only. It uses the JSON library of the locked OCUDU source. ASCII only.

#pragma once

#include "replay_core.h"
#include "replay_plan.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace isac_receiver {

constexpr const char* REPLAY_PLAN_SCHEMA = "replay-pusch-plan/v1";
constexpr const char* REPLAY_PLAN_ACCESS = "offline_stimulus_and_evaluator";

/// What the plan adds to one grant of the channel plan.
struct plan_grant {
  std::string           grant_id;
  std::string           record_sha256; ///< The hash the channel plan carries for this grant.
  isac_replay::file_ref record;        ///< grant.json of the grant's store entry.
  isac_replay::file_ref block;         ///< tb.bin of the store entry: the original transport block.
  std::size_t           block_bytes = 0;
};

struct payload_rule {
  std::string rule;      ///< "fixed" or "per_opportunity".
  std::string generator; ///< Of a generated payload. Empty for the fixed rule.
  std::string seed;
  std::size_t blocks = 0; ///< Distinct transport blocks of the run.
};

struct receiver_settings {
  std::vector<std::string> backends; ///< In dispatch order: "cpu", "cuda".
  unsigned                 ldpc_iterations = 6;
  bool                     early_stop      = true;
  unsigned                 timeout_ms      = 30000;
  std::string              tap; ///< "none": the decode path without an observer.
};

struct replay_plan {
  std::string             run_id;
  isac_replay::file_ref   channel_plan;
  payload_rule            payload;
  receiver_settings       receiver;
  std::vector<plan_grant> grants;
};

/// Reads a plan. plan_sha256, when given, receives the hash of the file. Throws input_error for
/// a file that is not a plan.
inline replay_plan load_replay_plan(const std::string& path, std::string* plan_sha256 = nullptr)
{
  using namespace isac_replay::plan_detail;
  using isac_replay::input_error;
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
  keys(plan, "the plan", {"schema", "run_id", "access", "channel_plan", "payload", "receiver", "grants"});
  if (text(plan, "schema", "the plan") != REPLAY_PLAN_SCHEMA) {
    throw input_error(std::string("the plan schema is not ") + REPLAY_PLAN_SCHEMA);
  }
  if (text(plan, "access", "the plan") != REPLAY_PLAN_ACCESS) {
    throw input_error(std::string("the plan access must be ") + REPLAY_PLAN_ACCESS);
  }
  replay_plan spec;
  spec.run_id       = text(plan, "run_id", "the plan");
  spec.channel_plan = file(plan.at("channel_plan"), "the channel plan");

  const json& payload = plan.at("payload");
  keys(payload, "payload", {"rule", "blocks"}, {"generator", "seed"});
  spec.payload.rule   = text(payload, "rule", "payload");
  spec.payload.blocks = count(payload, "blocks", "payload", 1U << 30);
  if (payload.contains("generator")) {
    spec.payload.generator = text(payload, "generator", "payload", true);
  }
  if (payload.contains("seed")) {
    spec.payload.seed = text(payload, "seed", "payload", true);
  }

  const json& receiver = plan.at("receiver");
  keys(receiver, "receiver", {"backends", "ldpc_iterations", "early_stop", "timeout_ms", "tap"});
  if (!receiver.at("backends").is_array()) {
    throw input_error("receiver.backends must be an array");
  }
  for (const json& item : receiver.at("backends")) {
    if (!item.is_string()) {
      throw input_error("receiver.backends must hold names");
    }
    spec.receiver.backends.push_back(item.get<std::string>());
  }
  spec.receiver.ldpc_iterations = static_cast<unsigned>(count(receiver, "ldpc_iterations", "receiver", 1U << 16));
  if (!receiver.at("early_stop").is_boolean()) {
    throw input_error("receiver.early_stop must be true or false");
  }
  spec.receiver.early_stop = receiver.at("early_stop").get<bool>();
  spec.receiver.timeout_ms = static_cast<unsigned>(count(receiver, "timeout_ms", "receiver", 1U << 30));
  spec.receiver.tap        = text(receiver, "tap", "receiver");

  const json& grants = plan.at("grants");
  if (!grants.is_array()) {
    throw input_error("grants must be an array");
  }
  for (const json& item : grants) {
    keys(item, "a grant", {"grant_id", "record_sha256", "record", "tb", "tb_bytes"});
    plan_grant grant;
    grant.grant_id      = text(item, "grant_id", "a grant");
    grant.record_sha256 = text(item, "record_sha256", "grant " + grant.grant_id);
    grant.record        = file(item.at("record"), "the record of grant " + grant.grant_id);
    grant.block         = file(item.at("tb"), "the transport block of grant " + grant.grant_id);
    grant.block_bytes   = count(item, "tb_bytes", "grant " + grant.grant_id, 1U << 24);
    spec.grants.push_back(grant);
  }
  return spec;
}

} // namespace isac_receiver
