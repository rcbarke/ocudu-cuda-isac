// The JSON lines a core writes: window lines and decisions of the reference schemas (Sprint 3.6,
// selector sense_core).
//
// A sensor of the reference runtime writes one cusense-window/v1 line and one cusense-decision/v1
// line per answer of its stream. A sensor on the native core writes the same lines, with the
// runtime entry "native" and with what only this runtime has: the engine backend, the hash of the
// plan and of the core sources, the timing marks of a submission and the outcome against its
// budgets. ocudu_cuda_isac.sensing.cusense.open_output verifies both.
//
// These functions format. They run after the publication time of a decision was taken, and the
// digests they print are handed in by a caller that took them after its last record.
// ASCII only.

#include "core/build.h"
#include "plan/plan.h"
#include <cstdio>
#include <cstring>

using namespace sense;

namespace {

void put_text(std::string& out, const char* key, const char* text)
{
  out += '"';
  out += key;
  out += "\": ";
  json::put_string(out, text);
}

void put_number(std::string& out, const char* key, std::int64_t number)
{
  out += '"';
  out += key;
  out += "\": ";
  json::put_int(out, number);
}

void put_real(std::string& out, const char* key, double number)
{
  out += '"';
  out += key;
  out += "\": ";
  json::put_double(out, number);
}

void put_hash(std::string& out, const char* key, const char* hash)
{
  out += '"';
  out += key;
  out += "\": ";
  if (hash == nullptr || *hash == '\0') {
    out += "null";
  } else {
    json::put_string(out, hash);
  }
}

void put_window(std::string& out, bool has, std::uint64_t epoch, std::uint64_t episode, std::uint64_t period)
{
  out += "\"window\": ";
  if (!has) {
    out += "null";
    return;
  }
  char id[80];
  std::snprintf(id, sizeof(id), "%llu/%016llx/%llu", static_cast<unsigned long long>(epoch),
                static_cast<unsigned long long>(episode), static_cast<unsigned long long>(period));
  out += "{";
  put_text(out, "id", id);
  out += ", ";
  put_number(out, "epoch", static_cast<std::int64_t>(epoch));
  out += ", \"episode\": ";
  json::put_key(out, episode);
  out += ", ";
  put_number(out, "period", static_cast<std::int64_t>(period));
  out += "}";
}

void put_trigger(std::string& out, bool has, std::uint64_t epoch, std::uint64_t sequence, std::uint64_t opportunity,
                 std::uint64_t slot)
{
  out += "\"trigger\": ";
  if (!has) {
    out += "null";
    return;
  }
  out += "{";
  put_number(out, "epoch", static_cast<std::int64_t>(epoch));
  out += ", ";
  put_number(out, "sequence", static_cast<std::int64_t>(sequence));
  out += ", \"opportunity\": ";
  json::put_key(out, opportunity);
  out += ", ";
  put_number(out, "absolute_slot", static_cast<std::int64_t>(slot));
  out += "}";
}

void put_contributing(std::string& out, const sense_contributor* items, std::uint32_t count)
{
  out += "\"contributing\": [";
  for (std::uint32_t n = 0; n != count; ++n) {
    if (n != 0) {
      out += ", ";
    }
    out += "{";
    put_number(out, "sequence", static_cast<std::int64_t>(items[n].sequence));
    out += ", \"opportunity\": ";
    json::put_key(out, items[n].opportunity);
    out += ", ";
    put_number(out, "absolute_slot", static_cast<std::int64_t>(items[n].absolute_slot));
    out += ", ";
    put_number(out, "first_position", items[n].first_position);
    out += ", ";
    put_number(out, "positions", items[n].positions);
    out += "}";
  }
  out += "]";
}

void put_time(std::string& out, std::uint64_t first, std::uint64_t last)
{
  out += "\"time\": {\"unit\": \"tc\", ";
  put_number(out, "tc_per_second", static_cast<std::int64_t>(TC_PER_SECOND));
  out += ", ";
  put_number(out, "t_first_sample", static_cast<std::int64_t>(first));
  out += ", ";
  put_number(out, "t_last_sample", static_cast<std::int64_t>(last));
  out += ", ";
  put_real(out, "acquisition_us", 1e6 * static_cast<double>(last - first) / static_cast<double>(TC_PER_SECOND));
  out += "}";
}

void put_optional(std::string& out, const char* key, bool has, std::int64_t value)
{
  out += '"';
  out += key;
  out += "\": ";
  if (has) {
    json::put_int(out, value);
  } else {
    out += "null";
  }
}

/// The fields every window line has.
void window_head(std::string& out, const sense_plan& plan, std::uint64_t line, const char* status, const char* reason,
                 bool has_window, std::uint64_t epoch, std::uint64_t episode, std::uint64_t period, bool has_trigger,
                 std::uint64_t trigger_epoch, std::uint64_t sequence, std::uint64_t opportunity, std::uint64_t slot,
                 std::uint32_t have,
                 std::int64_t t_ready, std::int64_t t_publish, std::int64_t t_window)
{
  out += "{\"schema\": \"cusense-window/v1\", ";
  put_number(out, "line", static_cast<std::int64_t>(line));
  out += ", ";
  put_text(out, "status", status);
  out += ", \"reason\": ";
  if (reason == nullptr || *reason == '\0') {
    out += "null";
  } else {
    json::put_string(out, reason);
  }
  out += ", ";
  put_window(out, has_window, epoch, episode, period);
  out += ", ";
  put_trigger(out, has_trigger, trigger_epoch, sequence, opportunity, slot);
  out += ", \"history\": {";
  put_number(out, "have", have);
  out += ", ";
  put_number(out, "need", plan.need);
  out += "}, ";
  put_text(out, "spec_sha256", plan.model.spec_sha256.c_str());
  out += ", \"runtime\": \"native\", ";
  put_optional(out, "t_ready_ns", has_trigger, t_ready);
  out += ", ";
  put_optional(out, "t_publish_ns", has_trigger, t_publish);
  out += ", ";
  put_optional(out, "t_window_ns", has_trigger && t_window != 0, t_window);
}

/// The fields every decision line has, up to the score.
void decision_head(std::string& out, const sense_plan& plan, std::uint64_t line, const char* status, const char* reason,
                   bool has_window, std::uint64_t epoch, std::uint64_t episode, std::uint64_t period, bool has_trigger,
                   std::uint64_t trigger_epoch, std::uint64_t sequence, std::uint64_t opportunity, std::uint64_t slot)
{
  const artifact& model = plan.model;
  out += "{\"schema\": \"cusense-decision/v1\", ";
  put_number(out, "line", static_cast<std::int64_t>(line));
  out += ", ";
  put_text(out, "status", status);
  out += ", \"reason\": ";
  if (reason == nullptr || *reason == '\0') {
    out += "null";
  } else {
    json::put_string(out, reason);
  }
  out += ", ";
  put_window(out, has_window, epoch, episode, period);
  out += ", ";
  put_trigger(out, has_trigger, trigger_epoch, sequence, opportunity, slot);
  out += ", \"runtime\": \"native\", ";
  put_text(out, "device", plan.device_name.c_str());
  out += ", \"model\": {";
  put_text(out, "model_id", model.model_id.c_str());
  out += ", ";
  put_text(out, "entry", model.entry_model.c_str());
  out += ", ";
  put_text(out, "artifact_sha256", model.sha256.c_str());
  out += ", ";
  put_text(out, "weights_sha256", model.weights_sha256.c_str());
  out += ", ";
  put_text(out, "template_sha256", model.template_sha256.c_str());
  out += ", ";
  put_text(out, "normalizer_sha256", model.normalizer_sha256.c_str());
  out += ", ";
  put_text(out, "grid_sha256", model.grid_sha256.c_str());
  out += ", ";
  put_text(out, "threshold_sha256", model.threshold_sha256.c_str());
  out += "}, \"engine\": {";
  put_text(out, "backend", plan.eng->backend());
  out += ", ";
  put_text(out, "sha256", plan.engine_sha256.c_str());
  out += "}, ";
  put_text(out, "plan_sha256", plan.plan_sha256.c_str());
  out += ", ";
  put_text(out, "core_sources_sha256", core_sources_sha256());
}

void put_smoothed(std::string& out, const sense_plan& plan, const sense_smoothed* made)
{
  if (made == nullptr || made->active == 0) {
    out += "null";
    return;
  }
  out += "{";
  put_text(out, "entry", plan.model.entry_smoothing.c_str());
  out += ", ";
  put_number(out, "maps", made->maps);
  out += ", \"cell\": [";
  json::put_uint(out, made->row);
  out += ", ";
  json::put_uint(out, made->col);
  out += "], \"position_m\": [";
  json::put_double(out, made->x_m);
  out += ", ";
  json::put_double(out, made->y_m);
  out += "], ";
  put_real(out, "probability", made->probability);
  out += ", \"kalman_position_m\": [";
  json::put_double(out, made->kalman_x_m);
  out += ", ";
  json::put_double(out, made->kalman_y_m);
  out += "], \"kalman_velocity_m_s\": [";
  json::put_double(out, made->kalman_vx_m_s);
  out += ", ";
  json::put_double(out, made->kalman_vy_m_s);
  out += "]}";
}

const char* status_of(int kind)
{
  switch (kind) {
    case SENSE_ANSWER_REJECTED: return "rejected";
    case SENSE_ANSWER_FILLING: return "insufficient_history";
    case SENSE_ANSWER_GAP: return "gap";
    case SENSE_ANSWER_INVALID: return "invalid";
    case SENSE_ANSWER_STALE_SKIPPED: return "stale_skipped";
    default: return nullptr;
  }
}

const char* status_of_decision(int status)
{
  switch (status) {
    case SENSE_DECISION_OK: return "ok";
    case SENSE_DECISION_INVALID: return "invalid";
    case SENSE_DECISION_STALE: return "stale";
    default: return nullptr;
  }
}

sense_status give(const std::string& text, char* json, size_t bytes)
{
  if (json == nullptr || text.size() + 1 > bytes) {
    return refuse(SENSE_E_ARGUMENT, "the buffer of " + std::to_string(bytes) + " bytes does not hold a line of " +
                                        std::to_string(text.size() + 1));
  }
  std::memcpy(json, text.c_str(), text.size() + 1);
  return SENSE_OK;
}

} // namespace

extern "C" {

sense_status sense_answer_window_json(const sense_plan* plan, const sense_answer* answer, uint64_t line, char* json,
                                      size_t bytes)
{
  if (plan == nullptr || answer == nullptr || status_of(answer->kind) == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "a window line is written for an answer that is no complete window");
  }
  std::string out;
  out.reserve(1024);
  window_head(out, *plan, line, status_of(answer->kind), answer->reason, answer->has_window != 0, answer->epoch,
              answer->episode, answer->period, answer->has_trigger != 0, answer->trigger_epoch, answer->trigger_sequence,
              answer->trigger_opportunity, answer->trigger_slot, answer->have, answer->t_ready_ns, answer->t_publish_ns,
              answer->t_notify_ns);
  if (answer->kind == SENSE_ANSWER_GAP) {
    out += ", \"missing\": [";
    for (std::uint32_t n = 0; n != answer->missing; ++n) {
      out += n != 0 ? ", {" : "{";
      put_number(out, "period", static_cast<std::int64_t>(answer->period));
      out += ", ";
      put_number(out, "slot_in_period", answer->missing_slots[n]);
      out += "}";
    }
    out += "]";
  }
  if (answer->kind == SENSE_ANSWER_INVALID || answer->kind == SENSE_ANSWER_STALE_SKIPPED) {
    out += ", ";
    put_contributing(out, answer->contributing, answer->contributors);
    out += ", ";
    put_time(out, answer->t_first_sample, answer->t_last_sample);
  }
  out += "}";
  return give(out, json, bytes);
}

sense_status sense_answer_decision_json(const sense_plan* plan, const sense_answer* answer, uint64_t line, char* json,
                                        size_t bytes)
{
  if (plan == nullptr || answer == nullptr || status_of(answer->kind) == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "a decision without a score is written for an answer that is no complete window");
  }
  std::string out;
  out.reserve(1280);
  decision_head(out, *plan, line, status_of(answer->kind), answer->reason, answer->has_window != 0, answer->epoch,
                answer->episode, answer->period, answer->has_trigger != 0, answer->trigger_epoch, answer->trigger_sequence,
                answer->trigger_opportunity, answer->trigger_slot);
  out += ", \"score\": null, \"decision\": null, \"location\": null, \"smoothed\": null, \"input\": null, "
         "\"frontend_ns\": null, \"infer_ns\": null, ";
  put_optional(out, "t_ready_ns", answer->has_trigger != 0, answer->t_ready_ns);
  out += ", ";
  put_optional(out, "t_publish_ns", answer->has_trigger != 0, answer->t_publish_ns);
  out += ", ";
  put_optional(out, "t_window_ns", answer->has_trigger != 0 && answer->t_notify_ns != 0, answer->t_notify_ns);
  out += ", \"t_decision_ns\": null, \"marks\": null, \"deadline\": null}";
  return give(out, json, bytes);
}

sense_status sense_decision_window_json(const sense_plan* plan, const sense_decision* decision, uint64_t line,
                                        uint64_t index, const char* mean_sha256, const char* count_sha256, char* json,
                                        size_t bytes)
{
  if (plan == nullptr || decision == nullptr || status_of_decision(decision->status) == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "a window line needs a plan and the decision of a complete window");
  }
  const artifact& model = plan->model;
  std::string     out;
  out.reserve(2048);
  window_head(out, *plan, line, status_of_decision(decision->status), decision->reason, true, decision->epoch,
              decision->episode, decision->period, true, decision->epoch, decision->trigger_sequence,
              decision->trigger_opportunity, decision->trigger_slot, plan->need, decision->t_ready_ns, decision->t_publish_ns, decision->t_notify_ns);
  out += ", ";
  put_contributing(out, decision->contributing, decision->contributors);
  if (decision->status != SENSE_DECISION_STALE) {
    out += ", \"validity\": {";
    put_number(out, "elements", decision->elements);
    out += ", ";
    put_number(out, "valid", decision->valid);
    out += ", ";
    put_number(out, "holes", decision->elements - decision->valid);
    out += ", ";
    put_real(out, "fill", static_cast<double>(decision->valid) / static_cast<double>(decision->elements));
    out += ", \"records_per_element\": {";
    put_number(out, "min", decision->records_min);
    out += ", ";
    put_number(out, "max", decision->records_max);
    out += "}, \"interpolation\": \"none: an element without a valid sample is left out of every average\"}";
  }
  out += ", ";
  put_time(out, decision->t_first_sample, decision->t_last_sample);
  // The input of the window: the entry of the specification and what the records said of themselves.
  out += ", \"input\": {";
  put_text(out, "entry", model.entry_input.c_str());
  out += ", ";
  put_text(out, "stage", model.input_stage.c_str());
  out += ", ";
  put_text(out, "source", model.input_source.c_str());
  out += ", ";
  put_text(out, "layout", model.input_layout.c_str());
  out += ", ";
  put_text(out, "dtype", model.input_dtype.c_str());
  out += ", ";
  put_number(out, "port", decision->port);
  out += ", ";
  put_real(out, "scale", decision->scale);
  out += ", \"producer_steps\": [";
  static const std::pair<std::uint32_t, const char*> steps[] = {{obs::STEP_PILOT_LS, "pilot_ls"},
                                                                {obs::STEP_FREQ_INTERP, "frequency_interpolation"},
                                                                {obs::STEP_SMOOTHING, "smoothing"},
                                                                {obs::STEP_TIME_INTERP, "time_interpolation"}};
  bool first = true;
  for (const auto& step : steps) {
    if ((decision->producer_steps & step.first) != 0) {
      out += first ? "\"" : ", \"";
      out += step.second;
      out += "\"";
      first = false;
    }
  }
  out += "], \"evidence\": [";
  first = true;
  for (const auto& kind : {std::make_pair(1U, "audit"), std::make_pair(2U, "timed"), std::make_pair(4U, "unknown")}) {
    if ((decision->evidence & kind.first) != 0) {
      out += first ? "\"" : ", \"";
      out += kind.second;
      out += "\"";
      first = false;
    }
  }
  out += "]}, \"rx\": {\"elements\": [";
  for (std::size_t a = 0; a != model.elements.size(); ++a) {
    if (a != 0) {
      out += ", ";
    }
    json::put_uint(out, model.elements[a]);
  }
  out += "], \"unit\": \"array element of the antenna map, rising\"}";
  if (decision->status == SENSE_DECISION_OK) {
    out += ", \"mean\": {";
    put_number(out, "index", static_cast<std::int64_t>(index));
    out += ", \"shape\": [";
    json::put_uint(out, model.rx);
    out += ", ";
    json::put_uint(out, model.subcarriers);
    out += ", ";
    json::put_uint(out, model.dmrs_symbols);
    out += "], \"dtype\": \"<c8\", ";
    put_hash(out, "sha256", mean_sha256);
    out += "}, \"count\": {\"dtype\": \"u1\", ";
    put_hash(out, "sha256", count_sha256);
    out += "}";
  }
  out += "}";
  return give(out, json, bytes);
}

sense_status sense_decision_json(const sense_plan* plan, const sense_decision* decision, uint64_t line, uint64_t index,
                                 const char* input_sha256, const char* map_sha256, const sense_smoothed* smoothed,
                                 char* json, size_t bytes)
{
  if (plan == nullptr || decision == nullptr || status_of_decision(decision->status) == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "a decision line needs a plan and the decision of a complete window");
  }
  const artifact& model  = plan->model;
  const bool      scored = decision->status == SENSE_DECISION_OK;
  std::string     out;
  out.reserve(3072);
  decision_head(out, *plan, line, status_of_decision(decision->status), decision->reason, true, decision->epoch,
                decision->episode, decision->period, true, decision->epoch, decision->trigger_sequence,
                decision->trigger_opportunity, decision->trigger_slot);
  if (scored) {
    out += ", \"score\": {";
    put_real(out, "logit", static_cast<double>(decision->logit));
    out += ", ";
    put_real(out, "probability", decision->probability);
    out += "}, \"decision\": {\"present\": ";
    out += decision->present != 0 ? "true" : "false";
    out += ", ";
    put_real(out, "threshold_logit", model.threshold_logit);
    out += ", ";
    put_text(out, "detector", model.entry_detector.c_str());
    out += "}, \"location\": {\"cell\": [";
    json::put_uint(out, decision->row);
    out += ", ";
    json::put_uint(out, decision->col);
    out += "], \"position_m\": [";
    json::put_double(out, decision->position_x_m);
    out += ", ";
    json::put_double(out, decision->position_y_m);
    out += "], ";
    put_real(out, "probability", static_cast<double>(decision->peak));
    out += ", \"map\": {\"shape\": [";
    json::put_uint(out, model.rows);
    out += ", ";
    json::put_uint(out, model.cols);
    out += "], \"dtype\": \"<f4\", ";
    put_hash(out, "sha256", map_sha256);
    out += ", ";
    put_number(out, "index", static_cast<std::int64_t>(index));
    out += "}}, \"smoothed\": ";
    put_smoothed(out, *plan, smoothed);
    out += ", \"input\": {";
    put_text(out, "spec_sha256", model.spec_sha256.c_str());
    out += ", ";
    put_text(out, "entry", model.entry_input.c_str());
    out += ", \"shape\": [1, ";
    json::put_uint(out, model.rx);
    out += ", ";
    json::put_uint(out, model.subcarriers);
    out += "], ";
    put_hash(out, "sha256", input_sha256);
    out += ", ";
    put_number(out, "index", static_cast<std::int64_t>(index));
    out += "}, ";
    put_number(out, "frontend_ns", decision->t_kernels_ns - decision->t_submit_ns);
    out += ", ";
    put_number(out, "infer_ns", decision->t_engine_ns - decision->t_kernels_ns);
  } else {
    out += ", \"score\": null, \"decision\": null, \"location\": null, \"smoothed\": null, \"input\": null, "
           "\"frontend_ns\": null, \"infer_ns\": null";
  }
  out += ", ";
  put_number(out, "t_ready_ns", decision->t_ready_ns);
  out += ", ";
  put_number(out, "t_publish_ns", decision->t_publish_ns);
  out += ", ";
  put_optional(out, "t_window_ns", decision->t_notify_ns != 0, decision->t_notify_ns);
  out += ", ";
  put_number(out, "t_decision_ns", decision->t_decision_ns);
  // What only the native runtime has: the marks of the submission and the outcome against its budgets.
  out += ", \"marks\": {";
  put_optional(out, "t_notify_ns", decision->t_notify_ns != 0, decision->t_notify_ns);
  out += ", ";
  put_number(out, "t_submit_ns", decision->t_submit_ns);
  out += ", ";
  put_number(out, "t_ingest_ns", decision->t_ingest_ns);
  out += ", ";
  put_number(out, "t_kernels_ns", decision->t_kernels_ns);
  out += ", ";
  put_number(out, "t_engine_ns", decision->t_engine_ns);
  out += "}, \"deadline\": ";
  const bool budgeted = plan->budget_ingest > 0 || plan->budget_kernels > 0 || plan->budget_engine > 0 ||
                        plan->budget_window > 0;
  if (!scored || !budgeted) {
    out += "null";
  } else {
    out += "{\"outcome\": ";
    out += decision->late != 0 ? "\"late\"" : "\"on_time\"";
    out += ", \"overrun\": [";
    bool first = true;
    for (const auto& stage : {std::make_pair(decision->overrun_ingest, "ingest"), std::make_pair(decision->overrun_kernels, "kernels"),
                              std::make_pair(decision->overrun_engine, "engine"), std::make_pair(decision->overrun_window, "window")}) {
      if (stage.first != 0) {
        out += first ? "\"" : ", \"";
        out += stage.second;
        out += "\"";
        first = false;
      }
    }
    out += "], \"budget_ns\": {";
    put_number(out, "ingest", plan->budget_ingest);
    out += ", ";
    put_number(out, "kernels", plan->budget_kernels);
    out += ", ";
    put_number(out, "engine", plan->budget_engine);
    out += ", ";
    put_number(out, "window", plan->budget_window);
    out += "}}";
  }
  if (scored) {
    out += ", ";
    put_contributing(out, decision->contributing, decision->contributors);
    out += ", ";
    put_time(out, decision->t_first_sample, decision->t_last_sample);
  }
  out += "}";
  return give(out, json, bytes);
}

sense_status sense_plan_smooth(sense_plan* plan, const sense_decision* decision, sense_smoothed* smoothed)
{
  if (plan == nullptr || decision == nullptr || smoothed == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "the optional stage needs a plan, a decision and a place for its output");
  }
  std::memset(smoothed, 0, sizeof(*smoothed));
  if (!plan->smooth || decision->status != SENSE_DECISION_OK || decision->map == nullptr) {
    return SENSE_OK;
  }
  const sense::smoothed made = plan->smooth->update(decision->epoch, decision->episode, decision->period, decision->map);
  smoothed->active        = 1;
  smoothed->maps          = made.maps;
  smoothed->row           = made.row;
  smoothed->col           = made.col;
  smoothed->x_m           = made.x_m;
  smoothed->y_m           = made.y_m;
  smoothed->probability   = made.probability;
  smoothed->kalman_x_m    = made.kalman_x_m;
  smoothed->kalman_y_m    = made.kalman_y_m;
  smoothed->kalman_vx_m_s = made.kalman_vx;
  smoothed->kalman_vy_m_s = made.kalman_vy;
  return SENSE_OK;
}

} // extern "C"
