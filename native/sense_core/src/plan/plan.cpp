// The plan of the cuSense-like branch: window tracking, submission and decision (Sprint 3.6,
// selector sense_core).
//
// This file is the native twin of three things in ocudu_cuda_isac.sensing.cusense: refusal(),
// which says whether a record is an input of the specification, WindowStream, which cuts one
// causal window per TDD period, and Model.decide, which turns the scores of a window into a
// decision at the frozen threshold. It also holds the detector entry frozen_threshold.
//
// ASCII only.

#include "plan/plan.h"
#include "core/build.h"
#include "core/sha256.h"
#include "core/tolerance.h"
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <time.h>

namespace sense {

namespace {

thread_local std::string last_error = "no call was refused";

std::int64_t monotonic(void*)
{
  timespec stamp;
  ::clock_gettime(CLOCK_MONOTONIC, &stamp);
  return static_cast<std::int64_t>(stamp.tv_sec) * 1000000000LL + stamp.tv_nsec;
}

void say(char* out, const char* format, ...)
{
  va_list arguments;
  va_start(arguments, format);
  std::vsnprintf(out, SENSE_TEXT_BYTES, format, arguments);
  va_end(arguments);
}

std::uint32_t steps_of_names(const std::vector<std::string>& names)
{
  std::uint32_t bits = 0;
  for (const std::string& name : names) {
    if (name == "pilot_ls") {
      bits |= obs::STEP_PILOT_LS;
    } else if (name == "frequency_interpolation") {
      bits |= obs::STEP_FREQ_INTERP;
    } else if (name == "smoothing") {
      bits |= obs::STEP_SMOOTHING;
    } else if (name == "time_interpolation") {
      bits |= obs::STEP_TIME_INTERP;
    } else {
      throw artifact_error("the artifact names the producer step " + name + ", which no record carries");
    }
  }
  return bits;
}

/// The geometry of a record from its header: its layout, its sizes and the receive plane behind
/// each array element. Returns false with the reason when the header is no record of a layout
/// this core has an ingest unit for.
bool geometry_of(const obs::record_header& header, record_geometry& made, char* reason)
{
  std::memset(&made, 0, sizeof(made));
  unsigned       symbols[SYMBOLS_PER_SLOT];
  const unsigned count = symbols_of(header, symbols);
  const unsigned rx    = header.rx_count;
  bool           unique = rx >= 1 && rx <= obs::MAX_PLANES;
  for (unsigned i = 0; unique && i != rx; ++i) {
    for (unsigned j = 0; j != i; ++j) {
      unique = unique && header.antenna_map[i] != header.antenna_map[j];
    }
  }
  if (count == 0 || header.layer_count != 1 || !unique) {
    say(reason, "the record does not hold one layer with one array element per receive plane");
    return false;
  }
  const std::uint32_t* shape   = header.shape;
  const std::uint32_t* strides = header.strides;
  std::uint64_t        elements = 0, width = 0;
  if (header.layout == obs::LAYOUT_OCUDU_O3_FP16) {
    const std::uint32_t blocks = header.subcarrier_count / SUBCARRIERS_PER_PRB;
    if (header.rank != 4 || shape[0] != count || shape[1] != blocks || shape[2] != rx || shape[3] != SUBCARRIERS_PER_PRB ||
        header.subcarrier_count != blocks * SUBCARRIERS_PER_PRB || blocks < 1 ||
        strides[0] != shape[1] * shape[2] * shape[3] || strides[1] != shape[2] * shape[3] || strides[2] != shape[3] ||
        strides[3] != 1 || header.dtype != obs::DTYPE_CFP16) {
      say(reason, "the shape is not [dmrs_symbol][prb][rx_port][subcarrier_in_prb] of whole resource blocks");
      return false;
    }
    elements = static_cast<std::uint64_t>(shape[0]) * shape[1] * shape[2] * shape[3];
    width    = 4;
  } else if (header.layout == obs::LAYOUT_OCUDU_O2_COMB_C64) {
    if (header.rank != 3 || shape[0] != rx || shape[1] != count || shape[2] != header.subcarrier_count ||
        strides[0] != shape[1] * shape[2] || strides[1] != shape[2] || strides[2] != 1 || header.dtype != obs::DTYPE_C64) {
      say(reason, "the shape is not [rx_plane][reference_symbol][pilot]");
      return false;
    }
    elements = static_cast<std::uint64_t>(shape[0]) * shape[1] * shape[2];
    width    = 8;
  } else if (header.layout == obs::LAYOUT_AERIAL_HEST_C64) {
    if (header.rank != 4 || shape[0] != count || shape[1] != header.subcarrier_count || shape[2] != rx || shape[3] != 1 ||
        strides[0] != shape[1] * shape[2] || strides[1] != shape[2] || strides[2] != 1 || strides[3] != 1 ||
        header.dtype != obs::DTYPE_C64) {
      say(reason, "the shape is not [dmrs][subcarrier][rx_antenna][layer] of one layer");
      return false;
    }
    elements = static_cast<std::uint64_t>(shape[0]) * shape[1] * shape[2];
    width    = 8;
  } else {
    say(reason, "the core has no ingest unit for the layout %s", layout_name(header.layout));
    return false;
  }
  if (header.payload_bytes != width * elements || header.mask_bytes != (elements + 7) / 8) {
    say(reason, "the payload or the validity bits do not have the size of the shape");
    return false;
  }
  made.layout    = header.layout;
  made.planes    = rx;
  made.positions = header.subcarrier_count;
  made.symbols   = count;
  // The receive planes in rising order of their array element.
  std::uint32_t order[obs::MAX_PLANES];
  for (unsigned i = 0; i != rx; ++i) {
    order[i] = i;
  }
  std::sort(order, order + rx,
            [&](std::uint32_t a, std::uint32_t b) { return header.antenna_map[a] < header.antenna_map[b]; });
  for (unsigned a = 0; a != rx; ++a) {
    made.plane_of[a] = order[a];
  }
  return true;
}

/// refusal() of the reference: why a record is no input of the specification. Nothing of the
/// payload is read. Returns true when the record is refused.
bool refused(const sense_plan& plan, const obs::record_header& header, record_geometry& geometry, std::uint32_t& offset,
             char* reason)
{
  const artifact& model = plan.model;
  if (header.stage != plan.expected_stage || header.source != obs::SOURCE_PUSCH_DMRS ||
      header.layout != plan.expected_layout || header.dtype != plan.expected_dtype) {
    say(reason, "the record is %s of %s in the layout %s as %s, and the specification reads %s of %s in the layout %s",
        stage_name(header.stage), source_name(header.source), layout_name(header.layout), dtype_name(header.dtype),
        model.input_stage.c_str(), model.input_source.c_str(), model.input_layout.c_str());
    return true;
  }
  if (!geometry_of(header, geometry, reason)) {
    return true;
  }
  if (header.layout == obs::LAYOUT_OCUDU_O3_FP16 &&
      ((header.producer_steps & (obs::STEP_PILOT_LS | obs::STEP_FREQ_INTERP)) != (obs::STEP_PILOT_LS | obs::STEP_FREQ_INTERP) ||
       (header.producer_steps & obs::STEP_TIME_INTERP) != 0)) {
    say(reason, "the estimate is not the stage after frequency interpolation and before time interpolation");
    return true;
  }
  if (geometry.planes != model.rx || geometry.symbols != model.dmrs_symbols) {
    say(reason, "the record holds %u receive planes and %u DM-RS symbols, and the specification %u and %u", geometry.planes,
        geometry.symbols, model.rx, model.dmrs_symbols);
    return true;
  }
  if (header.subcarrier_step != model.subcarrier_step) {
    say(reason, "the record has a subcarrier step of %u, and the support of the specification %u", header.subcarrier_step,
        model.subcarrier_step);
    return true;
  }
  // place() of the reference: where the allocation lies on the support.
  const std::int64_t distance = static_cast<std::int64_t>(header.first_subcarrier) - model.first_subcarrier;
  const std::int64_t step     = model.subcarrier_step;
  if (distance < 0 || distance % step != 0 || distance / step + header.subcarrier_count > model.subcarriers) {
    say(reason,
        "an allocation of %u positions from subcarrier %u does not lie on the support: %u positions from subcarrier %u "
        "at a step of %u",
        header.subcarrier_count, header.first_subcarrier, model.subcarriers, model.first_subcarrier, model.subcarrier_step);
    return true;
  }
  offset = static_cast<std::uint32_t>(distance / step);
  unsigned       symbols[SYMBOLS_PER_SLOT];
  const unsigned count = symbols_of(header, symbols);
  if (header.time_unit != obs::TIME_TC ||
      header.t_first_sample != symbol_time_tc(plan.mu, header.absolute_slot, symbols[0]) ||
      header.t_last_sample != symbol_time_tc(plan.mu, header.absolute_slot, symbols[count - 1]) + useful_tc(plan.mu)) {
    say(reason, "the sample times of the record are not those of its slot and its DM-RS symbols at the subcarrier spacing "
                "of the specification");
    return true;
  }
  return false;
}

void release(sense_plan& plan, held_view& held)
{
  if (held.used) {
    if (held.view.release != nullptr) {
      held.view.release(held.view.owner);
    }
    held.used = false;
    --plan.counters.views_held;
  }
}

void release(sense_plan& plan, window_slots& window)
{
  for (std::uint32_t index = 0; index != plan.need; ++index) {
    release(plan, window.slot[index]);
  }
  window.active = false;
  window.have   = 0;
}

void trigger_of(sense_answer& answer, const obs::record_header& header, std::int64_t t_notify)
{
  answer.has_trigger         = 1;
  answer.trigger_epoch       = header.epoch;
  answer.trigger_sequence    = header.sequence;
  answer.trigger_opportunity = header.opportunity;
  answer.trigger_slot        = header.absolute_slot;
  answer.t_ready_ns          = header.t_ready_ns;
  answer.t_publish_ns        = header.t_publish_ns;
  answer.t_notify_ns         = t_notify;
}

void window_of(sense_answer& answer, const sense_plan& plan, const window_slots& window)
{
  answer.has_window = 1;
  answer.epoch      = window.epoch;
  answer.episode    = window.episode;
  answer.period     = window.period;
  answer.have       = window.have;
  answer.need       = plan.need;
}

template <typename T>
void contributors_of(T& out, const sense_plan& plan, const window_slots& window)
{
  out.contributors = 0;
  for (std::uint32_t index = 0; index != plan.need; ++index) {
    const held_view& held = window.slot[index];
    if (!held.used) {
      continue;
    }
    sense_contributor& item = out.contributing[out.contributors++];
    item.sequence           = held.header.sequence;
    item.opportunity        = held.header.opportunity;
    item.absolute_slot      = held.header.absolute_slot;
    item.first_position     = held.offset;
    item.positions          = held.header.subcarrier_count;
  }
  out.t_first_sample = window.slot[0].used ? window.slot[0].header.t_first_sample : 0;
  out.t_last_sample  = window.slot[plan.need - 1].used ? window.slot[plan.need - 1].header.t_last_sample : 0;
}

/// A window that closed without one of its slots.
void gap_of(sense_answer& answer, const sense_plan& plan, const window_slots& window)
{
  std::memset(&answer, 0, sizeof(answer));
  answer.kind = SENSE_ANSWER_GAP;
  window_of(answer, plan, window);
  for (std::uint32_t index = 0; index != plan.need; ++index) {
    if (!window.slot[index].used) {
      answer.missing_slots[answer.missing++] = plan.model.slots[index];
    }
  }
  say(answer.reason, "%u of %u slots of the window never arrived", plan.need - window.have, plan.need);
}

/// The array elements of a record, rising: those behind its receive planes.
void elements_of(const sense_plan& plan, const obs::record_header& header, std::uint16_t out[obs::MAX_PLANES])
{
  std::copy(header.antenna_map, header.antenna_map + plan.model.rx, out);
  std::sort(out, out + plan.model.rx);
}

bool same_chain(const sense_plan& plan, const window_slots& window, char* reason)
{
  const obs::record_header& first = window.slot[0].header;
  std::uint16_t             elements[obs::MAX_PLANES], other[obs::MAX_PLANES];
  elements_of(plan, first, elements);
  for (std::uint32_t index = 1; index != plan.need; ++index) {
    const obs::record_header& header = window.slot[index].header;
    elements_of(plan, header, other);
    if (header.producer_steps != first.producer_steps || !std::equal(elements, elements + plan.model.rx, other)) {
      say(reason, "the slots of the window were not estimated on one array or by one chain of steps");
      return false;
    }
  }
  // Model.decide of the reference: the window must be the input the model was trained on.
  bool trained = first.producer_steps == plan.expected_steps;
  for (std::uint32_t a = 0; trained && a != plan.model.rx; ++a) {
    trained = elements[a] == plan.model.elements[a];
  }
  if (!trained) {
    say(reason, "the window was estimated on other array elements or by another chain of steps than the model was "
                "trained on");
    return false;
  }
  return true;
}

void put_stage(std::string& out, const char* stage, const std::string& entry, const unit* item, bool first)
{
  if (!first) {
    out += ", ";
  }
  out += "{\"stage\": ";
  json::put_string(out, stage);
  out += ", \"entry\": ";
  json::put_string(out, entry);
  out += ", \"runs_on\": ";
  json::put_string(out, item->runs_on);
  out += ", \"precision\": ";
  json::put_string(out, item->precision);
  out += ", \"unit_sha256\": ";
  json::put_string(out, item->sha256);
  out += "}";
}

} // namespace

std::int64_t now(const sense_plan& plan)
{
  return plan.clock(plan.clock_owner);
}

sense_status refuse(sense_status status, const std::string& text)
{
  last_error = text;
  return status;
}

} // namespace sense

using namespace sense;

extern "C" {

const char* sense_last_error(void)
{
  return last_error.c_str();
}

const char* sense_core_build_json(void)
{
  return build_json().c_str();
}

const char* sense_core_registry_json(void)
{
  return registry_json().c_str();
}

const char* sense_core_tolerance_json(void)
{
  return tolerance_json().c_str();
}

void sense_sha256_hex(const void* data, size_t bytes, char out[SENSE_HASH_BYTES])
{
  const std::string text = sha256_hex(data, bytes);
  std::memcpy(out, text.c_str(), SENSE_HASH_BYTES);
}

sense_status sense_plan_create(const char* artifact_dir, const sense_options* options, sense_plan** out)
{
  if (artifact_dir == nullptr || options == nullptr || out == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "a plan needs an artifact directory, its options and a place for its handle");
  }
  *out = nullptr;
  if (options->api_version != SENSE_CORE_API_VERSION) {
    return refuse(SENSE_E_ARGUMENT, "the caller was built for version " + std::to_string(options->api_version) +
                                        " of the API, and this core is version " + std::to_string(SENSE_CORE_API_VERSION));
  }
  std::unique_ptr<sense_plan> plan(new sense_plan());
  std::memset(&plan->counters, 0, sizeof(plan->counters));
  std::memset(&plan->geometry, 0, sizeof(plan->geometry));
  std::memset(plan->payloads, 0, sizeof(plan->payloads));
  try {
    plan->model = load_artifact(artifact_dir, options->expect_sha256);
  } catch (const artifact_error& fault) {
    return refuse(SENSE_E_ARTIFACT, fault.what());
  }
  const artifact& model = plan->model;

  // One registry: a plan is a list of names, and a name without a unit is refused.
  const std::pair<const char*, const std::string*> named[] = {
      {"input", &model.entry_input},           {"ingest", &model.entry_ingest},     {"window", &model.entry_window},
      {"clutter", &model.entry_clutter},       {"symbols", &model.entry_symbols},   {"normalizer", &model.entry_normalizer},
      {"model", &model.entry_model},           {"detector", &model.entry_detector}, {"smoothing", &model.entry_smoothing}};
  for (const auto& stage : named) {
    if (find_unit(stage.first, *stage.second) == nullptr) {
      return refuse(SENSE_E_REGISTRY, std::string("the artifact names the ") + stage.first + " entry " + *stage.second +
                                          ", and this core has no unit of that name");
    }
  }
  plan->mu = numerology(model.scs_khz);
  if (plan->mu < 0 || model.period_slots % (1 << plan->mu) != 0 || model.slots.size() > SENSE_MAX_WINDOW_RECORDS) {
    return refuse(SENSE_E_ARTIFACT, "the specification names a subcarrier spacing, a period or a window this core does "
                                    "not hold: at most " + std::to_string(SENSE_MAX_WINDOW_RECORDS) + " slots of a period "
                                    "of whole subframes");
  }
  plan->need            = static_cast<std::uint32_t>(model.slots.size());
  plan->expected_stage  = model.input_stage == "O3" ? obs::STAGE_O3 : model.input_stage == "O2" ? obs::STAGE_O2 : 0;
  plan->expected_layout = model.input_layout == "ocudu_o3_fp16"       ? obs::LAYOUT_OCUDU_O3_FP16
                          : model.input_layout == "ocudu_o2_comb_c64" ? obs::LAYOUT_OCUDU_O2_COMB_C64
                          : model.input_layout == "aerial_hest_c64"   ? obs::LAYOUT_AERIAL_HEST_C64
                                                                      : 0;
  plan->expected_dtype  = model.input_dtype == "<cfp16" ? obs::DTYPE_CFP16 : model.input_dtype == "<c8" ? obs::DTYPE_C64 : 0;
  if (plan->expected_stage == 0 || plan->expected_layout == 0 || plan->expected_dtype == 0 ||
      model.input_source != "pusch_dmrs") {
    return refuse(SENSE_E_REGISTRY, "the specification reads " + model.input_stage + " of " + model.input_source +
                                        " in the layout " + model.input_layout + ", and this core has no ingest unit for it");
  }
  try {
    plan->expected_steps = steps_of_names(model.producer_steps);
  } catch (const artifact_error& fault) {
    return refuse(SENSE_E_ARTIFACT, fault.what());
  }

  plan->clock       = options->clock != nullptr ? options->clock : monotonic;
  plan->clock_owner = options->clock_owner;
  plan->keep_mean   = options->keep_mean != 0;
  plan->budget_ingest  = options->budget_ingest_ns;
  plan->budget_kernels = options->budget_kernels_ns;
  plan->budget_engine  = options->budget_engine_ns;
  plan->budget_window  = options->budget_window_ns;

  // The device.
  const std::string wanted_device = options->device != nullptr && *options->device != '\0'
                                        ? options->device
                                        : (built_with_cuda() ? "cuda" : "host");
  try {
    if (wanted_device == "host") {
      plan->dev = make_host_device(plan->clock, plan->clock_owner);
    } else if (wanted_device == "cuda" || wanted_device.compare(0, 5, "cuda:") == 0) {
      const int index = wanted_device.size() > 5 ? std::atoi(wanted_device.c_str() + 5) : 0;
      plan->dev       = make_cuda_device(index, options->stream);
    } else {
      return refuse(SENSE_E_ARGUMENT, "the device is \"host\", \"cuda\" or \"cuda:N\", and \"" + wanted_device +
                                          "\" was asked for");
    }
    // The workspaces: every byte a window will ever need.
    device_layout& layout = plan->layout;
    layout.rx             = model.rx;
    layout.positions      = model.subcarriers;
    layout.symbols        = model.dmrs_symbols;
    layout.records        = plan->need;
    layout.cells          = model.rows * model.cols;
    const std::size_t elements = layout.elements();
    const std::size_t width    = plan->expected_dtype == obs::DTYPE_CFP16 ? 4 : 8;
    const auto        align    = [](std::size_t value) { return (value + 63) & ~static_cast<std::size_t>(63); };
    const std::size_t record   = align((elements + 7) / 8) + align(width * elements);
    // The warm-up stages the template as one complex64 record.
    layout.staging_bytes = std::max(plan->need * record, align((elements + 7) / 8) + align(8 * elements));
    lay_out(layout);
    // |H_B| in double precision, once, and a negative value where the template holds no sample.
    std::vector<float> magnitude(elements);
    for (std::size_t at = 0; at != elements; ++at) {
      magnitude[at] = model.template_count[at] == 0
                          ? -1.0f
                          : static_cast<float>(std::hypot(static_cast<double>(model.template_values[2 * at]),
                                                          static_cast<double>(model.template_values[2 * at + 1])));
    }
    plan->dev->configure(layout, magnitude.data());
  } catch (const device_error& fault) {
    return refuse(SENSE_E_DEVICE, fault.what());
  }

  // The engine.
  const bool        on_cuda = std::strcmp(plan->dev->kind(), "cuda") == 0;
  const std::string backend = options->engine != nullptr && *options->engine != '\0'
                                  ? options->engine
                                  : (on_cuda && !tensorrt_build_version().empty() ? "tensorrt" : "host_reference");
  try {
    if (backend == "host_reference") {
      plan->eng = make_host_reference_engine(model);
    } else if (backend == "tensorrt" || backend == "libtorch") {
      if (options->engine_path == nullptr || *options->engine_path == '\0') {
        return refuse(SENSE_E_ENGINE, "the backend " + backend + " needs the directory of its engine");
      }
      plan->eng = backend == "tensorrt" ? make_tensorrt_engine(options->engine_path, model, *plan->dev, options->use_graph != 0)
                                        : make_libtorch_engine(options->engine_path, model, *plan->dev);
    } else {
      return refuse(SENSE_E_ARGUMENT, "the engine is \"tensorrt\", \"libtorch\" or \"host_reference\", and \"" + backend +
                                          "\" was asked for");
    }
    engine_binding binding;
    std::uint8_t*  host_block = plan->dev->output_writable();
    if (plan->eng->on_stream()) {
      std::uint8_t* block = plan->dev->engine_block();
      binding.input       = reinterpret_cast<const float*>(block + plan->layout.input);
      binding.map         = reinterpret_cast<float*>(block + plan->layout.map);
      binding.logit       = reinterpret_cast<float*>(block + plan->layout.logit);
    } else {
      std::uint8_t* from = plan->eng->reads_device() ? plan->dev->engine_block() : host_block;
      binding.input      = reinterpret_cast<const float*>(from + plan->layout.input);
      binding.map        = reinterpret_cast<float*>(host_block + plan->layout.map);
      binding.logit      = reinterpret_cast<float*>(host_block + plan->layout.logit);
    }
    binding.stream = plan->dev->stream();
    plan->eng->bind(binding);
  } catch (const engine_error& fault) {
    return refuse(SENSE_E_ENGINE, fault.what());
  } catch (const artifact_error& fault) {
    return refuse(SENSE_E_ENGINE, fault.what());
  }

  if (model.entry_smoothing != "none") {
    const double period_s = static_cast<double>((static_cast<std::uint64_t>(model.period_slots) >> plan->mu) *
                                                (TC_PER_SECOND / 1000)) /
                            static_cast<double>(TC_PER_SECOND);
    plan->smooth.reset(new smoother(model.smoothing_maps, model.smoothing_process, model.smoothing_measurement,
                                    model.smoothing_speed, model.rows, model.cols, model.cell_m, model.origin_m, period_s));
  }

  // What a manifest names of this plan.
  plan->device_name        = plan->dev->name();
  plan->device_description = plan->dev->description();
  plan->engine_sha256      = plan->eng->sha256();
  std::string& text        = plan->plan_json;
  text                     = "{\"schema\": \"sense-plan/v1\", \"runtime\": \"native\", \"stages\": [";
  bool first               = true;
  for (const auto& stage : named) {
    put_stage(text, stage.first, *stage.second, find_unit(stage.first, *stage.second), first);
    first = false;
  }
  text += "], \"device\": ";
  text += plan->dev->identity_json();
  text += ", \"engine_backend\": ";
  json::put_string(text, plan->eng->backend());
  text += ", \"engine_sha256\": ";
  json::put_string(text, plan->engine_sha256);
  text += ", \"artifact_sha256\": ";
  json::put_string(text, model.sha256);
  text += ", \"core_sources_sha256\": ";
  json::put_string(text, core_sources_sha256());
  text += ", \"submission\": \"one upload, the ingest kernel, the front-end kernel, the engine and one download, behind "
          "one completion event\"}";
  plan->plan_sha256 = sha256_hex(text);
  // The same entries by stage, as the report of a sensor names them.
  std::string& entries = plan->entries_json;
  entries              = "{";
  first                = true;
  for (const auto& stage : named) {
    const unit* item = find_unit(stage.first, *stage.second);
    entries += first ? "\"" : ", \"";
    entries += stage.first;
    entries += "\": {\"entry\": ";
    json::put_string(entries, *stage.second);
    entries += ", \"runtime\": \"native\", \"runs_on\": ";
    json::put_string(entries, item->runs_on);
    entries += ", \"precision\": ";
    json::put_string(entries, item->precision);
    entries += ", \"unit_sha256\": ";
    json::put_string(entries, item->sha256);
    entries += "}";
    first = false;
  }
  entries += "}";
  plan->engine_detail = plan->eng->detail();
  *out = plan.release();
  return SENSE_OK;
}

void sense_plan_destroy(sense_plan* plan)
{
  if (plan == nullptr) {
    return;
  }
  release(*plan, plan->filling);
  release(*plan, plan->ready);
  release(*plan, plan->running);
  delete plan;
}

sense_status sense_plan_info_get(const sense_plan* plan, sense_plan_info* info)
{
  if (plan == nullptr || info == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "a plan and a place for its facts are needed");
  }
  const artifact& model  = plan->model;
  info->rx               = model.rx;
  info->subcarriers      = model.subcarriers;
  info->dmrs_symbols     = model.dmrs_symbols;
  info->window_records   = plan->need;
  info->rows             = model.rows;
  info->cols             = model.cols;
  info->period_slots     = static_cast<std::uint32_t>(model.period_slots);
  info->scs_khz          = static_cast<std::uint32_t>(model.scs_khz);
  info->threshold_logit  = model.threshold_logit;
  info->model_id         = model.model_id.c_str();
  info->artifact_sha256  = model.sha256.c_str();
  info->spec_sha256      = model.spec_sha256.c_str();
  info->device           = plan->device_name.c_str();
  info->device_name      = plan->device_description.c_str();
  info->engine_backend   = plan->eng->backend();
  info->engine_detail    = plan->engine_detail.c_str();
  info->engine_sha256    = plan->engine_sha256.c_str();
  info->core_sources_sha256 = core_sources_sha256();
  info->plan_json        = plan->plan_json.c_str();
  info->entries_json     = plan->entries_json.c_str();
  info->plan_sha256      = plan->plan_sha256.c_str();
  info->engine_on_stream = plan->eng->on_stream() ? 1 : 0;
  info->engine_graph     = plan->eng->graph() ? 1 : 0;
  info->timing_valid     = std::strcmp(plan->dev->kind(), "cuda") == 0 && std::strcmp(plan->eng->backend(), "tensorrt") == 0;
  return SENSE_OK;
}

sense_status sense_plan_warm_up(sense_plan* plan, uint32_t rounds, int64_t* elapsed_ns)
{
  if (plan == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "no plan");
  }
  if (plan->running.active) {
    return refuse(SENSE_E_STATE, "a window is running: the warm-up belongs before the first record");
  }
  const std::int64_t started = now(*plan);
  const artifact&    model   = plan->model;
  try {
    // One record that equals the template, staged as complex64 in the layout of the comb, with
    // every bit set where the template holds a sample.
    const std::size_t elements   = plan->layout.elements();
    const std::size_t payload_at = ((elements + 7) / 8 + 63) & ~static_cast<std::size_t>(63);
    std::uint8_t*     staging    = plan->dev->staging();
    window_geometry   window;
    std::memset(&window, 0, sizeof(window));
    window.records   = 1;
    window.rx        = model.rx;
    window.positions = model.subcarriers;
    window.symbols   = model.dmrs_symbols;
    window.tau2      = model.tau * model.tau;
    window.mean      = static_cast<float>(model.norm_mean);
    window.divisor   = static_cast<float>(model.norm_scale + model.norm_epsilon);
    record_geometry& record = window.record[0];
    record.layout           = LAYOUT_O2_COMB_C64;
    record.planes           = model.rx;
    record.positions        = model.subcarriers;
    record.symbols          = model.dmrs_symbols;
    record.payload_offset   = static_cast<std::uint32_t>(payload_at);
    for (std::uint32_t a = 0; a != model.rx; ++a) {
      record.plane_of[a] = a;
    }
    std::memset(staging, 0, payload_at);
    for (std::uint32_t a = 0; a != model.rx; ++a) {
      for (std::uint32_t k = 0; k != model.subcarriers; ++k) {
        for (std::uint32_t s = 0; s != model.dmrs_symbols; ++s) {
          const std::size_t   from    = (static_cast<std::size_t>(a) * model.subcarriers + k) * model.dmrs_symbols + s;
          const std::uint32_t element = element_of(record, a, k, s);
          std::memcpy(staging + payload_at + 8 * static_cast<std::size_t>(element), &model.template_values[2 * from], 8);
          if (model.template_count[from] != 0) {
            staging[element >> 3] = static_cast<std::uint8_t>(staging[element >> 3] | (1U << (element & 7U)));
          }
        }
      }
    }
    const void* none[MAX_WINDOW_RECORDS] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    plan->eng->prepare(); // a first use, and the capture of a graph, before any window
    for (uint32_t round = 0; round < (rounds == 0 ? 1U : rounds); ++round) {
      plan->dev->begin(window, none, payload_at + 8 * elements);
      if (plan->eng->on_stream()) {
        plan->eng->enqueue();
      }
      plan->dev->end(plan->keep_mean);
      while (!plan->dev->done()) {
        // Configure time: this loop may spin for as long as the device needs.
      }
      if (!plan->eng->on_stream()) {
        plan->eng->infer();
      }
    }
  } catch (const device_error& fault) {
    return refuse(SENSE_E_DEVICE, fault.what());
  } catch (const engine_error& fault) {
    return refuse(SENSE_E_ENGINE, fault.what());
  }
  plan->engine_detail = plan->eng->detail(); // the graph, or why there is none, is known now
  if (elapsed_ns != nullptr) {
    *elapsed_ns = now(*plan) - started;
  }
  return SENSE_OK;
}

sense_status sense_plan_offer(sense_plan* plan, const sense_view* view, sense_answer answers[2], uint32_t* count)
{
  if (plan == nullptr || view == nullptr || answers == nullptr || count == nullptr || view->record == nullptr ||
      view->payload == nullptr || view->mask == nullptr ||
      (view->memory != SENSE_MEMORY_HOST_SHARED && view->memory != SENSE_MEMORY_HOST_PINNED &&
       view->memory != SENSE_MEMORY_DEVICE)) {
    return refuse(SENSE_E_ARGUMENT, "an offer needs a plan, a place for two answers, and a view that names its record, "
                                    "its payload, its validity bits and where they live");
  }
  *count                          = 0;
  const std::uint64_t calls_before = plan->dev->calls;
  ++plan->counters.records;
  held_view held;
  held.view   = *view;
  held.header = read_header(view->record);
  held.used   = true;
  ++plan->counters.views_held;
  const obs::record_header& header = held.header;

  record_geometry geometry;
  sense_answer&   answer = answers[0];
  std::memset(&answer, 0, sizeof(answer));
  if (header.schema != obs::RECORD_SCHEMA || header.header_bytes != obs::RECORD_HEADER_BYTES) {
    answer.kind = SENSE_ANSWER_REJECTED;
    say(answer.reason, "the record is not an obs/v1 record: schema %u, header of %u bytes", header.schema, header.header_bytes);
    trigger_of(answer, header, view->t_notify_ns);
    answer.need = plan->need;
    ++plan->counters.rejected;
    release(*plan, held);
    *count = 1;
    return SENSE_OK;
  }
  if (refused(*plan, header, geometry, held.offset, answer.reason)) {
    answer.kind = SENSE_ANSWER_REJECTED;
    trigger_of(answer, header, view->t_notify_ns);
    answer.need = plan->need;
    ++plan->counters.rejected;
    release(*plan, held);
    *count = 1;
    return SENSE_OK;
  }
  // Another epoch, another episode or a time that went back: a new history.
  const std::int64_t slot = static_cast<std::int64_t>(header.absolute_slot);
  if (!plan->has_key || header.epoch != plan->key_epoch || header.episode != plan->key_episode || slot <= plan->last_slot) {
    if (plan->has_key) {
      ++plan->counters.resets;
    }
    release(*plan, plan->filling);
    plan->has_key     = true;
    plan->key_epoch   = header.epoch;
    plan->key_episode = header.episode;
  }
  plan->last_slot              = slot;
  const std::uint64_t period   = header.absolute_slot / static_cast<std::uint64_t>(plan->model.period_slots);
  const std::uint32_t position = static_cast<std::uint32_t>(header.absolute_slot % static_cast<std::uint64_t>(plan->model.period_slots));
  std::uint32_t       index    = plan->need;
  for (std::uint32_t n = 0; n != plan->need; ++n) {
    if (plan->model.slots[n] == position) {
      index = n;
    }
  }
  if (index == plan->need) {
    // A slot outside the window may carry another allocation. It is not looked at further.
    ++plan->counters.passed;
    release(*plan, held);
    return SENSE_OK;
  }

  window_slots& filling = plan->filling;
  if (filling.active && filling.period != period) {
    // A window whose last slot never came is closed by the first record of a later period.
    gap_of(answers[*count], *plan, filling);
    ++*count;
    ++plan->counters.gap;
    release(*plan, filling);
  }
  if (!filling.active) {
    filling.active  = true;
    filling.epoch   = header.epoch;
    filling.episode = header.episode;
    filling.period  = period;
    filling.have    = 0;
  }
  filling.slot[index] = held;
  ++filling.have;

  sense_answer& last = answers[*count];
  std::memset(&last, 0, sizeof(last));
  window_of(last, *plan, filling);
  trigger_of(last, header, view->t_notify_ns);
  if (index + 1 != plan->need) {
    last.kind = SENSE_ANSWER_FILLING;
    say(last.reason, "the window is filling");
    ++plan->counters.filling;
    ++*count;
  } else if (filling.have != plan->need) {
    gap_of(last, *plan, filling);
    trigger_of(last, header, view->t_notify_ns);
    ++plan->counters.gap;
    release(*plan, filling);
    ++*count;
  } else if (!same_chain(*plan, filling, last.reason)) {
    last.kind = SENSE_ANSWER_INVALID;
    contributors_of(last, *plan, filling);
    ++plan->counters.invalid;
    release(*plan, filling);
    ++*count;
  } else {
    // The window is complete. No answer precedes this one: a gap of an earlier period is answered
    // when the first slot of this period arrives, and that slot does not complete a window of
    // more than one slot, while a window of one slot is never open across two records.
    sense_answer done = last;
    done.kind         = SENSE_ANSWER_READY;
    contributors_of(done, *plan, filling);
    *count = 0;
    if (plan->ready.active) {
      // A complete window that did not run before a newer one completed: it is overtaken. The
      // sensor skips to the newest complete window and never works through a backlog.
      sense_answer& skipped = answers[0];
      std::memset(&skipped, 0, sizeof(skipped));
      skipped.kind = SENSE_ANSWER_STALE_SKIPPED;
      window_of(skipped, *plan, plan->ready);
      contributors_of(skipped, *plan, plan->ready);
      const held_view& closing = plan->ready.slot[plan->need - 1];
      trigger_of(skipped, closing.header, closing.view.t_notify_ns);
      say(skipped.reason, "a newer window was complete before this one ran: the sensor does not work through a backlog");
      ++plan->counters.stale_skipped;
      plan->counters.stale_skipped_records += plan->need;
      release(*plan, plan->ready);
      *count = 1;
    }
    answers[*count] = done;
    plan->ready     = filling;
    filling.active  = false;
    filling.have    = 0;
    for (std::uint32_t n = 0; n != plan->need; ++n) {
      filling.slot[n].used = false; // the views moved to the ready window
    }
    ++plan->counters.ready;
    ++*count;
  }
  plan->counters.device_calls_open += plan->dev->calls - calls_before;
  return SENSE_OK;
}

int sense_plan_ready(const sense_plan* plan)
{
  return plan != nullptr && plan->ready.active ? 1 : 0;
}

uint32_t sense_plan_held(const sense_plan* plan, uint64_t* oldest_sequence)
{
  if (plan == nullptr) {
    return 0;
  }
  std::uint32_t held   = 0;
  std::uint64_t oldest = ~0ULL;
  for (const window_slots* window : {&plan->filling, &plan->ready, &plan->running}) {
    for (std::uint32_t index = 0; index != plan->need; ++index) {
      if (window->slot[index].used) {
        ++held;
        oldest = std::min<std::uint64_t>(oldest, window->slot[index].header.sequence);
      }
    }
  }
  if (oldest_sequence != nullptr && held != 0) {
    *oldest_sequence = oldest;
  }
  return held;
}

int64_t sense_plan_ready_time(const sense_plan* plan)
{
  return plan != nullptr && plan->ready.active ? plan->ready.slot[plan->need - 1].header.t_ready_ns : 0;
}

sense_status sense_plan_submit(sense_plan* plan)
{
  if (plan == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "no plan");
  }
  if (!plan->ready.active || plan->running.active) {
    return refuse(SENSE_E_STATE, plan->running.active ? "a window is running: collect it first"
                                                      : "no complete window waits for its submission");
  }
  plan->running      = plan->ready;
  plan->ready.active = false;
  plan->ready.have   = 0;
  for (std::uint32_t n = 0; n != plan->need; ++n) {
    plan->ready.slot[n].used = false;
  }
  plan->submitted  = true;
  plan->stale      = false;
  plan->engine_ran = false;
  plan->complete   = false;
  const artifact& model = plan->model;

  // The geometry of the window, and the ingest copy of its host views into the staging block.
  window_geometry& window = plan->geometry;
  std::memset(&window, 0, sizeof(window));
  window.records     = plan->need;
  window.rx          = model.rx;
  window.positions   = model.subcarriers;
  window.symbols     = model.dmrs_symbols;
  window.tau2        = model.tau * model.tau;
  window.mean        = static_cast<float>(model.norm_mean);
  window.divisor     = static_cast<float>(model.norm_scale + model.norm_epsilon);
  std::uint8_t* staging = plan->dev->staging();
  std::size_t   at      = 0;
  char          unused[SENSE_TEXT_BYTES];
  plan->evidence        = 0;
  for (std::uint32_t r = 0; r != plan->need; ++r) {
    const held_view&  held   = plan->running.slot[r];
    record_geometry&  record = window.record[r];
    geometry_of(held.header, record, unused);
    record.offset      = held.offset;
    record.mask_offset = static_cast<std::uint32_t>(at);
    std::memcpy(staging + at, held.view.mask, held.header.mask_bytes);
    at = (at + held.header.mask_bytes + 63) & ~static_cast<std::size_t>(63);
    if (held.view.memory == SENSE_MEMORY_DEVICE) {
      record.on_device  = 1;
      plan->payloads[r] = held.view.payload;
    } else {
      record.payload_offset = static_cast<std::uint32_t>(at);
      plan->payloads[r]     = nullptr;
      std::memcpy(staging + at, held.view.payload, held.header.payload_bytes);
      at = (at + held.header.payload_bytes + 63) & ~static_cast<std::size_t>(63);
    }
    plan->evidence |= held.header.evidence == obs::EVIDENCE_AUDIT ? 1U : held.header.evidence == obs::EVIDENCE_TIMED ? 2U : 4U;
  }
  const obs::record_header& first = plan->running.slot[0].header;
  plan->port                      = first.port;
  plan->scale                     = first.scale;
  plan->producer_steps            = first.producer_steps;
  // The copy above is the ingest of a host view. A record whose memory changed meanwhile yields
  // no decision, and nothing of its window reaches the device.
  for (std::uint32_t r = 0; r != plan->need; ++r) {
    const sense_view& view = plan->running.slot[r].view;
    if (view.intact != nullptr && view.intact(view.owner) == 0) {
      plan->stale = true;
    }
  }
  plan->t_submit = now(*plan);
  if (plan->stale) {
    plan->complete = true;
    return SENSE_OK;
  }
  try {
    plan->dev->begin(window, plan->payloads, at);
    if (plan->eng->on_stream()) {
      plan->eng->enqueue();
    }
    plan->dev->end(plan->keep_mean);
  } catch (const engine_error& fault) {
    release(*plan, plan->running); // the window is given up: nothing of it may be published
    plan->submitted = false;
    return refuse(SENSE_E_ENGINE, fault.what());
  } catch (const device_error& fault) {
    release(*plan, plan->running);
    plan->submitted = false;
    return refuse(SENSE_E_DEVICE, fault.what());
  }
  ++plan->counters.submissions;
  return SENSE_OK;
}

int sense_plan_poll(sense_plan* plan)
{
  if (plan == nullptr || !plan->running.active) {
    return 0;
  }
  if (plan->complete) {
    return 1;
  }
  try {
    if (!plan->dev->done()) {
      return 0;
    }
    if (!plan->eng->on_stream() && !plan->engine_ran) {
      plan->eng->infer(); // a blocking backend: after the kernels, on the calling thread
      plan->engine_ran    = true;
      plan->t_engine_host = now(*plan);
    }
  } catch (const std::exception& fault) {
    refuse(SENSE_E_DEVICE, fault.what());
    plan->stale = true; // no score may be published for this window
  }
  plan->complete = true;
  return 1;
}

sense_status sense_plan_collect(sense_plan* plan, sense_decision* decision)
{
  if (plan == nullptr || decision == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "a plan and a place for its decision are needed");
  }
  if (!plan->running.active || !plan->complete) {
    return refuse(SENSE_E_STATE, plan->running.active ? "the window has not completed: poll it first"
                                                      : "no window is running");
  }
  const artifact&     model  = plan->model;
  const window_slots& window = plan->running;
  std::memset(decision, 0, sizeof(*decision));
  const std::int64_t seen = now(*plan);
  decision->epoch         = window.epoch;
  decision->episode       = window.episode;
  decision->period        = window.period;
  const held_view& closing = window.slot[plan->need - 1];
  decision->trigger_sequence    = closing.header.sequence;
  decision->trigger_opportunity = closing.header.opportunity;
  decision->trigger_slot        = closing.header.absolute_slot;
  decision->t_ready_ns          = closing.header.t_ready_ns;
  decision->t_publish_ns        = closing.header.t_publish_ns;
  decision->t_notify_ns         = closing.view.t_notify_ns;
  decision->t_submit_ns         = plan->t_submit;
  contributors_of(*decision, *plan, window);
  decision->port           = plan->port;
  decision->producer_steps = plan->producer_steps;
  decision->evidence       = plan->evidence;
  decision->scale          = plan->scale;
  const std::uint8_t* out = plan->dev->output();
  decision->elements      = static_cast<std::uint32_t>(plan->layout.elements());

  if (plan->stale) {
    decision->status = SENSE_DECISION_STALE;
    say(decision->reason, "a record of the window was rewritten while it was read, or its device work failed: no score");
    decision->t_ingest_ns = decision->t_kernels_ns = decision->t_engine_ns = plan->t_submit;
    ++plan->counters.stale;
  } else {
    stage_marks marks;
    try {
      marks = plan->dev->marks(seen);
    } catch (const std::exception& fault) {
      release(*plan, plan->running);
      return refuse(SENSE_E_DEVICE, fault.what());
    }
    if (!plan->eng->on_stream()) {
      marks.engine = plan->t_engine_host;
    }
    decision->t_ingest_ns  = marks.ingest;
    decision->t_kernels_ns = marks.kernels;
    decision->t_engine_ns  = marks.engine;
    decision->input        = reinterpret_cast<const float*>(out + plan->layout.input);
    decision->map          = reinterpret_cast<const float*>(out + plan->layout.map);
    decision->difference   = reinterpret_cast<const float*>(out + plan->layout.difference);
    decision->count        = out + plan->layout.count;
    decision->mean         = plan->keep_mean ? reinterpret_cast<const float*>(out + plan->layout.mean) : nullptr;
    // The validity of the window.
    std::uint32_t good = 0, least = 255, most = 0;
    for (std::uint32_t at = 0; at != decision->elements; ++at) {
      const std::uint32_t records = decision->count[at];
      good += records != 0 ? 1U : 0U;
      least = std::min(least, records);
      most  = std::max(most, records);
    }
    decision->valid       = good;
    decision->records_min = least;
    decision->records_max = most;
    const float         logit = *reinterpret_cast<const float*>(out + plan->layout.logit);
    const std::uint32_t cells = model.rows * model.cols;
    bool                finite = std::isfinite(logit);
    std::uint32_t       best   = 0;
    for (std::uint32_t cell = 0; cell != cells; ++cell) {
      finite = finite && std::isfinite(decision->map[cell]);
      if (decision->map[cell] > decision->map[best]) {
        best = cell; // the first of equal values stays: the first one in row order
      }
    }
    if (static_cast<double>(good) / static_cast<double>(decision->elements) < model.min_valid_fraction) {
      decision->status = SENSE_DECISION_INVALID;
      // The share as the specification writes it: the reference prints the same text.
      say(decision->reason, "%u of %u elements of the window hold a valid sample, below the share %s of the specification",
          good, decision->elements, model.min_valid_text.c_str());
      ++plan->counters.invalid;
    } else if (!finite) {
      decision->status = SENSE_DECISION_INVALID;
      say(decision->reason, "the score of the window is not finite");
      ++plan->counters.invalid;
    } else {
      // detector/frozen_threshold: present when the logit lies above the frozen threshold.
      decision->status       = SENSE_DECISION_OK;
      decision->logit        = logit;
      decision->probability  = 1.0 / (1.0 + std::exp(-static_cast<double>(logit)));
      decision->present      = static_cast<double>(logit) > model.threshold_logit ? 1 : 0;
      decision->row          = best / model.cols;
      decision->col          = best % model.cols;
      decision->position_x_m = model.origin_m[0] + (decision->col + 0.5) * model.cell_m[1];
      decision->position_y_m = model.origin_m[1] + (decision->row + 0.5) * model.cell_m[0];
      decision->peak         = decision->map[best];
      ++plan->counters.scored;
      plan->counters.present += decision->present != 0 ? 1U : 0U;
    }
  }
  decision->t_decision_ns = now(*plan); // the publication time
  // A stage that overran its budget is counted, and the decision is published as late.
  if (decision->status == SENSE_DECISION_OK) {
    decision->overrun_ingest  = plan->budget_ingest > 0 && decision->t_ingest_ns - decision->t_submit_ns > plan->budget_ingest;
    decision->overrun_kernels = plan->budget_kernels > 0 && decision->t_kernels_ns - decision->t_ingest_ns > plan->budget_kernels;
    decision->overrun_engine  = plan->budget_engine > 0 && decision->t_engine_ns - decision->t_kernels_ns > plan->budget_engine;
    decision->overrun_window  = plan->budget_window > 0 && decision->t_notify_ns > 0 &&
                                decision->t_decision_ns - decision->t_notify_ns > plan->budget_window;
    decision->late = (decision->overrun_ingest | decision->overrun_kernels | decision->overrun_engine | decision->overrun_window) != 0;
    plan->counters.overrun_ingest += decision->overrun_ingest;
    plan->counters.overrun_kernels += decision->overrun_kernels;
    plan->counters.overrun_engine += decision->overrun_engine;
    plan->counters.overrun_window += decision->overrun_window;
    plan->counters.late += decision->late != 0 ? 1U : 0U;
  }
  ++plan->counters.decisions;
  release(*plan, plan->running); // the window, in the order it was offered
  plan->submitted = false;
  return SENSE_OK;
}

sense_status sense_plan_skip(sense_plan* plan, sense_answer* answer)
{
  if (plan == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "no plan");
  }
  if (!plan->ready.active) {
    return refuse(SENSE_E_STATE, "no complete window waits: there is nothing to skip");
  }
  if (answer != nullptr) {
    std::memset(answer, 0, sizeof(*answer));
    answer->kind = SENSE_ANSWER_STALE_SKIPPED;
    window_of(*answer, *plan, plan->ready);
    contributors_of(*answer, *plan, plan->ready);
    const held_view& closing = plan->ready.slot[plan->need - 1];
    trigger_of(*answer, closing.header, closing.view.t_notify_ns);
    say(answer->reason, "a newer window was complete before this one ran: the sensor does not work through a backlog");
  }
  ++plan->counters.stale_skipped;
  plan->counters.stale_skipped_records += plan->need;
  release(*plan, plan->ready);
  return SENSE_OK;
}

sense_status sense_plan_close_open(sense_plan* plan, sense_answer* answer)
{
  if (plan == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "no plan");
  }
  if (!plan->filling.active) {
    return refuse(SENSE_E_STATE, "no window is filling: there is nothing to close");
  }
  if (answer != nullptr) {
    gap_of(*answer, *plan, plan->filling);
  }
  ++plan->counters.gap;
  release(*plan, plan->filling);
  return SENSE_OK;
}

sense_status sense_plan_finish(sense_plan* plan, uint32_t* have, uint32_t* need)
{
  if (plan == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "no plan");
  }
  if (have != nullptr) {
    *have = plan->filling.active ? plan->filling.have : 0;
  }
  if (need != nullptr) {
    *need = plan->need;
  }
  release(*plan, plan->filling);
  return SENSE_OK;
}

sense_status sense_plan_counters(const sense_plan* plan, sense_counters* counters)
{
  if (plan == nullptr || counters == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "a plan and a place for its counters are needed");
  }
  *counters                    = plan->counters;
  counters->device_calls       = plan->dev->calls;
  counters->device_allocations = plan->dev->allocations;
  return SENSE_OK;
}

sense_status sense_plan_ingest(sense_plan* plan, const sense_view* view, float* values, uint8_t* valid, size_t elements,
                               uint32_t shape[3], uint32_t* first_position)
{
  if (plan == nullptr || view == nullptr || values == nullptr || valid == nullptr || shape == nullptr ||
      view->record == nullptr || view->payload == nullptr || view->mask == nullptr) {
    return refuse(SENSE_E_ARGUMENT, "the ingest of one record needs a plan, a view and the two arrays it fills");
  }
  if (view->memory == SENSE_MEMORY_DEVICE) {
    return refuse(SENSE_E_ARGUMENT, "the unit entry of the ingest reads host memory");
  }
  const obs::record_header header = read_header(view->record);
  record_geometry          record;
  char                     reason[SENSE_TEXT_BYTES];
  if (!geometry_of(header, record, reason)) {
    return refuse(SENSE_E_ARGUMENT, reason);
  }
  const std::size_t wanted = static_cast<std::size_t>(record.planes) * record.positions * record.symbols;
  if (elements < wanted) {
    return refuse(SENSE_E_ARGUMENT, "the arrays hold " + std::to_string(elements) + " elements, and the record " +
                                        std::to_string(wanted));
  }
  shape[0] = record.planes;
  shape[1] = record.positions;
  shape[2] = record.symbols;
  if (first_position != nullptr) {
    *first_position = header.first_subcarrier;
  }
  try {
    plan->dev->ingest_one(record, plan->model.tau * plan->model.tau, static_cast<const std::uint8_t*>(view->payload),
                          static_cast<const std::uint8_t*>(view->mask), header.payload_bytes, header.mask_bytes, values,
                          valid);
  } catch (const device_error& fault) {
    return refuse(SENSE_E_DEVICE, fault.what());
  }
  return SENSE_OK;
}

} // extern "C"
