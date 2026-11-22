// Reader of a frozen cusense-artifact/v1 artifact (Sprint 3.6, selector sense_core). ASCII only.

#include "core/artifact.h"
#include "core/sha256.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>

namespace sense {

namespace {

std::uint32_t count_of(const json::value& object, const char* key, std::int64_t low, std::int64_t high)
{
  const std::int64_t value = object.integer_at(key);
  if (value < low || value > high) {
    throw artifact_error(std::string("the member \"") + key + "\" lies outside " + std::to_string(low) + " to " +
                         std::to_string(high));
  }
  return static_cast<std::uint32_t>(value);
}

void same_hash(const std::string& named, const std::string& found, const char* what)
{
  if (named != found) {
    throw artifact_error(std::string(what) + ": the manifest names " + named + ", and the artifact has " + found);
  }
}

void fitted_on_training(const json::value& part, const char* name)
{
  const json::value& fitted = part.object_at("fitted_on");
  if (fitted.string_at("split") != "training") {
    throw artifact_error(std::string(name) + " names the " + fitted.string_at("split") +
                         " split: the template and the statistics of the z-score are fitted on windows of the training "
                         "split and on no other record");
  }
}

template <typename T>
std::vector<T> values_of(const std::string& bytes)
{
  std::vector<T> values(bytes.size() / sizeof(T));
  std::memcpy(values.data(), bytes.data(), values.size() * sizeof(T)); // little-endian hosts only, as the record is
  return values;
}

} // namespace

std::string read_file(const std::string& path)
{
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    throw artifact_error("cannot read " + path);
  }
  std::string data;
  char        chunk[65536];
  std::size_t got;
  while ((got = std::fread(chunk, 1, sizeof(chunk), file)) != 0) {
    data.append(chunk, got);
  }
  const bool failed = std::ferror(file) != 0;
  std::fclose(file);
  if (failed) {
    throw artifact_error("cannot read " + path);
  }
  return data;
}

const tensor_entry& artifact::tensor(const std::string& name) const
{
  for (const tensor_entry& entry : tensors) {
    if (entry.name == name) {
      return entry;
    }
  }
  throw artifact_error("the weight file names no tensor " + name + ": another architecture wrote it");
}

artifact load_artifact(const std::string& directory, const char* expect_sha256)
{
  artifact made;
  made.directory     = directory;
  made.manifest_text = read_file(directory + "/" + MODEL_FILE);
  while (!made.manifest_text.empty() && (made.manifest_text.back() == '\n' || made.manifest_text.back() == '\r')) {
    made.manifest_text.pop_back();
  }
  try {
    made.manifest         = json::parse(made.manifest_text);
    const json::value& m  = made.manifest;
    if (!m.is_object() || m.string_at("schema") != ARTIFACT_SCHEMA) {
      throw artifact_error(std::string("model.json is not a ") + ARTIFACT_SCHEMA + " object");
    }
    made.model_id = m.string_at("model_id");
    made.sha256   = m.string_at("sha256");
    same_hash(made.sha256, sha256_hex(json::canonical(m, "sha256")),
              "model.json does not have the hash it names: the artifact was altered after it was frozen");

    const json::value& entries = m.object_at("entries");
    made.entry_input           = entries.string_at("input");
    made.entry_ingest          = entries.string_at("ingest");
    made.entry_window          = entries.string_at("window");
    made.entry_clutter         = entries.string_at("clutter");
    made.entry_symbols         = entries.string_at("symbols");
    made.entry_normalizer      = entries.string_at("normalizer");
    made.entry_model           = entries.string_at("model");
    made.entry_detector        = entries.string_at("detector");
    made.entry_smoothing       = entries.string_at("smoothing");

    // The specification of the windows.
    const json::value& input = m.object_at("input");
    const json::value& spec  = input.object_at("spec");
    made.spec_canonical      = json::canonical(spec);
    made.spec_sha256         = input.string_at("spec_sha256");
    same_hash(made.spec_sha256, sha256_hex(made.spec_canonical), "the specification does not have the hash the artifact names");
    if (spec.string_at("schema") != "cusense-spec/v1") {
      throw artifact_error("the specification of the artifact is not a cusense-spec/v1 object");
    }
    made.spec_id               = spec.string_at("spec_id");
    const json::value& wanted  = spec.object_at("input");
    made.input_stage           = wanted.string_at("stage");
    made.input_source          = wanted.string_at("source");
    made.input_layout          = wanted.string_at("layout");
    made.input_dtype           = wanted.string_at("dtype");
    const json::value& radio   = spec.object_at("radio");
    made.scs_khz               = static_cast<int>(count_of(radio, "scs_khz", 15, 120));
    made.period_slots          = static_cast<int>(count_of(radio, "period_slots", 1, 1 << 16));
    const json::value& window  = spec.object_at("window");
    for (const json::value& slot : window.array_at("slots_in_period").items) {
      if (!slot.is_number() || !slot.integer || slot.number < 0 || slot.number >= made.period_slots ||
          (!made.slots.empty() && slot.number <= made.slots.back())) {
        throw artifact_error("window.slots_in_period must be a rising list of slots of one period");
      }
      made.slots.push_back(static_cast<std::uint32_t>(slot.number));
    }
    if (made.slots.empty()) {
      throw artifact_error("window.slots_in_period names no slot");
    }
    made.min_valid_fraction     = window.number_at("min_valid_fraction");
    made.min_valid_text         = window.at("min_valid_fraction").raw;
    const json::value& support  = spec.object_at("support");
    made.first_subcarrier       = count_of(support, "first_subcarrier", 0, 1 << 16);
    made.subcarrier_step        = count_of(support, "subcarrier_step", 1, 12);
    made.subcarriers            = count_of(support, "subcarriers", 16, 1 << 16);
    made.dmrs_symbols           = count_of(support, "dmrs_symbols", 1, 14);
    made.rx                     = count_of(support, "rx", 1, 16);
    made.tau                    = spec.object_at("validity").number_at("tau");
    const json::value& steps    = spec.object_at("steps");
    if (steps.string_at("ingest") != made.entry_ingest || steps.string_at("window") != made.entry_window ||
        steps.string_at("clutter") != made.entry_clutter || steps.string_at("symbols") != made.entry_symbols ||
        steps.string_at("normalizer") != made.entry_normalizer || wanted.string_at("entry") != made.entry_input ||
        made.entry_ingest != made.input_layout) {
      throw artifact_error("the entries of the artifact are not the steps of its specification");
    }
    for (const json::value& element : input.array_at("elements").items) {
      if (!element.is_number() || !element.integer || element.number < 0 ||
          (!made.elements.empty() && element.number <= made.elements.back())) {
        throw artifact_error("input.elements must list the array elements of the input in rising order");
      }
      made.elements.push_back(static_cast<std::uint32_t>(element.number));
    }
    for (const json::value& step : input.array_at("producer_steps").items) {
      if (!step.is_string()) {
        throw artifact_error("input.producer_steps must list the steps of the producer by name");
      }
      made.producer_steps.push_back(step.text);
    }
    const json::value& shape = input.array_at("shape");
    if (made.elements.size() != made.rx || input.string_at("dtype") != "<f4" || shape.items.size() != 3 ||
        shape.items[0].number != 1 || shape.items[1].number != made.rx || shape.items[2].number != made.subcarriers) {
      throw artifact_error("the input of a model is one window of its specification as <f4, shape [1, A, K]");
    }

    // The template and its counts.
    const std::size_t  elements = static_cast<std::size_t>(made.rx) * made.subcarriers * made.dmrs_symbols;
    const json::value& tmpl     = m.object_at("template");
    const std::string  values   = read_file(directory + "/" + TEMPLATE_FILE);
    const std::string  counted  = read_file(directory + "/" + TEMPLATE_COUNT_FILE);
    if (tmpl.string_at("file") != TEMPLATE_FILE || tmpl.string_at("count_file") != TEMPLATE_COUNT_FILE ||
        tmpl.string_at("dtype") != "<c8" || tmpl.string_at("count_dtype") != "<u4" ||
        static_cast<std::size_t>(tmpl.integer_at("bytes")) != 8 * elements ||
        static_cast<std::size_t>(tmpl.integer_at("count_bytes")) != 4 * elements) {
      throw artifact_error("the template of a model is [A, K, S] as <c8 with its counts as <u4");
    }
    made.template_sha256 = tmpl.string_at("sha256");
    if (values.size() != 8 * elements || counted.size() != 4 * elements) {
      throw artifact_error("the template or its counts do not have the size that model.json names: the background was "
                           "altered");
    }
    same_hash(made.template_sha256, sha256_hex(values), "the template does not have the hash that model.json names: the "
                                                        "background was altered");
    same_hash(tmpl.string_at("count_sha256"), sha256_hex(counted),
              "the counts of the template do not have the hash that model.json names: the background was altered");
    fitted_on_training(tmpl, "the template");
    made.template_values = values_of<float>(values);
    made.template_count  = values_of<std::uint32_t>(counted);

    // The normalizer.
    const json::value& normalizer = m.object_at("normalizer");
    made.norm_mean                = normalizer.number_at("mean");
    made.norm_scale               = normalizer.number_at("scale");
    made.norm_epsilon             = normalizer.number_at("epsilon");
    made.normalizer_sha256        = normalizer.string_at("sha256");
    static const std::vector<std::string> hashed = {"entry", "mean", "scale", "epsilon"};
    same_hash(made.normalizer_sha256, sha256_hex(json::canonical(normalizer, nullptr, &hashed)),
              "the normalizer does not have the hash the artifact names: the preprocessing state changed");
    if (normalizer.string_at("entry") != made.entry_normalizer || !(made.norm_scale > 0.0) || made.norm_epsilon != 1e-8) {
      throw artifact_error("the normalizer must be the entry of the artifact with a positive scale and its epsilon");
    }
    fitted_on_training(normalizer, "the normalizer");

    // The grid, the threshold and the optional stage.
    const json::value& grid = m.object_at("grid");
    made.rows               = count_of(grid, "rows", 1, 4096);
    made.cols               = count_of(grid, "cols", 1, 4096);
    made.grid_sha256        = grid.string_at("sha256");
    same_hash(made.grid_sha256, sha256_hex(json::canonical(grid, "sha256")),
              "the grid does not have the hash it names: the geometry changed");
    const json::value& cell   = grid.array_at("cell_m");
    const json::value& origin = grid.array_at("origin_m");
    if (cell.items.size() != 2 || origin.items.size() != 2 || !cell.items[0].is_number() || !cell.items[1].is_number() ||
        !origin.items[0].is_number() || !origin.items[1].is_number()) {
      throw artifact_error("grid requires cell_m and origin_m, each of two numbers");
    }
    made.cell_m[0]   = cell.items[0].number;
    made.cell_m[1]   = cell.items[1].number;
    made.origin_m[0] = origin.items[0].number;
    made.origin_m[1] = origin.items[1].number;
    const json::value& threshold = m.object_at("threshold");
    made.threshold_logit         = threshold.number_at("logit");
    made.threshold_sha256        = threshold.string_at("sha256");
    same_hash(made.threshold_sha256, sha256_hex(json::canonical(threshold, "sha256")),
              "the threshold does not have the hash it names: it was moved after it was frozen");
    if (threshold.string_at("entry") != made.entry_detector) {
      throw artifact_error("the threshold is not that of the detector of the artifact");
    }
    const json::value& smoothing = m.object_at("smoothing");
    if (smoothing.string_at("entry") != made.entry_smoothing) {
      throw artifact_error("the optional stage is not the entry of the artifact");
    }
    if (made.entry_smoothing != "none") {
      made.smoothing_maps        = count_of(smoothing, "maps", 1, 4096);
      made.smoothing_process     = smoothing.number_at("process_noise_m2_s3");
      made.smoothing_measurement = smoothing.number_at("measurement_noise_m2");
      made.smoothing_speed       = smoothing.number_at("initial_speed_m_s");
    }

    // The model and its weights.
    const json::value& config = m.object_at("model").object_at("config");
    made.antennas             = count_of(config, "antennas", 1, 64);
    made.stem_channels        = count_of(config, "stem_channels", 1, 4096);
    made.stem_kernel          = count_of(config, "stem_kernel", 1, 63);
    made.block_kernel         = count_of(config, "block_kernel", 1, 63);
    const json::value& blocks = config.array_at("block_channels");
    const json::value& hidden = config.array_at("hidden");
    if (config.string_at("entry") != made.entry_model || blocks.items.size() != 3 || hidden.items.size() != 2 ||
        made.antennas != made.rx || count_of(config, "rows", 1, 4096) != made.rows ||
        count_of(config, "cols", 1, 4096) != made.cols || made.stem_kernel % 2 == 0 || made.block_kernel % 2 == 0) {
      throw artifact_error("the model is not the entry of the artifact on the array of its specification and on its grid");
    }
    for (unsigned i = 0; i != 3; ++i) {
      made.block_channels[i] = static_cast<std::uint32_t>(blocks.items[i].number);
    }
    for (unsigned i = 0; i != 2; ++i) {
      made.hidden[i] = static_cast<std::uint32_t>(hidden.items[i].number);
    }
    const json::value& weights = m.object_at("weights");
    const std::string  data    = read_file(directory + "/" + WEIGHTS_FILE);
    made.weights_sha256        = weights.string_at("sha256");
    if (weights.string_at("file") != WEIGHTS_FILE || weights.string_at("dtype") != "<f4" ||
        static_cast<std::size_t>(weights.integer_at("bytes")) != data.size() || data.size() % 4 != 0 ||
        static_cast<std::size_t>(weights.integer_at("count")) != data.size() / 4) {
      throw artifact_error("weights.f32 does not have the size that model.json names: the checkpoint was altered");
    }
    same_hash(made.weights_sha256, sha256_hex(data),
              "weights.f32 does not have the hash that model.json names: the checkpoint was altered");
    made.weights       = values_of<float>(data);
    std::size_t offset = 0;
    for (const json::value& item : weights.array_at("tensors").items) {
      tensor_entry entry;
      entry.name        = item.string_at("name");
      entry.offset      = static_cast<std::size_t>(item.integer_at("offset"));
      entry.count       = static_cast<std::size_t>(item.integer_at("count"));
      std::size_t total = 1;
      for (const json::value& side : item.array_at("shape").items) {
        entry.shape.push_back(static_cast<std::size_t>(side.number));
        total *= entry.shape.back();
      }
      if (entry.offset != offset || entry.count != total) {
        throw artifact_error("the layout of the weight file is not contiguous");
      }
      offset += entry.count;
      made.tensors.push_back(std::move(entry));
    }
    if (offset != made.weights.size() || made.tensors.empty()) {
      throw artifact_error("the layout of the weight file does not name every value of it");
    }
  } catch (const json::error& fault) {
    throw artifact_error(std::string("model.json is incomplete: ") + fault.what());
  }

  for (const float value : made.template_values) {
    if (!std::isfinite(value)) {
      throw artifact_error("the template holds a value that is not finite");
    }
  }
  for (const float value : made.weights) {
    if (!std::isfinite(value)) {
      throw artifact_error("the weights hold a value that is not finite");
    }
  }
  if (!std::isfinite(made.threshold_logit) || !std::isfinite(made.norm_mean) || !(made.tau >= 0.0) ||
      !(made.min_valid_fraction > 0.0 && made.min_valid_fraction <= 1.0)) {
    throw artifact_error("the threshold, the normalizer or the validity rule of the artifact is no usable number");
  }

  // A deployed model is four files and nothing else.
  DIR* listing = ::opendir(directory.c_str());
  if (listing == nullptr) {
    throw artifact_error("cannot list " + directory);
  }
  std::string extra;
  while (const dirent* found = ::readdir(listing)) {
    const std::string name = found->d_name;
    if (name != "." && name != ".." && name != MODEL_FILE && name != TEMPLATE_FILE && name != TEMPLATE_COUNT_FILE &&
        name != WEIGHTS_FILE && (extra.empty() || name < extra)) {
      extra = name;
    }
  }
  ::closedir(listing);
  if (!extra.empty()) {
    throw artifact_error("the artifact holds " + extra + ": a deployed model is model.json, template.c64, "
                         "template.count.u4, weights.f32 and nothing else");
  }
  if (expect_sha256 != nullptr && *expect_sha256 != '\0' && made.sha256 != expect_sha256) {
    throw artifact_error("the artifact has the hash " + made.sha256 + ", and " + expect_sha256 + " was asked for");
  }
  return made;
}

} // namespace sense
