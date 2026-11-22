// A frozen cusense-artifact/v1 artifact, read and verified without Python (Sprint 3.6, selector
// sense_core).
//
// The artifact is a directory of four files that ocudu_cuda_isac.sensing.cusense wrote:
// model.json, template.c64, template.count.u4 and weights.f32. load_artifact reads them and
// checks what the reference reader checks before a plan is built from them:
//
//   the manifest has the hash it names, over its canonical form;
//   the template, its counts and the weights have the sizes and the SHA-256 the manifest names;
//   the normalizer, the grid, the threshold and the specification have the hashes it names;
//   the template and the statistics were fitted on the training split;
//   every value is finite, the shapes follow from the specification and from the grid;
//   the directory holds nothing else;
//   and, when a freeze record names a hash, the artifact is that one.
//
// A plan is refused for an artifact that fails any of these. The registry check, whether this
// core has a unit for every entry the artifact names, is the plan's.
//
// ASCII only.

#pragma once

#include "core/json.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sense {

struct artifact_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

constexpr const char ARTIFACT_SCHEMA[]     = "cusense-artifact/v1";
constexpr const char MODEL_FILE[]          = "model.json";
constexpr const char TEMPLATE_FILE[]       = "template.c64";
constexpr const char TEMPLATE_COUNT_FILE[] = "template.count.u4";
constexpr const char WEIGHTS_FILE[]        = "weights.f32";

struct tensor_entry {
  std::string              name;
  std::vector<std::size_t> shape;
  std::size_t              offset = 0; ///< In values of the weight file.
  std::size_t              count  = 0;
};

struct artifact {
  std::string directory;
  std::string manifest_text; ///< model.json as the file holds it, without its last line break.
  json::value manifest;
  std::string model_id, sha256;

  // The entries of every stage, as the registry names them.
  std::string entry_input, entry_ingest, entry_window, entry_clutter, entry_symbols, entry_normalizer, entry_model,
      entry_detector, entry_smoothing;

  // The specification of the windows.
  std::string              spec_sha256, spec_id, spec_canonical;
  std::string              input_stage, input_source, input_layout, input_dtype;
  int                      scs_khz = 0, period_slots = 0;
  std::vector<std::uint32_t> slots; ///< Slots of a period that form a window, rising.
  double                   min_valid_fraction = 0.0, tau = 0.0;
  std::string              min_valid_text; ///< The share as model.json writes it, for the text of a reason.
  std::uint32_t            first_subcarrier = 0, subcarrier_step = 0, subcarriers = 0, dmrs_symbols = 0, rx = 0;
  std::vector<std::uint32_t> elements;       ///< Array elements of the input, rising.
  std::vector<std::string>   producer_steps; ///< The chain of steps the model was trained on.

  // Template: H_B[a, k, s] as pairs {real, imaginary}, and the records behind each element.
  std::vector<float>         template_values;
  std::vector<std::uint32_t> template_count;
  std::string                template_sha256;

  // Normalizer.
  double      norm_mean = 0.0, norm_scale = 0.0, norm_epsilon = 0.0;
  std::string normalizer_sha256;

  // Model.
  std::uint32_t antennas = 0, rows = 0, cols = 0, stem_channels = 0, stem_kernel = 0, block_kernel = 0;
  std::uint32_t block_channels[3] = {0, 0, 0}, hidden[2] = {0, 0};
  std::vector<tensor_entry> tensors;
  std::vector<float>        weights;
  std::string               weights_sha256;

  // Grid, threshold and the optional stage.
  double      cell_m[2] = {0.0, 0.0}, origin_m[2] = {0.0, 0.0};
  std::string grid_sha256;
  double      threshold_logit = 0.0;
  std::string threshold_sha256;
  std::uint32_t smoothing_maps = 0;
  double        smoothing_process = 0.0, smoothing_measurement = 0.0, smoothing_speed = 0.0;

  /// The tensor of a name. Throws artifact_error when the weight file names none.
  const tensor_entry& tensor(const std::string& name) const;
  const float*        values(const tensor_entry& entry) const { return weights.data() + entry.offset; }
};

/// Reads and verifies an artifact. expect_sha256 may be null.
artifact load_artifact(const std::string& directory, const char* expect_sha256);

/// The bytes of a file. Throws artifact_error.
std::string read_file(const std::string& path);

} // namespace sense
