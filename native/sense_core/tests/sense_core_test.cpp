// Unit cases of the native sensing core (Sprint 3.6, selector sense_core).
//
// One case per CTest name:  sense_core_test <case>
//
// The cases build what they need themselves: a small frozen artifact, written as the reference
// writer lays it out, and obs/v1 records in the three layouts the core has an ingest unit for.
// They run the units on the host twin, so they need no device. device_vectors runs the same
// window on the CUDA device and on the twin, and is skipped where no device is reachable.
//
// The cases that compare the core with the Python reference, value for value, are in
// tests/unit/test_sense_core.py: they go through the binding and through sense_host.
//
// ASCII only.

#include "core/artifact.h"
#include "core/build.h"
#include "core/json.h"
#include "core/record.h"
#include "core/sha256.h"
#include "core/tolerance.h"
#include "engine/engine.h"
#include "kernels/units.h"
#include "obs/ring.h"
#include "plan/smoother.h"
#include "sense_core/sense_core.h"
#include "sensor_loop.h"
#include "sources.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

std::uint64_t allocation_count(); // tests/alloc_count.cpp

namespace {

int failures = 0;

#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #condition);                               \
      ++failures;                                                                                                      \
    }                                                                                                                  \
  } while (0)

// The array of the fixture: [A, K, S] of a window, the grid, and the window of a period.
constexpr std::uint32_t RX = 4, POSITIONS = 48, SYMBOLS = 2, ROWS = 3, COLS = 4, PERIOD = 10;
constexpr std::uint32_t WINDOW_SLOTS[3] = {7, 8, 9};
constexpr unsigned      DMRS[2]         = {2, 11};
constexpr int           MU              = 1; // 30 kHz

// ---------------------------------------------------------------------------------------------- files

std::string scratch_directory()
{
  const char* base = std::getenv("TMPDIR");
  std::string path = std::string(base != nullptr && *base != '\0' ? base : "/tmp") + "/sense-core-XXXXXX";
  if (::mkdtemp(&path[0]) == nullptr) {
    std::fprintf(stderr, "cannot create a scratch directory under %s\n", path.c_str());
    std::exit(2);
  }
  return path;
}

void remove_tree(const std::string& path)
{
  DIR* listing = ::opendir(path.c_str());
  if (listing != nullptr) {
    while (const dirent* found = ::readdir(listing)) {
      const std::string name = found->d_name;
      if (name != "." && name != "..") {
        remove_tree(path + "/" + name);
      }
    }
    ::closedir(listing);
    ::rmdir(path.c_str());
  } else {
    ::unlink(path.c_str());
  }
}

struct scratch {
  std::string path = scratch_directory();
  ~scratch() { remove_tree(path); }
  std::string sub(const std::string& name) const
  {
    const std::string made = path + "/" + name;
    ::mkdir(made.c_str(), 0700);
    return made;
  }
};

void write_file(const std::string& path, const void* data, std::size_t bytes)
{
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr || (bytes != 0 && std::fwrite(data, 1, bytes, file) != bytes) || std::fclose(file) != 0) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(2);
  }
}

void write_text(const std::string& path, const std::string& text)
{
  write_file(path, text.data(), text.size());
}

// ---------------------------------------------------------------------------------------------- a frozen artifact

// JSON as the reference writer lays it out: an object with its keys sorted. The canonical form
// of a part, which its hash is taken over, is that text without white space. It is built here by
// hand, apart from the reader of the core.
using members = std::map<std::string, std::string>;

std::string object(const members& parts)
{
  std::string out = "{";
  for (const auto& part : parts) {
    out += out.size() > 1 ? "," : "";
    out += "\"" + part.first + "\":" + part.second;
  }
  return out + "}";
}

std::string quoted(const std::string& text)
{
  return "\"" + text + "\"";
}

std::string list(const std::vector<std::string>& items)
{
  std::string out = "[";
  for (const std::string& item : items) {
    out += out.size() > 1 ? "," : "";
    out += item;
  }
  return out + "]";
}

std::string number(std::uint64_t value)
{
  return std::to_string(value);
}

/// The text of a file: the separators the reference writer uses, ", " and ": ".
std::string spaced(const std::string& canonical)
{
  std::string out;
  bool        in_string = false;
  for (std::size_t at = 0; at != canonical.size(); ++at) {
    const char letter = canonical[at];
    out += letter;
    if (letter == '"' && (at == 0 || canonical[at - 1] != '\\')) {
      in_string = !in_string;
    } else if (!in_string && (letter == ',' || letter == ':')) {
      out += ' ';
    }
  }
  return out;
}

struct artifact_options {
  std::string layout    = "ocudu_o3_fp16";
  std::string clutter   = "background_magnitude";
  std::string smoothing = "none";
  std::string threshold = "0.0";
  std::string min_valid = "0.5";
  std::string mean      = "0.2";
  std::string scale     = "0.1";
  std::string split     = "training";
  float       template_real = 0.5f, template_imaginary = 0.0f;
  bool        template_hole = false; ///< The template holds no sample at [a=1, k=3, s=0].
  std::uint32_t seed = 1;
};

struct lcg {
  std::uint32_t state;
  float         next() // in [-0.5, 0.5)
  {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
  }
};

/// Writes a frozen cusense-artifact/v1 artifact of the fixture array. Returns its hash.
std::string write_artifact(const std::string& directory, const artifact_options& options = artifact_options())
{
  const bool        o3       = options.layout == "ocudu_o3_fp16";
  const std::size_t elements = static_cast<std::size_t>(RX) * POSITIONS * SYMBOLS;
  // The template and its counts.
  std::vector<float>         background(2 * elements);
  std::vector<std::uint32_t> counts(elements, 6);
  for (std::size_t at = 0; at != elements; ++at) {
    background[2 * at]     = options.template_real;
    background[2 * at + 1] = options.template_imaginary;
  }
  if (options.template_hole) {
    const std::size_t at   = (1 * POSITIONS + 3) * SYMBOLS + 0;
    background[2 * at]     = 0.0f;
    background[2 * at + 1] = 0.0f;
    counts[at]             = 0;
  }
  write_file(directory + "/template.c64", background.data(), 8 * elements);
  write_file(directory + "/template.count.u4", counts.data(), 4 * elements);

  // The weights of the model, in the order and under the names of its state.
  constexpr std::uint32_t STEM = 4, STEM_KERNEL = 5, BLOCK_KERNEL = 3, BLOCKS[3] = {4, 6, 8}, HIDDEN[2] = {8, 8};
  struct named {
    std::string name;
    std::size_t sides[3];
    unsigned    rank;
  };
  std::vector<named> tensors;
  const auto add = [&](const std::string& name, std::size_t first, std::size_t second = 0, std::size_t third = 0) {
    tensors.push_back(named{name, {first, second, third}, third != 0 ? 3U : second != 0 ? 2U : 1U});
  };
  const auto norm = [&](const std::string& name, std::size_t width) {
    for (const char* part : {".weight", ".bias", ".running_mean", ".running_var"}) {
      add(name + part, width);
    }
  };
  add("stem.0.weight", STEM, RX, STEM_KERNEL);
  norm("stem.1", STEM);
  std::size_t in = STEM;
  for (unsigned block = 0; block != 3; ++block) {
    const std::string name = "blocks." + std::to_string(block);
    add(name + ".0.weight", BLOCKS[block], in, BLOCK_KERNEL);
    norm(name + ".1", BLOCKS[block]);
    add(name + ".3.weight", BLOCKS[block], BLOCKS[block], BLOCK_KERNEL);
    norm(name + ".4", BLOCKS[block]);
    in = BLOCKS[block];
  }
  add("trunk.0.weight", HIDDEN[0], in);
  add("trunk.0.bias", HIDDEN[0]);
  add("trunk.3.weight", HIDDEN[1], HIDDEN[0]);
  add("trunk.3.bias", HIDDEN[1]);
  add("location.weight", ROWS * COLS, HIDDEN[1]);
  add("location.bias", ROWS * COLS);
  add("presence.weight", 1, HIDDEN[1]);
  add("presence.bias", 1);
  std::vector<float>       weights;
  std::vector<std::string> layout;
  lcg                      random{options.seed};
  for (const auto& tensor : tensors) {
    std::size_t              count = 1;
    std::vector<std::string> shape;
    for (unsigned side = 0; side != tensor.rank; ++side) {
      count *= tensor.sides[side];
      shape.push_back(number(tensor.sides[side]));
    }
    layout.push_back(object({{"name", quoted(tensor.name)},
                             {"offset", number(weights.size())},
                             {"count", number(count)},
                             {"shape", list(shape)}}));
    const bool variance = tensor.name.find("running_var") != std::string::npos;
    const bool gain     = tensor.name.find(".1.weight") != std::string::npos || tensor.name.find(".4.weight") != std::string::npos;
    for (std::size_t n = 0; n != count; ++n) {
      const float value = random.next();
      weights.push_back(variance ? 1.0f + value : gain ? 1.0f + 0.2f * value : value);
    }
  }
  write_file(directory + "/weights.f32", weights.data(), 4 * weights.size());

  const std::string spec = object(
      {{"schema", quoted("cusense-spec/v1")},
       {"spec_id", quoted("unit-fixture")},
       {"input", object({{"stage", quoted(o3 ? "O3" : "O2")},
                         {"source", quoted("pusch_dmrs")},
                         {"layout", quoted(options.layout)},
                         {"dtype", quoted(o3 ? "<cfp16" : "<c8")},
                         {"entry", quoted(o3 ? "o3_fp16" : "o2_comb")}})},
       {"radio", object({{"scs_khz", "30"}, {"period_slots", number(PERIOD)}})},
       {"window", object({{"slots_in_period", "[7,8,9]"}, {"min_valid_fraction", options.min_valid}})},
       {"support", object({{"first_subcarrier", "0"},
                           {"subcarrier_step", "1"},
                           {"subcarriers", number(POSITIONS)},
                           {"dmrs_symbols", number(SYMBOLS)},
                           {"rx", number(RX)}})},
       {"validity", object({{"tau", "1e-10"}})},
       {"steps", object({{"ingest", quoted(options.layout)},
                         {"window", quoted("coherent_mean")},
                         {"clutter", quoted(options.clutter)},
                         {"symbols", quoted("dmrs_mean")},
                         {"normalizer", quoted("zscore_global")}})}});
  const std::string fitted = object({{"split", quoted(options.split)}, {"windows", "6"}});
  const members     normalizer_hashed = {
      {"entry", quoted("zscore_global")}, {"mean", options.mean}, {"scale", options.scale}, {"epsilon", "1e-08"}};
  members normalizer            = normalizer_hashed;
  normalizer["sha256"]          = quoted(sense::sha256_hex(object(normalizer_hashed)));
  normalizer["fitted_on"]       = fitted;
  members grid                  = {{"rows", number(ROWS)}, {"cols", number(COLS)}, {"cell_m", "[0.5,0.25]"}, {"origin_m", "[-1.0,2.0]"}};
  grid["sha256"]                = quoted(sense::sha256_hex(object(grid)));
  members threshold             = {{"entry", quoted("frozen_threshold")}, {"logit", options.threshold}, {"target_pfa", "0.01"}};
  threshold["sha256"]           = quoted(sense::sha256_hex(object(threshold)));
  members smoothing             = {{"entry", quoted(options.smoothing)}};
  if (options.smoothing != "none") {
    smoothing["maps"]                 = "2";
    smoothing["process_noise_m2_s3"]  = "1.0";
    smoothing["measurement_noise_m2"] = "0.25";
    smoothing["initial_speed_m_s"]    = "1.5";
  }
  members manifest = {
      {"schema", quoted("cusense-artifact/v1")},
      {"model_id", quoted("unit-fixture-model")},
      {"entries", object({{"input", quoted(o3 ? "o3_fp16" : "o2_comb")},
                          {"ingest", quoted(options.layout)},
                          {"window", quoted("coherent_mean")},
                          {"clutter", quoted(options.clutter)},
                          {"symbols", quoted("dmrs_mean")},
                          {"normalizer", quoted("zscore_global")},
                          {"model", quoted("cusense_cnn")},
                          {"detector", quoted("frozen_threshold")},
                          {"smoothing", quoted(options.smoothing)}})},
      {"input", object({{"spec", spec},
                        {"spec_sha256", quoted(sense::sha256_hex(spec))},
                        {"elements", "[0,1,2,3]"},
                        {"producer_steps", o3 ? "[\"pilot_ls\",\"frequency_interpolation\"]" : "[\"pilot_ls\"]"},
                        {"shape", list({"1", number(RX), number(POSITIONS)})},
                        {"dtype", quoted("<f4")}})},
      {"template", object({{"file", quoted("template.c64")},
                           {"count_file", quoted("template.count.u4")},
                           {"dtype", quoted("<c8")},
                           {"count_dtype", quoted("<u4")},
                           {"bytes", number(8 * elements)},
                           {"count_bytes", number(4 * elements)},
                           {"sha256", quoted(sense::sha256_hex(background.data(), 8 * elements))},
                           {"count_sha256", quoted(sense::sha256_hex(counts.data(), 4 * elements))},
                           {"fitted_on", fitted}})},
      {"normalizer", object(normalizer)},
      {"grid", object(grid)},
      {"threshold", object(threshold)},
      {"smoothing", object(smoothing)},
      {"model", object({{"config", object({{"entry", quoted("cusense_cnn")},
                                           {"antennas", number(RX)},
                                           {"stem_channels", number(STEM)},
                                           {"stem_kernel", number(STEM_KERNEL)},
                                           {"block_kernel", number(BLOCK_KERNEL)},
                                           {"block_channels", "[4,6,8]"},
                                           {"hidden", "[8,8]"},
                                           {"rows", number(ROWS)},
                                           {"cols", number(COLS)}})}})},
      {"weights", object({{"file", quoted("weights.f32")},
                          {"dtype", quoted("<f4")},
                          {"bytes", number(4 * weights.size())},
                          {"count", number(weights.size())},
                          {"sha256", quoted(sense::sha256_hex(weights.data(), 4 * weights.size()))},
                          {"tensors", list(layout)}})}};
  const std::string hash = sense::sha256_hex(object(manifest));
  manifest["sha256"]     = quoted(hash);
  write_text(directory + "/model.json", spaced(object(manifest)) + "\n");
  return hash;
}

// ---------------------------------------------------------------------------------------------- records

/// A value that FP16 holds exactly, as its bits. The fixtures use multiples of 1/1024 up to 2.
std::uint16_t half_of(float value)
{
  if (value == 0.0f) {
    return 0;
  }
  const std::uint16_t sign = value < 0.0f ? 0x8000U : 0U;
  int                 exponent;
  const float         fraction = std::frexp(std::fabs(value), &exponent); // in [0.5, 1)
  const int           biased   = exponent - 1 + 15;
  if (biased <= 0 || biased >= 31) {
    std::fprintf(stderr, "half_of: %g is outside the fixture range\n", static_cast<double>(value));
    std::exit(2);
  }
  const float scaled = (fraction * 2.0f - 1.0f) * 1024.0f;
  if (scaled != std::floor(scaled)) {
    std::fprintf(stderr, "half_of: %g is no FP16 value\n", static_cast<double>(value));
    std::exit(2);
  }
  return static_cast<std::uint16_t>(sign | (static_cast<unsigned>(biased) << 10) | static_cast<unsigned>(scaled));
}

struct record_options {
  std::uint8_t  layout    = obs::LAYOUT_OCUDU_O3_FP16;
  std::uint64_t epoch = 1, sequence = 0, episode = 5, slot = 7;
  std::uint32_t first_subcarrier = 0, positions = POSITIONS;
  std::uint32_t steps     = obs::STEP_PILOT_LS | obs::STEP_FREQ_INTERP;
  std::int64_t  t_ready   = 1000;
  std::uint16_t antenna_map[RX] = {0, 1, 2, 3}; ///< antenna_map[plane]: the array element behind it.
  std::uint8_t  evidence  = obs::EVIDENCE_AUDIT;
};

/// The sample at array element a, position k of the record and DM-RS symbol s. Returns its
/// validity bit.
using sample_fn = std::function<bool(unsigned a, unsigned k, unsigned s, float& real, float& imaginary)>;

/// One obs/v1 record as a file of an archive holds it: the header, the validity bits, the payload.
std::vector<std::uint8_t> make_record(const record_options& options, const sample_fn& sample)
{
  obs::record_header header;
  std::memset(&header, 0, sizeof(header));
  const std::uint32_t k_count  = options.positions;
  const std::uint32_t elements = SYMBOLS * k_count * RX;
  const bool          o3       = options.layout == obs::LAYOUT_OCUDU_O3_FP16;
  header.schema           = obs::RECORD_SCHEMA;
  header.header_bytes     = obs::RECORD_HEADER_BYTES;
  header.epoch            = options.epoch;
  header.sequence         = options.sequence;
  header.episode          = options.episode;
  header.opportunity      = 100 + options.sequence;
  header.grant            = 7;
  header.absolute_slot    = options.slot;
  header.cell             = 1;
  header.ue               = 0x4601;
  header.stage            = options.layout == obs::LAYOUT_OCUDU_O2_COMB_C64 ? obs::STAGE_O2 : obs::STAGE_O3;
  header.source           = obs::SOURCE_PUSCH_DMRS;
  header.layout           = options.layout;
  header.dtype            = o3 ? obs::DTYPE_CFP16 : obs::DTYPE_C64;
  header.producer_steps   = options.steps;
  header.scale            = 1.0;
  header.first_subcarrier = options.first_subcarrier;
  header.subcarrier_step  = 1;
  header.subcarrier_count = k_count;
  header.dmrs_symbols     = (1U << DMRS[0]) | (1U << DMRS[1]);
  header.rx_count         = RX;
  header.layer_count      = 1;
  std::memcpy(header.antenna_map, options.antenna_map, sizeof(options.antenna_map));
  if (o3) {
    const std::uint32_t shape[4] = {SYMBOLS, k_count / 12, RX, 12};
    header.rank                  = 4;
    std::memcpy(header.shape, shape, sizeof(shape));
    header.strides[0] = shape[1] * shape[2] * shape[3];
    header.strides[1] = shape[2] * shape[3];
    header.strides[2] = shape[3];
    header.strides[3] = 1;
  } else if (options.layout == obs::LAYOUT_OCUDU_O2_COMB_C64) {
    const std::uint32_t shape[3] = {RX, SYMBOLS, k_count};
    header.rank                  = 3;
    std::memcpy(header.shape, shape, sizeof(shape));
    header.strides[0] = shape[1] * shape[2];
    header.strides[1] = shape[2];
    header.strides[2] = 1;
  } else {
    const std::uint32_t shape[4] = {SYMBOLS, k_count, RX, 1};
    header.rank                  = 4;
    std::memcpy(header.shape, shape, sizeof(shape));
    header.strides[0] = shape[1] * shape[2];
    header.strides[1] = shape[2];
    header.strides[2] = 1;
    header.strides[3] = 1;
  }
  header.mask_bytes     = (elements + 7) / 8;
  header.payload_bytes  = (o3 ? 4U : 8U) * elements;
  header.host_bytes     = 0;
  header.time_unit      = obs::TIME_TC;
  header.digest_kind    = obs::DIGEST_NONE;
  header.evidence       = options.evidence;
  header.port           = 0;
  header.t_first_sample = sense::symbol_time_tc(MU, options.slot, DMRS[0]);
  header.t_last_sample  = sense::symbol_time_tc(MU, options.slot, DMRS[1]) + sense::useful_tc(MU);
  header.t_ready_ns     = options.t_ready;
  header.t_publish_ns   = options.t_ready + 10;

  std::vector<std::uint8_t> record(sizeof(header) + header.mask_bytes + header.payload_bytes, 0);
  std::uint8_t*             mask    = record.data() + sizeof(header);
  std::uint8_t*             payload = mask + header.mask_bytes;
  // The receive plane behind each array element, in rising order of the element.
  unsigned plane_of[RX];
  for (unsigned a = 0; a != RX; ++a) {
    unsigned smaller = 0;
    for (unsigned other = 0; other != RX; ++other) {
      smaller += options.antenna_map[other] < options.antenna_map[a] ? 1U : 0U;
    }
    plane_of[smaller] = a;
  }
  for (unsigned a = 0; a != RX; ++a) {
    for (unsigned k = 0; k != k_count; ++k) {
      for (unsigned s = 0; s != SYMBOLS; ++s) {
        float       real = 0.0f, imaginary = 0.0f;
        const bool  bit   = sample(a, k, s, real, imaginary);
        const unsigned plane = plane_of[a];
        // Written out apart from units.h: the layouts as docs/interfaces/OBSERVATION.md states them.
        std::size_t element;
        if (o3) {
          element = ((static_cast<std::size_t>(s) * (k_count / 12) + k / 12) * RX + plane) * 12 + k % 12;
          const std::uint16_t parts[2] = {half_of(real), half_of(imaginary)};
          std::memcpy(payload + 4 * element, parts, 4);
        } else {
          element = options.layout == obs::LAYOUT_OCUDU_O2_COMB_C64
                        ? (static_cast<std::size_t>(plane) * SYMBOLS + s) * k_count + k
                        : (static_cast<std::size_t>(s) * k_count + k) * RX + plane;
          const float parts[2] = {real, imaginary};
          std::memcpy(payload + 8 * element, parts, 8);
        }
        if (bit) {
          mask[element >> 3] = static_cast<std::uint8_t>(mask[element >> 3] | (1U << (element & 7U)));
          ++header.valid;
        }
      }
    }
  }
  std::memcpy(record.data(), &header, sizeof(header));
  return record;
}

sense_view view_of(const std::vector<std::uint8_t>& record, std::int64_t t_notify = 0)
{
  obs::record_header header;
  std::memcpy(&header, record.data(), sizeof(header));
  sense_view view;
  std::memset(&view, 0, sizeof(view));
  view.record      = record.data();
  view.mask        = record.data() + sizeof(header) + header.host_bytes;
  view.payload     = record.data() + sizeof(header) + header.host_bytes + header.mask_bytes;
  view.memory      = SENSE_MEMORY_HOST_SHARED;
  view.device      = -1;
  view.t_notify_ns = t_notify;
  return view;
}

/// A clock that moves by one step with every question.
struct fake_clock {
  std::int64_t now  = 1000000;
  std::int64_t step = 1000;
  std::function<void(std::int64_t)> watch;
  static std::int64_t               tick(void* owner)
  {
    fake_clock* clock = static_cast<fake_clock*>(owner);
    clock->now += clock->step;
    if (clock->watch) {
      clock->watch(clock->now);
    }
    return clock->now;
  }
};

struct plan_options {
  fake_clock*  clock = nullptr;
  std::int64_t budgets[4] = {0, 0, 0, 0};
  const char*  device = "host";
  const char*  engine = "host_reference";
  const char*  expect = nullptr;
  const char*  engine_path = nullptr;
};

sense_status create(const std::string& artifact, sense_plan** plan, const plan_options& wanted = plan_options())
{
  sense_options options;
  std::memset(&options, 0, sizeof(options));
  options.api_version       = SENSE_CORE_API_VERSION;
  options.device            = wanted.device;
  options.engine            = wanted.engine;
  options.engine_path       = wanted.engine_path;
  options.expect_sha256     = wanted.expect;
  options.keep_mean         = 1;
  options.use_graph         = 1;
  options.budget_ingest_ns  = wanted.budgets[0];
  options.budget_kernels_ns = wanted.budgets[1];
  options.budget_engine_ns  = wanted.budgets[2];
  options.budget_window_ns  = wanted.budgets[3];
  if (wanted.clock != nullptr) {
    options.clock       = &fake_clock::tick;
    options.clock_owner = wanted.clock;
  }
  return sense_plan_create(artifact.c_str(), &options, plan);
}

struct plan_holder {
  sense_plan* plan = nullptr;
  ~plan_holder() { sense_plan_destroy(plan); }
};

/// The three records of one window: the same sample function in every slot unless `by_slot` says
/// otherwise.
std::vector<std::vector<std::uint8_t>> window_records(std::uint64_t period, std::uint64_t first_sequence,
                                                      const std::function<sample_fn(unsigned)>& by_slot,
                                                      record_options base = record_options())
{
  std::vector<std::vector<std::uint8_t>> made;
  for (unsigned n = 0; n != 3; ++n) {
    record_options options = base;
    options.slot           = period * PERIOD + WINDOW_SLOTS[n];
    options.sequence       = first_sequence + n;
    options.t_ready        = base.t_ready + 500000 * static_cast<std::int64_t>(n);
    made.push_back(make_record(options, by_slot(n)));
  }
  return made;
}

sample_fn constant(float real, float imaginary, bool valid = true)
{
  return [=](unsigned, unsigned, unsigned, float& re, float& im) {
    re = real;
    im = imaginary;
    return valid;
  };
}

/// A sample that differs at every element and in every slot, on the FP16 grid.
sample_fn varied(unsigned salt)
{
  return [=](unsigned a, unsigned k, unsigned s, float& re, float& im) {
    re = static_cast<float>(static_cast<int>((a * 131 + k * 17 + s * 7 + salt * 29) % 1500) - 700) / 1024.0f;
    im = static_cast<float>(static_cast<int>((a * 37 + k * 59 + s * 11 + salt * 13) % 1300) - 600) / 1024.0f;
    return (a + 3 * k + 5 * s + salt) % 23 != 0;
  };
}

/// Offers the records of a window, submits it and collects its decision.
sense_status run_window(sense_plan* plan, const std::vector<std::vector<std::uint8_t>>& records, sense_decision& decision,
                        std::int64_t t_notify = 0)
{
  sense_answer  answers[2];
  std::uint32_t count = 0;
  for (const auto& record : records) {
    const sense_view   view   = view_of(record, t_notify);
    const sense_status status = sense_plan_offer(plan, &view, answers, &count);
    if (status != SENSE_OK) {
      return status;
    }
  }
  if (sense_plan_ready(plan) == 0) {
    return SENSE_E_STATE;
  }
  sense_status status = sense_plan_submit(plan);
  if (status != SENSE_OK) {
    return status;
  }
  while (sense_plan_poll(plan) == 0) {
  }
  return sense_plan_collect(plan, &decision);
}

bool close_to(double found, double wanted, double relative)
{
  return std::fabs(found - wanted) <= relative * std::max(std::fabs(wanted), 1e-30);
}

// ---------------------------------------------------------------------------------------------- cases

void case_sha256()
{
  CHECK(sense::sha256_hex(std::string()) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sense::sha256_hex(std::string("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(sense::sha256_hex(std::string("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  // A message that crosses many blocks, fed in pieces of every length.
  const std::string million(1000000, 'a');
  CHECK(sense::sha256_hex(million) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  sense::sha256 pieces;
  for (std::size_t at = 0, step = 1; at < million.size(); at += step, step = step % 97 + 1) {
    pieces.update(million.data() + at, std::min(step, million.size() - at));
  }
  CHECK(pieces.hex() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  char text[SENSE_HASH_BYTES];
  sense_sha256_hex("abc", 3, text);
  CHECK(std::string(text) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

void case_json_canonical()
{
  using namespace sense::json;
  // Keys in another order, white space everywhere, and numbers whose text a formatter would change.
  const std::string text = "{ \"b\" : [1, 2.50, 1e-08, -0.0, 12345678901234567890],\n \"a\": {\"z\": \"x y\\n\\u00e9\", \"k\": null},"
                           " \"sha256\": \"h\", \"c\": true }";
  const value       item = parse(text);
  CHECK(canonical(item) ==
        "{\"a\":{\"k\":null,\"z\":\"x y\\n\\u00e9\"},\"b\":[1,2.50,1e-08,-0.0,12345678901234567890],\"c\":true,\"sha256\":\"h\"}");
  CHECK(canonical(item, "sha256") == "{\"a\":{\"k\":null,\"z\":\"x y\\n\\u00e9\"},\"b\":[1,2.50,1e-08,-0.0,12345678901234567890],\"c\":true}");
  const std::vector<std::string> only = {"c", "b"};
  CHECK(canonical(item, nullptr, &only) == "{\"b\":[1,2.50,1e-08,-0.0,12345678901234567890],\"c\":true}");
  CHECK(item.at("b").items[2].number == 1e-8 && item.at("b").items[0].integer && !item.at("b").items[1].integer);
  CHECK(item.at("a").string_at("z") == std::string("x y\n\xc3\xa9"));
  CHECK(item.find("missing") == nullptr && item.at("c").boolean);
  for (const char* bad : {"{\"a\": 1, \"a\": 2}", "{\"a\": NaN}", "{\"a\": Infinity}", "[1, 2", "{\"a\": 1} x", "{'a': 1}", ""}) {
    bool refused = false;
    try {
      parse(bad);
    } catch (const error&) {
      refused = true;
    }
    CHECK(refused);
  }
  bool named = false;
  try {
    item.string_at("c");
  } catch (const error& fault) {
    named = std::string(fault.what()).find("\"c\"") != std::string::npos;
  }
  CHECK(named);
  // A number that was written reads back to the same double, and reads as a float.
  for (const double wanted : {0.0, 1.0, -7.698940, 0.1, 1.0 / 3.0, 2.5e-4, 1e300, 5e-324, 123456789.0, -0.5}) {
    std::string out;
    put_double(out, wanted);
    const value back = parse(out);
    CHECK(back.is_number() && back.number == wanted && !back.integer);
  }
  std::string out;
  put_string(out, "a\"b\\c\n\x01");
  CHECK(out == "\"a\\\"b\\\\c\\n\\u0001\"");
  out.clear();
  put_key(out, 0x0123456789abcdefULL);
  CHECK(out == "\"0123456789abcdef\"");
}

void case_half_widening()
{
  // Every FP16 number against its value from its definition: sign, exponent, fraction.
  unsigned wrong = 0;
  for (std::uint32_t bits = 0; bits != 65536; ++bits) {
    const std::uint32_t sign     = bits >> 15, exponent = (bits >> 10) & 31U, fraction = bits & 1023U;
    const float         found    = sense::half_to_float(static_cast<std::uint16_t>(bits));
    if (exponent == 31) {
      const bool same = fraction == 0 ? std::isinf(found) && (found < 0) == (sign != 0) : std::isnan(found);
      wrong += same ? 0U : 1U;
      continue;
    }
    const double magnitude = exponent == 0 ? std::ldexp(static_cast<double>(fraction), -24)
                                           : std::ldexp(1024.0 + static_cast<double>(fraction), static_cast<int>(exponent) - 25);
    const double wanted    = sign != 0 ? -magnitude : magnitude;
    wrong += (static_cast<double>(found) == wanted && std::signbit(found) == (sign != 0)) ? 0U : 1U;
  }
  CHECK(wrong == 0);
  CHECK(half_of(1.0f) == 0x3c00 && half_of(-2.0f) == 0xc000 && half_of(0.5f) == 0x3800 && half_of(1.5f) == 0x3e00);
}

/// The window geometry of one record layout on the fixture array, for a case at the level of the units.
sense::window_geometry unit_window(std::uint32_t records)
{
  sense::window_geometry window;
  std::memset(&window, 0, sizeof(window));
  window.records   = records;
  window.rx        = 1;
  window.positions = 1;
  window.symbols   = 1;
  window.tau2      = 1e-20;
  window.mean      = 0.2f;
  window.divisor   = static_cast<float>(0.1 + 1e-8);
  for (std::uint32_t r = 0; r != records; ++r) {
    window.record[r].positions = 1;
    window.record[r].symbols   = 1;
    window.record[r].planes    = 1;
  }
  return window;
}

void case_vectors()
{
  using namespace sense::tolerance;
  // The vectors of the sprint card, at the level of the units.
  //   The mean of 1+1j, 3-1j and one sample that is not valid is 2+0j, over two records.
  {
    const sense::window_geometry window   = unit_window(3);
    const float                  cube[6]  = {1.0f, 1.0f, 3.0f, -1.0f, 0.0f, 0.0f};
    const std::uint8_t           valid[3] = {1, 1, 0};
    float                        real, imaginary;
    CHECK(sense::coherent_mean(cube, valid, window, 0, 0, 0, real, imaginary) == 2 && real == 2.0f && imaginary == 0.0f);
    const std::uint8_t none[3] = {0, 0, 0};
    CHECK(sense::coherent_mean(cube, none, window, 0, 0, 0, real, imaginary) == 0 && real == 0.0f && imaginary == 0.0f);
  }
  //   1+0j and 0+1j average to 0.5+0.5j with the magnitude 0.7071. Minus the template magnitude
  //   0.5 that is 0.2071, and the z-score with the mean 0.2 and the deviation 0.1 is 0.071.
  {
    const sense::window_geometry window   = unit_window(2);
    const float                  cube[4]  = {1.0f, 0.0f, 0.0f, 1.0f};
    const std::uint8_t           valid[2] = {1, 1};
    const float                  background[1] = {0.5f};
    float                        mean[2], difference[1], input[1];
    std::uint8_t                 count[1];
    sense::frontend_position(cube, valid, window, background, 0, 0, mean, count, difference, input);
    CHECK(mean[0] == 0.5f && mean[1] == 0.5f && count[0] == 2);
    CHECK(close_to(sense::magnitude(mean[0], mean[1]), std::sqrt(0.5), VECTOR_RELATIVE));
    CHECK(close_to(difference[0], std::sqrt(0.5) - 0.5, VECTOR_RELATIVE));
    const double wanted = (std::sqrt(0.5) - 0.5 - 0.2) / (0.1 + 1e-8);
    const double bound  = INPUT_ULPS * FP32_SPACING * std::sqrt(0.5) / (0.1 + 1e-8);
    CHECK(std::fabs(input[0] - wanted) <= bound && std::fabs(wanted - 0.0710678) < 1e-6);
    // A position where the template holds no sample takes no part: its input is zero.
    const float hole[1] = {-1.0f};
    sense::frontend_position(cube, valid, window, hole, 0, 0, mean, count, difference, input);
    CHECK(count[0] == 2 && difference[0] == 0.0f && input[0] == 0.0f);
  }

  // The same vector through a plan, from records to the input of the model.
  scratch     files;
  const std::string directory = files.sub("artifact");
  artifact_options  wanted;
  wanted.template_hole = true;
  write_artifact(directory, wanted);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    std::fprintf(stderr, "%s\n", sense_last_error());
    return;
  }
  // Slot 7 holds 1+0j, slot 8 holds 0+1j, and slot 9 a sample whose bit is not set.
  const auto records = window_records(0, 0, [](unsigned n) {
    return n == 0 ? constant(1.0f, 0.0f) : n == 1 ? constant(0.0f, 1.0f) : constant(1.5f, 1.5f, false);
  });
  sense_decision decision;
  CHECK(run_window(held.plan, records, decision) == SENSE_OK);
  CHECK(decision.status == SENSE_DECISION_OK && decision.elements == RX * POSITIONS * SYMBOLS);
  CHECK(decision.valid == decision.elements && decision.records_min == 2 && decision.records_max == 2);
  const double z     = (std::sqrt(0.5) - 0.5 - 0.2) / (0.1 + 1e-8);
  const double bound = INPUT_ULPS * FP32_SPACING * std::sqrt(0.5) / (0.1 + 1e-8);
  unsigned     off   = 0;
  for (unsigned a = 0; a != RX; ++a) {
    for (unsigned k = 0; k != POSITIONS; ++k) {
      const std::size_t at = static_cast<std::size_t>(a) * POSITIONS + k;
      // At [1, 3] the template holds a sample in one of the two symbols: the mean runs over it alone.
      off += std::fabs(decision.input[at] - z) <= bound ? 0U : 1U;
      off += close_to(decision.difference[at], std::sqrt(0.5) - 0.5, VECTOR_RELATIVE) ? 0U : 1U;
      for (unsigned s = 0; s != SYMBOLS; ++s) {
        const std::size_t e = at * SYMBOLS + s;
        off += decision.mean[2 * e] == 0.5f && decision.mean[2 * e + 1] == 0.5f && decision.count[e] == 2 ? 0U : 1U;
      }
    }
  }
  CHECK(off == 0);
  CHECK(std::isfinite(decision.logit) && decision.probability > 0.0 && decision.probability < 1.0);
  double total = 0.0;
  for (unsigned cell = 0; cell != ROWS * COLS; ++cell) {
    total += static_cast<double>(decision.map[cell]);
  }
  CHECK(std::fabs(total - 1.0) < 1e-5 && decision.row < ROWS && decision.col < COLS);
  CHECK(decision.peak == decision.map[decision.row * COLS + decision.col]);
  // cell_m is [rows, cols] and origin_m [x, y]: x follows the column, y the row.
  CHECK(decision.position_x_m == -1.0 + (decision.col + 0.5) * 0.25 && decision.position_y_m == 2.0 + (decision.row + 0.5) * 0.5);
  CHECK((decision.present != 0) == (static_cast<double>(decision.logit) > 0.0));
}

void case_placement()
{
  // A slot whose allocation starts two resource blocks into the support lands on subcarrier 24.
  scratch           files;
  const std::string directory = files.sub("artifact");
  write_artifact(directory);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  std::vector<std::vector<std::uint8_t>> records;
  for (unsigned n = 0; n != 3; ++n) {
    record_options options;
    options.slot     = WINDOW_SLOTS[n];
    options.sequence = n;
    if (n == 1) {
      options.first_subcarrier = 24; // PRB offset 2
      options.positions        = 12; // one resource block
    }
    records.push_back(make_record(options, n == 1 ? constant(1.0f, 0.0f) : constant(0.25f, 0.0f)));
  }
  sense_decision decision;
  CHECK(run_window(held.plan, records, decision) == SENSE_OK && decision.status == SENSE_DECISION_OK);
  CHECK(decision.contributors == 3 && decision.contributing[1].first_position == 24 && decision.contributing[1].positions == 12 &&
        decision.contributing[0].first_position == 0 && decision.contributing[0].positions == POSITIONS);
  unsigned off = 0;
  for (unsigned a = 0; a != RX; ++a) {
    for (unsigned k = 0; k != POSITIONS; ++k) {
      for (unsigned s = 0; s != SYMBOLS; ++s) {
        const std::size_t e      = (static_cast<std::size_t>(a) * POSITIONS + k) * SYMBOLS + s;
        const bool        inside = k >= 24 && k < 36;
        // Inside the block three slots average (0.25 + 1 + 0.25) / 3 = 0.5, outside two give 0.25.
        off += decision.count[e] == (inside ? 3 : 2) && decision.mean[2 * e] == (inside ? 0.5f : 0.25f) &&
                       decision.mean[2 * e + 1] == 0.0f
                   ? 0U
                   : 1U;
      }
    }
  }
  CHECK(off == 0 && decision.records_min == 2 && decision.records_max == 3);
  // An allocation that does not lie on the support is no input.
  record_options beyond;
  beyond.slot             = PERIOD + 7;
  beyond.sequence         = 3;
  beyond.first_subcarrier = 40;
  beyond.positions        = 12;
  const auto    record    = make_record(beyond, constant(1.0f, 0.0f));
  const sense_view view   = view_of(record);
  sense_answer  answers[2];
  std::uint32_t count = 0;
  CHECK(sense_plan_offer(held.plan, &view, answers, &count) == SENSE_OK && count == 1 &&
        answers[0].kind == SENSE_ANSWER_REJECTED && std::strstr(answers[0].reason, "does not lie on the support") != nullptr);
}

void case_o3_element()
{
  // Element [1][3][2][5] of an ocudu_o3_fp16 payload is DM-RS symbol 1, receive port 2 and
  // subcarrier 3 * 12 + 5 = 41.
  scratch           files;
  const std::string directory = files.sub("artifact");
  write_artifact(directory);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  std::vector<std::uint8_t> record = make_record(record_options(), constant(0.0f, 0.0f, false));
  obs::record_header        header;
  std::memcpy(&header, record.data(), sizeof(header));
  const std::size_t   element   = ((1 * (POSITIONS / 12) + 3) * RX + 2) * 12 + 5; // the C order of [2][4][4][12]
  std::uint8_t*       mask      = record.data() + sizeof(header);
  std::uint8_t*       payload   = mask + header.mask_bytes;
  const std::uint16_t parts[2]  = {half_of(0.75f), half_of(-1.25f)};
  std::memcpy(payload + 4 * element, parts, 4);
  mask[element >> 3] = static_cast<std::uint8_t>(mask[element >> 3] | (1U << (element & 7U)));
  const sense_view          view = view_of(record);
  const std::size_t         elements = static_cast<std::size_t>(RX) * POSITIONS * SYMBOLS;
  std::vector<float>        values(2 * elements, -9.0f);
  std::vector<std::uint8_t> valid(elements, 9);
  std::uint32_t             shape[3] = {0, 0, 0}, first = 99;
  CHECK(sense_plan_ingest(held.plan, &view, values.data(), valid.data(), elements, shape, &first) == SENSE_OK);
  CHECK(shape[0] == RX && shape[1] == POSITIONS && shape[2] == SYMBOLS && first == 0);
  unsigned set = 0, off = 0;
  for (unsigned a = 0; a != RX; ++a) {
    for (unsigned k = 0; k != POSITIONS; ++k) {
      for (unsigned s = 0; s != SYMBOLS; ++s) {
        const std::size_t at = (static_cast<std::size_t>(a) * POSITIONS + k) * SYMBOLS + s;
        if (valid[at] != 0) {
          ++set;
          off += a == 2 && k == 41 && s == 1 && values[2 * at] == 0.75f && values[2 * at + 1] == -1.25f ? 0U : 1U;
        } else {
          off += values[2 * at] == 0.0f && values[2 * at + 1] == 0.0f ? 0U : 1U;
        }
      }
    }
  }
  CHECK(set == 1 && off == 0);
  // A bit that is set over a value that is not finite, or below tau, is no sample.
  const std::uint16_t infinite[2] = {0x7c00, half_of(1.0f)};
  std::memcpy(payload + 4 * element, infinite, 4);
  CHECK(sense_plan_ingest(held.plan, &view, values.data(), valid.data(), elements, shape, &first) == SENSE_OK);
  const std::size_t at = (static_cast<std::size_t>(2) * POSITIONS + 41) * SYMBOLS + 1;
  CHECK(valid[at] == 0 && values[2 * at] == 0.0f && values[2 * at + 1] == 0.0f);
  const std::uint16_t nothing[2] = {0x0000, 0x8000}; // a set bit over zero: the magnitude is not above tau
  std::memcpy(payload + 4 * element, nothing, 4);
  CHECK(sense_plan_ingest(held.plan, &view, values.data(), valid.data(), elements, shape, &first) == SENSE_OK);
  CHECK(valid[at] == 0);
  CHECK(sense_plan_ingest(held.plan, &view, values.data(), valid.data(), elements - 1, shape, &first) == SENSE_E_ARGUMENT);
}

void case_aerial_same_cube()
{
  // The same values in the three layouts give one cube, and the receive planes follow the
  // array elements whatever order the planes come in.
  scratch           files;
  const std::string directory = files.sub("artifact");
  write_artifact(directory);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  const std::size_t elements = static_cast<std::size_t>(RX) * POSITIONS * SYMBOLS;
  const auto        cube_of  = [&](const record_options& options, std::vector<float>& values, std::vector<std::uint8_t>& valid) {
    const auto       record = make_record(options, varied(3));
    const sense_view view   = view_of(record);
    std::uint32_t    shape[3], first;
    values.assign(2 * elements, -9.0f);
    valid.assign(elements, 9);
    return sense_plan_ingest(held.plan, &view, values.data(), valid.data(), elements, shape, &first) == SENSE_OK &&
           shape[0] == RX && shape[1] == POSITIONS && shape[2] == SYMBOLS;
  };
  std::vector<float>        o3, o2, aerial, permuted;
  std::vector<std::uint8_t> o3_valid, o2_valid, aerial_valid, permuted_valid;
  record_options            options;
  CHECK(cube_of(options, o3, o3_valid));
  options.layout = obs::LAYOUT_AERIAL_HEST_C64;
  CHECK(cube_of(options, aerial, aerial_valid));
  options.layout = obs::LAYOUT_OCUDU_O2_COMB_C64;
  options.steps  = obs::STEP_PILOT_LS;
  CHECK(cube_of(options, o2, o2_valid));
  CHECK(o3 == aerial && o3_valid == aerial_valid && o3 == o2 && o3_valid == o2_valid);
  unsigned samples = 0;
  for (const std::uint8_t bit : o3_valid) {
    samples += bit;
  }
  CHECK(samples > elements / 2 && samples < elements);
  // Planes in another order: plane 0 is array element 2, and so on.
  record_options    mixed;
  mixed.layout             = obs::LAYOUT_AERIAL_HEST_C64;
  const std::uint16_t order[RX] = {2, 0, 3, 1};
  std::memcpy(mixed.antenna_map, order, sizeof(order));
  CHECK(cube_of(mixed, permuted, permuted_valid));
  CHECK(permuted == o3 && permuted_valid == o3_valid);
}

void case_artifact_hashes()
{
  scratch           files;
  const std::string directory = files.sub("artifact");
  const std::string hash      = write_artifact(directory);
  sense::artifact   model;
  try {
    model = sense::load_artifact(directory, hash.c_str());
  } catch (const sense::artifact_error& fault) {
    std::fprintf(stderr, "%s\n", fault.what());
  }
  CHECK(model.sha256 == hash && model.rx == RX && model.subcarriers == POSITIONS && model.dmrs_symbols == SYMBOLS &&
        model.rows == ROWS && model.cols == COLS && model.slots.size() == 3 && model.slots[2] == 9 &&
        model.norm_mean == 0.2 && model.norm_scale == 0.1 && model.tau == 1e-10 && model.tensors.size() == 43);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  sense_plan_info info;
  CHECK(held.plan != nullptr && sense_plan_info_get(held.plan, &info) == SENSE_OK && hash == info.artifact_sha256 &&
        std::string(info.engine_backend) == "host_reference" && info.timing_valid == 0 && info.window_records == 3 &&
        std::string(info.core_sources_sha256) == sense::core_sources_sha256() && std::strlen(info.plan_sha256) == 64);

  const auto refused = [&](const std::string& where, const char* expect, const char* reason) {
    sense_plan*  plan = nullptr;
    plan_options wanted;
    wanted.expect = expect;
    const sense_status status = create(where, &plan, wanted);
    const bool         said   = std::strstr(sense_last_error(), reason) != nullptr;
    if (status != SENSE_E_ARTIFACT || !said) {
      std::fprintf(stderr, "status %d: %s\n", static_cast<int>(status), sense_last_error());
    }
    sense_plan_destroy(plan);
    return status == SENSE_E_ARTIFACT && plan == nullptr && said;
  };
  // Another artifact than a freeze record names.
  CHECK(refused(directory, std::string(64, '0').c_str(), "was asked for"));
  // One bit of the template, of its counts and of the weights.
  for (const char* name : {"template.c64", "template.count.u4", "weights.f32"}) {
    const std::string other = files.sub(std::string("altered-") + name);
    write_artifact(other);
    std::string bytes = sense::read_file(other + "/" + name);
    bytes[bytes.size() / 2] = static_cast<char>(bytes[bytes.size() / 2] ^ 1);
    write_text(other + "/" + name, bytes);
    CHECK(refused(other, nullptr, "have the hash that model.json names"));
    bytes.pop_back();
    write_text(other + "/" + name, bytes);
    CHECK(refused(other, nullptr, "size"));
  }
  // A threshold that was moved after it was frozen, with and without its own hash recomputed.
  {
    const std::string other = files.sub("threshold");
    write_artifact(other);
    std::string text      = sense::read_file(other + "/model.json");
    const std::size_t at  = text.find("\"logit\": 0.0");
    CHECK(at != std::string::npos);
    text.replace(at, 12, "\"logit\": 0.5");
    write_text(other + "/model.json", text);
    CHECK(refused(other, nullptr, "model.json does not have the hash it names"));
  }
  // A fifth file.
  {
    const std::string other = files.sub("extra");
    write_artifact(other);
    write_text(other + "/notes.txt", "x");
    CHECK(refused(other, nullptr, "notes.txt"));
  }
  // A template that was fitted on another split.
  {
    const std::string other = files.sub("split");
    artifact_options  wanted;
    wanted.split = "validation";
    write_artifact(other, wanted);
    CHECK(refused(other, nullptr, "training"));
  }
  // No artifact at all.
  CHECK(refused(files.path + "/missing", nullptr, "cannot read"));
}

void case_registry_refused()
{
  scratch           files;
  const std::string directory = files.sub("artifact");
  artifact_options  wanted;
  wanted.clutter = "learned_clutter"; // a stage name this core has no unit for, in a well-formed artifact
  write_artifact(directory, wanted);
  sense_plan* plan = nullptr;
  CHECK(create(directory, &plan) == SENSE_E_REGISTRY && plan == nullptr);
  CHECK(std::strstr(sense_last_error(), "clutter entry learned_clutter") != nullptr);
  // Names outside the registry of devices and engines, and a caller of another API version.
  const std::string good = files.sub("good");
  write_artifact(good);
  plan_options other;
  other.device = "fpga";
  CHECK(create(good, &plan, other) == SENSE_E_ARGUMENT && plan == nullptr);
  other.device = "host";
  other.engine = "onnxruntime";
  CHECK(create(good, &plan, other) == SENSE_E_ARGUMENT && plan == nullptr);
  other.engine = "tensorrt";
  CHECK(create(good, &plan, other) == SENSE_E_ENGINE && plan == nullptr); // no engine directory was named
  sense_options options;
  std::memset(&options, 0, sizeof(options));
  options.api_version = SENSE_CORE_API_VERSION + 1;
  CHECK(sense_plan_create(good.c_str(), &options, &plan) == SENSE_E_ARGUMENT && plan == nullptr);
  // Every entry of the registry names the file that holds its unit and the hash of that file.
  const sense::json::value registry = sense::json::parse(sense_core_registry_json());
  CHECK(registry.is_array() && registry.items.size() >= 12);
  bool found = false;
  for (const sense::json::value& item : registry.items) {
    CHECK(item.string_at("unit_sha256").size() == 64);
    found = found || (item.string_at("stage") == "ingest" && item.string_at("entry") == "aerial_hest_c64");
  }
  CHECK(found && sense::find_unit("clutter", "background_magnitude") != nullptr &&
        sense::find_unit("clutter", "learned_clutter") == nullptr);
  const sense::json::value build = sense::json::parse(sense_core_build_json());
  CHECK(build.string_at("sources_sha256") == sense::core_sources_sha256() && build.integer_at("api_version") == SENSE_CORE_API_VERSION);
  const sense::json::value tolerance = sense::json::parse(sense_core_tolerance_json());
  CHECK(tolerance.is_object());
}

void case_engine_manifest()
{
  // An engine carries the hash of its artifact, the version of its library and the GPU it was
  // built for, and is refused anywhere else. No TensorRT is needed to check that.
  scratch           files;
  const std::string directory = files.sub("artifact");
  const std::string hash      = write_artifact(directory);
  const sense::artifact model = sense::load_artifact(directory, nullptr);
  const std::string engine    = files.sub("engine");
  const std::string plan      = "not a real engine: the manifest check reads its hash only";
  write_text(engine + "/model.plan", plan);
  const auto manifest = [&](const members& changes) {
    members parts = {{"schema", quoted("sense-engine/v1")},
                     {"backend", quoted("tensorrt")},
                     {"artifact_sha256", quoted(hash)},
                     {"weights_sha256", quoted(model.weights_sha256)},
                     {"library", object({{"name", quoted("tensorrt")}, {"version", quoted("10.14.1.48")}})},
                     {"precision", quoted("fp32")},
                     {"device", object({{"name", quoted("NVIDIA GB10")},
                                        {"capability", quoted("12.1")},
                                        {"uuid", quoted("GPU-1111")}})},
                     {"input", object({{"name", quoted("input")}, {"shape", list({"1", number(RX), number(POSITIONS)})}})},
                     {"outputs", object({{"map", list({"1", number(ROWS * COLS)})}, {"presence_logit", "[1]"}})},
                     {"file", quoted("model.plan")},
                     {"bytes", number(plan.size())},
                     {"sha256", quoted(sense::sha256_hex(plan))}};
    for (const auto& change : changes) {
      parts[change.first] = change.second;
    }
    write_text(engine + "/engine.json", spaced(object(parts)) + "\n");
  };
  sense::engine_host host;
  host.backend         = "tensorrt";
  host.library_version = "10.14.1.48";
  host.device_identity = "{\"kind\": \"cuda\", \"name\": \"NVIDIA GB10\", \"capability\": \"12.1\", \"uuid\": \"GPU-1111\"}";
  const auto refusal = [&](const sense::engine_host& on) {
    try {
      sense::check_engine_manifest(engine, model, on, sense::ENGINE_PLAN_FILE);
    } catch (const sense::engine_error& fault) {
      return std::string(fault.what());
    }
    return std::string();
  };
  manifest({});
  CHECK(refusal(host).empty());
  CHECK(sense::check_engine_manifest(engine, model, host, sense::ENGINE_PLAN_FILE) == sense::sha256_hex(plan));
  // Built for another GPU: by its name, by its capability, by its identifier.
  sense::engine_host other = host;
  other.device_identity = "{\"kind\": \"cuda\", \"name\": \"NVIDIA GH200\", \"capability\": \"9.0\", \"uuid\": \"GPU-2222\"}";
  CHECK(refusal(other).find("built on the GPU that runs it") != std::string::npos);
  other.device_identity = "{\"kind\": \"cuda\", \"name\": \"NVIDIA GB10\", \"capability\": \"12.1\", \"uuid\": \"GPU-3333\"}";
  CHECK(refusal(other).find("built on the GPU that runs it") != std::string::npos);
  // The host twin is no GPU at all.
  other.device_identity = "{\"kind\": \"host\", \"name\": \"host twin\", \"machine\": \"aarch64\"}";
  CHECK(refusal(other).find("built on the GPU that runs it") != std::string::npos);
  // Another version of the library.
  other                 = host;
  other.library_version = "10.13.3.9";
  CHECK(refusal(other).find("rebuild the engine") != std::string::npos);
  // Another backend than the plan asks for.
  other         = host;
  other.backend = "libtorch";
  CHECK(refusal(other).find("backend") != std::string::npos);
  // Built from another artifact, from other weights, for another input, or altered after the build.
  manifest({{"artifact_sha256", quoted(std::string(64, 'a'))}});
  CHECK(refusal(host).find("an engine carries the hash of its artifact") != std::string::npos);
  manifest({{"weights_sha256", quoted(std::string(64, 'b'))}});
  CHECK(refusal(host).find("other weights") != std::string::npos);
  manifest({{"input", object({{"name", quoted("input")}, {"shape", list({"1", number(RX), number(POSITIONS + 12)})}})}});
  CHECK(refusal(host).find("[1, A, K]") != std::string::npos);
  manifest({{"sha256", quoted(std::string(64, 'c'))}});
  CHECK(!refusal(host).empty());
  manifest({});
  write_text(engine + "/model.plan", plan + "x");
  CHECK(!refusal(host).empty());
  // No manifest: not an engine directory.
  CHECK(refusal(host).size() != 0);
  ::unlink((engine + "/engine.json").c_str());
  CHECK(refusal(host).find("holds no engine.json") != std::string::npos);
  // Through a plan: this device is the host twin, which runs no engine of a GPU.
  manifest({});
  write_text(engine + "/model.plan", plan);
  sense_plan*  made = nullptr;
  plan_options wanted;
  wanted.engine      = "tensorrt";
  wanted.engine_path = engine.c_str();
  CHECK(create(directory, &made, wanted) == SENSE_E_ENGINE && made == nullptr);
}

void case_no_allocation()
{
  // Every workspace exists when the plan is created. No window allocates, on the host or on the device.
  scratch           files;
  const std::string directory = files.sub("artifact");
  artifact_options  wanted;
  wanted.smoothing = "map_mean_kalman";
  write_artifact(directory, wanted);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  std::int64_t elapsed = -1;
  CHECK(sense_plan_warm_up(held.plan, 2, &elapsed) == SENSE_OK && elapsed >= 0);
  std::vector<std::vector<std::vector<std::uint8_t>>> windows;
  for (unsigned period = 0; period != 6; ++period) {
    windows.push_back(window_records(period, 3 * period, [period](unsigned n) { return varied(period * 3 + n); }));
  }
  sense_counters configured;
  CHECK(sense_plan_counters(held.plan, &configured) == SENSE_OK && configured.device_allocations > 0);
  sense_decision      decision;
  sense_smoothed      smoothed;
  unsigned            scored = 0;
  const std::uint64_t before = allocation_count();
  for (const auto& records : windows) {
    if (run_window(held.plan, records, decision) == SENSE_OK && decision.status == SENSE_DECISION_OK &&
        sense_plan_smooth(held.plan, &decision, &smoothed) == SENSE_OK && smoothed.active != 0) {
      ++scored;
    }
  }
  const std::uint64_t after = allocation_count();
  CHECK(scored == windows.size());
  CHECK(after == before);
  sense_counters counters;
  CHECK(sense_plan_counters(held.plan, &counters) == SENSE_OK);
  CHECK(counters.device_allocations == configured.device_allocations && counters.submissions == windows.size() &&
        counters.device_calls_open == 0 && counters.views_held == 0);
  // The sensor loop keeps what it will write in memory it had before the first record.
  struct listed final : sense::record_source {
    const std::vector<std::vector<std::vector<std::uint8_t>>>* windows = nullptr;
    std::size_t                                               at = 0;
    result next(sense_view& view, std::int64_t) override
    {
      if (at == 3 * windows->size()) {
        return result::closed;
      }
      view = view_of((*windows)[at / 3][at % 3]);
      ++at;
      return result::record;
    }
    std::uint64_t pending() override { return 0; }
    void          release(std::uint64_t) override {}
    std::uint64_t position() override { return at; }
    bool          live() const override { return false; }
  } source;
  std::vector<std::vector<std::vector<std::uint8_t>>> later;
  for (unsigned period = 10; period != 14; ++period) {
    later.push_back(window_records(period, 3 * period, [period](unsigned n) { return varied(period * 3 + n); }));
  }
  source.windows = &later;
  sense::loop_options options;
  options.lines     = 64;
  options.windows   = 8;
  options.keep_mean = true;
  sense::sensor_loop  loop(held.plan, options);
  const std::uint64_t loop_before = allocation_count();
  const sense_status  status      = loop.run(source);
  const std::uint64_t loop_after  = allocation_count();
  CHECK(status == SENSE_OK && loop.scored() == later.size() && loop.counters().lines == 3 * later.size());
  CHECK(loop_after == loop_before);
}

void case_one_submission()
{
  // The device as a counting shim: while a window is open no call reaches it, and a complete
  // window is one submission.
  scratch           files;
  const std::string directory = files.sub("artifact");
  write_artifact(directory);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  CHECK(sense_plan_warm_up(held.plan, 1, nullptr) == SENSE_OK);
  const auto     records = window_records(0, 0, [](unsigned n) { return varied(n); });
  sense_counters before, now;
  CHECK(sense_plan_counters(held.plan, &before) == SENSE_OK);
  sense_answer  answers[2];
  std::uint32_t count = 0;
  for (unsigned n = 0; n != 2; ++n) {
    const sense_view view = view_of(records[n]);
    CHECK(sense_plan_offer(held.plan, &view, answers, &count) == SENSE_OK && count == 1 &&
          answers[0].kind == SENSE_ANSWER_FILLING && answers[0].have == n + 1 && answers[0].need == 3);
    CHECK(sense_plan_counters(held.plan, &now) == SENSE_OK && now.device_calls == before.device_calls &&
          now.submissions == 0 && now.views_held == n + 1 && sense_plan_ready(held.plan) == 0);
    // Nothing to submit, nothing to collect and nothing to skip while the window is open.
    CHECK(sense_plan_submit(held.plan) == SENSE_E_STATE && sense_plan_poll(held.plan) == 0 &&
          sense_plan_skip(held.plan, nullptr) == SENSE_E_STATE);
  }
  const sense_view last = view_of(records[2]);
  CHECK(sense_plan_offer(held.plan, &last, answers, &count) == SENSE_OK && count == 1 && answers[0].kind == SENSE_ANSWER_READY &&
        answers[0].contributors == 3 && answers[0].trigger_sequence == 2);
  CHECK(sense_plan_counters(held.plan, &now) == SENSE_OK && now.device_calls == before.device_calls &&
        now.device_calls_open == 0 && now.submissions == 0 && now.views_held == 3 && sense_plan_ready(held.plan) == 1);
  std::uint64_t oldest = 99;
  CHECK(sense_plan_held(held.plan, &oldest) == 3 && oldest == 0);
  CHECK(sense_plan_submit(held.plan) == SENSE_OK);
  CHECK(sense_plan_counters(held.plan, &now) == SENSE_OK && now.submissions == 1 && now.device_calls > before.device_calls);
  CHECK(sense_plan_submit(held.plan) == SENSE_E_STATE); // one window runs at a time: nothing queues behind it
  while (sense_plan_poll(held.plan) == 0) {
  }
  sense_decision decision;
  CHECK(sense_plan_collect(held.plan, &decision) == SENSE_OK && decision.status == SENSE_DECISION_OK &&
        decision.trigger_sequence == 2 && decision.contributors == 3 && decision.contributing[0].sequence == 0);
  CHECK(sense_plan_counters(held.plan, &now) == SENSE_OK && now.submissions == 1 && now.decisions == 1 && now.scored == 1 &&
        now.views_held == 0 && now.device_allocations == before.device_allocations && now.device_calls_open == 0);
  CHECK(sense_plan_collect(held.plan, &decision) == SENSE_E_STATE);
  // The marks of a submission are in order.
  CHECK(decision.t_submit_ns <= decision.t_ingest_ns && decision.t_ingest_ns <= decision.t_kernels_ns &&
        decision.t_kernels_ns <= decision.t_engine_ns && decision.t_engine_ns <= decision.t_decision_ns);
}

int never_intact(void*)
{
  return 0;
}

void counted_release(void* owner)
{
  ++*static_cast<unsigned*>(owner);
}

void case_stale_record()
{
  // A record whose slot was rewritten while it was read yields no decision with a score, and
  // nothing of its window reaches the device.
  scratch           files;
  const std::string directory = files.sub("artifact");
  write_artifact(directory);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  const auto     records = window_records(0, 0, [](unsigned n) { return varied(n); });
  sense_answer   answers[2];
  std::uint32_t  count    = 0;
  unsigned       released = 0;
  sense_counters before, now;
  CHECK(sense_plan_counters(held.plan, &before) == SENSE_OK);
  for (unsigned n = 0; n != 3; ++n) {
    sense_view view = view_of(records[n]);
    view.release    = &counted_release;
    view.owner      = &released;
    if (n == 1) {
      view.intact = &never_intact; // the generation of this slot changed
    }
    CHECK(sense_plan_offer(held.plan, &view, answers, &count) == SENSE_OK);
  }
  CHECK(released == 0 && sense_plan_submit(held.plan) == SENSE_OK && sense_plan_poll(held.plan) == 1);
  sense_decision decision;
  CHECK(sense_plan_collect(held.plan, &decision) == SENSE_OK);
  CHECK(decision.status == SENSE_DECISION_STALE && decision.input == nullptr && decision.map == nullptr &&
        decision.logit == 0.0f && std::strstr(decision.reason, "rewritten") != nullptr);
  CHECK(sense_plan_counters(held.plan, &now) == SENSE_OK && now.stale == 1 && now.scored == 0 && now.submissions == 0 &&
        now.decisions == 1 && now.device_calls == before.device_calls && now.views_held == 0 && released == 3);
  std::vector<char> text(16384);
  CHECK(sense_decision_json(held.plan, &decision, 2, 0, nullptr, nullptr, nullptr, text.data(), text.size()) == SENSE_OK);
  const sense::json::value line = sense::json::parse(text.data());
  CHECK(line.string_at("status") == "stale" && line.at("score").is_null() && line.at("decision").is_null());
  // The window after it is scored as any other.
  const auto next = window_records(1, 3, [](unsigned n) { return varied(n); });
  CHECK(run_window(held.plan, next, decision) == SENSE_OK && decision.status == SENSE_DECISION_OK);

  // On the carrier the same fault shows before the plan sees the record: the loop is told that
  // the read was stale, and the window closes with a gap.
  struct stale_source final : sense::record_source {
    std::vector<std::vector<std::uint8_t>> records;
    std::size_t                            at = 0;
    result next(sense_view& view, std::int64_t) override
    {
      if (at == records.size()) {
        return result::closed;
      }
      const std::size_t n = at++;
      if (n == 1) {
        return result::stale;
      }
      view = view_of(records[n]);
      return result::record;
    }
    std::uint64_t pending() override { return 0; }
    void          release(std::uint64_t) override {}
    std::uint64_t position() override { return at; }
    bool          live() const override { return true; }
  } source;
  source.records = window_records(2, 6, [](unsigned n) { return varied(n); });
  sense::loop_options options;
  options.lines   = 16;
  options.windows = 4;
  sense::sensor_loop loop(held.plan, options);
  CHECK(loop.run(source) == SENSE_OK && loop.counters().stale_reads == 1 && loop.scored() == 0 && loop.counters().submissions == 0);
  CHECK(loop.counters().lines == 2 && !loop.lines()[1].is_decision && loop.lines()[1].answer.kind == SENSE_ANSWER_GAP &&
        loop.lines()[1].answer.missing == 1 && loop.lines()[1].answer.missing_slots[0] == 8);
}

void case_late_status()
{
  // A stage that takes longer than its budget is counted, and the decision is still published,
  // with the outcome late.
  scratch           files;
  const std::string directory = files.sub("artifact");
  write_artifact(directory);
  fake_clock   clock; // every question of the time moves it by one microsecond
  plan_options wanted;
  wanted.clock      = &clock;
  wanted.budgets[0] = 5000; // ingest: one step fits
  wanted.budgets[1] = 500;  // kernels: one step does not
  plan_holder held;
  CHECK(create(directory, &held.plan, wanted) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  const auto     records = window_records(0, 0, [](unsigned n) { return varied(n); });
  sense_decision decision;
  CHECK(run_window(held.plan, records, decision, clock.now) == SENSE_OK);
  CHECK(decision.status == SENSE_DECISION_OK && decision.late == 1 && decision.overrun_kernels == 1 &&
        decision.overrun_ingest == 0 && decision.overrun_engine == 0 && decision.overrun_window == 0);
  CHECK(decision.t_kernels_ns - decision.t_ingest_ns == 1000 && decision.t_ingest_ns - decision.t_submit_ns == 1000);
  sense_counters counters;
  CHECK(sense_plan_counters(held.plan, &counters) == SENSE_OK && counters.late == 1 && counters.overrun_kernels == 1 &&
        counters.overrun_ingest == 0 && counters.scored == 1 && counters.decisions == 1);
  std::vector<char> text(16384);
  char              hash[SENSE_HASH_BYTES];
  sense_sha256_hex("x", 1, hash);
  CHECK(sense_decision_json(held.plan, &decision, 2, 0, hash, hash, nullptr, text.data(), text.size()) == SENSE_OK);
  const sense::json::value line = sense::json::parse(text.data());
  CHECK(line.string_at("status") == "ok" && line.object_at("deadline").string_at("outcome") == "late" &&
        line.object_at("deadline").array_at("overrun").items.size() == 1 &&
        line.object_at("deadline").array_at("overrun").items[0].text == "kernels" &&
        line.object_at("deadline").object_at("budget_ns").integer_at("kernels") == 500 && !line.at("score").is_null());

  // The window budget runs from the notification of the last record to the decision.
  fake_clock   second;
  plan_options whole;
  whole.clock      = &second;
  whole.budgets[3] = 100000000; // 100 ms: on time
  plan_holder on_time;
  CHECK(create(directory, &on_time.plan, whole) == SENSE_OK);
  CHECK(run_window(on_time.plan, records, decision, second.now) == SENSE_OK && decision.status == SENSE_DECISION_OK &&
        decision.late == 0);
  CHECK(sense_decision_json(on_time.plan, &decision, 2, 0, hash, hash, nullptr, text.data(), text.size()) == SENSE_OK);
  CHECK(sense::json::parse(text.data()).object_at("deadline").string_at("outcome") == "on_time");
  fake_clock   third;
  whole.clock      = &third;
  whole.budgets[3] = 2000; // two microseconds: the window is late as a whole
  plan_holder late;
  CHECK(create(directory, &late.plan, whole) == SENSE_OK);
  CHECK(run_window(late.plan, records, decision, third.now) == SENSE_OK && decision.status == SENSE_DECISION_OK &&
        decision.late == 1 && decision.overrun_window == 1 && decision.overrun_kernels == 0);
  // A plan without a budget names no deadline.
  plan_holder plain;
  CHECK(create(directory, &plain.plan) == SENSE_OK);
  CHECK(run_window(plain.plan, records, decision) == SENSE_OK && decision.late == 0);
  CHECK(sense_decision_json(plain.plan, &decision, 2, 0, hash, hash, nullptr, text.data(), text.size()) == SENSE_OK);
  CHECK(sense::json::parse(text.data()).at("deadline").is_null());
}

/// A carrier whose records are all published before the sensor reads the first of them.
struct backlog_source final : sense::record_source {
  std::vector<std::vector<std::uint8_t>> records;
  std::size_t                            at = 0;
  std::vector<std::uint64_t>             releases;
  fake_clock*                            clock = nullptr;
  result next(sense_view& view, std::int64_t) override
  {
    if (at == records.size()) {
      return result::closed;
    }
    view = view_of(records[at]);
    ++at;
    return result::record;
  }
  std::uint64_t pending() override { return records.size() - at; }
  void          release(std::uint64_t sequence) override { releases.push_back(sequence); }
  std::uint64_t position() override { return at; }
  bool          live() const override { return true; }
};

void case_skip_backlog()
{
  // A sensor that starts two windows late takes the newest complete window and passes the two
  // before it: one decision, and three skipped records for each window it passed.
  scratch           files;
  const std::string directory = files.sub("artifact");
  write_artifact(directory);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  backlog_source source;
  for (unsigned period = 0; period != 3; ++period) {
    for (auto& record : window_records(period, 3 * period, [period](unsigned n) { return varied(period * 3 + n); })) {
      source.records.push_back(std::move(record));
    }
  }
  sense::loop_options options;
  options.lines   = 32;
  options.windows = 4;
  sense::sensor_loop loop(held.plan, options);
  sense_counters     before;
  CHECK(sense_plan_counters(held.plan, &before) == SENSE_OK);
  CHECK(loop.run(source) == SENSE_OK);
  sense_counters counters;
  CHECK(sense_plan_counters(held.plan, &counters) == SENSE_OK);
  CHECK(counters.submissions == 1 && counters.decisions == 1 && counters.scored == 1 && counters.stale_skipped == 2 &&
        counters.stale_skipped_records == 6 && counters.ready == 3 && counters.records == 9 && counters.views_held == 0 &&
        counters.device_calls_open == 0);
  CHECK(loop.scored() == 1 && loop.counters().submissions == 1 && loop.counters().read == 9 && loop.counters().closed);
  // Six lines of filling windows, two windows that were passed, and one decision: of the newest.
  unsigned filling = 0, skipped = 0, decisions = 0;
  for (std::size_t n = 0; n != loop.counters().lines; ++n) {
    const sense::kept_line& line = loop.lines()[n];
    if (line.is_decision) {
      ++decisions;
      CHECK(line.decision.period == 2 && line.decision.trigger_sequence == 8 && line.decision.status == SENSE_DECISION_OK);
    } else if (line.answer.kind == SENSE_ANSWER_STALE_SKIPPED) {
      CHECK(line.answer.period == skipped && line.answer.contributors == 3 && line.answer.contributing[0].sequence == 3 * skipped);
      ++skipped;
    } else {
      filling += line.answer.kind == SENSE_ANSWER_FILLING ? 1U : 0U;
    }
  }
  CHECK(loop.counters().lines == 9 && filling == 6 && skipped == 2 && decisions == 1);
  // Records are released in order: a release never goes back, and the last one names every record.
  bool ordered = !source.releases.empty();
  for (std::size_t n = 1; n < source.releases.size(); ++n) {
    ordered = ordered && source.releases[n] >= source.releases[n - 1];
  }
  CHECK(ordered && source.releases.back() == 9);
  // While the first window waited, its three records were held: nothing below them was released.
  CHECK(source.releases.size() >= 3 && source.releases[2] == 0);
  // The lines of a window that was passed.
  std::vector<std::string> window_lines, decision_lines;
  loop.seal(window_lines, decision_lines);
  CHECK(window_lines.size() == 9 && decision_lines.size() == 9);
  unsigned named = 0;
  for (const std::string& text : decision_lines) {
    const sense::json::value line = sense::json::parse(text);
    CHECK(line.string_at("schema") == "cusense-decision/v1" && line.string_at("runtime") == "native");
    named += line.string_at("status") == "stale_skipped" ? 1U : 0U;
  }
  CHECK(named == 2);
  // An archive is no live carrier: every window of it runs.
  plan_holder offline;
  CHECK(create(directory, &offline.plan) == SENSE_OK);
  struct archive final : sense::record_source {
    backlog_source* inner = nullptr;
    result          next(sense_view& view, std::int64_t limit) override { return inner->next(view, limit); }
    std::uint64_t   pending() override { return inner->pending(); }
    void            release(std::uint64_t) override {}
    std::uint64_t   position() override { return inner->position(); }
    bool            live() const override { return false; }
  } replay;
  source.at    = 0;
  replay.inner = &source;
  sense::sensor_loop every(offline.plan, options);
  CHECK(every.run(replay) == SENSE_OK && every.scored() == 3 && every.counters().submissions == 3);
}

void case_after_radio()
{
  // The release after_radio: the submission waits until the calibrated rest of the last uplink
  // slot has passed since t_ready of the last record, and makes no device call while it waits.
  scratch           files;
  const std::string directory = files.sub("artifact");
  write_artifact(directory);
  constexpr std::int64_t ALLOWANCE = 300000; // 300 us
  for (const bool after_radio : {true, false}) {
    fake_clock   clock;
    plan_options wanted;
    wanted.clock = &clock;
    plan_holder held;
    CHECK(create(directory, &held.plan, wanted) == SENSE_OK);
    if (held.plan == nullptr) {
      return;
    }
    record_options base;
    base.t_ready = clock.now + 2000000; // the records become ready after the plan exists
    backlog_source source;
    source.records = window_records(0, 0, [](unsigned n) { return varied(n); }, base);
    obs::record_header last;
    std::memcpy(&last, source.records[2].data(), sizeof(last));
    const std::int64_t target = last.t_ready_ns + ALLOWANCE;
    // The fake clock stands before t_ready of the last record: the radio still works on its slot.
    clock.now = last.t_ready_ns - 5000;
    std::uint64_t calls_low = ~0ULL, calls_high = 0;
    bool          watching  = false;
    sense_plan*   plan      = held.plan;
    clock.watch             = [&](std::int64_t now) {
      if (watching && now < target) {
        sense_counters seen;
        sense_plan_counters(plan, &seen);
        calls_low  = std::min<std::uint64_t>(calls_low, seen.device_calls);
        calls_high = std::max<std::uint64_t>(calls_high, seen.device_calls);
      }
    };
    sense::loop_options options;
    options.lines              = 16;
    options.windows            = 2;
    options.clock              = &fake_clock::tick;
    options.clock_owner        = &clock;
    options.release            = after_radio ? sense::loop_options::release_policy::after_radio
                                             : sense::loop_options::release_policy::immediate;
    options.radio_allowance_ns = ALLOWANCE;
    sense::sensor_loop loop(held.plan, options);
    sense_counters     before;
    sense_plan_counters(held.plan, &before);
    watching = true;
    CHECK(loop.run(source) == SENSE_OK && loop.scored() == 1);
    watching = false;
    const sense_decision& decision = loop.lines()[2].decision;
    CHECK(loop.lines()[2].is_decision && decision.status == SENSE_DECISION_OK);
    if (after_radio) {
      // Submitted at the first tick at or after the target, and not a tick earlier.
      CHECK(decision.t_submit_ns >= target && decision.t_submit_ns < target + 3 * clock.step);
      CHECK(loop.counters().waited_radio == 1);
      CHECK(calls_low == before.device_calls && calls_high == before.device_calls);
    } else {
      CHECK(decision.t_submit_ns < target && loop.counters().waited_radio == 0);
    }
  }
}

void case_gap_and_reset()
{
  scratch           files;
  const std::string directory = files.sub("artifact");
  artifact_options  wanted;
  wanted.min_valid = "0.9";
  write_artifact(directory, wanted);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  std::uint64_t sequence = 0;
  sense_answer  answers[2];
  std::uint32_t count = 0;
  std::vector<std::vector<std::uint8_t>> kept; // a view must stay valid while the plan holds it
  const auto offer = [&](std::uint64_t slot, const std::function<void(record_options&)>& change = nullptr,
                         const sample_fn& sample = constant(1.0f, 0.5f)) {
    record_options options;
    options.slot     = slot;
    options.sequence = sequence++;
    if (change) {
      change(options);
    }
    kept.push_back(make_record(options, sample));
    const sense_view view = view_of(kept.back());
    return sense_plan_offer(held.plan, &view, answers, &count);
  };
  // A slot of the window that never arrived: the window closes with a gap at its last slot.
  CHECK(offer(7) == SENSE_OK && count == 1 && answers[0].kind == SENSE_ANSWER_FILLING);
  CHECK(offer(9) == SENSE_OK && count == 1 && answers[0].kind == SENSE_ANSWER_GAP && answers[0].missing == 1 &&
        answers[0].missing_slots[0] == 8 && answers[0].have == 2 && answers[0].need == 3 && answers[0].period == 0);
  // A window whose last slot never came is closed by the first record of a later period.
  CHECK(offer(17) == SENSE_OK && count == 1 && answers[0].kind == SENSE_ANSWER_FILLING);
  CHECK(offer(27) == SENSE_OK && count == 2 && answers[0].kind == SENSE_ANSWER_GAP && answers[0].period == 1 &&
        answers[0].missing == 2 && answers[0].missing_slots[0] == 8 && answers[0].missing_slots[1] == 9 &&
        answers[1].kind == SENSE_ANSWER_FILLING && answers[1].period == 2 && answers[1].have == 1);
  // A slot outside the window is passed, and not answered.
  CHECK(offer(31) == SENSE_OK && count == 0);
  // Another episode starts a new history: the open window is dropped without an answer.
  CHECK(offer(37) == SENSE_OK && count == 2 && answers[0].kind == SENSE_ANSWER_GAP && answers[1].kind == SENSE_ANSWER_FILLING);
  CHECK(offer(38, [](record_options& options) { options.episode = 6; }) == SENSE_OK && count == 1 &&
        answers[0].kind == SENSE_ANSWER_FILLING && answers[0].have == 1 && answers[0].episode == 6);
  CHECK(offer(39, [](record_options& options) { options.episode = 6; }) == SENSE_OK && count == 1 &&
        answers[0].kind == SENSE_ANSWER_GAP && answers[0].missing == 1 && answers[0].missing_slots[0] == 7);
  // A time that went back starts one as well, and so does another epoch.
  CHECK(offer(47, [](record_options& options) { options.episode = 6; }) == SENSE_OK && count == 1);
  CHECK(offer(27, [](record_options& options) { options.episode = 6; }) == SENSE_OK && count == 1 &&
        answers[0].kind == SENSE_ANSWER_FILLING && answers[0].have == 1 && answers[0].period == 2);
  CHECK(offer(28, [](record_options& options) {
          options.episode = 6;
          options.epoch   = 2;
        }) == SENSE_OK &&
        count == 1 && answers[0].have == 1 && answers[0].epoch == 2);
  sense_counters counters;
  CHECK(sense_plan_counters(held.plan, &counters) == SENSE_OK && counters.resets == 3 && counters.gap == 4 &&
        counters.passed == 1 && counters.views_held == 1);
  // A record of another stage is no input of the specification.
  CHECK(offer(57, [](record_options& options) {
          options.layout = obs::LAYOUT_OCUDU_O2_COMB_C64;
          options.steps  = obs::STEP_PILOT_LS;
          options.epoch  = 2;
          options.episode = 6;
        }) == SENSE_OK &&
        count == 1 && answers[0].kind == SENSE_ANSWER_REJECTED && std::strstr(answers[0].reason, "the specification reads O3") != nullptr);
  // An estimate after time interpolation is another stage of the chain.
  CHECK(offer(57, [](record_options& options) {
          options.steps |= obs::STEP_TIME_INTERP;
          options.epoch   = 2;
          options.episode = 6;
        }) == SENSE_OK &&
        count == 1 && answers[0].kind == SENSE_ANSWER_REJECTED);
  // Slots that were not estimated by one chain of steps form no window.
  const auto in_epoch = [](record_options& options) {
    options.epoch   = 3;
    options.episode = 6;
  };
  CHECK(offer(67, in_epoch) == SENSE_OK && offer(68, [&](record_options& options) {
          in_epoch(options);
          options.steps |= obs::STEP_SMOOTHING;
        }) == SENSE_OK);
  CHECK(offer(69, in_epoch) == SENSE_OK && count == 1 && answers[0].kind == SENSE_ANSWER_INVALID &&
        answers[0].contributors == 3 && std::strstr(answers[0].reason, "one chain of steps") != nullptr);
  // A window on other array elements than the model was trained on is none either.
  const auto elsewhere = [&](record_options& options) {
    in_epoch(options);
    const std::uint16_t map[RX] = {0, 1, 2, 5};
    std::memcpy(options.antenna_map, map, sizeof(map));
  };
  CHECK(offer(77, elsewhere) == SENSE_OK && offer(78, elsewhere) == SENSE_OK);
  CHECK(offer(79, elsewhere) == SENSE_OK && count == 1 && answers[0].kind == SENSE_ANSWER_INVALID &&
        std::strstr(answers[0].reason, "trained on") != nullptr);
  CHECK(sense_plan_ready(held.plan) == 0);
  // A complete window with too few valid samples is decided as invalid: it is not scored.
  CHECK(offer(87, in_epoch, constant(1.0f, 0.5f, false)) == SENSE_OK && offer(88, in_epoch, constant(1.0f, 0.5f, false)) == SENSE_OK);
  const sample_fn sparse = [](unsigned, unsigned k, unsigned, float& real, float& imaginary) {
    real      = 1.0f;
    imaginary = 0.5f;
    return k % 2 == 0; // half of the elements: below the share 0.9 of the specification
  };
  CHECK(offer(89, in_epoch, sparse) == SENSE_OK && count == 1 && answers[0].kind == SENSE_ANSWER_READY);
  CHECK(sense_plan_submit(held.plan) == SENSE_OK);
  while (sense_plan_poll(held.plan) == 0) {
  }
  sense_decision decision;
  CHECK(sense_plan_collect(held.plan, &decision) == SENSE_OK && decision.status == SENSE_DECISION_INVALID &&
        decision.valid == decision.elements / 2 && std::strstr(decision.reason, "below the share") != nullptr);
  // A carrier that went silent: the window that is filling is given up as a gap, and its views return.
  CHECK(sense_plan_close_open(held.plan, &answers[0]) == SENSE_E_STATE);
  CHECK(offer(94, in_epoch) == SENSE_OK && count == 0 && offer(97, in_epoch) == SENSE_OK && offer(98, in_epoch) == SENSE_OK);
  CHECK(sense_plan_counters(held.plan, &counters) == SENSE_OK && counters.views_held == 2);
  CHECK(sense_plan_close_open(held.plan, &answers[0]) == SENSE_OK && answers[0].kind == SENSE_ANSWER_GAP &&
        answers[0].missing == 1 && answers[0].missing_slots[0] == 9 && answers[0].have == 2 && answers[0].period == 9);
  CHECK(sense_plan_counters(held.plan, &counters) == SENSE_OK && counters.views_held == 0 && counters.gap == 5);
  // The end of a stream closes an open window without an answer and says what it held.
  CHECK(offer(107, in_epoch) == SENSE_OK);
  std::uint32_t have = 0, need = 0;
  CHECK(sense_plan_finish(held.plan, &have, &need) == SENSE_OK && have == 1 && need == 3);
  CHECK(sense_plan_counters(held.plan, &counters) == SENSE_OK && counters.views_held == 0 && counters.rejected == 2 &&
        counters.invalid == 3 && counters.device_calls_open == 0 && counters.passed == 2);
  // Lines of the answers that carry no score.
  std::vector<char> text(16384);
  CHECK(offer(117, in_epoch) == SENSE_OK && offer(119, in_epoch) == SENSE_OK && answers[0].kind == SENSE_ANSWER_GAP);
  CHECK(sense_answer_window_json(held.plan, &answers[0], 4, text.data(), text.size()) == SENSE_OK);
  const sense::json::value window = sense::json::parse(text.data());
  CHECK(window.string_at("schema") == "cusense-window/v1" && window.string_at("status") == "gap" &&
        window.integer_at("line") == 4 && window.array_at("missing").items.size() == 1);
  CHECK(sense_answer_decision_json(held.plan, &answers[0], 4, text.data(), text.size()) == SENSE_OK);
  const sense::json::value line = sense::json::parse(text.data());
  CHECK(line.string_at("schema") == "cusense-decision/v1" && line.string_at("status") == "gap" && line.at("score").is_null());
  CHECK(sense_answer_window_json(held.plan, &answers[0], 4, text.data(), 16) == SENSE_E_ARGUMENT);
}

void case_smoothing()
{
  // The optional stage on its own: the mean of the newest maps, its largest cell, and a
  // constant-velocity Kalman filter on the centre of that cell.
  const double  cell[2] = {0.5, 0.25}, origin[2] = {-1.0, 2.0};
  sense::smoother filter(2, 1.0, 0.25, 1.5, 2, 3, cell, origin, 0.005);
  const float   first[6]  = {0.1f, 0.6f, 0.1f, 0.1f, 0.05f, 0.05f}; // the largest cell is [0, 1]
  const float   second[6] = {0.0f, 0.2f, 0.0f, 0.0f, 0.0f, 0.8f};  // the largest cell is [1, 2]
  sense::smoothed made = filter.update(1, 5, 10, first);
  CHECK(made.maps == 1 && made.row == 0 && made.col == 1 && made.x_m == -1.0 + 1.5 * 0.25 && made.y_m == 2.0 + 0.5 * 0.5);
  CHECK(made.probability == static_cast<double>(0.6f) && made.kalman_x_m == made.x_m && made.kalman_y_m == made.y_m &&
        made.kalman_vx == 0.0 && made.kalman_vy == 0.0);
  made = filter.update(1, 5, 11, second);
  // The mean of the two maps is largest at [1, 2]: (0.05 + 0.8) / 2 against (0.6 + 0.2) / 2.
  CHECK(made.maps == 2 && made.row == 1 && made.col == 2);
  CHECK(std::fabs(made.probability - (static_cast<double>(0.05f) + static_cast<double>(0.8f)) / 2) < 1e-15);
  // One step of the filter by hand, on the x axis, which is independent of y:
  //   P = [[r, 0], [0, v0^2]], F = [[1, dt], [0, 1]], Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]].
  const double dt = 0.005, q = 1.0, r = 0.25, v0 = 1.5;
  const double p00 = r + dt * dt * v0 * v0 + q * dt * dt * dt / 3, p01 = dt * v0 * v0 + q * dt * dt / 2;
  const double x0 = -1.0 + 1.5 * 0.25, z = -1.0 + 2.5 * 0.25;
  const double gain0 = p00 / (p00 + r), gain1 = p01 / (p00 + r);
  CHECK(std::fabs(made.kalman_x_m - (x0 + gain0 * (z - x0))) < sense::tolerance::SMOOTHED_POSITION_M);
  CHECK(std::fabs(made.kalman_vx - gain1 * (z - x0)) < 1e-9);
  CHECK(made.kalman_x_m > x0 && made.kalman_x_m < z && made.kalman_y_m > 2.25 && made.kalman_y_m < 2.75);
  // A third map pushes the first one out: the history holds two.
  made = filter.update(1, 5, 12, second);
  CHECK(made.maps == 2 && made.probability == static_cast<double>(0.8f));
  // Another episode, another epoch and a period that does not follow each start a new history.
  CHECK(filter.update(1, 6, 13, first).maps == 1 && filter.update(1, 6, 14, first).maps == 2);
  CHECK(filter.update(2, 6, 15, first).maps == 1 && filter.update(2, 6, 15, first).maps == 1);

  // Through a plan: once per scored decision, in their order, and off for an artifact without it.
  scratch           files;
  const std::string directory = files.sub("artifact");
  artifact_options  wanted;
  wanted.smoothing = "map_mean_kalman";
  write_artifact(directory, wanted);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  sense_decision      decision;
  sense_smoothed      smoothed;
  std::vector<double> sum(ROWS * COLS, 0.0);
  for (unsigned period = 0; period != 2; ++period) {
    const auto records = window_records(period, 3 * period, [period](unsigned n) { return varied(7 * period + n); });
    CHECK(run_window(held.plan, records, decision) == SENSE_OK && decision.status == SENSE_DECISION_OK);
    CHECK(sense_plan_smooth(held.plan, &decision, &smoothed) == SENSE_OK && smoothed.active == 1 && smoothed.maps == period + 1);
    for (unsigned c = 0; c != ROWS * COLS; ++c) {
      sum[c] += static_cast<double>(decision.map[c]);
    }
  }
  unsigned best = 0;
  for (unsigned c = 0; c != ROWS * COLS; ++c) {
    best = sum[c] / 2 > sum[best] / 2 ? c : best;
  }
  CHECK(smoothed.row == best / COLS && smoothed.col == best % COLS && std::fabs(smoothed.probability - sum[best] / 2) < 1e-15);
  CHECK(smoothed.x_m == -1.0 + (smoothed.col + 0.5) * 0.25 && smoothed.y_m == 2.0 + (smoothed.row + 0.5) * 0.5);
  std::vector<char> text(16384);
  char              hash[SENSE_HASH_BYTES];
  sense_sha256_hex("x", 1, hash);
  CHECK(sense_decision_json(held.plan, &decision, 5, 1, hash, hash, &smoothed, text.data(), text.size()) == SENSE_OK);
  const sense::json::value line = sense::json::parse(text.data());
  CHECK(line.object_at("smoothed").string_at("entry") == "map_mean_kalman" && line.object_at("smoothed").integer_at("maps") == 2);
  const std::string plain = files.sub("plain");
  write_artifact(plain);
  plan_holder without;
  CHECK(create(plain, &without.plan) == SENSE_OK);
  const auto records = window_records(0, 0, [](unsigned n) { return varied(n); });
  CHECK(run_window(without.plan, records, decision) == SENSE_OK &&
        sense_plan_smooth(without.plan, &decision, &smoothed) == SENSE_OK && smoothed.active == 0);
}

void case_lines()
{
  // The lines of a scored window, as the reference reader takes them.
  scratch           files;
  const std::string directory = files.sub("artifact");
  const std::string hash      = write_artifact(directory);
  plan_holder       held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  const auto     records = window_records(4, 12, [](unsigned n) { return varied(n); });
  sense_decision decision;
  CHECK(run_window(held.plan, records, decision, 777) == SENSE_OK && decision.status == SENSE_DECISION_OK);
  sense_plan_info info;
  CHECK(sense_plan_info_get(held.plan, &info) == SENSE_OK);
  char input_hash[SENSE_HASH_BYTES], map_hash[SENSE_HASH_BYTES], mean_hash[SENSE_HASH_BYTES], count_hash[SENSE_HASH_BYTES];
  sense_sha256_hex(decision.input, sizeof(float) * RX * POSITIONS, input_hash);
  sense_sha256_hex(decision.map, sizeof(float) * ROWS * COLS, map_hash);
  sense_sha256_hex(decision.mean, 2 * sizeof(float) * RX * POSITIONS * SYMBOLS, mean_hash);
  sense_sha256_hex(decision.count, RX * POSITIONS * SYMBOLS, count_hash);
  std::vector<char> text(32768);
  CHECK(sense_decision_json(held.plan, &decision, 2, 0, input_hash, map_hash, nullptr, text.data(), text.size()) == SENSE_OK);
  const sense::json::value line = sense::json::parse(text.data());
  CHECK(line.string_at("schema") == "cusense-decision/v1" && line.string_at("status") == "ok" &&
        line.string_at("runtime") == "native" && line.integer_at("line") == 2);
  CHECK(line.object_at("model").string_at("artifact_sha256") == hash);
  CHECK(line.object_at("engine").string_at("backend") == "host_reference" &&
        line.object_at("engine").string_at("sha256") == info.engine_sha256 && line.string_at("plan_sha256") == info.plan_sha256 &&
        line.string_at("core_sources_sha256") == sense::core_sources_sha256() && line.string_at("device") == "host");
  CHECK(line.object_at("score").number_at("logit") == static_cast<double>(decision.logit) &&
        line.object_at("score").number_at("probability") == decision.probability);
  CHECK(line.object_at("decision").at("present").boolean == (decision.present != 0) &&
        line.object_at("decision").number_at("threshold_logit") == 0.0);
  const sense::json::value& location = line.object_at("location");
  CHECK(location.array_at("cell").items[0].number == decision.row && location.array_at("cell").items[1].number == decision.col &&
        location.object_at("map").string_at("sha256") == map_hash && location.object_at("map").integer_at("index") == 0);
  CHECK(line.object_at("input").string_at("sha256") == input_hash && line.at("smoothed").is_null());
  CHECK(line.integer_at("t_decision_ns") == decision.t_decision_ns && line.integer_at("t_window_ns") == 777 &&
        line.object_at("marks").integer_at("t_submit_ns") == decision.t_submit_ns && line.at("deadline").is_null());
  CHECK(line.array_at("contributing").items.size() == 3 && line.integer_at("infer_ns") == decision.t_engine_ns - decision.t_kernels_ns);
  // Timed evidence: the digests are taken by nobody, and the line says so with null.
  CHECK(sense_decision_window_json(held.plan, &decision, 2, 0, nullptr, nullptr, text.data(), text.size()) == SENSE_OK);
  CHECK(sense::json::parse(text.data()).object_at("mean").at("sha256").is_null());
  CHECK(sense_decision_window_json(held.plan, &decision, 2, 0, mean_hash, count_hash, text.data(), text.size()) == SENSE_OK);
  const sense::json::value window = sense::json::parse(text.data());
  CHECK(window.string_at("schema") == "cusense-window/v1" && window.string_at("status") == "ok" &&
        window.string_at("runtime") == "native" && window.object_at("mean").string_at("sha256") == mean_hash);
  CHECK(window.object_at("validity").integer_at("elements") == RX * POSITIONS * SYMBOLS &&
        window.object_at("validity").integer_at("valid") == decision.valid);
  CHECK(window.array_at("contributing").items.size() == 3);
  // A buffer that is too small is refused, and nothing is cut.
  CHECK(sense_decision_json(held.plan, &decision, 2, 0, input_hash, map_hash, nullptr, text.data(), 64) == SENSE_E_ARGUMENT);
  // The plan as a manifest names it.
  const sense::json::value plan = sense::json::parse(info.plan_json);
  CHECK(plan.string_at("schema") == "sense-plan/v1" && plan.array_at("stages").items.size() == 9 &&
        plan.string_at("artifact_sha256") == hash && sense::sha256_hex(std::string(info.plan_json)) == info.plan_sha256);
  const sense::json::value entries = sense::json::parse(info.entries_json);
  CHECK(entries.object_at("clutter").string_at("entry") == "background_magnitude" &&
        entries.object_at("model").string_at("runtime") == "native");
}

void case_ring_backpressure()
{
  // The sensor on a real ring, slower than its producer. The window it works on is held in place:
  // its credits do not come back before the decision. The producer never waits for them: a record
  // that finds no credit is dropped and counted, and nothing is queued anywhere.
  scratch           files;
  const std::string directory = files.sub("artifact");
  write_artifact(directory);
  plan_holder held;
  CHECK(create(directory, &held.plan) == SENSE_OK);
  if (held.plan == nullptr) {
    return;
  }
  CHECK(sense_plan_warm_up(held.plan, 1, nullptr) == SENSE_OK);
  constexpr unsigned WINDOWS = 30;
  std::vector<std::vector<std::uint8_t>> records;
  for (unsigned period = 0; period != WINDOWS; ++period) {
    for (auto& record : window_records(period, 0, [period](unsigned n) { return varied(period * 3 + n); })) {
      records.push_back(std::move(record));
    }
  }
  obs::record_header shape;
  std::memcpy(&shape, records[0].data(), sizeof(shape));
  obs::ring_config config;
  config.socket_path      = files.path + "/obs.sock";
  config.slot_count       = 4;
  config.host_capacity    = 0;
  config.mask_capacity    = shape.mask_bytes;
  config.payload_capacity = shape.payload_bytes;
  obs::producer writer(config);

  sense::loop_options options;
  options.lines   = 256;
  options.windows = 64;
  options.hold_ns = 4000000; // the sensor works four milliseconds on each window
  options.idle_ns = 5000000000LL;
  sense::sensor_loop loop(held.plan, options);
  std::atomic<bool>  attached{false}, failed{false}, finished{false};
  sense_status       status = SENSE_E_STATE;
  obs::consumer_counts read_counts;
  std::thread          sensor([&]() {
    try {
      obs::consumer_config wanted;
      wanted.socket_path   = config.socket_path;
      wanted.mode          = obs::wait_mode::spin;
      wanted.connect_limit = std::chrono::milliseconds(5000);
      obs::consumer      reader(wanted);
      sense::ring_source source(reader, nullptr);
      attached = true;
      status   = loop.run(source);
      read_counts = reader.counts();
    } catch (const std::exception& fault) {
      std::fprintf(stderr, "sensor: %s\n", fault.what());
      failed = true;
    }
    finished = true;
  });
  while (!attached && !failed) {
    writer.service();
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  writer.service();
  std::uint64_t held_most = 0;
  std::int64_t  slowest   = 0;
  const std::int64_t started = obs::monotonic_ns();
  for (const std::vector<std::uint8_t>& record : records) {
    const std::int64_t before = obs::monotonic_ns();
    obs::slot          acquired;
    if (writer.try_acquire(acquired)) {
      obs::record_header header;
      std::memcpy(&header, record.data(), sizeof(header));
      header.t_ready_ns = obs::monotonic_ns();
      *acquired.header  = header;
      std::memcpy(acquired.mask, record.data() + sizeof(header), header.mask_bytes);
      std::memcpy(acquired.payload, record.data() + sizeof(header) + header.mask_bytes, header.payload_bytes);
      writer.publish(acquired);
    }
    slowest = std::max(slowest, obs::monotonic_ns() - before);
    writer.service();
    held_most = std::max<std::uint64_t>(held_most, writer.counts().pending);
    std::this_thread::sleep_for(std::chrono::microseconds(300));
  }
  const std::int64_t offered_ns = obs::monotonic_ns() - started;
  const std::uint64_t pending   = writer.drain(std::chrono::milliseconds(5000));
  writer.finish();
  sensor.join();
  const obs::producer_counts counts = writer.counts();
  sense_counters             counters;
  CHECK(sense_plan_counters(held.plan, &counters) == SENSE_OK);
  CHECK(!failed && status == SENSE_OK && pending == 0);
  // Every record was admitted or dropped at the ring, and the sensor read what was published.
  CHECK(counts.admitted + counts.dropped == records.size() && counts.published == counts.admitted);
  CHECK(read_counts.read + read_counts.stale + read_counts.skipped == counts.published && counts.faults == 0);
  // The sensor was slower than the producer: records were dropped, and counted.
  CHECK(counts.dropped > 0);
  // The credits of the window it worked on were held: at least one whole window was pending.
  CHECK(held_most >= 3 && held_most <= config.slot_count);
  // The producer never waited: it made one attempt per record, no system call between an
  // admission and its publication, and went on where it found no credit. The times are printed
  // and not graded: they are those of a loaded host.
  CHECK(counts.window_calls == 0);
  // A window that was still filling when the producer stopped was closed by the silence of the
  // carrier, so every credit came back before the producer finished.
  CHECK(loop.counters().closed && !loop.counters().idle);
  // Whatever it decided, it decided on three records in sequence of one period: a drop makes a
  // gap, never a window of the wrong records.
  unsigned decided = 0, wrong = 0;
  for (std::size_t n = 0; n != loop.counters().lines && n != loop.lines().size(); ++n) {
    const sense::kept_line& line = loop.lines()[n];
    if (!line.is_decision) {
      continue;
    }
    ++decided;
    const sense_contributor* part = line.decision.contributing;
    wrong += line.decision.contributors == 3 && part[1].sequence == part[0].sequence + 1 &&
                     part[2].sequence == part[1].sequence + 1 && part[0].absolute_slot / PERIOD == line.decision.period &&
                     part[2].absolute_slot == line.decision.period * PERIOD + 9
                 ? 0U
                 : 1U;
  }
  CHECK(decided >= 1 && decided < WINDOWS && wrong == 0);
  CHECK(counters.decisions == decided && counters.views_held == 0 && counters.device_calls_open == 0);
  std::printf("ring_backpressure: offered %zu in %.1f ms, dropped %llu, decisions %u, gaps %llu, skipped windows %llu, "
              "closed by silence %llu, most credits held %llu, slowest publication %.1f us\n",
              records.size(), static_cast<double>(offered_ns) / 1e6, static_cast<unsigned long long>(counts.dropped), decided,
              static_cast<unsigned long long>(counters.gap), static_cast<unsigned long long>(counters.stale_skipped),
              static_cast<unsigned long long>(loop.counters().closed_open), static_cast<unsigned long long>(held_most),
              static_cast<double>(slowest) / 1e3);
}

int case_device_vectors()
{
  // The kernels on the device against their twin on the host: one window, the same records.
  scratch           files;
  const std::string directory = files.sub("artifact");
  artifact_options  wanted;
  wanted.template_hole = true;
  write_artifact(directory, wanted);
  plan_options on_device;
  on_device.device = "cuda";
  plan_holder device, twin;
  const sense_status status = create(directory, &device.plan, on_device);
  if (status == SENSE_E_DEVICE) {
    std::printf("device_vectors: skipped, no device is reachable: %s\n", sense_last_error());
    return 77;
  }
  CHECK(status == SENSE_OK && create(directory, &twin.plan) == SENSE_OK);
  if (device.plan == nullptr || twin.plan == nullptr) {
    std::fprintf(stderr, "%s\n", sense_last_error());
    return 1;
  }
  CHECK(sense_plan_warm_up(device.plan, 2, nullptr) == SENSE_OK);
  sense_counters configured;
  CHECK(sense_plan_counters(device.plan, &configured) == SENSE_OK);
  bool          identical = true;
  double        worst     = 0.0;
  const std::size_t plane = static_cast<std::size_t>(RX) * POSITIONS, elements = plane * SYMBOLS;
  for (unsigned period = 0; period != 4; ++period) {
    const auto     records = window_records(period, 3 * period, [period](unsigned n) { return varied(period * 5 + n); });
    sense_decision on, off;
    CHECK(run_window(device.plan, records, on) == SENSE_OK && run_window(twin.plan, records, off) == SENSE_OK);
    CHECK(on.status == SENSE_DECISION_OK && off.status == SENSE_DECISION_OK);
    if (on.status != SENSE_DECISION_OK || off.status != SENSE_DECISION_OK) {
      return 1;
    }
    CHECK(std::memcmp(on.count, off.count, elements) == 0 && on.valid == off.valid);
    identical = identical && std::memcmp(on.input, off.input, sizeof(float) * plane) == 0 &&
                std::memcmp(on.mean, off.mean, 2 * sizeof(float) * elements) == 0 &&
                std::memcmp(on.difference, off.difference, sizeof(float) * plane) == 0;
    for (std::size_t at = 0; at != plane; ++at) {
      worst = std::max(worst, std::fabs(static_cast<double>(on.input[at]) - static_cast<double>(off.input[at])));
    }
    // The same model on the same input: the scores agree within the engine tolerance.
    CHECK(std::fabs(on.logit - off.logit) <= sense::tolerance::LOGIT_ABSOLUTE);
    CHECK(on.t_submit_ns <= on.t_ingest_ns && on.t_ingest_ns <= on.t_kernels_ns && on.t_kernels_ns <= on.t_engine_ns &&
          on.t_engine_ns <= on.t_decision_ns);
  }
  // The units are written once and compiled twice without fused multiply-add: the bound is the
  // declared one, and the bits are expected to be the same.
  CHECK(worst <= sense::tolerance::INPUT_ULPS * sense::tolerance::FP32_SPACING * 2.0 / (0.1 + 1e-8));
  sense_counters counters;
  CHECK(sense_plan_counters(device.plan, &counters) == SENSE_OK && counters.device_allocations == configured.device_allocations &&
        counters.device_calls_open == 0 && counters.submissions == 4);
  std::printf("device_vectors: kernels against the host twin: %s, largest input difference %.3g\n",
              identical ? "bit-identical" : "within tolerance, not bit-identical", worst);
  return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
  const std::map<std::string, void (*)()> cases = {
      {"sha256", case_sha256},
      {"json_canonical", case_json_canonical},
      {"half_widening", case_half_widening},
      {"vectors", case_vectors},
      {"placement", case_placement},
      {"o3_element", case_o3_element},
      {"aerial_same_cube", case_aerial_same_cube},
      {"artifact_hashes", case_artifact_hashes},
      {"registry_refused", case_registry_refused},
      {"engine_manifest", case_engine_manifest},
      {"no_allocation", case_no_allocation},
      {"one_submission", case_one_submission},
      {"stale_record", case_stale_record},
      {"late_status", case_late_status},
      {"skip_backlog", case_skip_backlog},
      {"after_radio", case_after_radio},
      {"gap_and_reset", case_gap_and_reset},
      {"smoothing", case_smoothing},
      {"lines", case_lines},
      {"ring_backpressure", case_ring_backpressure}};
  if (argc != 2) {
    std::fprintf(stderr, "Usage: sense_core_test <case>\n");
    for (const auto& item : cases) {
      std::fprintf(stderr, "  %s\n", item.first.c_str());
    }
    std::fprintf(stderr, "  device_vectors\n");
    return 2;
  }
  const std::string name = argv[1];
  if (name == "device_vectors") {
    return case_device_vectors();
  }
  const auto found = cases.find(name);
  if (found == cases.end()) {
    std::fprintf(stderr, "sense_core_test: no case %s\n", name.c_str());
    return 2;
  }
  try {
    found->second();
  } catch (const std::exception& fault) {
    std::fprintf(stderr, "%s: %s\n", name.c_str(), fault.what());
    return 1;
  }
  if (failures != 0) {
    std::fprintf(stderr, "%s: %d check(s) failed\n", name.c_str(), failures);
    return 1;
  }
  std::printf("%s: ok\n", name.c_str());
  return 0;
}
