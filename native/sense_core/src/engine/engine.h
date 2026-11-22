// The engine host of a plan: what runs the model of an artifact (Sprint 3.6, selector sense_core).
//
// A model is the registry entry cusense_cnn of the reference: a float32 tensor [1, A, K] to a
// location map over the position grid, the softmax of the grid logits, and one presence logit.
// Three backends run it.
//
//   tensorrt        The backend of every timed run. A TensorRT engine, built on the GPU that runs
//                   it from the ONNX export of the frozen artifact. It is enqueued on the stream
//                   of the submission, or launched there as a captured CUDA graph, reads its
//                   input where the front-end kernel wrote it and writes into the output block.
//   libtorch        The declared fallback of a host without a TensorRT build for its CUDA stack:
//                   a TorchScript module of the same artifact through the C++ library of the
//                   pinned PyTorch distribution. It runs after the kernels and blocks.
//   host_reference  The model in plain C++ on the CPU, from weights.f32. The twin of the unit
//                   gate and of a host without a device. Never a timed runtime.
//
// The backend is named in every manifest. An engine on the stream carries the hash of the
// artifact, the version of its library and the GPU it was built for in a manifest beside it, and
// check_engine_manifest refuses it anywhere else.
//
// ASCII only.

#pragma once

#include "core/artifact.h"
#include "kernels/device.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace sense {

struct engine_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

constexpr const char ENGINE_MANIFEST_SCHEMA[] = "sense-engine/v1";
constexpr const char ENGINE_MANIFEST_FILE[]   = "engine.json";
constexpr const char ENGINE_PLAN_FILE[]       = "model.plan";
constexpr const char ENGINE_MODULE_FILE[]     = "model.ts";
constexpr const char ENGINE_INPUT[]           = "input";
constexpr const char ENGINE_MAP[]             = "map";
constexpr const char ENGINE_LOGIT[]           = "presence_logit";

/// Where an engine reads and writes. For an engine on the stream these are device addresses
/// inside the output block of the device. For a blocking engine the outputs are host addresses,
/// and the input is a device address only where the backend says it reads the device.
struct engine_binding {
  const float* input = nullptr; ///< [1, A, K]
  float*       map   = nullptr; ///< [1, rows * cols]
  float*       logit = nullptr; ///< [1]
  void*        stream = nullptr;
};

class engine
{
public:
  virtual ~engine() = default;

  virtual const char* backend() const = 0;
  /// One line for a manifest: the versions and the precision of the backend.
  virtual std::string detail() const = 0;
  /// The hash of what the backend loaded: the engine file, the module file or the weights.
  virtual std::string sha256() const = 0;
  /// True: enqueue() queues the model on the stream of the submission and returns at once.
  /// False: infer() runs it and returns when the outputs are in host memory.
  virtual bool on_stream() const = 0;
  /// For a blocking engine: whether its input is read from device memory.
  virtual bool reads_device() const { return false; }
  /// Whether the model is launched as a captured CUDA graph.
  virtual bool graph() const { return false; }

  virtual void bind(const engine_binding& binding) = 0;
  /// Runs before the first window, after bind(): a first use, and the capture of a graph.
  virtual void prepare() {}
  virtual void enqueue() { throw engine_error("this engine does not run on a stream"); }
  virtual void infer() { throw engine_error("this engine runs on the stream of its submission"); }
};

/// What the running device and library are, for the check of an engine manifest.
struct engine_host {
  std::string backend;         ///< "tensorrt" or "libtorch".
  std::string library_version; ///< Of the library that would run the engine.
  std::string device_identity; ///< device::identity_json().
};

/// Checks the manifest beside an engine against the artifact and against what would run it.
/// Throws engine_error: another artifact, another library version, another GPU, another input
/// shape or a file that does not have the hash the manifest names. Returns the hash of the
/// engine file.
std::string check_engine_manifest(const std::string& directory, const artifact& model, const engine_host& host,
                                  const char* file_name);

/// The model in plain C++ on the CPU. Any device.
std::unique_ptr<engine> make_host_reference_engine(const artifact& model);

/// The TensorRT engine of a directory that sense_engine wrote. Throws engine_error where this
/// build has no TensorRT, or the engine does not fit.
std::unique_ptr<engine> make_tensorrt_engine(const std::string& directory, const artifact& model, device& on, bool graph);

/// The TorchScript module of a directory that the export wrote. Throws engine_error where this
/// build has no LibTorch.
std::unique_ptr<engine> make_libtorch_engine(const std::string& directory, const artifact& model, device& on);

/// What this build holds: the version of each library it was compiled against, or an empty text.
std::string tensorrt_build_version();
std::string libtorch_build_version();

} // namespace sense
