// The TensorRT backend of the engine host (Sprint 3.6, selector sense_core).
//
// It loads an engine that sense_engine built on this GPU from the ONNX export of the frozen
// artifact, after check_engine_manifest accepted it: the same artifact, the same TensorRT
// version and the same GPU. The engine has three tensors, all float32 and of fixed shape:
// "input" [1, A, K], "map" [1, rows * cols] and "presence_logit" [1].
//
// The engine runs on the stream of the submission. bind() gives it three addresses inside the
// output block of the device, so the front-end kernel writes its input in place and one download
// brings back its outputs. prepare() runs it once, so that no window pays for a first use, and
// then captures one enqueue into a CUDA graph where the plan asks for it and the runtime allows
// it. After that a window costs one cudaGraphLaunch. Where the capture is refused the engine is
// enqueued directly, and the manifest of the plan says which of the two runs.
//
// Compiled only where the build found TensorRT. ASCII only.

#include "core/sha256.h"
#include "engine/engine.h"
#include <NvInfer.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstring>

namespace sense {

namespace {

class quiet_logger final : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char* message) noexcept override
  {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "tensorrt: %s\n", message);
    }
  }
};

std::string library_version()
{
  char text[48];
  std::snprintf(text, sizeof(text), "%d.%d.%d.%d", static_cast<int>(getInferLibMajorVersion()),
                static_cast<int>(getInferLibMinorVersion()), static_cast<int>(getInferLibPatchVersion()),
                static_cast<int>(getInferLibBuildVersion()));
  return text;
}

std::string cuda_text(cudaError_t status)
{
  return std::string(cudaGetErrorName(status)) + ": " + cudaGetErrorString(status);
}

class tensorrt_engine final : public engine
{
public:
  tensorrt_engine(const std::string& directory, const artifact& model, device& on, bool graph) : want_graph(graph)
  {
    if (std::strcmp(on.kind(), "cuda") != 0) {
      throw engine_error("a TensorRT engine runs on the CUDA device, and this plan is on the host twin");
    }
    engine_host host;
    host.backend         = "tensorrt";
    host.library_version = library_version();
    host.device_identity = on.identity_json();
    hash                 = check_engine_manifest(directory, model, host, ENGINE_PLAN_FILE);
    std::string data;
    try {
      data     = read_file(directory + "/" + ENGINE_PLAN_FILE);
      manifest  = json::parse(read_file(directory + "/" + ENGINE_MANIFEST_FILE));
      precision = manifest.string_at("precision");
    } catch (const std::exception& fault) {
      throw engine_error(fault.what());
    }
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
      throw engine_error("TensorRT gave no runtime");
    }
    built.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
    if (!built) {
      throw engine_error("TensorRT refused the engine file: it was not built by this version on this GPU");
    }
    context.reset(built->createExecutionContext());
    if (!context) {
      throw engine_error("TensorRT gave no execution context");
    }
    // The three tensors, by name, type and shape.
    const std::int32_t cells = static_cast<std::int32_t>(model.rows * model.cols);
    require(ENGINE_INPUT, nvinfer1::TensorIOMode::kINPUT,
            {1, static_cast<std::int32_t>(model.rx), static_cast<std::int32_t>(model.subcarriers)});
    require(ENGINE_MAP, nvinfer1::TensorIOMode::kOUTPUT, {1, cells});
    require(ENGINE_LOGIT, nvinfer1::TensorIOMode::kOUTPUT, {1});
    if (built->getNbIOTensors() != 3) {
      throw engine_error("the engine has " + std::to_string(built->getNbIOTensors()) + " tensors, and the model has three");
    }
  }

  ~tensorrt_engine() override
  {
    if (launched != nullptr) {
      cudaGraphExecDestroy(launched);
    }
    if (captured != nullptr) {
      cudaGraphDestroy(captured);
    }
  }

  const char* backend() const override { return "tensorrt"; }
  std::string detail() const override
  {
    return "TensorRT " + library_version() + ", " + precision + ", " +
           (launched != nullptr ? "launched as a captured CUDA graph" : "enqueued on the stream" +
                                                                         (refusal.empty() ? std::string() : " (" + refusal + ")"));
  }
  std::string sha256() const override { return hash; }
  bool        on_stream() const override { return true; }
  bool        graph() const override { return launched != nullptr; }

  void bind(const engine_binding& binding) override
  {
    stream = static_cast<cudaStream_t>(binding.stream);
    if (!context->setTensorAddress(ENGINE_INPUT, const_cast<float*>(binding.input)) ||
        !context->setTensorAddress(ENGINE_MAP, binding.map) || !context->setTensorAddress(ENGINE_LOGIT, binding.logit) ||
        !context->allInputDimensionsSpecified()) {
      throw engine_error("TensorRT refused the addresses of the tensors");
    }
  }

  void prepare() override
  {
    // A first use outside every window. This is configure time, so it may wait for the device.
    if (!context->enqueueV3(stream)) {
      throw engine_error("TensorRT refused the first enqueue of the engine");
    }
    cudaError_t status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
      throw engine_error("the first run of the engine failed (" + cuda_text(status) + ")");
    }
    drop_graph(); // a second warm-up captures again and keeps no graph of the first
    refusal.clear();
    if (!want_graph) {
      refusal = "a graph was not asked for";
      return;
    }
    status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (status != cudaSuccess) {
      refusal = "no capture: " + cuda_text(status);
      cudaGetLastError();
      return;
    }
    const bool        enqueued = context->enqueueV3(stream);
    const cudaError_t ended    = cudaStreamEndCapture(stream, &captured); // always, or the stream stays in capture
    if (!enqueued || ended != cudaSuccess || captured == nullptr) {
      refusal = !enqueued ? "TensorRT cannot be captured into a graph" : "no capture: " + cuda_text(ended);
      drop_graph();
      return;
    }
    status = cudaGraphInstantiate(&launched, captured, 0);
    if (status != cudaSuccess) {
      refusal = "no graph: " + cuda_text(status);
      drop_graph();
      return;
    }
    // The graph once, so that its first launch is not a window's either.
    status = cudaGraphLaunch(launched, stream);
    if (status == cudaSuccess) {
      status = cudaStreamSynchronize(stream);
    }
    if (status != cudaSuccess) {
      refusal = "the graph does not launch: " + cuda_text(status);
      drop_graph();
    }
  }

  void enqueue() override
  {
    if (launched != nullptr) {
      const cudaError_t status = cudaGraphLaunch(launched, stream);
      if (status != cudaSuccess) {
        throw engine_error("the launch of the engine graph failed (" + cuda_text(status) + ")");
      }
    } else if (!context->enqueueV3(stream)) {
      throw engine_error("TensorRT refused to enqueue the engine");
    }
  }

private:
  void require(const char* name, nvinfer1::TensorIOMode mode, std::initializer_list<std::int32_t> shape)
  {
    const nvinfer1::Dims found = built->getTensorShape(name);
    bool                 same  = built->getTensorIOMode(name) == mode &&
                                 built->getTensorDataType(name) == nvinfer1::DataType::kFLOAT &&
                                 found.nbDims == static_cast<std::int32_t>(shape.size());
    std::int32_t         axis  = 0;
    for (const std::int32_t side : shape) {
      same = same && found.d[axis++] == side;
    }
    if (!same) {
      throw engine_error(std::string("the engine has no float32 tensor \"") + name + "\" of the shape the artifact needs");
    }
  }

  void drop_graph()
  {
    if (launched != nullptr) {
      cudaGraphExecDestroy(launched);
      launched = nullptr;
    }
    if (captured != nullptr) {
      cudaGraphDestroy(captured);
      captured = nullptr;
    }
    cudaGetLastError();
  }

  quiet_logger                                 logger;
  std::unique_ptr<nvinfer1::IRuntime>          runtime;
  std::unique_ptr<nvinfer1::ICudaEngine>       built;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  json::value                                  manifest;
  std::string                                  hash, precision, refusal;
  cudaStream_t                                 stream = nullptr;
  bool                                         want_graph;
  cudaGraph_t                                  captured = nullptr;
  cudaGraphExec_t                              launched = nullptr;
};

} // namespace

std::unique_ptr<engine> make_tensorrt_engine(const std::string& directory, const artifact& model, device& on, bool graph)
{
  return std::unique_ptr<engine>(new tensorrt_engine(directory, model, on, graph));
}

std::string tensorrt_build_version()
{
  char text[48];
  std::snprintf(text, sizeof(text), "%d.%d.%d.%d", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
                NV_TENSORRT_BUILD);
  return text;
}

} // namespace sense
