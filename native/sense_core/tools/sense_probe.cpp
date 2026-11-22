// sense_probe: whether TensorRT builds and runs an engine on this GPU (Sprint 3.6, selector
// sense_core, checkpoint D0).
//
// It is the first step of the sensing core on a host. It needs the CUDA runtime and TensorRT
// with its headers, and nothing of the core: scripts/native_core.py probe compiles this one file
// before anything else is built, so a host learns what it has before it builds the kernels.
//
// It records the TensorRT version of the headers and of the library, builds an engine of one
// layer on the GPU, executes it on a stream with device addresses, compares the output with the
// values that layer must give, captures the same enqueue into a CUDA graph and launches it, and
// reports the device it ran on. It prints one sense-probe/v1 JSON object and exits 0 when the
// engine was built, ran and gave the right values, and 1 otherwise with every problem named.
// A graph that cannot be captured is reported and is no failure: the core then enqueues.
//
//     sense_probe [--device N] [--rounds N]
//
// With SENSE_PROBE_ONNX defined it also checks that the ONNX parser of this TensorRT can be
// created, which the engine builder of the core needs.
//
// ASCII only.

#include <NvInfer.h>
#ifdef SENSE_PROBE_ONNX
#include <NvOnnxParser.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int CHANNELS = 4, LENGTH = 16, VALUES = CHANNELS * LENGTH;

std::string quoted(const std::string& text)
{
  std::string out = "\"";
  for (const char letter : text) {
    const unsigned char byte = static_cast<unsigned char>(letter);
    if (letter == '"' || letter == '\\') {
      out += '\\';
      out += letter;
    } else if (byte < 0x20 || byte > 0x7e) {
      char escape[8];
      std::snprintf(escape, sizeof(escape), "\\u%04x", byte);
      out += escape;
    } else {
      out += letter;
    }
  }
  return out + "\"";
}

class probe_logger final : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char* message) noexcept override
  {
    if (severity <= Severity::kWARNING && messages.size() < 16) {
      messages.emplace_back(message);
    }
  }
  std::vector<std::string> messages;
};

std::string cuda_text(cudaError_t status)
{
  return std::string(cudaGetErrorName(status)) + ": " + cudaGetErrorString(status);
}

struct report {
  std::vector<std::string> problems;
  std::string              json = "{\"schema\": \"sense-probe/v1\"";
  void                     add(const std::string& key, const std::string& value) { json += ", \"" + key + "\": " + value; }
  bool                     fail(const std::string& text)
  {
    problems.push_back(text);
    return false;
  }
};

int finish(report& out, const probe_logger& logger)
{
  std::string list = "[";
  for (std::size_t n = 0; n != out.problems.size(); ++n) {
    list += (n != 0 ? ", " : "") + quoted(out.problems[n]);
  }
  out.add("problems", list + "]");
  list = "[";
  for (std::size_t n = 0; n != logger.messages.size(); ++n) {
    list += (n != 0 ? ", " : "") + quoted(logger.messages[n]);
  }
  out.add("tensorrt_messages", list + "]");
  out.add("ok", out.problems.empty() ? "true" : "false");
  std::printf("%s}\n", out.json.c_str());
  return out.problems.empty() ? 0 : 1;
}

/// Runs the engine once, by an enqueue or by the launch of a graph, and compares its output.
bool run_once(report& out, nvinfer1::IExecutionContext& context, cudaGraphExec_t graph, cudaStream_t stream, void* input,
              void* output, const std::vector<float>& values, const char* what)
{
  std::vector<float> zeros(VALUES, -7.0f), found(VALUES, 0.0f);
  cudaError_t        status = cudaMemcpy(output, zeros.data(), sizeof(float) * VALUES, cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    status = cudaMemcpy(input, values.data(), sizeof(float) * VALUES, cudaMemcpyHostToDevice);
  }
  if (status != cudaSuccess) {
    return out.fail(std::string(what) + ": the upload failed (" + cuda_text(status) + ")");
  }
  if (graph != nullptr) {
    status = cudaGraphLaunch(graph, stream);
    if (status != cudaSuccess) {
      return out.fail(std::string(what) + ": the launch failed (" + cuda_text(status) + ")");
    }
  } else if (!context.enqueueV3(stream)) {
    return out.fail(std::string(what) + ": TensorRT refused the enqueue");
  }
  status = cudaStreamSynchronize(stream);
  if (status == cudaSuccess) {
    status = cudaMemcpy(found.data(), output, sizeof(float) * VALUES, cudaMemcpyDeviceToHost);
  }
  if (status != cudaSuccess) {
    return out.fail(std::string(what) + ": the run failed (" + cuda_text(status) + ")");
  }
  for (int n = 0; n != VALUES; ++n) {
    if (found[n] != (values[n] > 0.0f ? values[n] : 0.0f)) {
      return out.fail(std::string(what) + ": the output is not that of the layer at element " + std::to_string(n));
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv)
{
  long device_index = 0, rounds = 200;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::puts("Usage: sense_probe [--device N] [--rounds N]\n"
                "  --device N   the CUDA device to probe (default 0)\n"
                "  --rounds N   timed runs of the one-layer engine (default 200)");
      return 0;
    } else if (arg == "--device" && i + 1 < argc) {
      device_index = std::atol(argv[++i]);
    } else if (arg == "--rounds" && i + 1 < argc) {
      rounds = std::atol(argv[++i]);
    } else {
      std::fprintf(stderr, "sense_probe: unknown or incomplete option '%s'\n", arg.c_str());
      return 2;
    }
  }
  report       out;
  probe_logger logger;
  char         text[512];

  // TensorRT: what the headers say and what the library says.
  std::snprintf(text, sizeof(text), "{\"headers\": \"%d.%d.%d.%d\", \"library\": \"%d.%d.%d.%d\"}", NV_TENSORRT_MAJOR,
                NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD, static_cast<int>(getInferLibMajorVersion()),
                static_cast<int>(getInferLibMinorVersion()), static_cast<int>(getInferLibPatchVersion()),
                static_cast<int>(getInferLibBuildVersion()));
  out.add("tensorrt", text);
  if (getInferLibMajorVersion() != NV_TENSORRT_MAJOR || getInferLibMinorVersion() != NV_TENSORRT_MINOR ||
      getInferLibPatchVersion() != NV_TENSORRT_PATCH) {
    out.fail("the TensorRT headers and the TensorRT library are of different versions: build against the headers of the "
             "library that is installed");
  }

  // The device.
  int         count  = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count < 1) {
    out.fail("the CUDA runtime reaches no device" + (status != cudaSuccess ? " (" + cuda_text(status) + ")" : std::string()));
    return finish(out, logger);
  }
  if (device_index < 0 || device_index >= count) {
    out.fail("the CUDA device " + std::to_string(device_index) + " does not exist");
    return finish(out, logger);
  }
  cudaDeviceProp properties;
  int            runtime_version = 0, driver_version = 0;
  status = cudaSetDevice(static_cast<int>(device_index));
  if (status == cudaSuccess) {
    status = cudaGetDeviceProperties(&properties, static_cast<int>(device_index));
  }
  if (status != cudaSuccess) {
    out.fail("the CUDA runtime refused the device (" + cuda_text(status) + ")");
    return finish(out, logger);
  }
  cudaRuntimeGetVersion(&runtime_version);
  cudaDriverGetVersion(&driver_version);
  char uuid[33];
  for (unsigned i = 0; i != 16; ++i) {
    std::snprintf(uuid + 2 * i, 3, "%02x", static_cast<unsigned>(static_cast<unsigned char>(properties.uuid.bytes[i])));
  }
  std::snprintf(text, sizeof(text),
                "{\"kind\": \"cuda\", \"index\": %ld, \"name\": %s, \"uuid\": \"%s\", \"capability\": \"%d.%d\", "
                "\"cuda_runtime\": %d, \"cuda_driver\": %d, \"devices\": %d}",
                device_index, quoted(properties.name).c_str(), uuid, properties.major, properties.minor, runtime_version,
                driver_version, count);
  out.add("device", text);

  // An engine of one layer: a ReLU on a float32 tensor [1, 4, 16].
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    out.fail("TensorRT gave no builder");
    return finish(out, logger);
  }
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
  std::unique_ptr<nvinfer1::IBuilderConfig>     config(builder->createBuilderConfig());
  nvinfer1::ITensor* input = network ? network->addInput("x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{1, CHANNELS, LENGTH})
                                     : nullptr;
  nvinfer1::IActivationLayer* layer = input != nullptr ? network->addActivation(*input, nvinfer1::ActivationType::kRELU)
                                                       : nullptr;
  if (!config || layer == nullptr) {
    out.fail("TensorRT refused the network of one layer");
    return finish(out, logger);
  }
  layer->getOutput(0)->setName("y");
  network->markOutput(*layer->getOutput(0));
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<std::size_t>(64) << 20);
#ifdef SENSE_PROBE_ONNX
  {
    // The parser the engine builder needs. It is made and dropped: nothing is parsed here.
    std::unique_ptr<nvinfer1::INetworkDefinition> other(builder->createNetworkV2(0U));
    std::unique_ptr<nvonnxparser::IParser>        parser(other ? nvonnxparser::createParser(*other, logger) : nullptr);
    out.add("onnx_parser", parser ? "true" : "false");
    if (!parser) {
      out.fail("TensorRT gave no ONNX parser: the engine of an artifact cannot be built");
    }
  }
#else
  out.add("onnx_parser", "null");
#endif
  const auto                             started = std::chrono::steady_clock::now();
  std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  const double build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  if (!plan || plan->size() == 0) {
    out.fail("TensorRT built no engine on this GPU");
    return finish(out, logger);
  }
  std::unique_ptr<nvinfer1::IRuntime>          runtime(nvinfer1::createInferRuntime(logger));
  std::unique_ptr<nvinfer1::ICudaEngine>       engine(runtime ? runtime->deserializeCudaEngine(plan->data(), plan->size())
                                                              : nullptr);
  std::unique_ptr<nvinfer1::IExecutionContext> context(engine ? engine->createExecutionContext() : nullptr);
  if (!context) {
    out.fail("TensorRT did not load the engine it built");
    return finish(out, logger);
  }

  // Execute it on a stream, with device addresses.
  void*        device_input  = nullptr;
  void*        device_output = nullptr;
  cudaStream_t stream        = nullptr;
  cudaEvent_t  before = nullptr, after = nullptr;
  status = cudaMalloc(&device_input, sizeof(float) * VALUES);
  if (status == cudaSuccess) {
    status = cudaMalloc(&device_output, sizeof(float) * VALUES);
  }
  if (status == cudaSuccess) {
    status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
  if (status == cudaSuccess) {
    status = cudaEventCreate(&before);
  }
  if (status == cudaSuccess) {
    status = cudaEventCreate(&after);
  }
  if (status != cudaSuccess) {
    out.fail("the CUDA runtime gave no memory, stream or event (" + cuda_text(status) + ")");
    return finish(out, logger);
  }
  std::vector<float> values(VALUES);
  for (int n = 0; n != VALUES; ++n) {
    values[n] = static_cast<float>(n) * 0.25f - 7.875f; // both signs, exact in float32
  }
  bool executed = context->setTensorAddress("x", device_input) && context->setTensorAddress("y", device_output);
  if (!executed) {
    out.fail("TensorRT refused the device addresses of the tensors");
  }
  executed = executed && run_once(out, *context, nullptr, stream, device_input, device_output, values, "the enqueue");

  // The same enqueue as a captured graph, as the core launches an engine where it can.
  cudaGraph_t     captured = nullptr;
  cudaGraphExec_t launched = nullptr;
  std::string     graph_reason;
  if (executed) {
    status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (status != cudaSuccess) {
      graph_reason = "no capture: " + cuda_text(status);
      cudaGetLastError();
    } else {
      const bool        enqueued = context->enqueueV3(stream);
      const cudaError_t ended    = cudaStreamEndCapture(stream, &captured);
      if (!enqueued || ended != cudaSuccess || captured == nullptr) {
        graph_reason = !enqueued ? "TensorRT cannot be captured into a graph" : "no capture: " + cuda_text(ended);
        cudaGetLastError();
      } else if ((status = cudaGraphInstantiate(&launched, captured, 0)) != cudaSuccess) {
        graph_reason = "no graph: " + cuda_text(status);
        launched     = nullptr;
        cudaGetLastError();
      } else {
        report quiet;
        if (!run_once(quiet, *context, launched, stream, device_input, device_output, values, "the graph")) {
          graph_reason = quiet.problems.empty() ? "the graph gave other values" : quiet.problems[0];
          cudaGraphExecDestroy(launched);
          launched = nullptr;
        }
      }
    }
  }
  // The time of one run of this engine, between two events on its stream. It is the floor of
  // what a submission pays for an engine on this GPU, and no grade.
  std::vector<float> times;
  if (executed) {
    for (long round = 0; round < rounds; ++round) {
      cudaEventRecord(before, stream);
      if (launched != nullptr) {
        cudaGraphLaunch(launched, stream);
      } else {
        context->enqueueV3(stream);
      }
      cudaEventRecord(after, stream);
      float elapsed = 0.0f;
      if (cudaEventSynchronize(after) == cudaSuccess && cudaEventElapsedTime(&elapsed, before, after) == cudaSuccess) {
        times.push_back(elapsed * 1000.0f);
      }
    }
    std::sort(times.begin(), times.end());
  }
  int current = -1;
  cudaGetDevice(&current);
  std::snprintf(text, sizeof(text),
                "{\"layers\": %d, \"built\": true, \"bytes\": %zu, \"build_ms\": %.3f, \"device_memory_bytes\": %lld, "
                "\"executed\": %s, \"ran_on_device\": %d, \"runs\": %zu, \"run_us\": {\"p50\": %.3f, \"p99\": %.3f}}",
                static_cast<int>(engine->getNbLayers()), plan->size(), build_ms,
                static_cast<long long>(engine->getDeviceMemorySizeV2()), executed ? "true" : "false", current, times.size(),
                times.empty() ? 0.0 : static_cast<double>(times[times.size() / 2]),
                times.empty() ? 0.0 : static_cast<double>(times[(times.size() - 1) * 99 / 100]));
  out.add("engine", text);
  out.add("graph", std::string("{\"captured\": ") + (launched != nullptr ? "true" : "false") + ", \"reason\": " +
                       (graph_reason.empty() ? std::string("null") : quoted(graph_reason)) + "}");
  if (current != device_index) {
    out.fail("the engine ran on device " + std::to_string(current) + ", and device " + std::to_string(device_index) +
             " was asked for");
  }
  if (launched != nullptr) {
    cudaGraphExecDestroy(launched);
  }
  if (captured != nullptr) {
    cudaGraphDestroy(captured);
  }
  cudaEventDestroy(before);
  cudaEventDestroy(after);
  context.reset();
  engine.reset();
  runtime.reset();
  cudaStreamDestroy(stream);
  cudaFree(device_input);
  cudaFree(device_output);
  return finish(out, logger);
}
