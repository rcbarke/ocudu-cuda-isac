// sense_engine: builds the TensorRT engine of one frozen artifact on the GPU that will run it
// (Sprint 3.6, selector sense_core).
//
// It reads the ONNX export of the artifact, which ocudu_cuda_isac.sensing.export wrote from the
// weights of that artifact, builds a TensorRT engine for the CUDA device it runs on, and writes
// an engine directory:
//
//     model.plan     the serialized engine
//     engine.json    a sense-engine/v1 manifest: the hash of the artifact, the version of
//                    TensorRT, the precision, the GPU the engine was built on, and the hash of
//                    model.plan
//
// A plan loads that directory only on the same GPU, with the same TensorRT version and for the
// same artifact, so an engine is rebuilt on every GPU that runs it. This tool is the only place
// where an engine is built: scripts/native_core.py engine runs it. It is no part of a timed run.
//
//     sense_engine --onnx FILE --artifact DIR --output-dir DIR [--device N] [--precision fp32|fp16]
//                  [--tf32] [--workspace-mb N]
//
// The default is a float32 engine without TF32, so that its scores are compared with the
// reference by the float32 tolerance of the core. An fp16 engine is a declared variant with its
// own tolerance, graded only where a gate builds one.
//
// Built only where the build found TensorRT. ASCII only.

#include "core/artifact.h"
#include "core/build.h"
#include "core/json.h"
#include "core/sha256.h"
#include "engine/engine.h"
#include "kernels/device.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <memory>
#include <string>

namespace {

void usage()
{
  std::puts("Usage: sense_engine --onnx FILE --artifact DIR --output-dir DIR [--device N] [--precision fp32|fp16]\n"
            "                    [--tf32] [--workspace-mb N]\n"
            "  --onnx FILE         the ONNX export of the artifact: input [1, A, K], map and presence_logit\n"
            "  --artifact DIR      directory of the frozen cusense-artifact/v1 artifact the export was made from\n"
            "  --output-dir DIR    existing, empty directory for model.plan and engine.json\n"
            "  --device N          the CUDA device to build on and for (default 0)\n"
            "  --precision NAME    fp32 (default) or fp16\n"
            "  --tf32              allow TF32 in a float32 engine (default: not allowed)\n"
            "  --workspace-mb N    the workspace of the builder (default 1024)");
}

class build_logger final : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char* message) noexcept override
  {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "tensorrt: %s\n", message);
    }
  }
};

bool empty_directory(const std::string& path)
{
  DIR* listing = ::opendir(path.c_str());
  if (listing == nullptr) {
    return false;
  }
  bool empty = true;
  while (const dirent* found = ::readdir(listing)) {
    if (std::strcmp(found->d_name, ".") != 0 && std::strcmp(found->d_name, "..") != 0) {
      empty = false;
    }
  }
  ::closedir(listing);
  return empty;
}

bool write_file(const std::string& path, const void* data, std::size_t bytes)
{
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return false;
  }
  const bool ok = std::fwrite(data, 1, bytes, file) == bytes;
  return std::fclose(file) == 0 && ok;
}

std::string shape_text(const nvinfer1::Dims& dims)
{
  std::string text = "[";
  for (std::int32_t axis = 0; axis < dims.nbDims; ++axis) {
    text += (axis != 0 ? ", " : "") + std::to_string(dims.d[axis]);
  }
  return text + "]";
}

bool has_shape(const nvinfer1::ITensor* tensor, const char* name, std::initializer_list<std::int64_t> shape)
{
  if (tensor == nullptr || std::strcmp(tensor->getName(), name) != 0 || tensor->getType() != nvinfer1::DataType::kFLOAT) {
    return false;
  }
  const nvinfer1::Dims dims = tensor->getDimensions();
  bool                 same = dims.nbDims == static_cast<std::int32_t>(shape.size());
  std::int32_t         axis = 0;
  for (const std::int64_t side : shape) {
    same = same && dims.d[axis++] == side;
  }
  return same;
}

} // namespace

int main(int argc, char** argv)
{
  std::string onnx_path, artifact_dir, output_dir, precision = "fp32";
  long        device_index = 0, workspace_mb = 1024;
  bool        tf32 = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg  = argv[i];
    const bool        more = i + 1 < argc;
    if (arg == "--help") {
      usage();
      return 0;
    } else if (arg == "--onnx" && more) {
      onnx_path = argv[++i];
    } else if (arg == "--artifact" && more) {
      artifact_dir = argv[++i];
    } else if (arg == "--output-dir" && more) {
      output_dir = argv[++i];
    } else if (arg == "--device" && more) {
      device_index = std::atol(argv[++i]);
    } else if (arg == "--precision" && more) {
      precision = argv[++i];
    } else if (arg == "--tf32") {
      tf32 = true;
    } else if (arg == "--workspace-mb" && more) {
      workspace_mb = std::atol(argv[++i]);
    } else {
      usage();
      std::fprintf(stderr, "sense_engine: unknown or incomplete option '%s'\n", arg.c_str());
      return 2;
    }
  }
  if (onnx_path.empty() || artifact_dir.empty() || output_dir.empty() || (precision != "fp32" && precision != "fp16") ||
      device_index < 0 || workspace_mb < 16) {
    usage();
    std::fprintf(stderr, "sense_engine: --onnx, --artifact and --output-dir are required, --precision is fp32 or fp16, "
                         "and the workspace holds 16 MB or more\n");
    return 2;
  }
  if (!empty_directory(output_dir)) {
    std::fprintf(stderr, "sense_engine: the output directory %s does not exist or is not empty\n", output_dir.c_str());
    return 2;
  }
  try {
    const sense::artifact model = sense::load_artifact(artifact_dir, nullptr);
    const std::string     onnx  = sense::read_file(onnx_path);
    // The device first: TensorRT builds for the device that is current when the builder is made.
    const std::unique_ptr<sense::device> device = sense::make_cuda_device(static_cast<int>(device_index), nullptr);
    char                                 version[48];
    std::snprintf(version, sizeof(version), "%d.%d.%d.%d", static_cast<int>(getInferLibMajorVersion()),
                  static_cast<int>(getInferLibMinorVersion()), static_cast<int>(getInferLibPatchVersion()),
                  static_cast<int>(getInferLibBuildVersion()));
    if (sense::tensorrt_build_version() != version) {
      std::fprintf(stderr, "sense_engine: this core was compiled against TensorRT %s, and the library it loaded is %s\n",
                   sense::tensorrt_build_version().c_str(), version);
      return 1;
    }

    build_logger                                    logger;
    const std::unique_ptr<nvinfer1::IBuilder>       builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
      throw std::runtime_error("TensorRT gave no builder");
    }
    const std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    const std::unique_ptr<nvinfer1::IBuilderConfig>     config(builder->createBuilderConfig());
    if (!network || !config) {
      throw std::runtime_error("TensorRT gave no network or no builder configuration");
    }
    const std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) {
      throw std::runtime_error("TensorRT gave no ONNX parser");
    }
    if (!parser->parse(onnx.data(), onnx.size())) {
      std::string problems;
      for (std::int32_t n = 0; n < parser->getNbErrors(); ++n) {
        problems += std::string(n != 0 ? "; " : "") + parser->getError(n)->desc();
      }
      throw std::runtime_error("the ONNX parser refused " + onnx_path + ": " + problems);
    }
    // The model of the artifact and no other: one input and two outputs, by name and shape.
    const std::int64_t cells = static_cast<std::int64_t>(model.rows) * model.cols;
    bool               fits  = network->getNbInputs() == 1 && network->getNbOutputs() == 2 &&
                               has_shape(network->getInput(0), sense::ENGINE_INPUT, {1, model.rx, model.subcarriers});
    bool               has_map = false, has_logit = false;
    for (std::int32_t n = 0; fits && n != 2; ++n) {
      has_map   = has_map || has_shape(network->getOutput(n), sense::ENGINE_MAP, {1, cells});
      has_logit = has_logit || has_shape(network->getOutput(n), sense::ENGINE_LOGIT, {1});
    }
    if (!fits || !has_map || !has_logit) {
      throw std::runtime_error("the ONNX file is not the export of this artifact: it must take \"input\" [1, " +
                               std::to_string(model.rx) + ", " + std::to_string(model.subcarriers) + "] and give \"map\" [1, " +
                               std::to_string(cells) + "] and \"presence_logit\" [1] as float32");
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<std::size_t>(workspace_mb) << 20);
    if (precision == "fp16") {
      // The flag is the way of this TensorRT generation to allow half precision in a network that
      // is not strongly typed. Newer headers mark it as deprecated.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
#pragma GCC diagnostic pop
    }
    if (tf32) {
      config->setFlag(nvinfer1::BuilderFlag::kTF32);
    } else {
      config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }
    const auto                                        started = std::chrono::steady_clock::now();
    const std::unique_ptr<nvinfer1::IHostMemory>      plan(builder->buildSerializedNetwork(*network, *config));
    const double build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (!plan || plan->size() == 0) {
      throw std::runtime_error("TensorRT built no engine");
    }
    const std::string plan_sha256 = sense::sha256_hex(plan->data(), plan->size());
    if (!write_file(output_dir + "/" + sense::ENGINE_PLAN_FILE, plan->data(), plan->size())) {
      throw std::runtime_error("cannot write " + output_dir + "/" + sense::ENGINE_PLAN_FILE);
    }

    std::string       manifest = "{\"schema\": ";
    const std::size_t slash    = onnx_path.find_last_of('/');
    using sense::json::put_string;
    put_string(manifest, sense::ENGINE_MANIFEST_SCHEMA);
    manifest += ", \"backend\": \"tensorrt\", \"model_id\": ";
    put_string(manifest, model.model_id);
    manifest += ", \"artifact_sha256\": ";
    put_string(manifest, model.sha256);
    manifest += ", \"weights_sha256\": ";
    put_string(manifest, model.weights_sha256);
    manifest += ", \"library\": {\"name\": \"tensorrt\", \"version\": ";
    put_string(manifest, version);
    manifest += "}, \"precision\": ";
    put_string(manifest, precision);
    manifest += std::string(", \"tf32\": ") + (tf32 ? "true" : "false");
    manifest += ", \"device\": " + device->identity_json();
    manifest += ", \"input\": {\"name\": \"input\", \"shape\": [1, " + std::to_string(model.rx) + ", " +
                std::to_string(model.subcarriers) + "]}";
    manifest += ", \"outputs\": {\"map\": [1, " + std::to_string(cells) + "], \"presence_logit\": [1]}";
    manifest += ", \"onnx\": {\"file\": ";
    put_string(manifest, slash == std::string::npos ? onnx_path : onnx_path.substr(slash + 1));
    manifest += ", \"bytes\": " + std::to_string(onnx.size()) + ", \"sha256\": ";
    put_string(manifest, sense::sha256_hex(onnx));
    manifest += "}, \"layers\": " + std::to_string(network->getNbLayers());
    manifest += ", \"workspace_mb\": " + std::to_string(workspace_mb) + ", \"build_ms\": ";
    sense::json::put_double(manifest, build_ms);
    manifest += ", \"core_sources_sha256\": ";
    put_string(manifest, sense::core_sources_sha256());
    manifest += ", \"file\": ";
    put_string(manifest, sense::ENGINE_PLAN_FILE);
    manifest += ", \"bytes\": " + std::to_string(plan->size()) + ", \"sha256\": ";
    put_string(manifest, plan_sha256);
    manifest += "}\n";
    if (!write_file(output_dir + "/" + sense::ENGINE_MANIFEST_FILE, manifest.data(), manifest.size())) {
      throw std::runtime_error("cannot write " + output_dir + "/" + sense::ENGINE_MANIFEST_FILE);
    }
    // The directory, as a plan will load it: the manifest check, the deserialization and the tensors.
    const std::unique_ptr<sense::engine> loaded = sense::make_tensorrt_engine(output_dir, model, *device, false);
    std::printf("{\"schema\": \"sense-engine-build/v1\", \"directory\": ");
    std::string line;
    put_string(line, output_dir);
    line += ", \"artifact_sha256\": ";
    put_string(line, model.sha256);
    line += ", \"tensorrt\": ";
    put_string(line, version);
    line += ", \"precision\": ";
    put_string(line, precision);
    line += ", \"device\": " + device->identity_json();
    line += ", \"input\": " + shape_text(network->getInput(0)->getDimensions());
    line += ", \"layers\": " + std::to_string(network->getNbLayers()) + ", \"bytes\": " + std::to_string(plan->size());
    line += ", \"sha256\": ";
    put_string(line, loaded->sha256());
    line += ", \"build_ms\": ";
    sense::json::put_double(line, build_ms);
    std::printf("%s}\n", line.c_str());
    return 0;
  } catch (const std::exception& fault) {
    std::fprintf(stderr, "sense_engine: %s\n", fault.what());
    return 1;
  }
}
