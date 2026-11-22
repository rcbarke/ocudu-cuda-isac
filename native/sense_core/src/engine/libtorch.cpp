// The LibTorch backend of the engine host: the declared fallback (Sprint 3.6, selector sense_core).
//
// A host whose probe finds no TensorRT build for its CUDA stack runs the model through the C++
// library of the pinned PyTorch distribution. The backend loads a TorchScript module that the
// export wrote from the frozen artifact, with the same two outputs as the engine: the map
// [1, rows * cols] and the presence logit [1]. It is named in every manifest, and a run on it
// records the native showing as blocked: it does not stand in for the TensorRT result.
//
// It runs after the kernels of a window have completed and blocks until its outputs are in host
// memory. It allocates tensors in every call, so the bounded-execution contract of a plan does
// not hold on it, and the plan says so. On the CUDA device it reads its input in place, in
// device memory. On the host twin it runs on the CPU.
//
// Compiled only where the build asks for it. ASCII only.

#include "core/sha256.h"
#include "engine/engine.h"
#include <cstring>
#include <torch/script.h>
#include <torch/version.h>

namespace sense {

namespace {

class libtorch_engine final : public engine
{
public:
  libtorch_engine(const std::string& directory, const artifact& model, device& on) :
    rx(model.rx), positions(model.subcarriers), cells(static_cast<std::int64_t>(model.rows) * model.cols),
    place(torch::kCPU)
  {
    engine_host host;
    host.backend         = "libtorch";
    host.library_version = TORCH_VERSION;
    // A TorchScript module is not bound to a GPU: the manifest of a module names the device kind.
    host.device_identity = "{\"name\": \"any\", \"capability\": \"any\"}";
    hash                 = check_engine_manifest(directory, model, host, ENGINE_MODULE_FILE);
    on_device            = std::strcmp(on.kind(), "cuda") == 0;
    if (on_device) {
      const std::string name = on.name(); // "cuda:N"
      place                  = torch::Device(torch::kCUDA, static_cast<torch::DeviceIndex>(std::atoi(name.c_str() + 5)));
    }
    try {
      module = torch::jit::load(directory + "/" + ENGINE_MODULE_FILE, place);
      module.eval();
    } catch (const std::exception& fault) {
      throw engine_error(std::string("LibTorch refused the module: ") + fault.what());
    }
  }

  const char* backend() const override { return "libtorch"; }
  std::string detail() const override
  {
    return std::string("LibTorch ") + TORCH_VERSION + ", TorchScript, float32, on " + (on_device ? "cuda" : "cpu") +
           ", blocking, the declared fallback";
  }
  std::string sha256() const override { return hash; }
  bool        on_stream() const override { return false; }
  bool        reads_device() const override { return on_device; }
  void        bind(const engine_binding& binding) override { bound = binding; }
  void        prepare() override { infer(); }

  void infer() override
  {
    try {
      torch::InferenceMode guard;
      const torch::Tensor  input = torch::from_blob(const_cast<float*>(bound.input), {1, rx, positions},
                                                    torch::TensorOptions().dtype(torch::kFloat32).device(place));
      const auto           made  = module.forward({input}).toTuple();
      const torch::Tensor  map   = made->elements()[0].toTensor().to(torch::kCPU, torch::kFloat32).contiguous();
      const torch::Tensor  logit = made->elements()[1].toTensor().to(torch::kCPU, torch::kFloat32).contiguous();
      if (map.numel() != cells || logit.numel() != 1) {
        throw engine_error("the module does not give the map of the grid and one presence logit");
      }
      std::memcpy(bound.map, map.data_ptr<float>(), sizeof(float) * static_cast<std::size_t>(cells));
      std::memcpy(bound.logit, logit.data_ptr<float>(), sizeof(float));
    } catch (const engine_error&) {
      throw;
    } catch (const std::exception& fault) {
      throw engine_error(std::string("LibTorch failed to run the module: ") + fault.what());
    }
  }

private:
  std::int64_t              rx, positions, cells;
  torch::Device             place;
  torch::jit::script::Module module;
  std::string               hash;
  bool                      on_device = false;
  engine_binding            bound;
};

} // namespace

std::unique_ptr<engine> make_libtorch_engine(const std::string& directory, const artifact& model, device& on)
{
  return std::unique_ptr<engine>(new libtorch_engine(directory, model, on));
}

std::string libtorch_build_version()
{
  return TORCH_VERSION;
}

} // namespace sense
