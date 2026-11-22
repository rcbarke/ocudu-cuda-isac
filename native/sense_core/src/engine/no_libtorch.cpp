// A build of the core without the LibTorch backend (Sprint 3.6, selector sense_core). The backend
// is the declared fallback of a host without a TensorRT build for its CUDA stack, and is compiled
// only where the build asks for it. ASCII only.

#include "engine/engine.h"

namespace sense {

std::unique_ptr<engine> make_libtorch_engine(const std::string&, const artifact&, device&)
{
  throw engine_error("this build of the core holds no LibTorch backend. Build it with: python scripts/native_core.py "
                     "build --libtorch on");
}

std::string libtorch_build_version()
{
  return std::string();
}

} // namespace sense
