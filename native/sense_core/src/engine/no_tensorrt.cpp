// A build of the core without TensorRT (Sprint 3.6, selector sense_core). TensorRT is an optional
// component of the core: its kernels and its host twin build without it. A plan that asks for the
// backend is refused with a sentence that names the probe. ASCII only.

#include "engine/engine.h"

namespace sense {

std::unique_ptr<engine> make_tensorrt_engine(const std::string&, const artifact&, device&, bool)
{
  throw engine_error("this build of the core holds no TensorRT backend. Record what the host has with: python "
                     "scripts/native_core.py probe");
}

std::string tensorrt_build_version()
{
  return std::string();
}

} // namespace sense
