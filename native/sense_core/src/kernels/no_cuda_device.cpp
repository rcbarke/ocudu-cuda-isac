// A build of the core without the CUDA device: the host twin only (Sprint 3.6, selector
// sense_core). A plan that asks for the device is refused with a sentence that says how the
// device is built. ASCII only.

#include "kernels/device.h"

namespace sense {

std::unique_ptr<device> make_cuda_device(int, void*)
{
  throw device_error("this build of the core holds the host twin only. Build it with a CUDA compiler for the device: "
                     "python scripts/native_core.py build");
}

bool built_with_cuda()
{
  return false;
}

} // namespace sense
