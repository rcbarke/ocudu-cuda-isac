// The public header of the sensing core compiles as C++20 (Sprint 3.6, selector sense_core): the
// language level of the pinned Aerial tree, above the C++17 the core itself is written in. This
// file is compiled with -std=c++20 -Wpedantic -Werror and never linked. ASCII only.

#include "sense_core/sense_core.h"

int sense_core_api_compiles_as_cxx20()
{
  sense_options  options{};
  sense_view     view{};
  sense_decision decision{};
  options.api_version = SENSE_CORE_API_VERSION;
  view.memory         = SENSE_MEMORY_DEVICE;
  return static_cast<int>(options.api_version) + view.memory + decision.status;
}
