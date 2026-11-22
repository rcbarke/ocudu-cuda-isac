// Counts the allocations of the unit target (Sprint 3.6, selector sense_core).
//
// It replaces the global allocation functions of the C++ library for the target that links it,
// so that a case can show that a plan allocated nothing after it was created. It is linked by
// the unit target only, never by sense_host. ASCII only.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

namespace {
std::atomic<std::uint64_t> allocations{0};
}

std::uint64_t allocation_count()
{
  return allocations.load(std::memory_order_relaxed);
}

void* operator new(std::size_t bytes)
{
  allocations.fetch_add(1, std::memory_order_relaxed);
  void* memory = std::malloc(bytes == 0 ? 1 : bytes);
  if (memory == nullptr) {
    throw std::bad_alloc();
  }
  return memory;
}

void operator delete(void* memory) noexcept
{
  std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept
{
  std::free(memory);
}
