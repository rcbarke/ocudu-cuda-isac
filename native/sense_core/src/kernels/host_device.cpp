// The host twin of the device: the units of units.h in loops on the calling thread (Sprint 3.6,
// selector sense_core).
//
// It runs the arithmetic of the kernels where there is no device: on a development host, in the
// unit gate and behind the binding. A submission completes before begin() returns, and its marks
// are read from the clock of the plan. Nothing here is a timed runtime, and a plan on this device
// says so. ASCII only.

#include "kernels/device.h"
#include <cstdlib>
#include <cstring>
#include <sys/utsname.h>
#include <vector>

namespace sense {

void lay_out(device_layout& layout)
{
  const auto        align    = [](std::size_t value) { return (value + 255) & ~static_cast<std::size_t>(255); };
  const std::size_t elements = layout.elements();
  const std::size_t plane    = static_cast<std::size_t>(layout.rx) * layout.positions;
  layout.logit               = 0;
  layout.map                 = align(sizeof(float));
  layout.input               = align(layout.map + sizeof(float) * layout.cells);
  layout.difference          = align(layout.input + sizeof(float) * plane);
  layout.count               = align(layout.difference + sizeof(float) * plane);
  layout.bytes_without_mean  = align(layout.count + elements);
  layout.mean                = layout.bytes_without_mean;
  layout.bytes               = align(layout.mean + 2 * sizeof(float) * elements);
}

namespace {

/// Memory aligned to a page, zeroed. The twin allocates when a plan is configured and never after.
std::uint8_t* aligned_zeroed(std::size_t bytes)
{
  void* memory = nullptr;
  if (::posix_memalign(&memory, 4096, bytes == 0 ? 4096 : bytes) != 0 || memory == nullptr) {
    throw device_error("the host has no memory for a workspace of " + std::to_string(bytes) + " bytes");
  }
  std::memset(memory, 0, bytes);
  return static_cast<std::uint8_t*>(memory);
}

class host_device final : public device
{
public:
  host_device(clock_fn clock_, void* owner_) : clock(clock_), owner(owner_) {}
  ~host_device() override
  {
    std::free(staging_block);
    std::free(output_block);
    std::free(cube);
    std::free(cube_valid);
    std::free(magnitudes);
  }

  const char* kind() const override { return "host"; }
  std::string name() const override { return "host"; }
  std::string description() const override
  {
    struct utsname names;
    return ::uname(&names) == 0 ? std::string("host twin on ") + names.machine : std::string("host twin");
  }
  std::string identity_json() const override
  {
    struct utsname names;
    const std::string machine = ::uname(&names) == 0 ? names.machine : "unknown";
    return "{\"kind\": \"host\", \"name\": \"host twin\", \"machine\": \"" + machine + "\"}";
  }

  void configure(const device_layout& wanted, const float* template_magnitude) override
  {
    layout              = wanted;
    const std::size_t n = layout.elements();
    staging_block       = aligned_zeroed(layout.staging_bytes);
    output_block        = aligned_zeroed(layout.bytes);
    cube                = reinterpret_cast<float*>(aligned_zeroed(2 * sizeof(float) * n * layout.records));
    cube_valid          = aligned_zeroed(n * layout.records);
    magnitudes          = reinterpret_cast<float*>(aligned_zeroed(sizeof(float) * n));
    std::memcpy(magnitudes, template_magnitude, sizeof(float) * n);
    allocations += 5;
    calls += 5;
  }

  std::uint8_t*       staging() override { return staging_block; }
  const std::uint8_t* output() const override { return output_block; }
  std::uint8_t*       output_writable() override { return output_block; }
  std::uint8_t*       engine_block() override { return output_block; }
  void*               stream() override { return nullptr; }

  void begin(const window_geometry& window, const void* const payload[MAX_WINDOW_RECORDS], std::size_t) override
  {
    ++calls;
    const std::size_t per_record = layout.elements();
    // Ingest: every element of every record through the unit of its layout.
    for (std::uint32_t r = 0; r != window.records; ++r) {
      const record_geometry& record = window.record[r];
      const std::uint8_t*    data   = record.on_device != 0 ? static_cast<const std::uint8_t*>(payload[r])
                                                           : staging_block + record.payload_offset;
      const std::uint8_t*    bits   = staging_block + record.mask_offset;
      for (std::uint32_t a = 0; a != window.rx; ++a) {
        for (std::uint32_t k = 0; k != record.positions; ++k) {
          for (std::uint32_t s = 0; s != window.symbols; ++s) {
            const std::size_t at =
                static_cast<std::size_t>(r) * per_record + (static_cast<std::size_t>(a) * window.positions + k) * window.symbols + s;
            float real, imaginary;
            cube_valid[at]   = ingest_element(data, bits, record, window.tau2, a, k, s, real, imaginary) ? 1 : 0;
            cube[2 * at]     = real;
            cube[2 * at + 1] = imaginary;
          }
        }
      }
    }
    mark.ingest = clock(owner);
    // Front end: every position of the support.
    float*        mean       = reinterpret_cast<float*>(output_block + layout.mean);
    std::uint8_t* count      = output_block + layout.count;
    float*        difference = reinterpret_cast<float*>(output_block + layout.difference);
    float*        input      = reinterpret_cast<float*>(output_block + layout.input);
    for (std::uint32_t a = 0; a != window.rx; ++a) {
      for (std::uint32_t k = 0; k != window.positions; ++k) {
        frontend_position(cube, cube_valid, window, magnitudes, a, k, mean, count, difference, input);
      }
    }
    mark.kernels = clock(owner);
  }

  void end(bool) override
  {
    ++calls;
    mark.engine = clock(owner);
  }

  bool done() override
  {
    ++calls;
    return true;
  }

  stage_marks marks(std::int64_t now) override
  {
    stage_marks made = mark;
    made.complete    = now;
    return made;
  }

  void ingest_one(const record_geometry& record, double tau2, const std::uint8_t* payload, const std::uint8_t* mask,
                  std::size_t, std::size_t, float* values, std::uint8_t* valid) override
  {
    ++calls;
    for (std::uint32_t a = 0; a != record.planes; ++a) {
      for (std::uint32_t k = 0; k != record.positions; ++k) {
        for (std::uint32_t s = 0; s != record.symbols; ++s) {
          const std::size_t at = (static_cast<std::size_t>(a) * record.positions + k) * record.symbols + s;
          float             real, imaginary;
          valid[at]            = ingest_element(payload, mask, record, tau2, a, k, s, real, imaginary) ? 1 : 0;
          values[2 * at]       = real;
          values[2 * at + 1]   = imaginary;
        }
      }
    }
  }

private:
  clock_fn      clock;
  void*         owner;
  device_layout layout;
  std::uint8_t* staging_block = nullptr;
  std::uint8_t* output_block  = nullptr;
  float*        cube          = nullptr;
  std::uint8_t* cube_valid    = nullptr;
  float*        magnitudes    = nullptr;
  stage_marks   mark;
};

} // namespace

std::unique_ptr<device> make_host_device(clock_fn clock, void* owner)
{
  return std::unique_ptr<device>(new host_device(clock, owner));
}

} // namespace sense
