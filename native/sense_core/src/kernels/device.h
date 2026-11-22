// The device of a plan: where its units run (Sprint 3.6, selector sense_core).
//
// A plan does everything it needs of a device through this interface: the memory of its
// workspaces, which is allocated once when the plan is configured, and the submission of one
// window. Two devices implement it.
//
//   cuda_device   cuda_device.cu. The units as CUDA kernels on one stream, page-locked staging
//                 and output blocks, and timing events around each stage.
//   host_device   host_device.cpp. The twin: the same units, compiled by the host compiler, in
//                 loops on the calling thread. It serves hosts without a device and the unit
//                 gate, and it is never a timed runtime.
//
// A submission is one upload of the staging block, the ingest kernel, the front-end kernel, the
// engine and one download of the output block, with one completion event behind them. A device
// never synchronizes: done() asks the completion event for its state.
//
// The output block has one layout on the host and on the device:
//
//     [ logit ][ map ][ input ][ difference ][ count ][ mean ]
//
// each part at a multiple of 256 bytes. The engine writes the first two parts and reads the
// third in place, so that a window needs one download. The mean is downloaded only for a plan
// that keeps it.
//
// The counting device of the unit tests wraps either device and counts the calls.
// ASCII only.

#pragma once

#include "kernels/units.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace sense {

struct device_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// Sizes of the workspaces of one plan. Fixed when the plan is configured.
struct device_layout {
  std::uint32_t rx = 0, positions = 0, symbols = 0; ///< [A, K, S] of a window.
  std::uint32_t records = 0;                        ///< Slots of a window.
  std::uint32_t cells   = 0;                        ///< rows * cols of the map.
  std::size_t   staging_bytes = 0;                  ///< Validity bits and payloads of one window.
  // Offsets of the parts of the output block, in bytes.
  std::size_t logit = 0, map = 0, input = 0, difference = 0, count = 0, mean = 0;
  std::size_t bytes_without_mean = 0, bytes = 0;

  std::size_t elements() const { return static_cast<std::size_t>(rx) * positions * symbols; }
};

/// Fills the offsets of the output block from the sizes.
void lay_out(device_layout& layout);

/// Host times of the marks of the last submission, in nanoseconds on the clock of the plan.
struct stage_marks {
  std::int64_t ingest = 0, kernels = 0, engine = 0, complete = 0;
};

class device
{
public:
  virtual ~device() = default;

  virtual const char* kind() const = 0;        ///< "host" or "cuda".
  virtual std::string name() const = 0;        ///< "host" or "cuda:N".
  virtual std::string description() const = 0; ///< The name of the GPU, or of the machine.
  virtual std::string identity_json() const = 0; ///< What an engine is built for, as a JSON object.

  /// Allocates every workspace and loads the template. Called once.
  virtual void configure(const device_layout& layout, const float* template_magnitude) = 0;

  /// The page-locked block the plan fills with the validity bits and the payloads of a window.
  virtual std::uint8_t* staging() = 0;
  /// The block the results of a window arrive in, on the host.
  virtual const std::uint8_t* output() const = 0;
  virtual std::uint8_t*       output_writable() = 0;
  /// The output block where an engine on the stream reads its input and writes its outputs:
  /// device memory for cuda_device, and the host block for host_device.
  virtual std::uint8_t* engine_block() = 0;
  /// The stream of the submissions, as an engine takes it. Null for host_device.
  virtual void* stream() = 0;

  /// Begins the submission of one window: the upload of `staged` bytes of the staging block,
  /// the ingest kernel and the front-end kernel. payload[r] is the device address of a record
  /// whose geometry says on_device, and null otherwise.
  virtual void begin(const window_geometry& window, const void* const payload[MAX_WINDOW_RECORDS], std::size_t staged) = 0;
  /// Ends it: the mark behind the engine, the download of the output block and the completion
  /// event. An engine on the stream was enqueued between begin and end.
  virtual void end(bool with_mean) = 0;
  /// Whether the completion event of the last submission has occurred. Never blocks.
  virtual bool done() = 0;
  /// The marks of the last submission, once it is done. now is the time done() became true.
  virtual stage_marks marks(std::int64_t now) = 0;

  /// One record through its ingest unit alone, synchronously: for the unit gate.
  virtual void ingest_one(const record_geometry& record, double tau2, const std::uint8_t* payload,
                          const std::uint8_t* mask, std::size_t payload_bytes, std::size_t mask_bytes, float* values,
                          std::uint8_t* valid) = 0;

  /// Calls that reached the device: allocations, copies, launches, events and queries.
  std::uint64_t calls = 0;
  /// Of those, the allocations of device and page-locked memory.
  std::uint64_t allocations = 0;
};

/// The clock a host device stamps its stages with.
typedef std::int64_t (*clock_fn)(void* owner);

std::unique_ptr<device> make_host_device(clock_fn clock, void* owner);

/// The CUDA device with the given index, on the given stream or on one of its own. Throws
/// device_error where this build has no device, or the runtime reaches none.
std::unique_ptr<device> make_cuda_device(int index, void* stream);

/// Whether this build holds the CUDA device.
bool built_with_cuda();

} // namespace sense
