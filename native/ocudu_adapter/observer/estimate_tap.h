// Device side of the capture of the receiver's estimate (Sprint 5.2, selector native_estimate).
//
// This header exists only in the adapter build against the patched receiver: ISAC_O3_ROUTE is
// defined there, with ENABLE_CUDA. It joins three things.
//
//   receiver_tap     implements ocudu::pusch_resident_estimate_tap of the receiver patch
//                    (patches/ocudu/0001-pusch-resident-estimate-tap.patch) and passes what the
//                    demodulator reports to the capture core, field by field.
//   device_copier    implements isac_estimate::copier with the CUDA runtime: one asynchronous
//                    device-to-host copy per estimate, ordered after the producer, and one event
//                    that is recorded behind it. The receive path asks that event for its state
//                    and never waits for it.
//   device_memory    page-locked host memory for the copy: staging from cudaHostAlloc, and the
//                    registration of the ring segment, so that a copy can land in its slot.
//
// Ordering. Every kernel that writes the estimate of a PDU is enqueued on the stream the receiver
// reports before it calls the tap. Two ways to order the copy after them:
//
//   producer   the copy is enqueued on that stream. The stream runs it before every later kernel
//              of the same demodulator, so no later PDU can overwrite or free the buffer first.
//              This is the baseline of the sprint. The copy lies between the kernels of the
//              demodulation and those of the decode.
//   side       an event is recorded on the producer's stream, and the copy is enqueued on a
//              stream of this observer that waits for that event. The producer's stream gets one
//              event record and no copy. Nothing orders the copy before the next PDU, so this is
//              valid only where the owner resolves a capture before it starts the next decode,
//              as the staged harness does. The capture core refuses a copy that was overtaken.
//
// No call here synchronizes a stream or the device while a decode runs. drain() does, and the
// owner calls it after the last decode only.
//
// It uses the CUDA runtime API and no kernel. ASCII only.

#pragma once

#if !defined(ISAC_O3_ROUTE) || !defined(ENABLE_CUDA)
#error "estimate_tap.h belongs to the adapter build against the patched receiver (ISAC_O3_ROUTE with ENABLE_CUDA)"
#endif

#include "estimate_core.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_resident_estimate_tap.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace isac_estimate {

static_assert(ocudu::PUSCH_RESIDENT_ESTIMATE_TAP_VERSION == 1,
              "this observer was written for version 1 of the receiver's estimate tap");

inline std::string cuda_text(cudaError_t status)
{
  return std::string(cudaGetErrorName(status)) + ": " + cudaGetErrorString(status);
}

/// Page-locked host memory of the CUDA runtime for one payload.
class device_staging
{
public:
  explicit device_staging(std::size_t bytes_) : bytes(bytes_)
  {
    void*             memory = nullptr;
    const cudaError_t status = cudaHostAlloc(&memory, bytes, cudaHostAllocDefault);
    if (status != cudaSuccess || memory == nullptr) {
      throw std::runtime_error("no page-locked staging of " + std::to_string(bytes) + " bytes for the estimate (" +
                               cuda_text(status) + ")");
    }
    data = static_cast<std::uint8_t*>(memory);
  }
  device_staging(const device_staging&)            = delete;
  device_staging& operator=(const device_staging&) = delete;
  ~device_staging() { cudaFreeHost(data); }

  std::uint8_t* get() const noexcept { return data; }
  std::size_t   size() const noexcept { return bytes; }

private:
  std::uint8_t* data = nullptr;
  std::size_t   bytes;
};

/// Registers the mapping of a ring with the CUDA runtime, so that an asynchronous copy can land
/// in a slot. A runtime that cannot register that mapping reports why, and the copy lands in the
/// staging instead: nothing is ever copied asynchronously into memory that is not page-locked,
/// because such a copy makes its caller wait for the device.
class ring_registration
{
public:
  ring_registration(const void* base_, std::size_t bytes) : base(const_cast<void*>(base_))
  {
    const cudaError_t status = cudaHostRegister(base, bytes, cudaHostRegisterDefault);
    registered               = status == cudaSuccess;
    if (!registered) {
      reason = cuda_text(status);
      cudaGetLastError(); // the refusal is recorded here and must not surface in a later call
    }
  }
  ring_registration(const ring_registration&)            = delete;
  ring_registration& operator=(const ring_registration&) = delete;
  ~ring_registration()
  {
    if (registered) {
      cudaHostUnregister(base);
    }
  }

  bool               ok() const noexcept { return registered; }
  const std::string& why() const noexcept { return reason; }

private:
  void*       base;
  bool        registered = false;
  std::string reason;
};

/// The copier of the tool. One copy is in flight at a time.
class device_copier final : public copier
{
public:
  /// side: enqueue the copy on a stream of this observer, ordered after the producer by an event.
  explicit device_copier(bool side_) : side(side_)
  {
    check(cudaGetDevice(&device), "the current CUDA device");
    check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "the completion event of the copy");
    if (side) {
      check(cudaEventCreateWithFlags(&produced, cudaEventDisableTiming), "the event of the producer");
      check(cudaStreamCreateWithFlags(&own, cudaStreamNonBlocking), "the stream of the copy");
    }
  }
  device_copier(const device_copier&)            = delete;
  device_copier& operator=(const device_copier&) = delete;
  ~device_copier() override
  {
    if (own != nullptr) {
      cudaStreamDestroy(own);
    }
    if (produced != nullptr) {
      cudaEventDestroy(produced);
    }
    if (done != nullptr) {
      cudaEventDestroy(done);
    }
  }

  bool submit(const producer_view& view, void* destination) noexcept override
  {
    last = cudaSuccess;
    if (view.device != device) {
      last = cudaErrorInvalidDevice;
      return false;
    }
    const cudaStream_t producer = static_cast<cudaStream_t>(view.stream);
    cudaStream_t       stream   = producer;
    if (side) {
      // After every writer of the estimate, and on no stream of the receiver.
      if ((last = cudaEventRecord(produced, producer)) != cudaSuccess ||
          (last = cudaStreamWaitEvent(own, produced, 0)) != cudaSuccess) {
        return false;
      }
      stream = own;
    }
    if ((last = cudaMemcpyAsync(destination, view.estimates, view.bytes, cudaMemcpyDeviceToHost, stream)) != cudaSuccess) {
      return false;
    }
    // From here on the copy is enqueued. Without its event its completion cannot be observed:
    // the capture ends as a failed submission and the destination is not read.
    if ((last = cudaEventRecord(done, stream)) != cudaSuccess) {
      return false;
    }
    ++copies;
    return true;
  }

  copy_state poll() noexcept override
  {
    const cudaError_t status = cudaEventQuery(done);
    if (status == cudaSuccess) {
      return copy_state::complete;
    }
    if (status == cudaErrorNotReady) {
      return copy_state::pending;
    }
    last = status;
    return copy_state::failed;
  }

  copy_state drain() noexcept override
  {
    const cudaError_t status = cudaEventSynchronize(done);
    if (status != cudaSuccess) {
      last = status;
      return copy_state::failed;
    }
    return copy_state::complete;
  }

  bool        on_producer_stream() const noexcept override { return !side; }
  const char* stream_name() const noexcept override { return side ? COPY_SIDE : COPY_PRODUCER; }

  int           device_id() const noexcept { return device; }
  std::uint64_t submitted() const noexcept { return copies; }
  /// The name of the last CUDA status that was not a success, or an empty text.
  std::string last_error() const { return last == cudaSuccess ? std::string() : cuda_text(last); }

private:
  static void check(cudaError_t status, const char* what)
  {
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("the CUDA runtime gave no ") + what + " (" + cuda_text(status) + ")");
    }
  }

  bool          side;
  int           device   = 0;
  cudaEvent_t   done     = nullptr;
  cudaEvent_t   produced = nullptr;
  cudaStream_t  own      = nullptr;
  cudaError_t   last     = cudaSuccess;
  std::uint64_t copies   = 0;
};

/// The observer that the patched demodulator calls. It translates and forwards, and holds no state.
class receiver_tap final : public ocudu::pusch_resident_estimate_tap
{
public:
  explicit receiver_tap(capture& taker_) : taker(taker_) {}

  void on_estimate(const ocudu::pusch_resident_estimate& estimate) noexcept override
  {
    producer_view view;
    view.estimates               = estimate.estimates;
    view.bytes                   = estimate.nof_bytes;
    view.stream                  = estimate.stream;
    view.device                  = estimate.device_id;
    view.launch                  = estimate.launch;
    view.rnti                    = estimate.rnti;
    view.sfn                     = estimate.sfn;
    view.slot                    = estimate.slot_index;
    view.numerology              = estimate.numerology;
    view.start_prb               = estimate.start_prb;
    view.nof_prb                 = estimate.nof_prb;
    view.contiguous              = estimate.contiguous;
    view.dmrs_symbol_mask        = estimate.dmrs_symbol_mask;
    view.nof_dmrs_symbols        = estimate.nof_dmrs_symbols;
    view.rx_ports                = estimate.nof_rx_ports;
    view.layers                  = estimate.nof_tx_layers;
    view.dmrs_type               = estimate.dmrs_type;
    view.dmrs_scaling            = estimate.dmrs_scaling;
    view.frequency_interpolation = estimate.frequency_interpolation;
    view.frequency_smoothing     = estimate.frequency_smoothing;
    view.resident                = estimate.resident_mode;
    taker.on_estimate(view);
  }

private:
  capture& taker;
};

/// Registers a tap with the receiver for as long as it lives. The receiver has one tap per process.
class tap_registration
{
public:
  explicit tap_registration(ocudu::pusch_resident_estimate_tap& tap_) : tap(&tap_)
  {
    ocudu::pusch_resident_estimate_tap* before = ocudu::set_pusch_resident_estimate_tap(tap);
    if (before != nullptr) {
      ocudu::set_pusch_resident_estimate_tap(before);
      throw std::runtime_error("the receiver already has an observer of its estimate");
    }
  }
  tap_registration(const tap_registration&)            = delete;
  tap_registration& operator=(const tap_registration&) = delete;
  ~tap_registration()
  {
    if (ocudu::get_pusch_resident_estimate_tap() == tap) {
      ocudu::set_pusch_resident_estimate_tap(nullptr);
    }
  }

private:
  ocudu::pusch_resident_estimate_tap* tap;
};

/// Everything the tool holds for the capture of one run: the device objects, the page-locked
/// memory, the capture and the tap. Members are declared in the order they are created, so that
/// the tap is taken away from the receiver before the capture goes and the memory is freed last.
class device_route
{
public:
  /// ring is the ring of the O3 stream, or null for an audited run that publishes nothing.
  /// wanted names the evidence mode, whether a timed copy is asked to land in its slot, and the
  /// time a capture may ask for a completion. The staging is set here.
  device_route(bool side, std::size_t payload_bytes, obs::producer* ring, capture_config wanted) :
    device(side), staging(payload_bytes)
  {
    wanted.staging       = staging.get();
    wanted.staging_bytes = staging.size();
    asked_slot           = wanted.mode == evidence::timed && wanted.slot_landing && ring != nullptr;
    if (asked_slot) {
      pinned.reset(new ring_registration(&ring->header(), ring->segment_bytes()));
    }
    wanted.slot_landing = asked_slot && pinned->ok();
    in_slot             = wanted.slot_landing;
    taker.reset(new capture(device, ring, wanted));
    tap.reset(new receiver_tap(*taker));
  }
  device_route(const device_route&)            = delete;
  device_route& operator=(const device_route&) = delete;
  ~device_route()
  {
    close();
    taker->settle(); // a copy that ran out of time must not write memory that is freed below
  }

  /// Registers the tap with the receiver. Call it once every decode of the run is prepared.
  void open() { registered.reset(new tap_registration(*tap)); }
  /// Takes the tap away from the receiver. After the last decode.
  void close() noexcept { registered.reset(); }

  capture&             captures() noexcept { return *taker; }
  const device_copier& copier_of() const noexcept { return device; }
  /// Where a timed copy lands: its slot, or the staging.
  bool slot_landing() const noexcept { return in_slot; }
  /// Why a copy that was asked to land in its slot lands in the staging, or an empty text.
  std::string landing_refusal() const { return asked_slot && !in_slot ? pinned->why() : std::string(); }

private:
  device_copier                      device;
  device_staging                     staging;
  std::unique_ptr<ring_registration> pinned;
  std::unique_ptr<capture>           taker;
  std::unique_ptr<receiver_tap>      tap;
  std::unique_ptr<tap_registration>  registered;
  bool                               asked_slot = false;
  bool                               in_slot    = false;
};

} // namespace isac_estimate
