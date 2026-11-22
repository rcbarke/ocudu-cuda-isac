// The CUDA device of a plan: the units of units.h as kernels on one stream (Sprint 3.6, selector
// sense_core).
//
// Two kernels serve one window. The ingest kernel gathers every record of the window from the
// layout of its producer into the cube of the core, one thread per element. The front-end kernel
// runs the window average, the template subtraction, the mean over the DM-RS symbols and the
// z-score, one thread per antenna and subcarrier, and writes the input of the model where the
// engine reads it. Both are compiled with --fmad=false, so that they execute the operations of
// units.h one by one, as the host twin does.
//
// One submission is: an upload of the staging block, the two kernels, the engine, a download of
// the output block, and five events. Four of them are timing marks and the fifth is the
// completion event. Nothing here waits for the device: done() queries that event. The only
// synchronizing calls are in configure() and in ingest_one(), which no window passes through.
//
// Memory is allocated in configure() and nowhere else: device memory for the staging block, the
// cube, the template magnitudes and the output block, and page-locked host memory for the two
// blocks that cross the bus.
//
// It uses the CUDA runtime API. ASCII only.

#include "kernels/device.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <string>

namespace sense {

namespace {

constexpr unsigned BLOCK = 128;

struct payload_table {
  const std::uint8_t* at[MAX_WINDOW_RECORDS];
};

/// ingest/<layout>: one thread per element of the cube, [record][a][position][s].
__global__ void ingest_kernel(const std::uint8_t* staging, window_geometry window, payload_table table, float* cube,
                              std::uint8_t* valid)
{
  const std::uint32_t per_record = window.rx * window.positions * window.symbols;
  const std::uint32_t t          = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= window.records * per_record) {
    return;
  }
  const std::uint32_t    r      = t / per_record;
  const std::uint32_t    rest   = t % per_record;
  const std::uint32_t    a      = rest / (window.positions * window.symbols);
  const std::uint32_t    k      = (rest / window.symbols) % window.positions;
  const std::uint32_t    s      = rest % window.symbols;
  const record_geometry& record = window.record[r];
  if (k >= record.positions) {
    return; // beyond the allocation of this slot: the window unit never reads it
  }
  const std::uint8_t* data = record.on_device != 0 ? table.at[r] : staging + record.payload_offset;
  float               real, imaginary;
  valid[t]        = ingest_element(data, staging + record.mask_offset, record, window.tau2, a, k, s, real, imaginary) ? 1 : 0;
  cube[2 * t]     = real;
  cube[2 * t + 1] = imaginary;
}

/// window/coherent_mean, clutter/background_magnitude, symbols/dmrs_mean and
/// normalizer/zscore_global: one thread per position [a][k] of the support.
__global__ void frontend_kernel(const float* cube, const std::uint8_t* valid, window_geometry window,
                                const float* template_magnitude, float* mean, std::uint8_t* count, float* difference,
                                float* input)
{
  const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= window.rx * window.positions) {
    return;
  }
  frontend_position(cube, valid, window, template_magnitude, t / window.positions, t % window.positions, mean, count,
                    difference, input);
}

std::string cuda_text(cudaError_t status)
{
  return std::string(cudaGetErrorName(status)) + ": " + cudaGetErrorString(status);
}

void check(cudaError_t status, const char* what)
{
  if (status != cudaSuccess) {
    throw device_error(std::string("the CUDA runtime refused ") + what + " (" + cuda_text(status) + ")");
  }
}

class cuda_device final : public device
{
public:
  cuda_device(int index_, void* external) : index(index_)
  {
    int               found  = 0;
    const cudaError_t status = cudaGetDeviceCount(&found);
    if (status != cudaSuccess || found < 1) {
      cudaGetLastError();
      throw device_error("the CUDA runtime reaches no device" +
                         (status != cudaSuccess ? " (" + cuda_text(status) + ")" : std::string()));
    }
    if (index < 0 || index >= found) {
      throw device_error("the CUDA device " + std::to_string(index) + " does not exist: the runtime reaches " +
                         std::to_string(found));
    }
    check(cudaSetDevice(index), "the selection of the device");
    check(cudaGetDeviceProperties(&properties, index), "the properties of the device");
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    if (external != nullptr) {
      work = static_cast<cudaStream_t>(external);
    } else {
      check(cudaStreamCreateWithFlags(&work, cudaStreamNonBlocking), "a stream");
      own_stream = true;
    }
    for (cudaEvent_t& event : events) {
      check(cudaEventCreate(&event), "a timing event"); // with timing: the marks of a stage
    }
    calls += 4 + EVENTS;
  }

  ~cuda_device() override
  {
    if (work != nullptr) {
      cudaStreamSynchronize(work); // nothing of a window may still write into what is freed below
    }
    cudaFree(staging_device);
    cudaFree(output_device);
    cudaFree(cube);
    cudaFree(cube_valid);
    cudaFree(magnitudes);
    if (staging_host != nullptr) {
      cudaFreeHost(staging_host);
    }
    if (output_host != nullptr) {
      cudaFreeHost(output_host);
    }
    for (cudaEvent_t event : events) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
    if (own_stream && work != nullptr) {
      cudaStreamDestroy(work);
    }
  }

  const char* kind() const override { return "cuda"; }
  std::string name() const override { return "cuda:" + std::to_string(index); }
  std::string description() const override { return properties.name; }
  std::string identity_json() const override
  {
    char        uuid[33];
    static const char digits[] = "0123456789abcdef";
    for (unsigned i = 0; i != 16; ++i) {
      const unsigned byte = static_cast<unsigned char>(properties.uuid.bytes[i]);
      uuid[2 * i]         = digits[byte >> 4];
      uuid[2 * i + 1]     = digits[byte & 15];
    }
    uuid[32] = '\0';
    char text[640];
    std::snprintf(text, sizeof(text),
                  "{\"kind\": \"cuda\", \"index\": %d, \"name\": \"%s\", \"uuid\": \"%s\", \"capability\": \"%d.%d\", "
                  "\"cuda_runtime\": %d, \"cuda_driver\": %d}",
                  index, properties.name, uuid, properties.major, properties.minor, runtime_version, driver_version);
    return text;
  }

  void configure(const device_layout& wanted, const float* template_magnitude) override
  {
    layout              = wanted;
    const std::size_t n = layout.elements();
    check(cudaMalloc(&staging_device, layout.staging_bytes), "device memory for the staging block");
    check(cudaMalloc(&output_device, layout.bytes), "device memory for the output block");
    check(cudaMalloc(&cube, 2 * sizeof(float) * n * layout.records), "device memory for the cube");
    check(cudaMalloc(&cube_valid, n * layout.records), "device memory for the validity of the cube");
    check(cudaMalloc(&magnitudes, sizeof(float) * n), "device memory for the template");
    check(cudaHostAlloc(&staging_host, layout.staging_bytes, cudaHostAllocDefault), "page-locked memory for the staging block");
    check(cudaHostAlloc(&output_host, layout.bytes, cudaHostAllocDefault), "page-locked memory for the output block");
    allocations += 7;
    std::memset(staging_host, 0, layout.staging_bytes);
    std::memset(output_host, 0, layout.bytes);
    check(cudaMemset(staging_device, 0, layout.staging_bytes), "the clearing of the staging block");
    check(cudaMemset(output_device, 0, layout.bytes), "the clearing of the output block");
    check(cudaMemset(cube, 0, 2 * sizeof(float) * n * layout.records), "the clearing of the cube");
    check(cudaMemset(cube_valid, 0, n * layout.records), "the clearing of the validity of the cube");
    check(cudaMemcpy(magnitudes, template_magnitude, sizeof(float) * n, cudaMemcpyHostToDevice), "the upload of the template");
    calls += 12;
  }

  std::uint8_t*       staging() override { return static_cast<std::uint8_t*>(staging_host); }
  const std::uint8_t* output() const override { return static_cast<const std::uint8_t*>(output_host); }
  std::uint8_t*       output_writable() override { return static_cast<std::uint8_t*>(output_host); }
  std::uint8_t*       engine_block() override { return static_cast<std::uint8_t*>(output_device); }
  void*               stream() override { return work; }

  void begin(const window_geometry& window, const void* const payload[MAX_WINDOW_RECORDS], std::size_t staged) override
  {
    payload_table table;
    for (std::uint32_t r = 0; r != MAX_WINDOW_RECORDS; ++r) {
      table.at[r] = r < window.records ? static_cast<const std::uint8_t*>(payload[r]) : nullptr;
    }
    std::uint8_t* out = static_cast<std::uint8_t*>(output_device);
    check(cudaEventRecord(events[START], work), "the mark at the start of a window");
    if (staged != 0) {
      check(cudaMemcpyAsync(staging_device, staging_host, staged, cudaMemcpyHostToDevice, work), "the upload of a window");
    }
    const std::uint32_t elements = window.records * window.rx * window.positions * window.symbols;
    ingest_kernel<<<(elements + BLOCK - 1) / BLOCK, BLOCK, 0, work>>>(static_cast<const std::uint8_t*>(staging_device),
                                                                      window, table, static_cast<float*>(cube),
                                                                      static_cast<std::uint8_t*>(cube_valid));
    check(cudaGetLastError(), "the launch of the ingest kernel");
    check(cudaEventRecord(events[INGEST], work), "the mark behind the ingest");
    const std::uint32_t positions = window.rx * window.positions;
    frontend_kernel<<<(positions + BLOCK - 1) / BLOCK, BLOCK, 0, work>>>(
        static_cast<const float*>(cube), static_cast<const std::uint8_t*>(cube_valid), window,
        static_cast<const float*>(magnitudes), reinterpret_cast<float*>(out + layout.mean), out + layout.count,
        reinterpret_cast<float*>(out + layout.difference), reinterpret_cast<float*>(out + layout.input));
    check(cudaGetLastError(), "the launch of the front-end kernel");
    check(cudaEventRecord(events[KERNELS], work), "the mark behind the kernels");
    calls += staged != 0 ? 6 : 5;
  }

  void end(bool with_mean) override
  {
    check(cudaEventRecord(events[ENGINE], work), "the mark behind the engine");
    check(cudaMemcpyAsync(output_host, output_device, with_mean ? layout.bytes : layout.bytes_without_mean,
                          cudaMemcpyDeviceToHost, work),
          "the download of a window");
    check(cudaEventRecord(events[COMPLETE], work), "the completion event");
    calls += 3;
  }

  bool done() override
  {
    ++calls;
    const cudaError_t status = cudaEventQuery(events[COMPLETE]);
    if (status == cudaErrorNotReady) {
      return false;
    }
    check(status, "the query of the completion event");
    return true;
  }

  stage_marks marks(std::int64_t now) override
  {
    // The device measured the time between its marks. The completion was seen on the host at
    // `now`, so every mark lies that much device time before it.
    float download = 0.0f, engine = 0.0f, kernels = 0.0f;
    check(cudaEventElapsedTime(&download, events[ENGINE], events[COMPLETE]), "the time of the download");
    check(cudaEventElapsedTime(&engine, events[KERNELS], events[ENGINE]), "the time of the engine");
    check(cudaEventElapsedTime(&kernels, events[INGEST], events[KERNELS]), "the time of the kernels");
    calls += 3;
    stage_marks made;
    made.complete = now;
    made.engine   = now - static_cast<std::int64_t>(static_cast<double>(download) * 1.0e6);
    made.kernels  = made.engine - static_cast<std::int64_t>(static_cast<double>(engine) * 1.0e6);
    made.ingest   = made.kernels - static_cast<std::int64_t>(static_cast<double>(kernels) * 1.0e6);
    return made;
  }

  void ingest_one(const record_geometry& record, double tau2, const std::uint8_t* payload, const std::uint8_t* mask,
                  std::size_t payload_bytes, std::size_t mask_bytes, float* values, std::uint8_t* valid) override
  {
    // A unit entry, never the path of a window: it stages one record, runs the ingest kernel
    // alone and waits for it.
    const std::size_t payload_at = (mask_bytes + 63) & ~static_cast<std::size_t>(63);
    const std::size_t elements   = static_cast<std::size_t>(record.planes) * record.positions * record.symbols;
    if (payload_at + payload_bytes > layout.staging_bytes || record.planes != layout.rx || record.symbols != layout.symbols ||
        record.positions > layout.positions) {
      throw device_error("the record does not fit the workspaces of this plan");
    }
    std::uint8_t* staged = static_cast<std::uint8_t*>(staging_host);
    std::memcpy(staged, mask, mask_bytes);
    std::memcpy(staged + payload_at, payload, payload_bytes);
    window_geometry window;
    std::memset(&window, 0, sizeof(window));
    window.records                  = 1;
    window.rx                       = record.planes;
    window.positions                = record.positions;
    window.symbols                  = record.symbols;
    window.tau2                     = tau2;
    window.record[0]                = record;
    window.record[0].offset         = 0;
    window.record[0].mask_offset    = 0;
    window.record[0].payload_offset = static_cast<std::uint32_t>(payload_at);
    window.record[0].on_device      = 0;
    payload_table table;
    std::memset(&table, 0, sizeof(table));
    check(cudaMemcpyAsync(staging_device, staging_host, payload_at + payload_bytes, cudaMemcpyHostToDevice, work),
          "the upload of a record");
    ingest_kernel<<<(static_cast<std::uint32_t>(elements) + BLOCK - 1) / BLOCK, BLOCK, 0, work>>>(
        static_cast<const std::uint8_t*>(staging_device), window, table, static_cast<float*>(cube),
        static_cast<std::uint8_t*>(cube_valid));
    check(cudaGetLastError(), "the launch of the ingest kernel");
    check(cudaMemcpyAsync(values, cube, 2 * sizeof(float) * elements, cudaMemcpyDeviceToHost, work), "the download of a cube");
    check(cudaMemcpyAsync(valid, cube_valid, elements, cudaMemcpyDeviceToHost, work), "the download of the validity");
    check(cudaStreamSynchronize(work), "the wait for a unit entry");
    calls += 5;
  }

private:
  enum { START = 0, INGEST = 1, KERNELS = 2, ENGINE = 3, COMPLETE = 4, EVENTS = 5 };

  int            index;
  cudaDeviceProp properties;
  int            runtime_version = 0, driver_version = 0;
  cudaStream_t   work       = nullptr;
  bool           own_stream = false;
  cudaEvent_t    events[EVENTS] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  device_layout  layout;
  void*          staging_device = nullptr;
  void*          output_device  = nullptr;
  void*          cube           = nullptr;
  void*          cube_valid     = nullptr;
  void*          magnitudes     = nullptr;
  void*          staging_host   = nullptr;
  void*          output_host    = nullptr;
};

} // namespace

std::unique_ptr<device> make_cuda_device(int index, void* stream)
{
  return std::unique_ptr<device>(new cuda_device(index, stream));
}

bool built_with_cuda()
{
  return true;
}

} // namespace sense
