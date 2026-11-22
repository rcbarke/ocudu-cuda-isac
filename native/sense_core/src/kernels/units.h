// The units of the cuSense-like branch: ingest and front end (Sprint 3.6, selector sense_core).
//
// Every function here is one registry entry of the reference, ocudu_cuda_isac.sensing.cusense,
// under the same name, written once and compiled twice: by nvcc into the kernels of
// cuda_device.cu, and by the host compiler into the twin of host_device.cpp. The two builds run
// the same operations in the same order, with no fused multiply-add on either side, so the twin
// is a bit-level model of the kernels on a host without a device.
//
//   ingest/ocudu_o3_fp16       [dmrs_symbol][prb][rx_port][subcarrier_in_prb], complex FP16
//   ingest/ocudu_o2_comb_c64   [rx_plane][reference_symbol][pilot], complex64
//   ingest/aerial_hest_c64     [dmrs][subcarrier][rx_antenna][layer] of one layer, complex64
//   window/coherent_mean       the slots of one period, each at the offset of its allocation,
//                              averaged in the complex domain over their valid samples
//   clutter/background_magnitude   |H_t| - |H_B| where the window and the template hold a sample
//   symbols/dmrs_mean          the mean over the DM-RS symbols that hold one
//   normalizer/zscore_global   (d - mean) / (scale + epsilon); a position without a sample is 0
//
// Working precision: FP32. An ingest unit widens the payload once, FP16 to FP32 without a change
// of value, and applies no scale. The validity rule of the reference is evaluated on exact
// products in double precision, so that it does not depend on the working precision: a sample
// is valid when its bit is set, both parts are finite and its magnitude lies above tau.
//
// The reference stores a window mean as complex64 and computes the input in double precision
// before it rounds to float32. tolerance.h declares what the FP32 units may differ by.
//
// No allocation, no branch on data outside the validity rule, no library call but sqrtf.
// ASCII only.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define SENSE_HD __host__ __device__
#else
#define SENSE_HD
#endif

namespace sense {

constexpr std::uint32_t MAX_WINDOW_RECORDS = 8;
constexpr std::uint32_t MAX_RX             = 16;
// The identifiers of obs::layout_id that this core has an ingest unit for.
constexpr std::uint32_t LAYOUT_O3_FP16         = 1;
constexpr std::uint32_t LAYOUT_O2_COMB_C64     = 2;
constexpr std::uint32_t LAYOUT_AERIAL_HEST_C64 = 4;

/// One record of a window, as an ingest unit and the window unit read it. Plain data: it is
/// copied into the kernels by value.
struct record_geometry {
  std::uint32_t layout;         ///< LAYOUT_*.
  std::uint32_t planes;         ///< Receive planes of the payload.
  std::uint32_t positions;      ///< Positions on the subcarrier axis of the record.
  std::uint32_t symbols;        ///< DM-RS symbols.
  std::uint32_t offset;         ///< Where position 0 of the record lies on the support of the window.
  std::uint32_t payload_offset; ///< Of the payload in the staging block, when it was staged.
  std::uint32_t mask_offset;    ///< Of the validity bits in the staging block.
  std::uint32_t on_device;      ///< 1: the payload is read from the address of the view, on the device.
  std::uint32_t plane_of[MAX_RX]; ///< plane_of[a]: the receive plane behind array element a, rising.
};

/// Everything the two kernels of one window need. Plain data.
struct window_geometry {
  std::uint32_t   records;   ///< Slots of this window.
  std::uint32_t   rx;        ///< A.
  std::uint32_t   positions; ///< K: the support of the specification.
  std::uint32_t   symbols;   ///< S.
  double          tau2;      ///< The square of validity.tau.
  float           mean;      ///< Of the z-score, as float32.
  float           divisor;   ///< scale + epsilon of the z-score, as float32.
  record_geometry record[MAX_WINDOW_RECORDS];
};

SENSE_HD inline float bits_to_float(std::uint32_t bits)
{
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/// An FP16 number widened to FP32. Every FP16 value is an FP32 value: nothing is rounded.
SENSE_HD inline float half_to_float(std::uint16_t half)
{
  const std::uint32_t sign     = static_cast<std::uint32_t>(half & 0x8000U) << 16;
  const std::uint32_t exponent = (half >> 10) & 0x1fU;
  std::uint32_t       fraction = half & 0x3ffU;
  if (exponent == 0) {
    if (fraction == 0) {
      return bits_to_float(sign);
    }
    std::uint32_t shift = 0;
    while ((fraction & 0x400U) == 0) { // a subnormal: at most ten steps
      fraction <<= 1;
      ++shift;
    }
    return bits_to_float(sign | ((113U - shift) << 23) | ((fraction & 0x3ffU) << 13));
  }
  if (exponent == 31) {
    return bits_to_float(sign | 0x7f800000U | (fraction << 13));
  }
  return bits_to_float(sign | ((exponent + 112U) << 23) | (fraction << 13));
}

SENSE_HD inline std::uint16_t read_u16(const std::uint8_t* at)
{
  return static_cast<std::uint16_t>(static_cast<std::uint16_t>(at[0]) | (static_cast<std::uint16_t>(at[1]) << 8));
}

SENSE_HD inline float read_f32(const std::uint8_t* at)
{
  return bits_to_float(static_cast<std::uint32_t>(at[0]) | (static_cast<std::uint32_t>(at[1]) << 8) |
                       (static_cast<std::uint32_t>(at[2]) << 16) | (static_cast<std::uint32_t>(at[3]) << 24));
}

/// The position of one element in the payload of its layout: array element a in rising order,
/// position k on the subcarrier axis of the record, DM-RS symbol s.
SENSE_HD inline std::uint32_t element_of(const record_geometry& record, std::uint32_t a, std::uint32_t k, std::uint32_t s)
{
  const std::uint32_t plane = record.plane_of[a];
  if (record.layout == LAYOUT_O3_FP16) {
    // [dmrs_symbol][prb][rx_port][subcarrier_in_prb]
    const std::uint32_t blocks = record.positions / 12U;
    return ((s * blocks + k / 12U) * record.planes + plane) * 12U + k % 12U;
  }
  if (record.layout == LAYOUT_O2_COMB_C64) {
    // [rx_plane][reference_symbol][pilot]
    return (plane * record.symbols + s) * record.positions + k;
  }
  // LAYOUT_AERIAL_HEST_C64: [dmrs][subcarrier][rx_antenna][layer], one layer
  return (s * record.positions + k) * record.planes + plane;
}

/// The ingest units. Reads one element of a payload as FP32 and says whether it is a valid
/// sample. An element that is not valid is returned as zero.
SENSE_HD inline bool ingest_element(const std::uint8_t* payload, const std::uint8_t* mask, const record_geometry& record,
                                    double tau2, std::uint32_t a, std::uint32_t k, std::uint32_t s, float& real,
                                    float& imaginary)
{
  const std::uint32_t element = element_of(record, a, k, s);
  if (record.layout == LAYOUT_O3_FP16) {
    const std::uint8_t* at = payload + 4U * static_cast<std::size_t>(element);
    real                   = half_to_float(read_u16(at));
    imaginary              = half_to_float(read_u16(at + 2));
  } else {
    const std::uint8_t* at = payload + 8U * static_cast<std::size_t>(element);
    real                   = read_f32(at);
    imaginary              = read_f32(at + 4);
  }
  const bool bit    = ((mask[element >> 3] >> (element & 7U)) & 1U) != 0;
  // x - x is zero for a finite x and not a number otherwise: a test without a library call.
  const bool finite = (real - real == 0.0f) && (imaginary - imaginary == 0.0f);
  if (!bit || !finite) {
    real      = 0.0f;
    imaginary = 0.0f;
    return false;
  }
  const double power = static_cast<double>(real) * static_cast<double>(real) +
                       static_cast<double>(imaginary) * static_cast<double>(imaginary);
  return power > tau2;
}

/// window/coherent_mean at one element of the support: the mean of the valid samples of the
/// records that cover position k, in the order of the records, and their count.
/// cube holds pairs {real, imaginary} as [record][a][position of the record][s].
SENSE_HD inline std::uint32_t coherent_mean(const float* cube, const std::uint8_t* valid, const window_geometry& window,
                                            std::uint32_t a, std::uint32_t k, std::uint32_t s, float& real,
                                            float& imaginary)
{
  float         sum_real = 0.0f, sum_imaginary = 0.0f;
  std::uint32_t count = 0;
  for (std::uint32_t r = 0; r != window.records; ++r) {
    const record_geometry& record = window.record[r];
    if (k < record.offset || k - record.offset >= record.positions) {
      continue; // the allocation of this slot does not cover the position
    }
    const std::size_t at =
        ((static_cast<std::size_t>(r) * window.rx + a) * window.positions + (k - record.offset)) * window.symbols + s;
    if (valid[at] != 0) {
      sum_real += cube[2 * at];
      sum_imaginary += cube[2 * at + 1];
      ++count;
    }
  }
  if (count == 0) {
    real      = 0.0f;
    imaginary = 0.0f;
    return 0;
  }
  const float divisor = static_cast<float>(count);
  real                = sum_real / divisor;
  imaginary           = sum_imaginary / divisor;
  return count;
}

/// The magnitude of a complex value in FP32.
SENSE_HD inline float magnitude(float real, float imaginary)
{
  return sqrtf(real * real + imaginary * imaginary);
}

/// clutter/background_magnitude: the magnitude of the window minus that of the template.
SENSE_HD inline float background_magnitude(float real, float imaginary, float template_magnitude)
{
  return magnitude(real, imaginary) - template_magnitude;
}

/// symbols/dmrs_mean: the mean over the DM-RS symbols that took part.
SENSE_HD inline float dmrs_mean(float sum, std::uint32_t symbols)
{
  return symbols == 0 ? 0.0f : sum / static_cast<float>(symbols);
}

/// normalizer/zscore_global.
SENSE_HD inline float zscore_global(float difference, float mean, float divisor)
{
  return (difference - mean) / divisor;
}

/// The front end at one position (a, k) of the support: the three units after the window mean.
/// template_magnitude holds |H_B| as [a][k][s], and a negative value where the template has no
/// sample. Writes the window mean and its count per DM-RS symbol, the difference before the
/// z-score and the input of the model.
SENSE_HD inline void frontend_position(const float* cube, const std::uint8_t* valid, const window_geometry& window,
                                       const float* template_magnitude, std::uint32_t a, std::uint32_t k, float* mean,
                                       std::uint8_t* count, float* difference, float* input)
{
  float         sum     = 0.0f;
  std::uint32_t symbols = 0;
  for (std::uint32_t s = 0; s != window.symbols; ++s) {
    const std::size_t   at = (static_cast<std::size_t>(a) * window.positions + k) * window.symbols + s;
    float               real, imaginary;
    const std::uint32_t records = coherent_mean(cube, valid, window, a, k, s, real, imaginary);
    mean[2 * at]                = real;
    mean[2 * at + 1]            = imaginary;
    count[at]                   = static_cast<std::uint8_t>(records);
    if (records != 0 && template_magnitude[at] >= 0.0f) {
      sum += background_magnitude(real, imaginary, template_magnitude[at]);
      ++symbols;
    }
  }
  const std::size_t position = static_cast<std::size_t>(a) * window.positions + k;
  const float       d        = dmrs_mean(sum, symbols);
  difference[position]       = d;
  input[position]            = symbols == 0 ? 0.0f : zscore_global(d, window.mean, window.divisor);
}

} // namespace sense
