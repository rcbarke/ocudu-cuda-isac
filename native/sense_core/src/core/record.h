// What the core reads of an obs/v1 record header, and the radio time of a slot and a symbol
// (Sprint 3.6, selector sense_core).
//
// The byte-level definition of the header is native/obs_ring/include/obs/record.h, a header of
// constants and one plain structure. The core includes that definition and nothing else of the
// ring: a view arrives from an adapter, whatever carried the record. The time rules are those of
// TS 38.211 5.3.1, as ocudu_cuda_isac.sensing.pilot_ls and features state them.
//
// ASCII only.

#pragma once

#include "obs/record.h"
#include <cstdint>
#include <cstring>

namespace sense {

constexpr std::uint64_t TC_PER_SECOND    = 480000ULL * 4096ULL;
constexpr unsigned      SYMBOLS_PER_SLOT = 14;
constexpr unsigned      KAPPA            = 64;
constexpr unsigned      SUBCARRIERS_PER_PRB = 12;

/// The numerology of a subcarrier spacing in kHz, or -1.
inline int numerology(int scs_khz)
{
  switch (scs_khz) {
    case 15: return 0;
    case 30: return 1;
    case 60: return 2;
    case 120: return 3;
    default: return -1;
  }
}

/// Length of the useful part of a symbol in T_c.
inline std::uint64_t useful_tc(int mu)
{
  return (2048ULL * KAPPA) >> mu;
}

/// Start of a slot in T_c, counted from slot 0 of the epoch. The first symbol of every half
/// subframe carries the longer cyclic prefix.
inline std::uint64_t slot_start_tc(int mu, std::uint64_t absolute_slot)
{
  const std::uint64_t per      = 1ULL << mu;
  std::uint64_t       start    = (absolute_slot / per) * (TC_PER_SECOND / 1000);
  const std::uint64_t useful   = (2048ULL * KAPPA) >> mu, prefix = (144ULL * KAPPA) >> mu, boundary = 7ULL << mu;
  const std::uint64_t symbols  = (absolute_slot % per) * SYMBOLS_PER_SLOT;
  for (std::uint64_t symbol = 0; symbol != symbols; ++symbol) {
    start += useful + prefix + (symbol % boundary == 0 ? 16ULL * KAPPA : 0);
  }
  return start;
}

/// Start of the useful part of one symbol in T_c from slot 0 of the epoch.
inline std::uint64_t symbol_time_tc(int mu, std::uint64_t absolute_slot, unsigned symbol)
{
  const std::uint64_t useful = (2048ULL * KAPPA) >> mu, prefix = (144ULL * KAPPA) >> mu, boundary = 7ULL << mu;
  const std::uint64_t first  = (absolute_slot % (1ULL << mu)) * SYMBOLS_PER_SLOT;
  std::uint64_t       start  = slot_start_tc(mu, absolute_slot);
  for (std::uint64_t index = first; index != first + symbol; ++index) {
    start += useful + prefix + (index % boundary == 0 ? 16ULL * KAPPA : 0);
  }
  return start + prefix + ((first + symbol) % boundary == 0 ? 16ULL * KAPPA : 0);
}

/// The header of a view, copied out of memory that another process may rewrite.
inline obs::record_header read_header(const void* record)
{
  obs::record_header header;
  std::memcpy(&header, record, sizeof(header));
  return header;
}

/// The DM-RS symbols a header names: their indices in rising order. Returns how many.
inline unsigned symbols_of(const obs::record_header& header, unsigned out[SYMBOLS_PER_SLOT])
{
  unsigned count = 0;
  for (unsigned symbol = 0; symbol != SYMBOLS_PER_SLOT; ++symbol) {
    if ((header.dmrs_symbols >> symbol) & 1U) {
      out[count++] = symbol;
    }
  }
  return count;
}

inline const char* stage_name(std::uint8_t stage)
{
  switch (stage) {
    case obs::STAGE_O1: return "O1";
    case obs::STAGE_O2: return "O2";
    case obs::STAGE_O3: return "O3";
    default: return "None";
  }
}

inline const char* source_name(std::uint8_t source)
{
  return source == obs::SOURCE_PUSCH_DMRS ? "pusch_dmrs" : "None";
}

inline const char* layout_name(std::uint8_t layout)
{
  switch (layout) {
    case obs::LAYOUT_OCUDU_O3_FP16: return "ocudu_o3_fp16";
    case obs::LAYOUT_OCUDU_O2_COMB_C64: return "ocudu_o2_comb_c64";
    case obs::LAYOUT_OCUDU_O1_CBF16: return "ocudu_o1_cbf16";
    case obs::LAYOUT_AERIAL_HEST_C64: return "aerial_hest_c64";
    case obs::LAYOUT_AERIAL_IQ_FP16: return "aerial_iq_fp16";
    default: return "None";
  }
}

inline const char* dtype_name(std::uint8_t dtype)
{
  switch (dtype) {
    case obs::DTYPE_C64: return "<c8";
    case obs::DTYPE_CFP16: return "<cfp16";
    case obs::DTYPE_CBF16: return "<cbf16";
    default: return "None";
  }
}

inline const char* evidence_name(std::uint8_t evidence)
{
  switch (evidence) {
    case obs::EVIDENCE_AUDIT: return "audit";
    case obs::EVIDENCE_TIMED: return "timed";
    default: return "unknown";
  }
}

} // namespace sense
