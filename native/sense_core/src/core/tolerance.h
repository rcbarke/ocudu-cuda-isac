// What the native units may differ from their reference by (Sprint 3.6, selector sense_core).
//
// The reference of the front end, ocudu_cuda_isac.sensing.cusense, stores a window mean as
// complex64 and computes the input of the model in double precision before it rounds to float32.
// The native units work in FP32. The bound follows from their operations.
//
//   window mean     three FP32 additions and one division against an exact sum rounded once:
//                   each part within MEAN_ULPS spacings of FP32 at the magnitude of the mean.
//   input           the magnitude, the subtraction of the template, the mean over the symbols
//                   and the z-score: within INPUT_ULPS spacings of FP32 at the larger of the two
//                   magnitudes, divided by the divisor of the z-score. A wrong index or a wrong
//                   offset is off by the size of the signal, many orders above this.
//   hand vectors    the values of the sprint card, exact in FP32: within VECTOR_RELATIVE.
//
// A payload that was stored as FP16 holds each part within one half spacing of FP16, a relative
// 2^-11. That is the FP16 tolerance, declared apart: it bounds what a value may differ by after
// the producer stored it, and no unit of this core adds to it, because the ingest unit widens
// FP16 to FP32 without rounding.
//
// The engine is compared with the reference model by the tolerances of the Sprint 3.5 parity,
// which hold between two devices of one runtime: a logit by an absolute term plus a relative
// one, a probability and a cell of the map by an absolute one. An engine built in FP16 has its
// own, wider set, declared here and graded only where a gate builds such an engine.
//
// ASCII only.

#pragma once

namespace sense {
namespace tolerance {

constexpr double FP32_SPACING       = 1.1920928955078125e-07; // 2^-23
constexpr double FP16_RELATIVE      = 4.8828125e-04;          // 2^-11
constexpr double VECTOR_RELATIVE    = 1.0e-6;
constexpr double MEAN_ULPS          = 4.0;
constexpr double INPUT_ULPS         = 16.0;
// Engine against the reference model, FP32.
constexpr double LOGIT_ABSOLUTE       = 1.0e-3;
constexpr double LOGIT_RELATIVE       = 1.0e-4;
constexpr double PROBABILITY_ABSOLUTE = 2.5e-4;
constexpr double MAP_ABSOLUTE         = 1.0e-4;
// Engine against the reference model, built in FP16.
constexpr double FP16_LOGIT_ABSOLUTE       = 5.0e-2;
constexpr double FP16_LOGIT_RELATIVE       = 1.0e-2;
constexpr double FP16_PROBABILITY_ABSOLUTE = 1.0e-2;
constexpr double FP16_MAP_ABSOLUTE         = 5.0e-3;
// The optional stage on the host, in double precision against NumPy.
constexpr double SMOOTHED_POSITION_M = 1.0e-9;

} // namespace tolerance
} // namespace sense
