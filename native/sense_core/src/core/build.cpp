// What this build of the core is (Sprint 3.6, selector sense_core). ASCII only.

#include "core/build.h"
#include "core/json.h"
#include "core/tolerance.h"
#include "engine/engine.h"
#include "sense_build_info.h"
#include "sense_core/sense_core.h"
#include <cstring>

namespace sense {

namespace {

const unit UNITS[] = {
    {"input", "o3_fp16", SENSE_UNIT_UNITS_SHA256, "src/kernels/units.h", "device", "fp32"},
    {"input", "o2_comb", SENSE_UNIT_UNITS_SHA256, "src/kernels/units.h", "device", "fp32"},
    {"ingest", "ocudu_o3_fp16", SENSE_UNIT_UNITS_SHA256, "src/kernels/units.h", "device", "fp16 widened to fp32"},
    {"ingest", "ocudu_o2_comb_c64", SENSE_UNIT_UNITS_SHA256, "src/kernels/units.h", "device", "fp32"},
    {"ingest", "aerial_hest_c64", SENSE_UNIT_UNITS_SHA256, "src/kernels/units.h", "device", "fp32"},
    {"window", "coherent_mean", SENSE_UNIT_UNITS_SHA256, "src/kernels/units.h", "device", "fp32"},
    {"clutter", "background_magnitude", SENSE_UNIT_UNITS_SHA256, "src/kernels/units.h", "device", "fp32"},
    {"symbols", "dmrs_mean", SENSE_UNIT_UNITS_SHA256, "src/kernels/units.h", "device", "fp32"},
    {"normalizer", "zscore_global", SENSE_UNIT_UNITS_SHA256, "src/kernels/units.h", "device", "fp32"},
    {"model", "cusense_cnn", SENSE_UNIT_ENGINE_SHA256, "src/engine/engine.h", "engine", "fp32"},
    {"detector", "frozen_threshold", SENSE_UNIT_PLAN_SHA256, "src/plan/plan.cpp", "host", "fp64"},
    {"smoothing", "none", SENSE_UNIT_SMOOTHER_SHA256, "src/plan/smoother.cpp", "host", "fp64"},
    {"smoothing", "map_mean_kalman", SENSE_UNIT_SMOOTHER_SHA256, "src/plan/smoother.cpp", "host", "fp64"},
    {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}};

} // namespace

const char* core_sources_sha256()
{
  return SENSE_CORE_SOURCES_SHA256;
}

const unit* registry()
{
  return UNITS;
}

const unit* find_unit(const char* stage, const std::string& entry)
{
  for (const unit* item = UNITS; item->stage != nullptr; ++item) {
    if (std::strcmp(item->stage, stage) == 0 && entry == item->entry) {
      return item;
    }
  }
  return nullptr;
}

const std::string& build_json()
{
  static const std::string text = [] {
    std::string out = "{\"schema\": \"sense-core-build/v1\", \"api_version\": ";
    json::put_int(out, SENSE_CORE_API_VERSION);
    out += ", \"sources_sha256\": \"" SENSE_CORE_SOURCES_SHA256 "\", \"source_files\": ";
    json::put_int(out, SENSE_CORE_SOURCE_FILES);
    out += ", \"device\": \"" SENSE_CORE_DEVICE_KIND "\", \"cuda\": ";
    out += built_with_cuda() ? "true" : "false";
    out += ", \"cuda_architectures\": ";
    json::put_string(out, SENSE_CORE_CUDA_ARCHITECTURES);
    out += ", \"cuda_compiler\": ";
    json::put_string(out, SENSE_CORE_CUDA_COMPILER);
    out += ", \"cxx_compiler\": ";
    json::put_string(out, SENSE_CORE_CXX_COMPILER);
    out += ", \"build_type\": ";
    json::put_string(out, SENSE_CORE_BUILD_TYPE);
    out += ", \"processor\": ";
    json::put_string(out, SENSE_CORE_PROCESSOR);
    out += ", \"tensorrt\": ";
    const std::string tensorrt = tensorrt_build_version();
    if (tensorrt.empty()) {
      out += "null";
    } else {
      json::put_string(out, tensorrt);
    }
    out += ", \"libtorch\": ";
    const std::string libtorch = libtorch_build_version();
    if (libtorch.empty()) {
      out += "null";
    } else {
      json::put_string(out, libtorch);
    }
    out += ", \"engine_sources\": {\"host_reference\": \"" SENSE_UNIT_HOST_REFERENCE_SHA256 "\", \"tensorrt\": \""
           SENSE_UNIT_TENSORRT_SHA256 "\", \"libtorch\": \"" SENSE_UNIT_LIBTORCH_SHA256 "\"}";
    out += ", \"record_header_bytes\": ";
    json::put_int(out, SENSE_RECORD_HEADER_BYTES);
    out += "}";
    return out;
  }();
  return text;
}

const std::string& registry_json()
{
  static const std::string text = [] {
    std::string out = "[";
    for (const unit* item = UNITS; item->stage != nullptr; ++item) {
      if (item != UNITS) {
        out += ", ";
      }
      out += "{\"stage\": ";
      json::put_string(out, item->stage);
      out += ", \"entry\": ";
      json::put_string(out, item->entry);
      out += ", \"runtime\": \"native\", \"runs_on\": ";
      json::put_string(out, item->runs_on);
      out += ", \"precision\": ";
      json::put_string(out, item->precision);
      out += ", \"source\": ";
      json::put_string(out, item->source);
      out += ", \"unit_sha256\": ";
      json::put_string(out, item->sha256);
      out += "}";
    }
    out += "]";
    return out;
  }();
  return text;
}

const std::string& tolerance_json()
{
  static const std::string text = [] {
    std::string out = "{\"schema\": \"sense-core-tolerance/v1\", \"fp32\": {\"working_precision\": \"fp32\", "
                      "\"spacing\": ";
    json::put_double(out, tolerance::FP32_SPACING);
    out += ", \"vector_relative\": ";
    json::put_double(out, tolerance::VECTOR_RELATIVE);
    out += ", \"mean_ulps\": ";
    json::put_double(out, tolerance::MEAN_ULPS);
    out += ", \"input_ulps\": ";
    json::put_double(out, tolerance::INPUT_ULPS);
    out += ", \"input_rule\": \"input_ulps spacings of FP32 at the larger magnitude of the window mean and of the "
           "template, divided by the divisor of the z-score\", \"logit_absolute\": ";
    json::put_double(out, tolerance::LOGIT_ABSOLUTE);
    out += ", \"logit_relative\": ";
    json::put_double(out, tolerance::LOGIT_RELATIVE);
    out += ", \"probability_absolute\": ";
    json::put_double(out, tolerance::PROBABILITY_ABSOLUTE);
    out += ", \"map_absolute\": ";
    json::put_double(out, tolerance::MAP_ABSOLUTE);
    out += "}, \"fp16\": {\"payload_relative\": ";
    json::put_double(out, tolerance::FP16_RELATIVE);
    out += ", \"payload_rule\": \"one half spacing of FP16 on each part of a value the producer stored: the ingest "
           "unit widens without rounding and adds nothing\", \"logit_absolute\": ";
    json::put_double(out, tolerance::FP16_LOGIT_ABSOLUTE);
    out += ", \"logit_relative\": ";
    json::put_double(out, tolerance::FP16_LOGIT_RELATIVE);
    out += ", \"probability_absolute\": ";
    json::put_double(out, tolerance::FP16_PROBABILITY_ABSOLUTE);
    out += ", \"map_absolute\": ";
    json::put_double(out, tolerance::FP16_MAP_ABSOLUTE);
    out += "}, \"smoothed_position_m\": ";
    json::put_double(out, tolerance::SMOOTHED_POSITION_M);
    out += "}";
    return out;
  }();
  return text;
}

} // namespace sense
