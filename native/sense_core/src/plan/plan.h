// The plan of the cuSense-like branch: a configured chain of registry entries with its
// workspaces (Sprint 3.6, selector sense_core).
//
// A plan is built from one frozen artifact on one device. It tracks the causal window of the
// reference stream, holds the views of an open window in place, and runs ingest, kernels and
// engine once when the window is complete. This header is the C++ side of the C API in
// include/sense_core/sense_core.h, shared by plan.cpp, which tracks and submits, and lines.cpp,
// which writes the JSON lines of the reference schemas.
//
// Rules a plan keeps:
//   No device call while a window is open. offer() reads the header of a view and nothing else.
//   One submission per window: one upload, two kernels, the engine, one download, one event.
//   Nothing is allocated after the plan is created. Texts of an answer live in fixed buffers.
//   A view is released when the plan is done with its memory, in the order it was offered.
//   A window that was overtaken by a newer complete one is skipped and counted, never queued.
//
// ASCII only.

#pragma once

#include "core/artifact.h"
#include "core/record.h"
#include "engine/engine.h"
#include "kernels/device.h"
#include "plan/smoother.h"
#include "sense_core/sense_core.h"
#include <memory>
#include <string>

struct held_view {
  sense_view         view;
  obs::record_header header;
  std::uint32_t      offset = 0; ///< Of the allocation on the support of the window.
  bool               used   = false;
};

struct window_slots {
  bool          active = false;
  std::uint64_t epoch = 0, episode = 0, period = 0;
  std::uint32_t have = 0;
  held_view     slot[SENSE_MAX_WINDOW_RECORDS];
};

struct sense_plan {
  sense::artifact                 model;
  std::unique_ptr<sense::device>  dev;
  std::unique_ptr<sense::engine>  eng;
  std::unique_ptr<sense::smoother> smooth;
  sense::device_layout            layout;
  sense_clock                     clock = nullptr;
  void*                           clock_owner = nullptr;
  bool                            keep_mean = false;
  std::int64_t                    budget_ingest = 0, budget_kernels = 0, budget_engine = 0, budget_window = 0;
  int                             mu = 0;
  std::uint32_t                   need = 0;
  std::uint8_t                    expected_stage = 0, expected_layout = 0, expected_dtype = 0;
  std::uint32_t                   expected_steps = 0; ///< The producer steps of the artifact, as bits.

  // The causal window.
  bool          has_key = false;
  std::uint64_t key_epoch = 0, key_episode = 0;
  std::int64_t  last_slot = -1;
  window_slots  filling, ready, running;
  bool          submitted = false, stale = false, engine_ran = false, complete = false;
  std::int64_t  t_submit = 0, t_engine_host = 0;
  sense::window_geometry geometry;
  const void*   payloads[SENSE_MAX_WINDOW_RECORDS];
  std::uint32_t port = 0, producer_steps = 0, evidence = 0; ///< Of the running window.
  double        scale = 0.0;

  sense_counters counters;

  // Texts a caller may keep for the life of the plan.
  std::string device_name, device_description, engine_detail, engine_sha256, plan_json, plan_sha256, entries_json;
};

namespace sense {

std::int64_t now(const sense_plan& plan);
/// Stores the text of a refusal for sense_last_error and returns the status.
sense_status refuse(sense_status status, const std::string& text);

} // namespace sense
