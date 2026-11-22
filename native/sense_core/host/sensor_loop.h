// The loop of the sensor process: from the records of a carrier to decisions (Sprint 3.6, selector
// sense_core).
//
// sense_host runs this loop on the observation ring. The unit target runs it on doubles, and an
// offline replay runs it on the records of a consumer archive. The loop keeps the rules of the
// window submission, decided on 2026-11-21:
//
//   It holds the records of an open window in place and makes no device call for them.
//   When the last record of a window arrives it submits ingest, kernels and engine once, waits
//   for the one completion event by asking for its state, and then releases the window in order.
//   It keeps no queue. Before it submits it reads every record that is already published: a
//   complete window that a newer complete window overtook is skipped, its records are released
//   in order and counted as stale_skipped. It never works through a backlog.
//   Its release is immediate, or after_radio: the submission waits until the calibrated
//   remaining service time of the last uplink slot of the window has passed since t_ready of its
//   last record.
//   A window that is still filling when its carrier has been silent for longer than a window can
//   stay open is closed as a gap, and its records are released: a producer that drains at the
//   end of a run gets every credit back.
//
// Nothing is allocated, hashed or written to a file of the run while records arrive. The loop
// keeps every answer and every decision as plain data in memory that was allocated before it
// attached, with the input and the map of each scored window. seal() takes the digests and
// formats the lines after the last record. The one exception is the live feed of a status page:
// where a descriptor is given, the line of a decision is formatted and written to it after the
// publication time of that decision was taken.
//
// ASCII only.

#pragma once

#include "sense_core/sense_core.h"
#include <cstdint>
#include <string>
#include <vector>

namespace sense {

/// Where records come from: the ring, an archive, or a double.
class record_source
{
public:
  enum class result { record, none, stale, closed };
  virtual ~record_source() = default;
  /// The next record in the order of its carrier, in place. Waits at most limit_ns.
  virtual result next(sense_view& view, std::int64_t limit_ns) = 0;
  /// Records that are published and not read yet.
  virtual std::uint64_t pending() = 0;
  /// The work on every record below sequence is complete.
  virtual void release(std::uint64_t sequence) = 0;
  /// The sequence of the record that next() delivers next.
  virtual std::uint64_t position() = 0;
  /// True for a carrier that keeps publishing: the loop skips a window that was overtaken.
  /// False for an archive: every window runs.
  virtual bool live() const = 0;
};

struct loop_options {
  enum class release_policy { immediate, after_radio };
  release_policy release            = release_policy::immediate;
  std::int64_t   radio_allowance_ns = 0;
  std::int64_t   poll_ns            = 20000000;    ///< How long one next() may wait.
  std::int64_t   idle_ns            = 30000000000; ///< Give up when no record arrived for this long.
  /// Close a window that is filling as a gap when no record arrived for this long, so that its
  /// credits return to the producer. Two periods and more: its last slot cannot arrive any more.
  /// 0: never.
  std::int64_t   open_ns            = 20000000;
  std::int64_t   hold_ns            = 0;           ///< Test switch: work this long on each window.
  std::size_t    lines              = 16384;       ///< Answers and decisions to keep.
  std::size_t    windows            = 4096;        ///< Scored windows whose tensors are kept.
  bool           keep_mean          = false;       ///< Keep the window mean and its count as well.
  int            live_fd            = -1;          ///< Decisions are appended here, after their publication time.
  sense_clock    clock              = nullptr;
  void*          clock_owner        = nullptr;
};

/// One line of the stream: an answer without a score, or the decision of a complete window.
struct kept_line {
  bool           is_decision = false;
  sense_answer   answer;
  sense_decision decision;
  sense_smoothed smoothed;
  std::int64_t   tensors = -1; ///< Index of the kept tensors of a scored window, or -1.
};

struct loop_counters {
  std::uint64_t read = 0, stale_reads = 0, lines = 0, lines_not_kept = 0, windows_not_kept = 0, submissions = 0;
  std::uint64_t waited_radio = 0; ///< Submissions that waited for the allowance of after_radio.
  std::uint64_t closed_open  = 0; ///< Windows that were filling when the carrier went silent.
  std::int64_t  first_record_ns = 0, last_record_ns = 0;
  bool          closed = false, idle = false;
};

class sensor_loop
{
public:
  /// Allocates everything the loop will keep. The plan is warm and the source not attached yet.
  sensor_loop(sense_plan* plan, const loop_options& options);

  /// Runs until the source says that no record follows, or was idle for too long. Returns
  /// SENSE_OK, or the status of a refusal of the plan, with the text in sense_last_error.
  sense_status run(record_source& source);

  /// After the last record: the digests of the kept tensors and the lines of both files.
  /// timed leaves the digests of the window mean and of its count out.
  void seal(std::vector<std::string>& window_lines, std::vector<std::string>& decision_lines) const;

  const std::vector<kept_line>& lines() const { return kept; }
  const loop_counters&          counters() const { return counts; }
  std::size_t                   scored() const { return kept_windows; }
  /// The kept tensors of scored window n: the input [A, K], the map [rows, cols], and, with
  /// keep_mean, the mean [A, K, S] as pairs and its count.
  const float*        input_of(std::size_t n) const { return inputs.data() + n * input_values; }
  const float*        map_of(std::size_t n) const { return maps.data() + n * map_values; }
  const float*        mean_of(std::size_t n) const { return means.data() + n * mean_values; }
  const std::uint8_t* count_of(std::size_t n) const { return window_counts.data() + n * count_values; }
  std::size_t         input_size() const { return input_values; }
  std::size_t         map_size() const { return map_values; }
  std::size_t         mean_size() const { return mean_values; }
  std::size_t         count_size() const { return count_values; }

private:
  std::int64_t now() const;
  void         keep(const sense_answer& answer);
  void         keep(const sense_decision& decision, const sense_smoothed& smoothed);
  void         release(record_source& source);
  sense_status decide(record_source& source);

  sense_plan*                plan;
  loop_options               options;
  sense_plan_info            info;
  std::vector<kept_line>     kept;
  std::size_t                kept_lines = 0, kept_windows = 0;
  std::vector<float>         inputs, maps, means;
  std::vector<std::uint8_t>  window_counts;
  std::size_t                input_values = 0, map_values = 0, mean_values = 0, count_values = 0;
  loop_counters              counts;
};

} // namespace sense
