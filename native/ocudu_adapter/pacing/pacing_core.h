// Paced release of receiver jobs on the staged harness (Sprint 4.1, selector mps_pacing). Host-free core.
//
// A paced run offers the receiver one job per scheduled uplink slot at an absolute release time
// on the host's monotonic clock. Two things live here, and neither knows a PUSCH processor.
//
// The pacer. The schedule is a list of release offsets from the start of the run, with a
// processing allowance and a latest start. The offered schedule is fixed before the first job
// and nothing that happens in the run moves it:
//
//     release(n) = start + offset(n)                                  never shifted
//     the worker is free at t <= release(n) + latest_start            job n runs, at release(n) or,
//                                                                     when its input was ready later,
//                                                                     as soon as it is ready: a late start
//     the worker is free at t >  release(n) + latest_start            job n is a skipped release. It keeps
//                                                                     its key and stays in every denominator
//     job n is done later than release(n) + allowance                 an overrun, counted
//
// A worker that takes 12 ms on a schedule released every 5 ms therefore skips the two releases
// it worked through and starts the next one on its own release time. A reference in another
// language repeats the rule (ocudu_cuda_isac.episodes.mps_pacing.Pacer).
//
// The paced book. The ledger of Sprint 2.4 allocates and takes a mutex for every decode, between
// a ready grid and the decoder's callback. That is no part of a receiver, and a paced run must
// not time it. The paced book holds the same job records with everything of a decode named and
// allocated before the first fill: its key, its original block, its output buffer with the guard
// and the request its decoder is handed. In the loop an admission is a check of the grid's state
// and generation, a completion is a few stores and one atomic flag, and the dispatch thread asks
// that flag for its state. It keeps the rules of the ledger in the evidence mode timed: a grid is
// admitted and held by the ticket of its fill, a decode that timed out keeps its grid held, the
// decoded bytes are compared with the original block after the callback, and the outcome follows
// the one rule of the dispatch core (isac_receiver::classify). Its job records are those of the
// ledger, field by field, and the unit target compares the two.
//
// Threads. complete() may be called from any thread, as a processor does. Every other function
// belongs to the one dispatch thread, which is also the only user of the grid pool.
//
// This file includes the dispatch core and the channel application core and nothing of OCUDU.
// ASCII only.

#pragma once

#include "dispatch_core.h"
#include "replay_core.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace isac_pacing {

using isac_replay::input_error;
using isac_replay::state_error;

/// Nanoseconds on the host's monotonic clock: the clock of the observation ring and of the sensor.
std::int64_t monotonic_ns();

/// Waits for an absolute time on that clock without entering the kernel. Returns the time it saw.
std::int64_t spin_until(std::int64_t target_ns);

// ---------------------------------------------------------------------------------------------- pacer

struct schedule {
  std::vector<std::int64_t> release_ns;          ///< Offsets from the start of the run, one per job, rising.
  std::int64_t              allowance_ns    = 0; ///< The processing allowance of one job.
  std::int64_t              latest_start_ns = 0; ///< A job that cannot start within this of its release is skipped.
};

/// Throws input_error for an empty schedule, a negative or falling offset, an allowance that is
/// not positive and a latest start that is negative or beyond the allowance.
void check(const schedule& plan);

enum class release_outcome { pending, on_time, late, skipped };

const char* to_string(release_outcome value);

/// What became of one release. Every time is absolute, on the host's monotonic clock.
struct release_mark {
  std::int64_t    release_ns = 0; ///< start + offset: the offered release time.
  std::int64_t    free_ns    = 0; ///< The worker was free to take the job.
  std::int64_t    ready_ns   = 0; ///< The input of the job was prepared. Zero for a skipped release.
  std::int64_t    start_ns   = 0; ///< The job was handed to the receiver. Zero for a skipped release.
  std::int64_t    done_ns    = 0; ///< Every decode of the job was resolved. Zero for a skipped release.
  release_outcome outcome    = release_outcome::pending;
  bool            overrun    = false; ///< Done later than release + allowance.
};

struct pace_counts {
  std::size_t  offered         = 0; ///< Releases of the schedule.
  std::size_t  taken           = 0; ///< Releases the run reached: on time, late or skipped.
  std::size_t  on_time         = 0;
  std::size_t  late            = 0; ///< Started after their release because their input was ready later.
  std::size_t  skipped         = 0;
  std::size_t  overruns        = 0;
  std::int64_t max_lateness_ns = 0; ///< Largest start minus release of a job that ran.
  std::int64_t max_excess_ns   = 0; ///< Largest done minus release minus allowance of an overrun.
};

/// The release rule. It reads no clock: every call is told the time, so a unit case drives it
/// with a clock of its own. Jobs are taken in the order of the schedule.
class pacer
{
public:
  /// Checks the schedule and allocates one mark per release. Nothing is allocated afterwards.
  explicit pacer(const schedule& plan);

  /// Fixes the offered schedule: release(n) = start_ns + offset(n). Once.
  void begin(std::int64_t start_ns);
  bool begun() const { return has_begun; }

  /// The worker is free at now_ns. True: job n runs. False: its release is skipped and counted.
  bool         take(std::size_t n, std::int64_t now_ns);
  /// The input of job n is prepared.
  void         ready(std::size_t n, std::int64_t now_ns);
  /// The offered release time of job n.
  std::int64_t release_time(std::size_t n) const;
  /// Job n is handed to the receiver at now_ns.
  void         start(std::size_t n, std::int64_t now_ns);
  /// Every decode of job n is resolved at now_ns.
  void         done(std::size_t n, std::int64_t now_ns);
  /// Job n was taken and its input could not be prepared: it becomes a skipped release.
  void         give_up(std::size_t n, std::int64_t now_ns);

  std::size_t                      size() const { return marks_.size(); }
  const std::vector<release_mark>& marks() const { return marks_; }
  const schedule&                  plan() const { return plan_; }
  pace_counts                      counts() const;

private:
  release_mark& at(std::size_t n, const char* action);

  enum class stage : std::uint8_t { offered, taken, prepared, started, finished };

  schedule                  plan_;
  std::vector<release_mark> marks_;
  std::vector<stage>        stages_;        ///< How far each release got. A time of zero is a time.
  std::size_t               next_     = 0;  ///< The job take() expects.
  bool                      has_begun = false;
};

// ---------------------------------------------------------------------------------------------- paced book

/// Why a prepared decode was never started.
enum class skip_reason { none, missed_release, no_free_grid, backend_stopped };

const char* to_string(skip_reason value);

/// Every decode of a paced run: named before the first fill, allocation-free in the loop.
class paced_book final : public isac_receiver::completion_sink
{
public:
  using clock_fn = std::int64_t (*)();

  /// The pool is not owned and must outlive the book. clock is the monotonic clock, or one of a unit case.
  explicit paced_book(isac_replay::o1_pool& pool_, clock_fn clock_ = &monotonic_ns);
  ~paced_book() override;
  paced_book(const paced_book&)            = delete;
  paced_book& operator=(const paced_book&) = delete;

  /// Before the first fill: names one decode and allocates everything of it. Tokens count from one
  /// in the order of the calls. A key that was named before and an empty block are refused.
  std::uint64_t prepare(const isac_receiver::job_key& key,
                        const std::string&            grant_id,
                        const std::string&            grant_record_sha256,
                        const isac_receiver::payload& original);

  /// Admits a prepared decode on a ready grid and holds the grid for it. A decode that was
  /// admitted or skipped before, a grid that is not ready, a stale generation and another grant
  /// record are refused and recorded. The admitted path allocates nothing and takes no lock.
  isac_receiver::admission admit(std::uint64_t token, const isac_replay::ticket& grid, std::string* reason = nullptr);

  /// What the decoder of an admitted decode is handed. The reference stays valid until the book ends.
  const isac_receiver::request& describe(std::uint64_t token) const;
  /// Marks the start of the decode: the service time is counted from here.
  void started(std::uint64_t token);

  // See completion_sink. The first callback of a decode stands. A callback for an unknown decode,
  // a second callback and a callback after a timeout are counted and reported as anomalies.
  void complete(std::uint64_t token, bool crc_ok, const std::string& error) override;

  /// Asks for the completion of a decode until it arrived or limit_ns passed. It never sleeps.
  bool wait(std::uint64_t token, std::int64_t limit_ns);
  /// Gives the decode its outcome. A decode without a completion becomes a timeout and keeps its grid held.
  isac_receiver::outcome resolve(std::uint64_t token);
  /// A prepared decode that is never started: a skipped release, no free grid, a stopped backend.
  void skip(std::uint64_t token, skip_reason why, std::int64_t late_ns = 0);

  /// Releases a grid to the pool when no admitted decode still holds it.
  bool        release_grid(const isac_replay::ticket& grid);
  std::size_t holds(const isac_replay::ticket& grid) const;
  std::size_t held_grids() const;
  std::size_t size() const { return entries.size(); }

  // ---- After the run. These allocate. ----
  isac_receiver::job                  snapshot(std::uint64_t token) const;
  std::vector<isac_receiver::job>     jobs() const;
  std::vector<isac_receiver::anomaly> anomalies() const;
  std::vector<std::uint8_t>           decoded(std::uint64_t token) const;
  std::size_t count(isac_receiver::outcome value, const std::string& backend = std::string()) const;
  /// The start of a decode and its first completion callback, absolute. Zero when there was none.
  std::int64_t started_ns(std::uint64_t token) const;
  std::int64_t completed_ns(std::uint64_t token) const;
  skip_reason  skipped(std::uint64_t token) const;

private:
  struct entry;
  entry&       find(std::uint64_t token, const char* action);
  const entry& find(std::uint64_t token, const char* action) const;
  void         refuse(const entry& item, const char* kind, const std::string& detail);

  isac_replay::o1_pool&                        pool;
  clock_fn                                     clock;
  std::vector<std::unique_ptr<entry>>          entries; ///< Index token - 1.
  std::vector<std::vector<std::uint64_t>>      holders; ///< Per grid of the pool: the tokens that hold it.
  std::vector<isac_receiver::anomaly>          noted;   ///< Refusals. Dispatch thread only.
  std::atomic<std::uint64_t>                   unknown_completions{0};
};

/// The result of one paced dispatch.
struct paced_result {
  isac_receiver::admission admitted = isac_receiver::admission::admitted;
  isac_receiver::outcome   result   = isac_receiver::outcome::pending;
};

/// Hands one ready grid to one decoder for a prepared decode and waits for its outcome: admit,
/// start, ask for the completion, resolve. An exception of the decoder's start becomes a
/// decoder_error. A decoder that never calls back becomes a timeout after limit_ns.
paced_result paced_dispatch(paced_book&                book,
                            isac_receiver::decoder&    processor,
                            std::uint64_t              token,
                            const isac_replay::ticket& grid,
                            std::int64_t               limit_ns,
                            std::string*               reason = nullptr);

} // namespace isac_pacing
