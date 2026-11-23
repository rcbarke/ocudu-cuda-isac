// Paced release of receiver jobs on the staged harness (Sprint 4.1, selector mps_pacing). ASCII only.

#include "pacing_core.h"
#include <algorithm>
#include <time.h>

namespace isac_pacing {

using isac_receiver::admission;
using isac_receiver::anomaly;
using isac_receiver::comparison;
using isac_receiver::job;
using isac_receiver::job_key;
using isac_receiver::outcome;
using isac_receiver::payload;
using isac_receiver::request;
using isac_replay::o1_pool;
using isac_replay::ticket;

namespace {

inline void relax()
{
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

/// A lock of a few stores between the completing thread and the dispatch thread. It never
/// enters the kernel.
struct spin_guard {
  explicit spin_guard(std::atomic_flag& flag_) : flag(flag_)
  {
    while (flag.test_and_set(std::memory_order_acquire)) {
      relax();
    }
  }
  ~spin_guard() { flag.clear(std::memory_order_release); }
  spin_guard(const spin_guard&)            = delete;
  spin_guard& operator=(const spin_guard&) = delete;

  std::atomic_flag& flag;
};

} // namespace

std::int64_t monotonic_ns()
{
  timespec stamp;
  ::clock_gettime(CLOCK_MONOTONIC, &stamp);
  return static_cast<std::int64_t>(stamp.tv_sec) * 1000000000LL + stamp.tv_nsec;
}

std::int64_t spin_until(std::int64_t target_ns)
{
  std::int64_t now = monotonic_ns();
  while (now < target_ns) {
    relax();
    now = monotonic_ns();
  }
  return now;
}

// ---------------------------------------------------------------------------------------------- pacer

void check(const schedule& plan)
{
  if (plan.release_ns.empty()) {
    throw input_error("a paced run names at least one release");
  }
  if (plan.allowance_ns <= 0) {
    throw input_error("the processing allowance of a paced run is positive");
  }
  if (plan.latest_start_ns < 0 || plan.latest_start_ns > plan.allowance_ns) {
    throw input_error("the latest start of a paced run lies between zero and the processing allowance");
  }
  std::int64_t before = -1;
  for (std::size_t n = 0; n != plan.release_ns.size(); ++n) {
    if (plan.release_ns[n] < 0 || plan.release_ns[n] <= before) {
      throw input_error("release " + std::to_string(n) + " of a paced run does not lie after the release before it: "
                        "the offsets are not negative and rise");
    }
    before = plan.release_ns[n];
  }
}

void check(const schedule& plan, const lead_in& warm, std::int64_t start_delay_ns)
{
  check(plan);
  if (warm.releases == 0) {
    return;
  }
  if (warm.releases > plan.release_ns.size()) {
    throw input_error("the lead-in names " + std::to_string(warm.releases) + " releases, and the schedule holds " +
                      std::to_string(plan.release_ns.size()) + ": a lead-in repeats the first releases of its schedule");
  }
  if (warm.lead_ns <= 0) {
    throw input_error("the lead of a lead-in is positive: its decodes are issued ahead of the schedule");
  }
  // The last decode of the lead-in, done on its allowance, has ended when the first release is offered.
  if (plan.release_ns[warm.releases - 1] - warm.lead_ns + plan.allowance_ns > plan.release_ns.front()) {
    throw input_error("the lead-in does not end before the schedule: its last decode is issued " +
                      std::to_string(warm.lead_ns - plan.release_ns[warm.releases - 1] + plan.release_ns.front()) +
                      " ns ahead of the first release, and the processing allowance is " +
                      std::to_string(plan.allowance_ns) + " ns");
  }
  // The first decode of the lead-in is not issued before the schedule was fixed.
  if (warm.lead_ns - plan.release_ns.front() > start_delay_ns) {
    throw input_error("the lead-in begins before the schedule is fixed: its lead is " + std::to_string(warm.lead_ns) +
                      " ns, and the schedule starts " + std::to_string(start_delay_ns) + " ns after it was fixed");
  }
}

std::int64_t lead_time(const schedule& plan, const lead_in& warm, std::int64_t start_ns, std::size_t k)
{
  if (k >= warm.releases || k >= plan.release_ns.size()) {
    throw state_error("the lead-in holds no decode " + std::to_string(k));
  }
  return start_ns + plan.release_ns[k] - warm.lead_ns;
}

const char* to_string(release_outcome value)
{
  switch (value) {
    case release_outcome::pending:
      return "pending";
    case release_outcome::on_time:
      return "on_time";
    case release_outcome::late:
      return "late";
    case release_outcome::skipped:
      return "skipped";
  }
  return "pending";
}

pacer::pacer(const schedule& plan) : plan_(plan)
{
  check(plan_);
  marks_.assign(plan_.release_ns.size(), release_mark());
  stages_.assign(plan_.release_ns.size(), stage::offered);
}

void pacer::begin(std::int64_t start_ns)
{
  if (has_begun) {
    throw state_error("begin: the offered schedule of a paced run is fixed once");
  }
  for (std::size_t n = 0; n != marks_.size(); ++n) {
    marks_[n].release_ns = start_ns + plan_.release_ns[n];
  }
  has_begun = true;
}

release_mark& pacer::at(std::size_t n, const char* action)
{
  if (!has_begun) {
    throw state_error(std::string(action) + ": the paced run has not begun");
  }
  if (n >= marks_.size()) {
    throw state_error(std::string(action) + ": the schedule holds no release " + std::to_string(n));
  }
  return marks_[n];
}

bool pacer::take(std::size_t n, std::int64_t now_ns)
{
  release_mark& mark = at(n, "take");
  if (n != next_) {
    throw state_error("take: release " + std::to_string(n) + " is taken out of order, release " + std::to_string(next_) +
                      " is next");
  }
  ++next_;
  mark.free_ns = now_ns;
  if (now_ns > mark.release_ns + plan_.latest_start_ns) {
    mark.outcome = release_outcome::skipped;
    stages_[n]   = stage::finished;
    return false;
  }
  stages_[n] = stage::taken;
  return true;
}

void pacer::ready(std::size_t n, std::int64_t now_ns)
{
  release_mark& mark = at(n, "ready");
  if (stages_[n] != stage::taken) {
    throw state_error("ready: release " + std::to_string(n) + " was not taken, or was prepared before");
  }
  mark.ready_ns = now_ns;
  stages_[n]    = stage::prepared;
}

std::int64_t pacer::release_time(std::size_t n) const
{
  if (!has_begun || n >= marks_.size()) {
    throw state_error("release_time: the paced run has not begun, or holds no release " + std::to_string(n));
  }
  return marks_[n].release_ns;
}

void pacer::start(std::size_t n, std::int64_t now_ns)
{
  release_mark& mark = at(n, "start");
  if (stages_[n] != stage::prepared) {
    throw state_error("start: release " + std::to_string(n) + " was not prepared, or was started before");
  }
  mark.start_ns = now_ns;
  stages_[n]    = stage::started;
  // A job starts on its release time unless its input was ready later than that.
  mark.outcome = mark.ready_ns > mark.release_ns ? release_outcome::late : release_outcome::on_time;
}

void pacer::done(std::size_t n, std::int64_t now_ns)
{
  release_mark& mark = at(n, "done");
  if (stages_[n] != stage::started) {
    throw state_error("done: release " + std::to_string(n) + " was not started, or was done before");
  }
  mark.done_ns = now_ns;
  stages_[n]   = stage::finished;
  mark.overrun = now_ns > mark.release_ns + plan_.allowance_ns;
}

void pacer::give_up(std::size_t n, std::int64_t now_ns)
{
  release_mark& mark = at(n, "give_up");
  if (stages_[n] != stage::taken) {
    throw state_error("give_up: release " + std::to_string(n) + " was not taken, or was prepared before");
  }
  mark.free_ns = now_ns;
  mark.outcome = release_outcome::skipped;
  stages_[n]   = stage::finished;
}

pace_counts pacer::counts() const
{
  pace_counts total;
  total.offered = marks_.size();
  for (const release_mark& mark : marks_) {
    switch (mark.outcome) {
      case release_outcome::pending:
        continue;
      case release_outcome::skipped:
        ++total.skipped;
        break;
      case release_outcome::late:
        ++total.late;
        break;
      case release_outcome::on_time:
        ++total.on_time;
        break;
    }
    ++total.taken;
    if (mark.outcome != release_outcome::skipped) {
      total.max_lateness_ns = std::max(total.max_lateness_ns, mark.start_ns - mark.release_ns);
    }
    if (mark.overrun) {
      ++total.overruns;
      total.max_excess_ns = std::max(total.max_excess_ns, mark.done_ns - mark.release_ns - plan_.allowance_ns);
    }
  }
  return total;
}

// ---------------------------------------------------------------------------------------------- paced book

const char* to_string(skip_reason value)
{
  switch (value) {
    case skip_reason::none:
      return "none";
    case skip_reason::missed_release:
      return "missed_release";
    case skip_reason::no_free_grid:
      return "no_free_grid";
    case skip_reason::backend_stopped:
      return "backend_stopped";
  }
  return "none";
}

struct paced_book::entry {
  job                       record;   ///< Everything that is known before the run, and the outcome.
  request                   handed;   ///< What the decoder is handed. Its grid is set at the admission.
  std::vector<std::uint8_t> buffer;   ///< The decoded bytes, then the guard.
  std::vector<std::uint8_t> original; ///< A copy: the comparison needs no object of the caller.
  std::int64_t              started_ns = 0;
  skip_reason               skipped    = skip_reason::none;
  std::int64_t              late_ns    = 0;
  bool                      resolved   = false;
  // Shared with the completing thread. The fields below the lock are written under it, once.
  std::atomic<bool>     admitted{false};
  std::atomic<bool>     completed{false};
  std::atomic<unsigned> completions{0};
  std::atomic_flag      busy = ATOMIC_FLAG_INIT;
  bool                  timed_out    = false; ///< Resolved without a completion. Under the lock.
  bool                  late         = false; ///< The completion arrived after that. Under the lock.
  bool                  stray        = false; ///< A completion named this decode although it was never started.
  bool                  crc_ok       = false;
  std::int64_t          completed_ns = 0;
  std::string           error;
};

paced_book::paced_book(o1_pool& pool_, clock_fn clock_, std::uint64_t first_token) :
  pool(pool_), clock(clock_), first(first_token), holders(pool_.size())
{
  if (first == 0) {
    throw input_error("the tokens of a paced book count from one or from a later token: zero names no decode");
  }
  for (std::vector<std::uint64_t>& held : holders) {
    held.reserve(16);
  }
}

paced_book::~paced_book() = default;

paced_book::entry& paced_book::find(std::uint64_t token, const char* action)
{
  if (!known(token)) {
    throw state_error(std::string(action) + ": the paced book holds no job " + std::to_string(token));
  }
  return *entries[token - first];
}

const paced_book::entry& paced_book::find(std::uint64_t token, const char* action) const
{
  return const_cast<paced_book*>(this)->find(token, action);
}

void paced_book::refuse(const entry& item, const char* kind, const std::string& detail)
{
  anomaly found;
  found.kind   = kind;
  found.key    = item.record.key.text();
  found.detail = detail;
  noted.push_back(found);
}

std::uint64_t paced_book::prepare(const job_key&     key,
                                  const std::string& grant_id,
                                  const std::string& grant_record_sha256,
                                  const payload&     original)
{
  if (key.opportunity_id.empty() || key.backend.empty()) {
    throw input_error("a decode needs the identifier of its opportunity and the name of its backend");
  }
  if (original.bytes.empty()) {
    throw input_error("the decode " + key.text() + " names no original transport block to compare with");
  }
  const std::string name = key.text();
  for (const std::unique_ptr<entry>& known : entries) {
    if (known->record.key.opportunity_id == key.opportunity_id && known->record.key.backend == key.backend) {
      throw input_error("duplicate delivery: the decode " + name + " was named before");
    }
  }
  std::unique_ptr<entry> item(new entry);
  item->record.token               = first + entries.size();
  item->record.sequence            = entries.size();
  item->record.key                 = key;
  item->record.grant_id            = grant_id;
  item->record.grant_record_sha256 = grant_record_sha256;
  item->record.payload_id          = original.id;
  item->record.payload_sha256      = original.sha256;
  item->record.payload_bytes       = original.bytes.size();
  item->original                   = original.bytes;
  item->buffer.assign(original.bytes.size() + isac_receiver::GUARD_BYTES, isac_receiver::FILL_VALUE);
  std::fill(item->buffer.begin() + static_cast<std::ptrdiff_t>(original.bytes.size()), item->buffer.end(),
            isac_receiver::GUARD_VALUE);
  item->handed.token               = item->record.token;
  item->handed.key                 = key;
  item->handed.grant_id            = grant_id;
  item->handed.grant_record_sha256 = grant_record_sha256;
  item->handed.output              = item->buffer.data();
  item->handed.output_bytes        = original.bytes.size();
  const std::uint64_t token = item->record.token;
  entries.push_back(std::move(item));
  return token;
}

admission paced_book::admit(std::uint64_t token, const ticket& grid, std::string* reason)
{
  entry& item = find(token, "admit");
  auto   fail = [&](admission why, const char* kind, const std::string& detail) {
    refuse(item, kind, detail);
    if (reason != nullptr) {
      *reason = detail;
    }
    return why;
  };
  if (item.admitted.load(std::memory_order_relaxed) || item.resolved) {
    return fail(admission::duplicate_delivery, "duplicate_delivery",
                "duplicate delivery: the decode " + item.record.key.text() + " was admitted or skipped before");
  }
  // The grid must be ready, of this generation and filled for this grant record. The view proves it.
  try {
    (void)pool.view(grid, item.record.grant_record_sha256);
  } catch (const state_error& error) {
    const std::string text       = error.what();
    const bool        allocation = text.find("changed allocation") != std::string::npos;
    return fail(allocation ? admission::changed_allocation : admission::stale_grid, "refused_grid", text);
  }
  if (grid.grid >= holders.size()) {
    return fail(admission::stale_grid, "refused_grid", "the grid " + std::to_string(grid.grid) + " is not one of the pool");
  }
  item.record.grid = grid;
  item.handed.grid = grid;
  holders[grid.grid].push_back(token);
  item.admitted.store(true, std::memory_order_release);
  return admission::admitted;
}

const request& paced_book::describe(std::uint64_t token) const
{
  const entry& item = find(token, "describe");
  if (!item.admitted.load(std::memory_order_relaxed)) {
    throw state_error("describe: the job " + item.record.key.text() + " was never dispatched");
  }
  return item.handed;
}

void paced_book::started(std::uint64_t token)
{
  entry& item         = find(token, "start");
  item.record.started = true;
  item.started_ns     = clock();
}

void paced_book::complete(std::uint64_t token, bool crc_ok, const std::string& error)
{
  const std::int64_t now = clock();
  if (!known(token)) {
    unknown_completions.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  entry& item = *entries[token - first];
  if (item.completions.fetch_add(1, std::memory_order_relaxed) != 0) {
    return; // a second completion: counted. The first one stands.
  }
  spin_guard lock(item.busy);
  if (!item.admitted.load(std::memory_order_acquire)) {
    item.stray = true;
    return;
  }
  if (item.timed_out) {
    // The job timed out before. Its outcome stays, and its grid is no longer read.
    item.late = true;
  } else {
    item.crc_ok       = crc_ok;
    item.completed_ns = now;
    if (!error.empty()) {
      item.error = error;
    }
  }
  item.completed.store(true, std::memory_order_release);
}

bool paced_book::wait(std::uint64_t token, std::int64_t limit_ns)
{
  const entry&       item = find(token, "wait");
  const std::int64_t end  = clock() + limit_ns;
  while (!item.completed.load(std::memory_order_acquire)) {
    if (clock() > end) {
      return item.completed.load(std::memory_order_acquire);
    }
    relax();
  }
  return true;
}

outcome paced_book::resolve(std::uint64_t token)
{
  entry& item = find(token, "resolve");
  if (item.resolved) {
    return item.record.result;
  }
  if (!item.admitted.load(std::memory_order_relaxed)) {
    throw state_error("resolve: the job " + item.record.key.text() + " was never dispatched");
  }
  {
    spin_guard lock(item.busy);
    if (!item.completed.load(std::memory_order_acquire)) {
      // The processor may still read the grid and write the buffer: neither is touched here.
      item.timed_out     = true;
      item.resolved      = true;
      item.record.result = isac_receiver::classify(false, false, false, false, false, false);
      return item.record.result;
    }
  }
  // The callback arrived, so the decoder is done with the grid and with the buffer.
  bool intact = false;
  try {
    (void)pool.view(item.record.grid, item.record.grant_record_sha256);
    intact = true;
  } catch (const state_error&) {
    intact = false;
  }
  const std::size_t bytes = item.record.payload_bytes;
  const bool        guard = std::all_of(item.buffer.begin() + static_cast<std::ptrdiff_t>(bytes), item.buffer.end(),
                                        [](std::uint8_t value) { return value == isac_receiver::GUARD_VALUE; });
  const comparison  compared =
      isac_receiver::compare_bytes(item.buffer.data(), bytes, item.original.data(), item.original.size());
  item.record.completed    = true;
  item.record.crc_ok       = item.crc_ok;
  item.record.input_intact = intact;
  item.record.guard_intact = guard;
  if (item.crc_ok) {
    item.record.compared = compared;
  }
  if (item.error.empty() && !guard) {
    item.error = "the decoder wrote past the transport block";
  } else if (item.error.empty() && !intact) {
    item.error = "the grid changed while it was decoded";
  }
  if (item.record.started) {
    item.record.service_us = static_cast<double>(item.completed_ns - item.started_ns) / 1000.0;
  }
  item.record.result = isac_receiver::classify(true, !item.error.empty(), item.crc_ok, compared.equal, guard, intact);
  item.resolved      = true;
  return item.record.result;
}

void paced_book::skip(std::uint64_t token, skip_reason why, std::int64_t late_ns)
{
  entry& item = find(token, "skip");
  if (item.admitted.load(std::memory_order_relaxed) || item.resolved) {
    refuse(item, "duplicate_delivery",
           "duplicate delivery: the decode " + item.record.key.text() + " was admitted or skipped before");
    return;
  }
  item.skipped       = why == skip_reason::none ? skip_reason::missed_release : why;
  item.late_ns       = late_ns;
  item.record.result = outcome::not_dispatched;
  item.resolved      = true;
}

bool paced_book::release_grid(const ticket& grid)
{
  if (holds(grid) != 0) {
    return false;
  }
  if (grid.grid < holders.size()) {
    // Every decode on this grid has called back: none of them is listed any longer.
    std::vector<std::uint64_t>& held = holders[grid.grid];
    held.erase(std::remove_if(held.begin(), held.end(),
                              [&](std::uint64_t token) {
                                return entries[token - first]->record.grid.generation == grid.generation;
                              }),
               held.end());
  }
  pool.release(grid);
  return true;
}

std::size_t paced_book::holds(const ticket& grid) const
{
  if (grid.grid >= holders.size()) {
    return 0;
  }
  std::size_t total = 0;
  for (const std::uint64_t token : holders[grid.grid]) {
    const entry& item = *entries[token - first];
    if (item.record.grid.generation == grid.generation && !item.completed.load(std::memory_order_acquire)) {
      ++total;
    }
  }
  return total;
}

std::size_t paced_book::held_grids() const
{
  std::size_t total = 0;
  for (std::size_t index = 0; index != holders.size(); ++index) {
    bool held = false;
    for (const std::uint64_t token : holders[index]) {
      held = held || !entries[token - first]->completed.load(std::memory_order_acquire);
    }
    total += held ? 1 : 0;
  }
  return total;
}

job paced_book::snapshot(std::uint64_t token) const
{
  const entry& item   = find(token, "read");
  job          record = item.record;
  record.completions  = item.completions.load(std::memory_order_relaxed);
  record.error        = item.error;
  if (!item.resolved) {
    // Named before the run and never reached.
    record.result = outcome::not_dispatched;
    record.error  = "the run stopped before this decode was started";
  } else if (record.result == outcome::not_dispatched) {
    switch (item.skipped) {
      case skip_reason::missed_release:
        record.error = "skipped release: the receiver was free " + std::to_string(item.late_ns / 1000) +
                       " us after the release of this job, beyond its latest start";
        break;
      case skip_reason::no_free_grid:
        record.error = "no free grid: every owned grid is held by a decode without a completion";
        break;
      case skip_reason::backend_stopped:
        record.error = "the backend was stopped: an earlier decode on it did not complete within the declared time";
        break;
      case skip_reason::none:
        record.error = "the run stopped before this decode was started";
        break;
    }
  } else if (record.result == outcome::timeout) {
    record.error = "no completion within the declared time";
  }
  return record;
}

std::vector<job> paced_book::jobs() const
{
  std::vector<job> result;
  result.reserve(entries.size());
  for (std::size_t index = 0; index != entries.size(); ++index) {
    result.push_back(snapshot(first + index));
  }
  return result;
}

std::vector<anomaly> paced_book::anomalies() const
{
  std::vector<anomaly> result = noted;
  auto                 add    = [&result](const char* kind, const std::string& key, const std::string& detail) {
    anomaly found;
    found.kind   = kind;
    found.key    = key;
    found.detail = detail;
    result.push_back(found);
  };
  for (const std::unique_ptr<entry>& item : entries) {
    const unsigned seen = item->completions.load(std::memory_order_relaxed);
    if (item->stray) {
      add("unknown_completion", item->record.key.text(), "a completion named a decode that was never started");
    }
    if (item->late) {
      add("late_completion", item->record.key.text(), "the completion arrived after the decode had timed out");
    }
    if (seen > 1) {
      add("duplicate_completion", item->record.key.text(),
          std::to_string(seen - 1) + " further completion(s) arrived. The first one stands.");
    }
  }
  const std::uint64_t unknown = unknown_completions.load(std::memory_order_relaxed);
  if (unknown != 0) {
    add("unknown_completion", std::string(), std::to_string(unknown) + " completion(s) named a job the book does not hold");
  }
  return result;
}

std::vector<std::uint8_t> paced_book::decoded(std::uint64_t token) const
{
  const entry& item = find(token, "read");
  if (!item.record.completed || !item.resolved || item.record.result == outcome::timeout) {
    throw state_error("read: the decode " + item.record.key.text() + " has no completed result");
  }
  return std::vector<std::uint8_t>(item.buffer.begin(),
                                   item.buffer.begin() + static_cast<std::ptrdiff_t>(item.record.payload_bytes));
}

std::size_t paced_book::count(outcome value, const std::string& backend) const
{
  std::size_t total = 0;
  for (const std::unique_ptr<entry>& item : entries) {
    const outcome result = item->resolved ? item->record.result : outcome::not_dispatched;
    if (result == value && (backend.empty() || item->record.key.backend == backend)) {
      ++total;
    }
  }
  return total;
}

std::int64_t paced_book::started_ns(std::uint64_t token) const
{
  return find(token, "read").started_ns;
}

std::int64_t paced_book::completed_ns(std::uint64_t token) const
{
  return find(token, "read").completed_ns;
}

skip_reason paced_book::skipped(std::uint64_t token) const
{
  return find(token, "read").skipped;
}

// ---------------------------------------------------------------------------------------------- dispatch

paced_result paced_dispatch(paced_book&             book,
                            isac_receiver::decoder& processor,
                            std::uint64_t           token,
                            const ticket&           grid,
                            std::int64_t            limit_ns,
                            std::string*            reason)
{
  paced_result result;
  result.admitted = book.admit(token, grid, reason);
  if (result.admitted != admission::admitted) {
    return result;
  }
  const request& handed = book.describe(token);
  book.started(token);
  try {
    processor.start(handed, book);
  } catch (const std::exception& error) {
    book.complete(token, false, std::string("the decoder raised: ") + error.what());
  }
  book.wait(token, limit_ns);
  result.result = book.resolve(token);
  return result;
}

} // namespace isac_pacing
