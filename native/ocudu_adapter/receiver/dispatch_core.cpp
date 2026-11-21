// Dispatch of one receiver grid to the PUSCH processors (Sprint 2.4, selector replay_pusch). Host-free core.
// See dispatch_core.h for the rules. ASCII only.

#include "dispatch_core.h"
#include <algorithm>
#include <stdexcept>

namespace isac_receiver {

using isac_replay::o1_pool;
using isac_replay::o1_view;
using isac_replay::ticket;

// ---------------------------------------------------------------------------------------------- outcomes

const char* to_string(outcome value)
{
  switch (value) {
    case outcome::pending:
      return "pending";
    case outcome::decoded:
      return "decoded";
    case outcome::payload_mismatch:
      return "payload_mismatch";
    case outcome::crc_fail:
      return "crc_fail";
    case outcome::decoder_error:
      return "decoder_error";
    case outcome::timeout:
      return "timeout";
    case outcome::not_dispatched:
      return "not_dispatched";
  }
  return "unknown";
}

const char* to_string(admission value)
{
  switch (value) {
    case admission::admitted:
      return "admitted";
    case admission::duplicate_delivery:
      return "duplicate_delivery";
    case admission::stale_grid:
      return "stale_grid";
    case admission::changed_allocation:
      return "changed_allocation";
    case admission::changed_grid:
      return "changed_grid";
    case admission::empty_payload:
      return "empty_payload";
  }
  return "unknown";
}

outcome classify(bool completed, bool error, bool crc_ok, bool bytes_equal, bool guard_intact, bool input_intact)
{
  if (!completed) {
    return outcome::timeout;
  }
  if (error || !guard_intact || !input_intact) {
    return outcome::decoder_error;
  }
  if (!crc_ok) {
    return outcome::crc_fail;
  }
  return bytes_equal ? outcome::decoded : outcome::payload_mismatch;
}

comparison compare_bytes(const std::uint8_t* decoded,
                         std::size_t         decoded_bytes,
                         const std::uint8_t* original,
                         std::size_t         original_bytes)
{
  comparison result;
  result.bytes_compared = std::min(decoded_bytes, original_bytes);
  for (std::size_t index = 0; index != result.bytes_compared; ++index) {
    if (decoded[index] != original[index]) {
      result.first_differing_byte = static_cast<long long>(index);
      break;
    }
  }
  result.equal = decoded_bytes == original_bytes && original_bytes != 0 && result.first_differing_byte < 0;
  return result;
}

// ---------------------------------------------------------------------------------------------- ledger

struct ledger::entry {
  job                                   record;
  std::vector<std::uint8_t>             buffer;   ///< The decoded bytes, then the guard.
  std::vector<std::uint8_t>             original; ///< A copy: the comparison needs no object of the caller.
  std::chrono::steady_clock::time_point started_at;
  bool                                  resolved   = false;
  bool                                  holds_grid = false;
};

ledger::ledger(o1_pool& pool_, isac_replay::evidence mode_) : pool(pool_), mode(mode_) {}

ledger::~ledger() = default;

void ledger::note(const std::string& kind, const std::string& key, const std::string& detail)
{
  // The caller holds the mutex.
  anomaly item;
  item.kind   = kind;
  item.key    = key;
  item.detail = detail;
  noted.push_back(item);
}

ledger::entry& ledger::find(std::uint64_t token, const char* action)
{
  if (token == 0 || token > entries.size()) {
    throw state_error(std::string(action) + ": the ledger holds no job " + std::to_string(token));
  }
  return *entries[token - 1];
}

const ledger::entry& ledger::find(std::uint64_t token, const char* action) const
{
  return const_cast<ledger*>(this)->find(token, action);
}

admission ledger::admit(const job_key&     key,
                        const ticket&      grid,
                        const std::string& grant_record_sha256,
                        const payload&     original,
                        std::uint64_t&     token,
                        std::string*       reason)
{
  if (key.opportunity_id.empty() || key.backend.empty()) {
    throw input_error("a decode needs the identifier of its opportunity and the name of its backend");
  }
  const std::string name   = key.text();
  auto              refuse = [&](admission why, const char* kind, const std::string& detail) {
    std::lock_guard<std::mutex> lock(mutex);
    note(kind, name, detail);
    if (reason != nullptr) {
      *reason = detail;
    }
    return why;
  };
  {
    std::unique_lock<std::mutex> lock(mutex);
    if (keys.count(name) != 0) {
      lock.unlock();
      return refuse(admission::duplicate_delivery, "duplicate_delivery",
                    "duplicate delivery: the decode " + name + " was admitted before");
    }
  }
  if (original.bytes.empty()) {
    return refuse(admission::empty_payload, "refused_grid",
                  "the decode " + name + " names no original transport block to compare with");
  }
  // The grid must be ready, of this generation, filled for this grant record, and unchanged.
  std::string digest;
  std::string grant_id;
  try {
    const o1_view view = pool.view(grid, grant_record_sha256);
    grant_id           = view.descriptor().grant_id;
    // Timed: the view itself proves the grid ready, of this generation and of this grant record.
    digest = mode == isac_replay::evidence::audit ? view.digest() : std::string();
    if (mode == isac_replay::evidence::audit && digest != view.descriptor().readback_sha256) {
      return refuse(admission::changed_grid, "refused_grid",
                    "grid " + std::to_string(grid.grid) + " changed after it became ready");
    }
  } catch (const state_error& error) {
    const std::string text       = error.what();
    const bool        allocation = text.find("changed allocation") != std::string::npos;
    return refuse(allocation ? admission::changed_allocation : admission::stale_grid, "refused_grid", text);
  }

  std::unique_ptr<entry> item(new entry);
  item->record.key                 = key;
  item->record.grid                = grid;
  item->record.grant_id            = grant_id;
  item->record.grant_record_sha256 = grant_record_sha256;
  item->record.ingress_sha256      = digest;
  item->record.payload_id          = original.id;
  item->record.payload_sha256      = original.sha256;
  item->record.payload_bytes       = original.bytes.size();
  item->original                   = original.bytes;
  item->buffer.assign(original.bytes.size() + GUARD_BYTES, FILL_VALUE);
  std::fill(item->buffer.begin() + static_cast<std::ptrdiff_t>(original.bytes.size()), item->buffer.end(), GUARD_VALUE);
  item->holds_grid = true;

  std::lock_guard<std::mutex> lock(mutex);
  item->record.sequence = entries.size();
  item->record.token    = entries.size() + 1;
  token                 = item->record.token;
  keys[name]            = token;
  ++held[grid_id(grid.grid, grid.generation)];
  entries.push_back(std::move(item));
  return admission::admitted;
}

request ledger::describe(std::uint64_t token) const
{
  std::lock_guard<std::mutex> lock(mutex);
  const entry&                item = find(token, "describe");
  if (!item.holds_grid && !item.record.completed) {
    throw state_error("describe: the job " + item.record.key.text() + " was never dispatched");
  }
  request result;
  result.token               = token;
  result.key                 = item.record.key;
  result.grid                = item.record.grid;
  result.grant_id            = item.record.grant_id;
  result.grant_record_sha256 = item.record.grant_record_sha256;
  result.output              = const_cast<std::uint8_t*>(item.buffer.data());
  result.output_bytes        = item.record.payload_bytes;
  return result;
}

o1_view ledger::view(std::uint64_t token) const
{
  ticket      grid;
  std::string record;
  {
    std::lock_guard<std::mutex> lock(mutex);
    const entry&                item = find(token, "read");
    grid                             = item.record.grid;
    record                           = item.record.grant_record_sha256;
  }
  return pool.view(grid, record);
}

void ledger::started(std::uint64_t token)
{
  std::lock_guard<std::mutex> lock(mutex);
  entry&                      item = find(token, "start");
  item.record.started              = true;
  item.started_at                  = std::chrono::steady_clock::now();
}

void ledger::complete(std::uint64_t token, bool crc_ok, const std::string& error)
{
  const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex>                 lock(mutex);
  if (token == 0 || token > entries.size()) {
    note("unknown_completion", std::string(), "a completion named the unknown job " + std::to_string(token));
    return;
  }
  entry& item = *entries[token - 1];
  ++item.record.completions;
  if (item.record.result == outcome::not_dispatched) {
    note("unknown_completion", item.record.key.text(), "a completion named a decode that was never started");
    return;
  }
  if (item.record.completed) {
    note("duplicate_completion", item.record.key.text(), "a second completion arrived. The first one stands.");
    return;
  }
  item.record.completed = true;
  if (item.resolved) {
    // The job timed out before. Its outcome stays, and its grid is no longer read.
    note("late_completion", item.record.key.text(), "the completion arrived after the decode had timed out");
  } else {
    item.record.crc_ok = crc_ok;
    item.record.error  = error;
    if (item.record.started) {
      item.record.service_us = std::chrono::duration<double, std::micro>(now - item.started_at).count();
    }
  }
  if (item.holds_grid) {
    item.holds_grid = false;
    auto found      = held.find(grid_id(item.record.grid.grid, item.record.grid.generation));
    if (found != held.end() && --found->second == 0) {
      held.erase(found);
    }
  }
  arrived.notify_all();
}

bool ledger::wait(std::uint64_t token, std::chrono::milliseconds limit)
{
  std::unique_lock<std::mutex> lock(mutex);
  const entry&                 item = find(token, "wait");
  return arrived.wait_for(lock, limit, [&item]() { return item.record.completed; });
}

outcome ledger::resolve(std::uint64_t token)
{
  entry* item = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex);
    item = &find(token, "resolve");
    if (item->resolved) {
      return item->record.result;
    }
    if (!item->record.completed) {
      // The processor may still read the grid and write the buffer: neither is touched here.
      item->resolved      = true;
      item->record.result = classify(false, false, false, false, false, false);
      item->record.error  = "no completion within the declared time";
      return item->record.result;
    }
  }
  // The callback arrived, so the decoder is done with the grid and with the buffer.
  bool intact = false;
  try {
    const o1_view view = pool.view(item->record.grid, item->record.grant_record_sha256);
    intact             = mode == isac_replay::evidence::audit ? view.digest() == item->record.ingress_sha256 : true;
  } catch (const state_error&) {
    intact = false;
  }
  const std::size_t bytes = item->record.payload_bytes;
  const bool        guard = std::all_of(item->buffer.begin() + static_cast<std::ptrdiff_t>(bytes), item->buffer.end(),
                                        [](std::uint8_t value) { return value == GUARD_VALUE; });
  const comparison  compared = compare_bytes(item->buffer.data(), bytes, item->original.data(), item->original.size());
  const std::string digest   =
      mode == isac_replay::evidence::audit ? isac_replay::sha256_hex(item->buffer.data(), bytes) : std::string();

  std::lock_guard<std::mutex> lock(mutex);
  item->record.input_intact = intact;
  item->record.guard_intact = guard;
  if (item->record.crc_ok) {
    item->record.compared       = compared;
    item->record.decoded_sha256 = digest;
  }
  if (item->record.error.empty() && !guard) {
    item->record.error = "the decoder wrote past the transport block";
  } else if (item->record.error.empty() && !intact) {
    item->record.error = "the grid changed while it was decoded";
  }
  item->record.result = classify(true, !item->record.error.empty(), item->record.crc_ok, compared.equal, guard, intact);
  item->resolved      = true;
  return item->record.result;
}

void ledger::skip(const job_key&     key,
                  const std::string& grant_id,
                  const std::string& grant_record_sha256,
                  const std::string& why)
{
  if (key.opportunity_id.empty() || key.backend.empty()) {
    throw input_error("a decode needs the identifier of its opportunity and the name of its backend");
  }
  std::lock_guard<std::mutex> lock(mutex);
  const std::string           name = key.text();
  if (keys.count(name) != 0) {
    note("duplicate_delivery", name, "duplicate delivery: the decode " + name + " was admitted before");
    return;
  }
  std::unique_ptr<entry> item(new entry);
  item->record.key                 = key;
  item->record.grant_id            = grant_id;
  item->record.grant_record_sha256 = grant_record_sha256;
  item->record.result              = outcome::not_dispatched;
  item->record.error               = why;
  item->record.sequence            = entries.size();
  item->record.token               = entries.size() + 1;
  item->resolved                   = true;
  keys[name]                       = item->record.token;
  entries.push_back(std::move(item));
}

bool ledger::release_grid(const ticket& grid)
{
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (held.count(grid_id(grid.grid, grid.generation)) != 0) {
      return false;
    }
  }
  pool.release(grid);
  return true;
}

std::size_t ledger::holds(const ticket& grid) const
{
  std::lock_guard<std::mutex> lock(mutex);
  const auto                  found = held.find(grid_id(grid.grid, grid.generation));
  return found == held.end() ? 0 : found->second;
}

std::size_t ledger::held_grids() const
{
  std::lock_guard<std::mutex> lock(mutex);
  return held.size();
}

job ledger::snapshot(std::uint64_t token) const
{
  std::lock_guard<std::mutex> lock(mutex);
  return find(token, "read").record;
}

std::vector<job> ledger::jobs() const
{
  std::lock_guard<std::mutex> lock(mutex);
  std::vector<job>            result;
  result.reserve(entries.size());
  for (const std::unique_ptr<entry>& item : entries) {
    result.push_back(item->record);
  }
  return result;
}

std::vector<anomaly> ledger::anomalies() const
{
  std::lock_guard<std::mutex> lock(mutex);
  return noted;
}

std::vector<std::uint8_t> ledger::decoded(std::uint64_t token) const
{
  std::lock_guard<std::mutex> lock(mutex);
  const entry&                item = find(token, "read");
  if (!item.record.completed || !item.resolved || item.record.result == outcome::timeout) {
    throw state_error("read: the decode " + item.record.key.text() + " has no completed result");
  }
  return std::vector<std::uint8_t>(item.buffer.begin(),
                                   item.buffer.begin() + static_cast<std::ptrdiff_t>(item.record.payload_bytes));
}

std::size_t ledger::count(outcome value, const std::string& backend) const
{
  std::lock_guard<std::mutex> lock(mutex);
  std::size_t                 total = 0;
  for (const std::unique_ptr<entry>& item : entries) {
    if (item->record.result == value && (backend.empty() || item->record.key.backend == backend)) {
      ++total;
    }
  }
  return total;
}

// ---------------------------------------------------------------------------------------------- dispatch

dispatched dispatch(ledger&                   book,
                    decoder&                  processor,
                    const job_key&            key,
                    const ticket&             grid,
                    const std::string&        grant_record_sha256,
                    const payload&            original,
                    std::chrono::milliseconds limit)
{
  dispatched result;
  result.admitted = book.admit(key, grid, grant_record_sha256, original, result.token, &result.reason);
  if (result.admitted != admission::admitted) {
    result.token = 0;
    return result;
  }
  const request job = book.describe(result.token);
  book.started(result.token);
  try {
    processor.start(job, book);
  } catch (const std::exception& error) {
    book.complete(result.token, false, std::string("the decoder raised: ") + error.what());
  }
  book.wait(result.token, limit);
  result.result = book.resolve(result.token);
  return result;
}

} // namespace isac_receiver
