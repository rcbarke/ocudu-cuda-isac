// Dispatch of one receiver grid to the PUSCH processors (Sprint 2.4, selector replay_pusch). Host-free core.
//
// A replay hands every filled grid to the receiver: the same owned grid, the PDU of its grant and
// the identity of the original transport block go to the host processor and to the CUDA processor,
// and each decode ends in one explicit outcome under a key that is unique in the run.
//
//     decoded            the decoder reported a CRC pass and every decoded byte equals the original
//     payload_mismatch   the decoder reported a CRC pass and a byte differs: the comparison fails
//     crc_fail           the decoder reported a CRC failure. It stays in the denominator.
//     decoder_error      the decoder raised or reported an error, wrote past the transport block,
//                        or its input grid changed while it was read
//     timeout            no completion arrived within the declared time
//     not_dispatched     the run stopped before this decode was started
//
// The CRC flag is the decoder's own. Nothing here sets it, and nothing forces an acknowledgement.
//
// Ownership. A grid that a decoder reads is held from its admission until the completion callback
// of every decode on it has arrived. Until then the ledger refuses to release it, so the replay
// cannot fill it again. A decode that timed out keeps its grid held: the processor may still read
// it. The decoded bytes go to a buffer the ledger owns, with guard bytes behind the payload.
//
// Threads. complete() may be called from any thread, as a processor does. Every other function
// belongs to the one dispatch thread, which is also the only user of the grid pool.
//
// This file includes the channel application core and nothing of OCUDU. The unit target links it
// with doubles only. The pinned PUSCH processors are one implementation of decoder
// (replay_pusch.cpp). All text in this directory is ASCII.

#pragma once

#include "replay_core.h"
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace isac_receiver {

using isac_replay::input_error;
using isac_replay::state_error;

/// Bytes behind every decoded transport block that the decoder must leave untouched.
constexpr std::size_t GUARD_BYTES = 64;
constexpr std::uint8_t GUARD_VALUE = 0x7e;
constexpr std::uint8_t FILL_VALUE  = 0xa5;

// ---------------------------------------------------------------------------------------------- outcomes

enum class outcome { pending, decoded, payload_mismatch, crc_fail, decoder_error, timeout, not_dispatched };

const char* to_string(outcome value);

/// The one rule that turns what a decoder did into an outcome. A reference in another language
/// repeats it (ocudu_cuda_isac.episodes.replay_pusch.classify).
///   no completion                                  -> timeout
///   an error, a damaged guard, a changed input     -> decoder_error
///   CRC failure                                    -> crc_fail
///   CRC pass, a byte differs from the original     -> payload_mismatch
///   CRC pass, every byte equal                     -> decoded
outcome classify(bool completed, bool error, bool crc_ok, bool bytes_equal, bool guard_intact, bool input_intact);

/// Why a grid was not admitted for a decode. Each is recorded. None is a job.
enum class admission { admitted, duplicate_delivery, stale_grid, changed_allocation, changed_grid, empty_payload };

const char* to_string(admission value);

// ---------------------------------------------------------------------------------------------- payload

/// The original transport block of one grant. The receiver never reads it: the ledger compares
/// the decoded bytes with it after the decoder's callback.
struct payload {
  std::string               id;     ///< Opaque identifier of the block, the grant it belongs to.
  std::string               sha256; ///< Hash of the bytes, as the grant store records it.
  std::vector<std::uint8_t> bytes;
};

struct comparison {
  bool        equal                = false;
  long long   first_differing_byte = -1; ///< -1 when no compared byte differs.
  std::size_t bytes_compared       = 0;  ///< The shorter of the two lengths.
};

/// Compares decoded bytes with the original ones. Blocks of different lengths are never equal.
comparison compare_bytes(const std::uint8_t* decoded,
                         std::size_t         decoded_bytes,
                         const std::uint8_t* original,
                         std::size_t         original_bytes);

// ---------------------------------------------------------------------------------------------- jobs

/// Names one decode. Unique in a run: one opportunity is decoded once by each backend.
struct job_key {
  std::string opportunity_id;
  std::string backend;

  std::string text() const { return opportunity_id + "/" + backend; }
};

/// Something that should not happen and did. It is counted and never changes an outcome.
struct anomaly {
  std::string kind; ///< duplicate_delivery, duplicate_completion, unknown_completion, late_completion, refused_grid.
  std::string key;
  std::string detail;
};

/// One admitted decode, as the ledger knows it.
struct job {
  std::uint64_t       token    = 0; ///< Names the job in the completion callback. Never reused.
  std::size_t         sequence = 0; ///< Order of admission in the run.
  job_key             key;
  isac_replay::ticket grid;
  std::string         grant_id;
  std::string         grant_record_sha256;
  std::string         ingress_sha256;     ///< Readback digest of the grid at admission.
  bool                input_intact = false; ///< The grid had that digest again after the completion.
  std::string         payload_id;
  std::string         payload_sha256;
  std::size_t         payload_bytes = 0;
  outcome             result        = outcome::pending;
  bool                started       = false;
  bool                completed     = false; ///< The completion callback arrived.
  bool                crc_ok        = false; ///< As the decoder reported it.
  std::string         error;                 ///< As the decoder reported it, or the exception of its start.
  bool                guard_intact = false;
  comparison          compared;
  std::string         decoded_sha256; ///< Of the bytes the decoder left, when it reported a CRC pass.
  unsigned            completions = 0; ///< Callbacks that named this job.
  double              service_us  = 0.0; ///< From the start of the decode to its first completion callback.
};

/// Where a decoder reports its result.
class completion_sink
{
public:
  virtual ~completion_sink() = default;
  /// The decode named by token finished. error, when not empty, says why it produced no result.
  virtual void complete(std::uint64_t token, bool crc_ok, const std::string& error) = 0;
};

/// What a decoder is handed for one decode: the key, the grid and the buffer for the decoded bytes.
struct request {
  std::uint64_t       token = 0;
  job_key             key;
  isac_replay::ticket grid;
  std::string         grant_id;
  std::string         grant_record_sha256;
  std::uint8_t*       output       = nullptr; ///< output_bytes bytes, owned by the ledger until the run ends.
  std::size_t         output_bytes = 0;
};

/// One PUSCH processor behind the interface the dispatch needs.
class decoder
{
public:
  virtual ~decoder() = default;
  /// Starts the decode of one grid. The result arrives at sink, during the call or later, on any thread.
  virtual void start(const request& job, completion_sink& sink) = 0;
};

/// Every decode of a run: its key, its grid, its buffer and its outcome.
class ledger final : public completion_sink
{
public:
  /// The pool is not owned and must outlive the ledger.
  explicit ledger(isac_replay::o1_pool& pool_);
  ~ledger() override;
  ledger(const ledger&)            = delete;
  ledger& operator=(const ledger&) = delete;

  /// Admits one decode of a ready grid and holds the grid for it. A key that was admitted before,
  /// a grid that is not ready, a stale generation, another grant record and a grid whose values
  /// changed since it became ready are refused and recorded. token names the job afterwards.
  admission admit(const job_key&             key,
                  const isac_replay::ticket& grid,
                  const std::string&         grant_record_sha256,
                  const payload&             original,
                  std::uint64_t&             token,
                  std::string*               reason = nullptr);

  /// What the decoder of an admitted job is handed.
  request describe(std::uint64_t token) const;
  /// Read access to the grid of an admitted job, for a decoder that reads through the core.
  isac_replay::o1_view view(std::uint64_t token) const;
  /// Marks the start of the decode: the service time is counted from here.
  void started(std::uint64_t token);

  // See completion_sink. A callback for an unknown job, a second callback and a callback after a
  // timeout are recorded as anomalies. The first callback of a job stands.
  void complete(std::uint64_t token, bool crc_ok, const std::string& error) override;

  /// Waits for the completion callback of a job. Returns false when it did not arrive in time.
  bool wait(std::uint64_t token, std::chrono::milliseconds limit);

  /// Gives the job its outcome. A job without a completion becomes a timeout and keeps its grid held.
  outcome resolve(std::uint64_t token);

  /// Records a decode of the plan that was never started. It has a key and an outcome, and no grid.
  void skip(const job_key& key, const std::string& grant_id, const std::string& grant_record_sha256,
            const std::string& why);

  /// Releases a grid to the pool when no admitted decode still holds it. Returns false, and leaves
  /// the grid as it is, while a completion is outstanding.
  bool release_grid(const isac_replay::ticket& grid);

  /// Decodes that still hold a grid.
  std::size_t holds(const isac_replay::ticket& grid) const;
  std::size_t held_grids() const;

  job                       snapshot(std::uint64_t token) const;
  std::vector<job>          jobs() const;
  std::vector<anomaly>      anomalies() const;
  /// The decoded bytes of a job, as the decoder left them. Meaningful after a reported CRC pass.
  std::vector<std::uint8_t> decoded(std::uint64_t token) const;
  /// Jobs with a given outcome and backend. An empty backend counts every backend.
  std::size_t count(outcome value, const std::string& backend = std::string()) const;

private:
  struct entry;
  using grid_id = std::pair<std::size_t, std::uint64_t>;

  entry&       find(std::uint64_t token, const char* action);
  const entry& find(std::uint64_t token, const char* action) const;
  void         note(const std::string& kind, const std::string& key, const std::string& detail);

  isac_replay::o1_pool&                 pool;
  mutable std::mutex                    mutex;
  std::condition_variable               arrived;
  std::vector<std::unique_ptr<entry>>   entries; ///< Index token - 1. Addresses stay valid until the ledger ends.
  std::map<std::string, std::uint64_t>  keys;
  std::map<grid_id, std::size_t>        held;
  std::vector<anomaly>                  noted;
};

/// The result of one dispatch: whether the grid was admitted, and the outcome of the decode.
struct dispatched {
  admission     admitted = admission::admitted;
  outcome       result   = outcome::pending;
  std::uint64_t token    = 0; ///< Zero when the grid was not admitted.
  std::string   reason;       ///< Why the grid was refused.
};

/// Hands one ready grid to one decoder and waits for its outcome: admit, start, wait, resolve.
/// An exception of the decoder's start becomes a decoder_error. A decoder that never calls back
/// becomes a timeout after limit.
dispatched dispatch(ledger&                    book,
                    decoder&                   processor,
                    const job_key&             key,
                    const isac_replay::ticket& grid,
                    const std::string&         grant_record_sha256,
                    const payload&             original,
                    std::chrono::milliseconds  limit);

} // namespace isac_receiver
