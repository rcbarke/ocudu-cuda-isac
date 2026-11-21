// Channel application for receiver replay (Sprint 2.3, selector grid_apply). Host-free core.
//
// The receiver grid of one PUSCH opportunity is
//
//     Y[rx, symbol, subcarrier] = g * (sum over tx of H[rx, tx, symbol, subcarrier] * X[tx, symbol, subcarrier]
//                                      + W[rx, symbol, subcarrier])
//
// over the whole carrier grid, computed once in double precision. X is the transmit grid of the
// opportunity's grant, H and W are one opportunity of a channel-episode entry, and g is the grid
// gain of that entry: real, positive and the same for every opportunity of a replay run. The gain
// is applied to the signal and to the noise before the value is converted to the receiver grid's
// storage format. A stored grid is never rescaled.
//
// Order of the arithmetic, so that a reference in another language reproduces every bit:
//   1. each product H * X as (hr * xr - hi * xi, hr * xi + hi * xr), in transmit-port order;
//   2. the products are summed in that order, then W is added;
//   3. the sum is multiplied by g, each part on its own;
//   4. each part is converted to single precision (round to nearest, ties to even) and handed to
//      the grid, which rounds it to its storage format.
// The operands of step 1 are single-precision values, so every product is exact in double
// precision and the result does not depend on whether a compiler fuses it into the sum.
//
// This file includes nothing of OCUDU. The unit target links it with doubles only. The pinned
// resource grid is one implementation of grid_sink (pinned_grid.h), and the plan reader is in
// replay_plan.h. All text in this directory is ASCII.
//
// Amended by Sprint 3.2 (decided 2026-11-21), and by nothing else since Sprint 2.3 was accepted:
// the hash counts its own work (sha256_count), and a grid can become ready without a readback
// digest (evidence::timed). The arithmetic, the gain rules and every default are unchanged, so a
// caller that names no evidence mode runs the code that was accepted.

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace isac_replay {

using cf32 = std::complex<float>;

/// A plan, an array or a gain violates the documented interface. Raised before a grid is written.
struct input_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// A grid was used out of its fill, ready and release order, with a stale generation or for another allocation.
struct state_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------------------------- hashing

/// SHA-256 (FIPS 180-4). Binds every array to the hash its record names and digests a stored grid.
class sha256
{
public:
  sha256();
  void        update(const void* data, std::size_t bytes);
  std::string hex();

private:
  void          compress(const std::uint8_t* block);
  std::uint32_t state[8];
  std::uint64_t length = 0;
  std::uint8_t  buffer[64];
  std::size_t   filled   = 0;
  bool          finished = false;
  std::string   result;
};

std::string sha256_hex(const void* data, std::size_t bytes);

/// What the hash did in this process so far: digests that were finished, calls that handed it
/// data and the bytes of those calls. A timed path must leave all three unchanged.
struct hash_count {
  std::uint64_t digests = 0;
  std::uint64_t updates = 0;
  std::uint64_t bytes   = 0;
};

hash_count sha256_count();

// ---------------------------------------------------------------------------------------------- evidence

/// The two evidence modes of a run. One arithmetic serves both.
///   audit  every digest: a grid is read back when it becomes ready and whenever it is checked.
///          The gates of Sprints 2.3 to 3.1 ran this mode, and it is the default everywhere.
///   timed  no digest. A grid is owned by the ticket of its fill, that is by its index and its
///          generation, and the evaluator digests what was archived after the run.
enum class evidence { audit, timed };

const char* to_string(evidence mode);

// ---------------------------------------------------------------------------------------------- storage

/// The value the pinned resource grid returns for a written single-precision value. The grid keeps
/// the upper 16 bits of each component, rounded to nearest with ties to even: eight significant
/// bits. A stored value lies within 2^-8 of its magnitude of the value written.
float bf16_value(float value);

constexpr const char* BF16_STORAGE = "complex bf16: the upper 16 bits of each single-precision component, nearest, ties to even";

// ---------------------------------------------------------------------------------------------- arrays

/// Read-only mapping of a little-endian, C-ordered NumPy file of format 1.0 holding complex64 or int32.
class array_file
{
public:
  explicit array_file(const std::string& path);
  ~array_file();
  array_file(const array_file&)            = delete;
  array_file& operator=(const array_file&) = delete;

  const std::string&              path() const { return file_path; }
  const std::string&              descr() const { return type; }
  const std::vector<std::size_t>& shape() const { return dimensions; }
  std::size_t                     count() const { return values; }
  std::size_t                     file_bytes() const { return mapped_bytes; }
  /// Hash of the whole file, as the manifests of the stores and of the episode entries record it.
  std::string file_sha256() const;
  /// The values. Each refuses the other element type.
  const cf32*         complex64() const;
  const std::int32_t* int32() const;

private:
  std::string              file_path;
  std::string              type;
  std::vector<std::size_t> dimensions;
  std::size_t              values       = 0;
  std::size_t              mapped_bytes = 0;
  std::size_t              offset       = 0;
  void*                    mapping      = nullptr;
};

// ---------------------------------------------------------------------------------------------- gain

/// Relative tolerance of the rule g = 1 / sqrt(reference_power).
constexpr double GAIN_RULE_TOLERANCE = 1e-6;

struct gain_record {
  bool        present = false; ///< The run names a gain. A run without one is refused.
  double      value   = 0.0;
  std::string source;          ///< "entry": calibration.grid_gain of an episode entry. "override": a fixture without an entry.
  bool        has_reference_power = false;
  double      reference_power     = 0.0; ///< Of the entry: mean |H|^2 that the gain normalizes.
  std::string reason;                    ///< Override only: why the fixture sets the gain. It is recorded.
};

// ---------------------------------------------------------------------------------------------- plan

struct file_ref {
  std::string path;
  std::string sha256; ///< The hash that binds the file. Empty: nothing binds it, and the run is refused.
};

/// One grant of the run: its transmit grid and its occupied resource elements.
struct grant_spec {
  std::string           grant_id;
  std::string           record_sha256; ///< Identity of the grant record, hence of the allocation.
  file_ref              x;             ///< X[tx_port, symbol, subcarrier], complex64, whole carrier grid.
  std::vector<file_ref> active;        ///< int32 [n, 2] files of (symbol, subcarrier): together the occupied elements.
};

/// One channel realization: H and W of every opportunity of one case of an episode, or of a fixture.
struct source_spec {
  std::string realization_id; ///< Opaque. It names no case and no label.
  bool        from_entry            = false;
  double      entry_grid_gain       = 0.0; ///< calibration.grid_gain recorded by the entry.
  double      entry_reference_power = 0.0;
  file_ref    h; ///< H[opportunity, rx_port, tx_port, symbol, subcarrier], complex64.
  file_ref    w; ///< W[opportunity, rx_port, symbol, subcarrier], complex64.
};

struct opportunity_spec {
  std::string opportunity_id; ///< Opaque join key, unique in the run.
  std::string realization_id;
  std::size_t index = 0; ///< Opportunity index into H and W of the realization.
  std::string grant_id;  ///< Empty: the opportunity has no bound grid, and the run is refused.
  unsigned    sfn  = 0;
  unsigned    slot = 0;
};

struct run_spec {
  std::string                   run_id;
  unsigned                      rx_ports    = 0;
  unsigned                      tx_ports    = 0;
  unsigned                      symbols     = 0;
  unsigned                      subcarriers = 0;
  gain_record                   gain;
  std::vector<grant_spec>       grants;
  std::vector<source_spec>      sources;
  std::vector<opportunity_spec> opportunities;
};

/// The gain rules of the ticket. Throws input_error for a gain that is missing, zero, negative or
/// not finite, for an entry gain that differs from 1 / sqrt(reference_power) by more than the
/// tolerance, for two gains in one run and for an override on a run that reads an episode entry.
void check_gain(const run_spec& spec);

// ---------------------------------------------------------------------------------------------- grid

struct grid_shape {
  unsigned rx_ports    = 0;
  unsigned tx_ports    = 0;
  unsigned symbols     = 0;
  unsigned subcarriers = 0;

  std::size_t plane() const { return static_cast<std::size_t>(symbols) * subcarriers; }
};

/// One receive grid in the receiver's storage format: the writer the channel is applied into and
/// the reader its stored values come back from.
class grid_sink
{
public:
  virtual ~grid_sink() = default;

  virtual unsigned planes() const      = 0;
  virtual unsigned symbols() const     = 0;
  virtual unsigned subcarriers() const = 0;
  /// Sets every resource element to zero.
  virtual void clear() = 0;
  /// Writes subcarriers 0 to count-1 of one symbol of one receive plane.
  virtual void put(unsigned plane, unsigned symbol, const cf32* row, unsigned count) = 0;
  /// Reads the stored values of subcarriers 0 to count-1, widened to single precision without change.
  virtual void get(unsigned plane, unsigned symbol, cf32* row, unsigned count) const = 0;
  /// Name of the storage representation, for the record.
  virtual std::string storage() const = 0;
};

/// A grid that stores what the pinned resource grid stores, with no OCUDU code. The unit target
/// applies the channel into it, and the tool measures the pinned grid against it on every element.
class model_grid final : public grid_sink
{
public:
  model_grid(unsigned planes_, unsigned symbols_, unsigned subcarriers_);

  unsigned    planes() const override { return nof_planes; }
  unsigned    symbols() const override { return nof_symbols; }
  unsigned    subcarriers() const override { return nof_subcarriers; }
  void        clear() override;
  void        put(unsigned plane, unsigned symbol, const cf32* row, unsigned count) override;
  void        get(unsigned plane, unsigned symbol, cf32* row, unsigned count) const override;
  std::string storage() const override { return BF16_STORAGE; }

private:
  unsigned          nof_planes;
  unsigned          nof_symbols;
  unsigned          nof_subcarriers;
  std::vector<cf32> cells;
};

/// Applies the channel of one opportunity. x holds [tx_port, symbol, subcarrier], h holds
/// [rx_port, tx_port, symbol, subcarrier] and w holds [rx_port, symbol, subcarrier], last axis
/// fastest. prestore, when given, receives the single-precision values handed to the grid, in the
/// order of w. Throws input_error for a gain that is not finite and positive, for a grid smaller
/// than the shape and for a nonfinite result.
void apply(const grid_shape& shape,
           const cf32*       x,
           const cf32*       h,
           const cf32*       w,
           double            gain,
           grid_sink&        sink,
           cf32*             prestore = nullptr);

/// SHA-256 over the stored values of a grid: little-endian complex64 [rx_port, symbol, subcarrier]
/// in C order over the carrier grid, read back through the grid's reader.
std::string readback_sha256(const grid_sink& grid, const grid_shape& shape);

// ---------------------------------------------------------------------------------------------- owned grids

enum class grid_state { free, filling, ready };

const char* to_string(grid_state state);

/// What the receiver side may know of a filled grid. It names no channel, no noise, no transport
/// block and no label.
struct o1_descriptor {
  std::string run_id;
  std::string opportunity_id;
  std::string grant_id;
  std::string grant_record_sha256;
  std::string active_sha256; ///< Hash of the sorted occupied elements, as little-endian uint32 flat indices.
  std::size_t active_re = 0;
  unsigned    sfn       = 0;
  unsigned    slot      = 0;
  grid_shape  shape;
  double      grid_gain = 0.0;
  std::string storage;
  std::string readback_sha256; ///< Set when the grid becomes ready in the mode audit. Empty in the mode timed.
};

/// Names one fill of one grid. A ticket of an earlier fill is stale.
struct ticket {
  std::size_t   grid       = 0;
  std::uint64_t generation = 0;
};

class o1_pool;

/// Read access to one ready grid. It holds no route to H, W, X or any evaluator record.
class o1_view
{
public:
  const o1_descriptor& descriptor() const;
  ticket               key() const { return id; }
  /// Stored values of one symbol of one plane. Throws state_error once the grid was released or filled again.
  void get(unsigned plane, unsigned symbol, cf32* row, unsigned count) const;
  /// The readback digest, computed again from the stored values.
  std::string digest() const;

private:
  friend class o1_pool;
  o1_view(const o1_pool& owner_, ticket id_) : owner(&owner_), id(id_) {}
  const o1_pool* owner;
  ticket         id;
};

/// Grids that the replay owns, with their fill, ready and release state. One producer thread:
/// the cross-thread ownership of an observation is the ring of Sprint 3.2.
class o1_pool
{
public:
  /// The grids are not owned by the pool and must outlive it.
  explicit o1_pool(std::vector<grid_sink*> grids_);

  std::size_t size() const { return slots.size(); }
  /// free -> filling with a new generation. Returns false when no grid is free. It never waits.
  bool try_acquire(ticket& out);
  /// The writer of a grid that is being filled.
  grid_sink& writer(const ticket& key);
  /// filling -> ready. In the mode audit it reads the grid back and records its digest in the
  /// descriptor. In the mode timed it reads nothing, and the digest of the descriptor stays empty.
  void commit(const ticket& key, o1_descriptor descriptor, evidence mode = evidence::audit);
  /// filling -> free after a failed fill. The grid is cleared.
  void abort(const ticket& key);
  /// A ready grid, for the consumer that decodes this allocation. Throws state_error for a grid
  /// that is not ready, for a stale generation and for another allocation.
  o1_view view(const ticket& key, const std::string& grant_record_sha256) const;
  /// ready -> free, exactly once.
  void release(const ticket& key);

  grid_state    state(std::size_t grid) const;
  std::uint64_t generation(std::size_t grid) const;

private:
  friend class o1_view;
  struct slot_t {
    grid_sink*    grid       = nullptr;
    grid_state    state      = grid_state::free;
    std::uint64_t generation = 0;
    o1_descriptor descriptor;
  };
  const slot_t& ready_slot(const ticket& key) const;
  slot_t&       slot_for(const ticket& key, grid_state expected, const char* action);
  std::vector<slot_t> slots;
};

// ---------------------------------------------------------------------------------------------- session

/// What a fill uses of one grant. The arrays stay mapped for the life of the session.
struct allocation {
  std::string                grant_id;
  std::string                record_sha256;
  std::string                x_sha256;
  std::string                active_sha256;
  const cf32*                x = nullptr;
  std::vector<std::uint32_t> active; ///< Sorted flat indices symbol * subcarriers + subcarrier.
};

struct verified_file {
  std::string role;
  std::string path;
  std::string sha256;
  std::size_t bytes = 0;
};

struct plane_power {
  double all    = 0.0; ///< Mean |Y|^2 of the stored plane over the carrier grid.
  double active = 0.0; ///< Mean |Y|^2 of the stored plane over the occupied elements.
};

/// What storage did to one grid, measured on every component against the values handed to the writer.
struct storage_stats {
  std::vector<plane_power> planes;
  std::size_t              components         = 0;
  std::size_t              components_changed = 0;   ///< Stored value differs from the value written.
  std::size_t              model_mismatches   = 0;   ///< Stored value differs from bf16_value of the value written.
  double                   max_relative_error = 0.0; ///< Largest |stored - written| / |written| of a component.
  double                   error_power        = 0.0; ///< Mean |stored - written|^2 per resource element.
  double                   written_power      = 0.0; ///< Mean |written|^2 per resource element.
};

/// The preloaded stimulus of one replay run: every array mapped and verified against its hash,
/// every grant bound to its grid and its occupied elements, the gain checked.
class session
{
public:
  /// Validates the plan and every file. Throws input_error before any grid is touched.
  explicit session(const run_spec& spec_);
  ~session();

  const run_spec&                   spec() const { return plan; }
  grid_shape                        shape() const;
  std::size_t                       opportunities() const { return plan.opportunities.size(); }
  const std::vector<verified_file>& files() const { return verified; }
  /// The grant an opportunity is scheduled with.
  const allocation& scheduled(std::size_t opportunity) const;
  /// A grant of the run by name. Throws input_error for a grant the run does not bind.
  const allocation& bound(const std::string& grant_id) const;
  /// g^2 times the mean |H|^2 of an opportunity over the receive ports and the occupied elements of its grant.
  double unit_power(std::size_t opportunity) const;
  /// Applies the channel of one opportunity with its scheduled grant.
  void fill(std::size_t opportunity, grid_sink& sink, cf32* prestore = nullptr) const;
  /// The same with another grant of the run: the grant changed, so X and the occupied elements are
  /// taken from that grant and H at the coordinates it transmits on. A grant the run does not bind is refused.
  void fill(std::size_t opportunity, const std::string& grant_id, grid_sink& sink, cf32* prestore = nullptr) const;
  /// Acquires a grid, clears it, applies the channel and makes it ready. Returns false when no grid is free.
  bool deliver(std::size_t opportunity, o1_pool& pool, ticket& out, cf32* prestore = nullptr) const;
  /// The descriptor of an opportunity filled with a grant, without the readback digest.
  o1_descriptor describe(std::size_t opportunity, const allocation& grant, const grid_sink& sink) const;

private:
  struct loaded;
  run_spec                   plan;
  std::unique_ptr<loaded>    data;
  std::vector<verified_file> verified;
};

/// Measures a ready grid against the values that were handed to its writer.
storage_stats measure(const o1_view& view, const allocation& grant, const cf32* prestore);

} // namespace isac_replay
