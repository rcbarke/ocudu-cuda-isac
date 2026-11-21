// Channel application for receiver replay (Sprint 2.3). See replay_core.h for the contract.
// This file includes nothing of OCUDU and uses only ASCII characters.

#include "replay_core.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace isac_replay {

namespace {

bool little_endian_host()
{
  const std::uint16_t probe = 1;
  std::uint8_t        first = 0;
  std::memcpy(&first, &probe, 1);
  return first == 1;
}

std::string shape_text(const std::vector<std::size_t>& shape)
{
  std::string text = "[";
  for (std::size_t i = 0; i != shape.size(); ++i) {
    text += (i ? ", " : "") + std::to_string(shape[i]);
  }
  return text + "]";
}

std::string number_text(double value)
{
  char text[40];
  std::snprintf(text, sizeof(text), "%.17g", value);
  return text;
}

inline std::uint32_t rotr(std::uint32_t value, unsigned bits)
{
  return (value >> bits) | (value << (32U - bits));
}

const std::uint32_t SHA256_K[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

} // namespace

// ---------------------------------------------------------------------------------------------- hashing

namespace {

// Counted in the hash itself, so that no caller can hash without being counted.
std::atomic<std::uint64_t> hash_digests{0};
std::atomic<std::uint64_t> hash_updates{0};
std::atomic<std::uint64_t> hash_bytes{0};

} // namespace

hash_count sha256_count()
{
  hash_count now;
  now.digests = hash_digests.load(std::memory_order_relaxed);
  now.updates = hash_updates.load(std::memory_order_relaxed);
  now.bytes   = hash_bytes.load(std::memory_order_relaxed);
  return now;
}

const char* to_string(evidence mode)
{
  return mode == evidence::timed ? "timed" : "audit";
}

sha256::sha256()
{
  static const std::uint32_t initial[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                           0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::memcpy(state, initial, sizeof(state));
}

void sha256::compress(const std::uint8_t* block)
{
  std::uint32_t words[64];
  for (unsigned i = 0; i != 16; ++i) {
    words[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24) | (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
               (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) | static_cast<std::uint32_t>(block[4 * i + 3]);
  }
  for (unsigned i = 16; i != 64; ++i) {
    const std::uint32_t s0 = rotr(words[i - 15], 7) ^ rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
    const std::uint32_t s1 = rotr(words[i - 2], 17) ^ rotr(words[i - 2], 19) ^ (words[i - 2] >> 10);
    words[i]               = words[i - 16] + s0 + words[i - 7] + s1;
  }
  std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (unsigned i = 0; i != 64; ++i) {
    const std::uint32_t s1    = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const std::uint32_t pick  = (e & f) ^ (~e & g);
    const std::uint32_t temp1 = h + s1 + pick + SHA256_K[i] + words[i];
    const std::uint32_t s0    = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const std::uint32_t major = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temp2 = s0 + major;
    h                         = g;
    g                         = f;
    f                         = e;
    e                         = d + temp1;
    d                         = c;
    c                         = b;
    b                         = a;
    a                         = temp1 + temp2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

void sha256::update(const void* data, std::size_t bytes)
{
  if (finished) {
    throw state_error("a finished SHA-256 takes no more data");
  }
  const std::uint8_t* input = static_cast<const std::uint8_t*>(data);
  length += bytes;
  hash_updates.fetch_add(1, std::memory_order_relaxed);
  hash_bytes.fetch_add(bytes, std::memory_order_relaxed);
  if (filled != 0) {
    const std::size_t take = std::min(bytes, sizeof(buffer) - filled);
    std::memcpy(buffer + filled, input, take);
    filled += take;
    input += take;
    bytes -= take;
    if (filled != sizeof(buffer)) {
      return;
    }
    compress(buffer);
    filled = 0;
  }
  for (; bytes >= sizeof(buffer); input += sizeof(buffer), bytes -= sizeof(buffer)) {
    compress(input);
  }
  if (bytes != 0) {
    std::memcpy(buffer, input, bytes);
    filled = bytes;
  }
}

std::string sha256::hex()
{
  if (!finished) {
    const std::uint64_t bits = length * 8U;
    std::uint8_t        tail[72];
    std::size_t         pad = (filled < 56) ? 56 - filled : 120 - filled;
    std::memset(tail, 0, sizeof(tail));
    tail[0] = 0x80;
    for (unsigned i = 0; i != 8; ++i) {
      tail[pad + i] = static_cast<std::uint8_t>(bits >> (56U - 8U * i));
    }
    // update() must not count the padding as message bytes.
    const std::uint64_t message_length = length;
    update(tail, pad + 8);
    length   = message_length;
    finished = true;
    // The padding is not data of a caller. The finished digest is counted once.
    hash_updates.fetch_sub(1, std::memory_order_relaxed);
    hash_bytes.fetch_sub(pad + 8, std::memory_order_relaxed);
    hash_digests.fetch_add(1, std::memory_order_relaxed);
    static const char digits[] = "0123456789abcdef";
    for (unsigned i = 0; i != 8; ++i) {
      for (int shift = 28; shift >= 0; shift -= 4) {
        result.push_back(digits[(state[i] >> shift) & 0xfU]);
      }
    }
  }
  return result;
}

std::string sha256_hex(const void* data, std::size_t bytes)
{
  sha256 hasher;
  hasher.update(data, bytes);
  return hasher.hex();
}

// ---------------------------------------------------------------------------------------------- storage

float bf16_value(float value)
{
  std::uint32_t raw = 0;
  std::memcpy(&raw, &value, sizeof(raw));
  raw += 0x7fffU + ((raw >> 16) & 1U);
  raw &= 0xffff0000U;
  float stored = 0.0F;
  std::memcpy(&stored, &raw, sizeof(stored));
  return stored;
}

// ---------------------------------------------------------------------------------------------- arrays

array_file::array_file(const std::string& path_) : file_path(path_)
{
  if (!little_endian_host()) {
    throw input_error("the arrays are little-endian and this host is not");
  }
  const int descriptor = ::open(file_path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    throw input_error("cannot read " + file_path);
  }
  struct stat info;
  if (::fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 10) {
    ::close(descriptor);
    throw input_error(file_path + " is not a .npy file");
  }
  mapped_bytes = static_cast<std::size_t>(info.st_size);
  mapping      = ::mmap(nullptr, mapped_bytes, PROT_READ, MAP_PRIVATE, descriptor, 0);
  ::close(descriptor);
  if (mapping == MAP_FAILED) {
    mapping = nullptr;
    throw input_error("cannot map " + file_path);
  }
  try {
    const std::uint8_t* bytes = static_cast<const std::uint8_t*>(mapping);
    if (std::memcmp(bytes, "\x93NUMPY\x01\x00", 8) != 0) {
      throw input_error(file_path + " is not a version 1.0 .npy file");
    }
    const std::size_t header_bytes = static_cast<std::size_t>(bytes[8]) | (static_cast<std::size_t>(bytes[9]) << 8);
    offset                         = 10 + header_bytes;
    if (offset > mapped_bytes) {
      throw input_error(file_path + " has a truncated array header");
    }
    const std::string header(reinterpret_cast<const char*>(bytes) + 10, header_bytes);

    auto value_of = [&](const char* key) {
      const std::size_t found = header.find(key);
      if (found == std::string::npos) {
        throw input_error(file_path + " has an unreadable array header");
      }
      std::size_t next = found + std::strlen(key);
      while (next < header.size() && header[next] == ' ') {
        ++next;
      }
      return next;
    };

    std::size_t at = value_of("'descr':");
    if (at >= header.size() || header[at] != '\'') {
      throw input_error(file_path + " has an unreadable array header");
    }
    const std::size_t close = header.find('\'', at + 1);
    if (close == std::string::npos) {
      throw input_error(file_path + " has an unreadable array header");
    }
    type = header.substr(at + 1, close - at - 1);
    if (type != "<c8" && type != "<i4") {
      throw input_error(file_path + " holds " + type + ", expected little-endian complex64 or int32");
    }
    at = value_of("'fortran_order':");
    if (header.compare(at, 5, "False") != 0) {
      throw input_error(file_path + " is not in C order");
    }
    at = value_of("'shape':");
    if (at >= header.size() || header[at] != '(') {
      throw input_error(file_path + " has an unreadable array header");
    }
    ++at;
    values = 1;
    while (at < header.size() && header[at] != ')') {
      if (header[at] == ' ' || header[at] == ',') {
        ++at;
        continue;
      }
      if (header[at] < '0' || header[at] > '9') {
        throw input_error(file_path + " has an unreadable array shape");
      }
      std::size_t dimension = 0;
      while (at < header.size() && header[at] >= '0' && header[at] <= '9') {
        const std::size_t digit = static_cast<std::size_t>(header[at] - '0');
        if (dimension > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
          throw input_error(file_path + " has an array shape that is too large");
        }
        dimension = dimension * 10 + digit;
        ++at;
      }
      if (dimension != 0 && values > std::numeric_limits<std::size_t>::max() / dimension) {
        throw input_error(file_path + " has an array shape that is too large");
      }
      values *= dimension;
      dimensions.push_back(dimension);
    }
    if (at >= header.size() || dimensions.empty()) {
      throw input_error(file_path + " has an unreadable array shape");
    }
    const std::size_t item = (type == "<c8") ? 8 : 4;
    if (values > (std::numeric_limits<std::size_t>::max() / item) || mapped_bytes - offset != values * item) {
      throw input_error(file_path + " holds " + std::to_string(mapped_bytes - offset) + " bytes of data, its shape " +
                        shape_text(dimensions) + " needs " + std::to_string(values * item));
    }
    if (offset % 4 != 0) {
      throw input_error(file_path + " has an array header that leaves the data unaligned");
    }
  } catch (...) {
    ::munmap(mapping, mapped_bytes);
    mapping = nullptr;
    throw;
  }
}

array_file::~array_file()
{
  if (mapping != nullptr) {
    ::munmap(mapping, mapped_bytes);
  }
}

std::string array_file::file_sha256() const
{
  return sha256_hex(mapping, mapped_bytes);
}

const cf32* array_file::complex64() const
{
  if (type != "<c8") {
    throw input_error(file_path + " holds " + type + ", expected complex64");
  }
  return reinterpret_cast<const cf32*>(static_cast<const std::uint8_t*>(mapping) + offset);
}

const std::int32_t* array_file::int32() const
{
  if (type != "<i4") {
    throw input_error(file_path + " holds " + type + ", expected int32");
  }
  return reinterpret_cast<const std::int32_t*>(static_cast<const std::uint8_t*>(mapping) + offset);
}

// ---------------------------------------------------------------------------------------------- gain

void check_gain(const run_spec& spec)
{
  const gain_record& gain = spec.gain;
  if (!gain.present) {
    throw input_error("the run names no grid gain. An episode entry records it as calibration.grid_gain.");
  }
  if (!std::isfinite(gain.value)) {
    throw input_error("the grid gain is not finite");
  }
  if (!(gain.value > 0.0)) {
    throw input_error("the grid gain is " + number_text(gain.value) + ". It must be positive.");
  }
  const bool any_entry =
      std::any_of(spec.sources.begin(), spec.sources.end(), [](const source_spec& item) { return item.from_entry; });
  if (gain.source == "override") {
    if (any_entry) {
      throw input_error("a gain override is allowed only for a fixture that has no entry, and this run reads an "
                        "episode entry");
    }
    if (gain.reason.empty()) {
      throw input_error("a gain override records its reason, and this one names none");
    }
    return;
  }
  if (gain.source != "entry") {
    throw input_error("gain.source must be 'entry' or 'override'");
  }
  if (!gain.has_reference_power || !std::isfinite(gain.reference_power) || !(gain.reference_power > 0.0)) {
    throw input_error("the grid gain of an entry needs the entry's positive reference power");
  }
  auto follows_rule = [](double value, double power) {
    const double rule = 1.0 / std::sqrt(power);
    return std::fabs(value - rule) <= GAIN_RULE_TOLERANCE * rule;
  };
  if (!follows_rule(gain.value, gain.reference_power)) {
    throw input_error("the grid gain " + number_text(gain.value) + " differs from 1 / sqrt(reference_power) = " +
                      number_text(1.0 / std::sqrt(gain.reference_power)) + " by more than 1e-6");
  }
  for (const source_spec& item : spec.sources) {
    if (!item.from_entry) {
      throw input_error("realization " + item.realization_id + " has no entry, and the gain of the run is an entry's. "
                        "A run reads episode entries or a fixture, not both.");
    }
    if (item.entry_grid_gain != gain.value) {
      throw input_error("two gains in one run: " + number_text(gain.value) + " and " +
                        number_text(item.entry_grid_gain) + " (realization " + item.realization_id + ")");
    }
    if (!std::isfinite(item.entry_reference_power) || !(item.entry_reference_power > 0.0) ||
        !follows_rule(item.entry_grid_gain, item.entry_reference_power)) {
      throw input_error("the grid gain of realization " + item.realization_id +
                        " differs from 1 / sqrt(reference_power) of its entry by more than 1e-6");
    }
  }
}

// ---------------------------------------------------------------------------------------------- grid

model_grid::model_grid(unsigned planes_, unsigned symbols_, unsigned subcarriers_) :
  nof_planes(planes_),
  nof_symbols(symbols_),
  nof_subcarriers(subcarriers_),
  cells(static_cast<std::size_t>(planes_) * symbols_ * subcarriers_)
{
}

void model_grid::clear()
{
  std::fill(cells.begin(), cells.end(), cf32(0.0F, 0.0F));
}

void model_grid::put(unsigned plane, unsigned symbol, const cf32* row, unsigned count)
{
  if (plane >= nof_planes || symbol >= nof_symbols || count > nof_subcarriers) {
    throw state_error("a write leaves the grid");
  }
  cf32* cell = cells.data() + (static_cast<std::size_t>(plane) * nof_symbols + symbol) * nof_subcarriers;
  for (unsigned k = 0; k != count; ++k) {
    cell[k] = cf32(bf16_value(row[k].real()), bf16_value(row[k].imag()));
  }
}

void model_grid::get(unsigned plane, unsigned symbol, cf32* row, unsigned count) const
{
  if (plane >= nof_planes || symbol >= nof_symbols || count > nof_subcarriers) {
    throw state_error("a read leaves the grid");
  }
  const cf32* cell = cells.data() + (static_cast<std::size_t>(plane) * nof_symbols + symbol) * nof_subcarriers;
  std::copy(cell, cell + count, row);
}

void apply(const grid_shape& shape,
           const cf32*       x,
           const cf32*       h,
           const cf32*       w,
           double            gain,
           grid_sink&        sink,
           cf32*             prestore)
{
  if (shape.rx_ports == 0 || shape.tx_ports == 0 || shape.symbols == 0 || shape.subcarriers == 0) {
    throw input_error("the carrier grid has an empty axis");
  }
  if (!std::isfinite(gain) || !(gain > 0.0)) {
    throw input_error("the grid gain must be finite and positive");
  }
  if (sink.planes() < shape.rx_ports || sink.symbols() < shape.symbols || sink.subcarriers() < shape.subcarriers) {
    throw input_error("the receiver grid holds " + std::to_string(sink.planes()) + " planes of " +
                      std::to_string(sink.symbols()) + " by " + std::to_string(sink.subcarriers()) +
                      ", too small for " + std::to_string(shape.rx_ports) + " planes of " +
                      std::to_string(shape.symbols) + " by " + std::to_string(shape.subcarriers));
  }
  const std::size_t plane = shape.plane();
  std::vector<cf32> row(shape.subcarriers);
  for (unsigned rx = 0; rx != shape.rx_ports; ++rx) {
    const cf32* h_rx = h + static_cast<std::size_t>(rx) * shape.tx_ports * plane;
    const cf32* w_rx = w + static_cast<std::size_t>(rx) * plane;
    for (unsigned symbol = 0; symbol != shape.symbols; ++symbol) {
      const std::size_t base = static_cast<std::size_t>(symbol) * shape.subcarriers;
      for (unsigned k = 0; k != shape.subcarriers; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (unsigned tx = 0; tx != shape.tx_ports; ++tx) {
          const cf32   hv         = h_rx[tx * plane + base + k];
          const cf32   xv         = x[tx * plane + base + k];
          const double hr         = hv.real();
          const double hi         = hv.imag();
          const double xr         = xv.real();
          const double xi         = xv.imag();
          const double product_re = hr * xr - hi * xi;
          const double product_im = hr * xi + hi * xr;
          if (tx == 0) {
            re = product_re;
            im = product_im;
          } else {
            re += product_re;
            im += product_im;
          }
        }
        const cf32 wv = w_rx[base + k];
        re += static_cast<double>(wv.real());
        im += static_cast<double>(wv.imag());
        re *= gain;
        im *= gain;
        const float out_re = static_cast<float>(re);
        const float out_im = static_cast<float>(im);
        if (!std::isfinite(out_re) || !std::isfinite(out_im)) {
          throw input_error("channel application produced a nonfinite value at receive port " + std::to_string(rx) +
                            ", symbol " + std::to_string(symbol) + ", subcarrier " + std::to_string(k));
        }
        row[k] = cf32(out_re, out_im);
      }
      sink.put(rx, symbol, row.data(), shape.subcarriers);
      if (prestore != nullptr) {
        std::copy(row.begin(), row.end(), prestore + static_cast<std::size_t>(rx) * plane + base);
      }
    }
  }
}

std::string readback_sha256(const grid_sink& grid, const grid_shape& shape)
{
  sha256            hasher;
  std::vector<cf32> row(shape.subcarriers);
  for (unsigned rx = 0; rx != shape.rx_ports; ++rx) {
    for (unsigned symbol = 0; symbol != shape.symbols; ++symbol) {
      grid.get(rx, symbol, row.data(), shape.subcarriers);
      hasher.update(row.data(), row.size() * sizeof(cf32));
    }
  }
  return hasher.hex();
}

// ---------------------------------------------------------------------------------------------- owned grids

const char* to_string(grid_state state)
{
  switch (state) {
    case grid_state::free:
      return "free";
    case grid_state::filling:
      return "filling";
    case grid_state::ready:
      return "ready";
  }
  return "unknown";
}

o1_pool::o1_pool(std::vector<grid_sink*> grids_)
{
  if (grids_.empty()) {
    throw input_error("a grid pool needs at least one grid");
  }
  for (grid_sink* grid : grids_) {
    if (grid == nullptr) {
      throw input_error("a grid pool takes no null grid");
    }
    slot_t slot;
    slot.grid = grid;
    slots.push_back(slot);
  }
}

bool o1_pool::try_acquire(ticket& out)
{
  for (std::size_t index = 0; index != slots.size(); ++index) {
    slot_t& slot = slots[index];
    if (slot.state == grid_state::free) {
      slot.state = grid_state::filling;
      ++slot.generation;
      slot.descriptor = o1_descriptor();
      out.grid        = index;
      out.generation  = slot.generation;
      return true;
    }
  }
  return false;
}

o1_pool::slot_t& o1_pool::slot_for(const ticket& key, grid_state expected, const char* action)
{
  if (key.grid >= slots.size()) {
    throw state_error(std::string(action) + ": the pool has no grid " + std::to_string(key.grid));
  }
  slot_t& slot = slots[key.grid];
  if (key.generation != slot.generation) {
    throw state_error(std::string(action) + ": stale generation " + std::to_string(key.generation) + " of grid " +
                      std::to_string(key.grid) + ", which is at generation " + std::to_string(slot.generation));
  }
  if (slot.state != expected) {
    throw state_error(std::string(action) + ": grid " + std::to_string(key.grid) + " is " + to_string(slot.state) +
                      ", not " + to_string(expected));
  }
  return slot;
}

const o1_pool::slot_t& o1_pool::ready_slot(const ticket& key) const
{
  return const_cast<o1_pool*>(this)->slot_for(key, grid_state::ready, "read");
}

grid_sink& o1_pool::writer(const ticket& key)
{
  return *slot_for(key, grid_state::filling, "write").grid;
}

void o1_pool::commit(const ticket& key, o1_descriptor descriptor, evidence mode)
{
  slot_t& slot = slot_for(key, grid_state::filling, "commit");
  if (mode == evidence::audit) {
    descriptor.readback_sha256 = readback_sha256(*slot.grid, descriptor.shape);
  } else {
    descriptor.readback_sha256.clear(); // owned by its ticket: no grid is read back in a timed path
  }
  slot.descriptor            = std::move(descriptor);
  slot.state                 = grid_state::ready;
}

void o1_pool::abort(const ticket& key)
{
  slot_t& slot = slot_for(key, grid_state::filling, "abort");
  slot.grid->clear();
  slot.descriptor = o1_descriptor();
  slot.state      = grid_state::free;
}

o1_view o1_pool::view(const ticket& key, const std::string& grant_record_sha256) const
{
  const slot_t& slot = ready_slot(key);
  if (slot.descriptor.grant_record_sha256 != grant_record_sha256) {
    throw state_error("changed allocation: grid " + std::to_string(key.grid) + " holds the grant " +
                      slot.descriptor.grant_id + ", and the consumer decodes another grant record");
  }
  return o1_view(*this, key);
}

void o1_pool::release(const ticket& key)
{
  slot_for(key, grid_state::ready, "release").state = grid_state::free;
}

grid_state o1_pool::state(std::size_t grid) const
{
  return slots.at(grid).state;
}

std::uint64_t o1_pool::generation(std::size_t grid) const
{
  return slots.at(grid).generation;
}

const o1_descriptor& o1_view::descriptor() const
{
  return owner->ready_slot(id).descriptor;
}

void o1_view::get(unsigned plane, unsigned symbol, cf32* row, unsigned count) const
{
  const o1_pool::slot_t& slot = owner->ready_slot(id);
  const grid_shape&      size = slot.descriptor.shape;
  if (plane >= size.rx_ports || symbol >= size.symbols || count > size.subcarriers) {
    throw state_error("a read leaves the carrier grid of the opportunity");
  }
  slot.grid->get(plane, symbol, row, count);
}

std::string o1_view::digest() const
{
  const o1_pool::slot_t& slot = owner->ready_slot(id);
  return readback_sha256(*slot.grid, slot.descriptor.shape);
}

// ---------------------------------------------------------------------------------------------- session

struct session::loaded {
  struct source_t {
    const cf32* h             = nullptr;
    const cf32* w             = nullptr;
    std::size_t opportunities = 0;
  };
  struct bound_t {
    std::size_t source = 0;
    std::size_t grant  = 0;
  };
  std::map<std::string, std::unique_ptr<array_file>> files;
  std::map<std::string, std::string>                 hashes;
  std::vector<allocation>                            grants;
  std::map<std::string, std::size_t>                 grant_index;
  std::vector<source_t>                              sources;
  std::map<std::string, std::size_t>                 source_index;
  std::vector<bound_t>                               opportunities;

  /// Maps a file once and checks it against the hash that binds it.
  const array_file& open(const file_ref& ref, const std::string& role, std::vector<verified_file>& verified)
  {
    if (ref.path.empty()) {
      throw input_error(role + " names no file");
    }
    if (ref.sha256.empty()) {
      throw input_error(role + " is bound by no hash");
    }
    auto found = files.find(ref.path);
    if (found == files.end()) {
      std::unique_ptr<array_file> file(new array_file(ref.path));
      hashes[ref.path] = file->file_sha256();
      found            = files.emplace(ref.path, std::move(file)).first;
    }
    const std::string& measured = hashes[ref.path];
    if (measured != ref.sha256) {
      throw input_error(role + " has the hash " + measured + ", and its record binds " + ref.sha256 + " (" +
                        ref.path + ")");
    }
    verified_file entry;
    entry.role   = role;
    entry.path   = ref.path;
    entry.sha256 = measured;
    entry.bytes  = found->second->file_bytes();
    verified.push_back(entry);
    return *found->second;
  }
};

namespace {

bool all_finite(const cf32* values, std::size_t count)
{
  const float* parts = reinterpret_cast<const float*>(values);
  for (std::size_t i = 0; i != 2 * count; ++i) {
    if (!std::isfinite(parts[i])) {
      return false;
    }
  }
  return true;
}

} // namespace

session::session(const run_spec& spec_) : plan(spec_), data(new loaded)
{
  if (plan.run_id.empty()) {
    throw input_error("the run has no identifier");
  }
  const grid_shape size = shape();
  if (size.rx_ports == 0 || size.tx_ports == 0 || size.symbols == 0 || size.subcarriers == 0) {
    throw input_error("the carrier grid of the run has an empty axis");
  }
  check_gain(plan);
  if (plan.grants.empty() || plan.sources.empty() || plan.opportunities.empty()) {
    throw input_error("the run needs at least one grant, one channel realization and one opportunity");
  }
  const std::size_t plane = size.plane();
  if (plane > std::numeric_limits<std::uint32_t>::max()) {
    throw input_error("the carrier grid of the run is too large");
  }

  // ---- Grants: the transmit grid bound by its hash and the occupied elements it transmits on. ----
  for (const grant_spec& grant : plan.grants) {
    if (grant.grant_id.empty() || grant.record_sha256.empty()) {
      throw input_error("a grant needs its identifier and the hash of its record");
    }
    if (data->grant_index.count(grant.grant_id) != 0) {
      throw input_error("the grant " + grant.grant_id + " is listed twice");
    }
    if (grant.x.sha256.empty()) {
      throw input_error("the grant " + grant.grant_id + " has no bound transmit grid: no hash binds it");
    }
    const array_file&              x_file = data->open(grant.x, "the transmit grid of grant " + grant.grant_id, verified);
    const std::vector<std::size_t> wanted = {size.tx_ports, size.symbols, size.subcarriers};
    if (x_file.descr() != "<c8" || x_file.shape() != wanted) {
      throw input_error("changed allocation: the transmit grid of grant " + grant.grant_id + " is " +
                        x_file.descr() + " " + shape_text(x_file.shape()) + ", and the carrier grid of the run is " +
                        shape_text(wanted));
    }
    allocation item;
    item.grant_id      = grant.grant_id;
    item.record_sha256 = grant.record_sha256;
    item.x_sha256      = grant.x.sha256;
    item.x             = x_file.complex64();
    if (!all_finite(item.x, x_file.count())) {
      throw input_error("the transmit grid of grant " + grant.grant_id + " holds a nonfinite value");
    }
    if (grant.active.empty()) {
      throw input_error("the grant " + grant.grant_id + " lists no occupied resource elements");
    }
    for (const file_ref& ref : grant.active) {
      const array_file& index = data->open(ref, "the occupied elements of grant " + grant.grant_id, verified);
      if (index.descr() != "<i4" || index.shape().size() != 2 || index.shape()[1] != 2) {
        throw input_error(ref.path + " is not an int32 array of (symbol, subcarrier) rows");
      }
      const std::int32_t* rows = index.int32();
      for (std::size_t row = 0; row != index.shape()[0]; ++row) {
        const std::int32_t symbol = rows[2 * row];
        const std::int32_t k      = rows[2 * row + 1];
        if (symbol < 0 || k < 0 || static_cast<unsigned>(symbol) >= size.symbols ||
            static_cast<unsigned>(k) >= size.subcarriers) {
          throw input_error("changed allocation: grant " + grant.grant_id + " occupies symbol " +
                            std::to_string(symbol) + ", subcarrier " + std::to_string(k) +
                            ", outside the carrier grid of the run");
        }
        item.active.push_back(static_cast<std::uint32_t>(symbol) * size.subcarriers + static_cast<std::uint32_t>(k));
      }
    }
    std::sort(item.active.begin(), item.active.end());
    if (item.active.empty() || std::adjacent_find(item.active.begin(), item.active.end()) != item.active.end()) {
      throw input_error("the occupied elements of grant " + grant.grant_id + " are empty or list one element twice");
    }
    item.active_sha256 = sha256_hex(item.active.data(), item.active.size() * sizeof(std::uint32_t));
    // The grid must transmit on exactly the elements the grant occupies.
    std::size_t next = 0;
    for (std::size_t flat = 0; flat != plane; ++flat) {
      bool nonzero = false;
      for (unsigned tx = 0; tx != size.tx_ports; ++tx) {
        nonzero = nonzero || item.x[tx * plane + flat] != cf32(0.0F, 0.0F);
      }
      const bool occupied = next != item.active.size() && item.active[next] == flat;
      next += occupied ? 1 : 0;
      if (nonzero != occupied) {
        throw input_error("the transmit grid of grant " + grant.grant_id + (nonzero ? " transmits" : " is empty") +
                          " at symbol " + std::to_string(flat / size.subcarriers) + ", subcarrier " +
                          std::to_string(flat % size.subcarriers) + ", which the grant " +
                          (nonzero ? "does not occupy" : "occupies"));
      }
    }
    data->grant_index[grant.grant_id] = data->grants.size();
    data->grants.push_back(std::move(item));
  }

  // ---- Channel realizations: H and W of every opportunity, on the carrier grid of the run. ----
  for (const source_spec& source : plan.sources) {
    if (source.realization_id.empty()) {
      throw input_error("a channel realization needs an identifier");
    }
    if (data->source_index.count(source.realization_id) != 0) {
      throw input_error("the realization " + source.realization_id + " is listed twice");
    }
    const array_file& h_file = data->open(source.h, "the channel of realization " + source.realization_id, verified);
    const array_file& w_file = data->open(source.w, "the noise of realization " + source.realization_id, verified);
    const std::vector<std::size_t>& h_shape = h_file.shape();
    const std::vector<std::size_t>& w_shape = w_file.shape();
    if (h_file.descr() != "<c8" || h_shape.size() != 5 || h_shape[0] == 0 || h_shape[1] != size.rx_ports ||
        h_shape[2] != size.tx_ports || h_shape[3] != size.symbols || h_shape[4] != size.subcarriers) {
      throw input_error("the channel of realization " + source.realization_id + " is " + h_file.descr() + " " +
                        shape_text(h_shape) + ", expected complex64 [opportunity, " + std::to_string(size.rx_ports) +
                        ", " + std::to_string(size.tx_ports) + ", " + std::to_string(size.symbols) + ", " +
                        std::to_string(size.subcarriers) + "]");
    }
    if (w_file.descr() != "<c8" || w_shape.size() != 4 || w_shape[0] != h_shape[0] || w_shape[1] != size.rx_ports ||
        w_shape[2] != size.symbols || w_shape[3] != size.subcarriers) {
      throw input_error("the noise of realization " + source.realization_id + " is " + w_file.descr() + " " +
                        shape_text(w_shape) + ", expected complex64 [" + std::to_string(h_shape[0]) + ", " +
                        std::to_string(size.rx_ports) + ", " + std::to_string(size.symbols) + ", " +
                        std::to_string(size.subcarriers) + "]");
    }
    loaded::source_t item;
    item.h             = h_file.complex64();
    item.w             = w_file.complex64();
    item.opportunities = h_shape[0];
    if (!all_finite(item.h, h_file.count()) || !all_finite(item.w, w_file.count())) {
      throw input_error("the channel or the noise of realization " + source.realization_id +
                        " holds a nonfinite value");
    }
    data->source_index[source.realization_id] = data->sources.size();
    data->sources.push_back(item);
  }

  // ---- Opportunities: each one names its realization, its index there and its bound grant. ----
  std::map<std::string, bool> seen;
  for (const opportunity_spec& opportunity : plan.opportunities) {
    if (opportunity.opportunity_id.empty()) {
      throw input_error("an opportunity needs an identifier");
    }
    if (seen.count(opportunity.opportunity_id) != 0) {
      throw input_error("the opportunity " + opportunity.opportunity_id + " is listed twice");
    }
    seen[opportunity.opportunity_id] = true;
    const auto source                = data->source_index.find(opportunity.realization_id);
    if (source == data->source_index.end()) {
      throw input_error("the opportunity " + opportunity.opportunity_id + " names the realization " +
                        opportunity.realization_id + ", which the run does not hold");
    }
    if (opportunity.index >= data->sources[source->second].opportunities) {
      throw input_error("the opportunity " + opportunity.opportunity_id + " reads index " +
                        std::to_string(opportunity.index) + " of a realization with " +
                        std::to_string(data->sources[source->second].opportunities) + " opportunities");
    }
    if (opportunity.grant_id.empty()) {
      throw input_error("the opportunity " + opportunity.opportunity_id + " has no bound grid");
    }
    const auto grant = data->grant_index.find(opportunity.grant_id);
    if (grant == data->grant_index.end()) {
      throw input_error("the opportunity " + opportunity.opportunity_id + " names the grant " + opportunity.grant_id +
                        ", which the run does not bind");
    }
    loaded::bound_t item;
    item.source = source->second;
    item.grant  = grant->second;
    data->opportunities.push_back(item);
  }
}

session::~session() = default;

grid_shape session::shape() const
{
  grid_shape size;
  size.rx_ports    = plan.rx_ports;
  size.tx_ports    = plan.tx_ports;
  size.symbols     = plan.symbols;
  size.subcarriers = plan.subcarriers;
  return size;
}

const allocation& session::scheduled(std::size_t opportunity) const
{
  return data->grants.at(data->opportunities.at(opportunity).grant);
}

const allocation& session::bound(const std::string& grant_id) const
{
  const auto found = data->grant_index.find(grant_id);
  if (found == data->grant_index.end()) {
    throw input_error("changed allocation: the run does not bind the grant " + grant_id);
  }
  return data->grants[found->second];
}

double session::unit_power(std::size_t opportunity) const
{
  const loaded::bound_t&  item   = data->opportunities.at(opportunity);
  const loaded::source_t& source = data->sources[item.source];
  const allocation&       grant  = data->grants[item.grant];
  const grid_shape        size   = shape();
  const std::size_t       plane  = size.plane();
  const cf32* h = source.h + plan.opportunities[opportunity].index * size.rx_ports * size.tx_ports * plane;
  double      total = 0.0;
  for (std::size_t port = 0; port != static_cast<std::size_t>(size.rx_ports) * size.tx_ports; ++port) {
    double partial = 0.0;
    for (std::uint32_t flat : grant.active) {
      const cf32 value = h[port * plane + flat];
      partial += static_cast<double>(value.real()) * value.real() + static_cast<double>(value.imag()) * value.imag();
    }
    total += partial;
  }
  const double count = static_cast<double>(grant.active.size()) * size.rx_ports * size.tx_ports;
  return plan.gain.value * plan.gain.value * total / count;
}

void session::fill(std::size_t opportunity, grid_sink& sink, cf32* prestore) const
{
  fill(opportunity, scheduled(opportunity).grant_id, sink, prestore);
}

void session::fill(std::size_t opportunity, const std::string& grant_id, grid_sink& sink, cf32* prestore) const
{
  const loaded::bound_t&  item   = data->opportunities.at(opportunity);
  const loaded::source_t& source = data->sources[item.source];
  const allocation&       grant  = bound(grant_id);
  const grid_shape        size   = shape();
  const std::size_t       plane  = size.plane();
  const std::size_t       index  = plan.opportunities[opportunity].index;
  apply(size,
        grant.x,
        source.h + index * size.rx_ports * size.tx_ports * plane,
        source.w + index * size.rx_ports * plane,
        plan.gain.value,
        sink,
        prestore);
}

o1_descriptor session::describe(std::size_t opportunity, const allocation& grant, const grid_sink& sink) const
{
  const opportunity_spec& item = plan.opportunities.at(opportunity);
  o1_descriptor           descriptor;
  descriptor.run_id              = plan.run_id;
  descriptor.opportunity_id      = item.opportunity_id;
  descriptor.grant_id            = grant.grant_id;
  descriptor.grant_record_sha256 = grant.record_sha256;
  descriptor.active_sha256       = grant.active_sha256;
  descriptor.active_re           = grant.active.size();
  descriptor.sfn                 = item.sfn;
  descriptor.slot                = item.slot;
  descriptor.shape               = shape();
  descriptor.grid_gain           = plan.gain.value;
  descriptor.storage             = sink.storage();
  return descriptor;
}

bool session::deliver(std::size_t opportunity, o1_pool& pool, ticket& out, cf32* prestore) const
{
  const allocation& grant = scheduled(opportunity);
  ticket            key;
  if (!pool.try_acquire(key)) {
    return false;
  }
  try {
    grid_sink& sink = pool.writer(key);
    sink.clear();
    fill(opportunity, sink, prestore);
    pool.commit(key, describe(opportunity, grant, sink));
  } catch (...) {
    pool.abort(key);
    throw;
  }
  out = key;
  return true;
}

storage_stats measure(const o1_view& view, const allocation& grant, const cf32* prestore)
{
  const grid_shape  size  = view.descriptor().shape;
  const std::size_t plane = size.plane();
  storage_stats     stats;
  std::vector<cf32> stored(plane);
  double            error_total   = 0.0;
  double            written_total = 0.0;
  for (unsigned rx = 0; rx != size.rx_ports; ++rx) {
    for (unsigned symbol = 0; symbol != size.symbols; ++symbol) {
      view.get(rx, symbol, stored.data() + static_cast<std::size_t>(symbol) * size.subcarriers, size.subcarriers);
    }
    const cf32* written = prestore + static_cast<std::size_t>(rx) * plane;
    plane_power power;
    for (std::size_t flat = 0; flat != plane; ++flat) {
      const float parts[2][2] = {{stored[flat].real(), written[flat].real()}, {stored[flat].imag(), written[flat].imag()}};
      double      magnitude   = 0.0;
      for (const float* part : parts) {
        const double kept  = part[0];
        const double given = part[1];
        const double error = kept - given;
        magnitude += kept * kept;
        error_total += error * error;
        written_total += given * given;
        stats.components += 1;
        stats.components_changed += (part[0] != part[1]) ? 1 : 0;
        stats.model_mismatches += (part[0] != bf16_value(part[1])) ? 1 : 0;
        if (given != 0.0) {
          stats.max_relative_error = std::max(stats.max_relative_error, std::fabs(error) / std::fabs(given));
        }
      }
      power.all += magnitude;
    }
    for (std::uint32_t flat : grant.active) {
      power.active += static_cast<double>(stored[flat].real()) * stored[flat].real() +
                      static_cast<double>(stored[flat].imag()) * stored[flat].imag();
    }
    power.all /= static_cast<double>(plane);
    power.active /= static_cast<double>(grant.active.size());
    stats.planes.push_back(power);
  }
  const double elements = static_cast<double>(plane) * size.rx_ports;
  stats.error_power     = error_total / elements;
  stats.written_power   = written_total / elements;
  return stats;
}

} // namespace isac_replay
