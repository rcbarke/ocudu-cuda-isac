// Unit cases of the channel application core (Sprint 2.3, selector grid_apply).
//
// The target links replay_core and nothing else: no OCUDU library, no test framework. The grid is
// the storage model of the core, and the fixtures are arrays written by this file.
//
//     replay_core_test            run every case
//     replay_core_test NAME       run one case (CTest registers each)
//     replay_core_test --list     print the case names
//     replay_core_test --digest SEED RX TX SYMBOLS SUBCARRIERS GAIN
//                                 apply a generated fixture and print the digests of the values
//                                 handed to the grid and of the stored grid, for a reference in
//                                 another language to reproduce
//
// ASCII only.

#include "replay_core.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace isac_replay;

namespace {

int failures = 0;

void failed(const char* file, int line, const std::string& what)
{
  ++failures;
  std::cerr << file << ":" << line << ": " << what << std::endl;
}

#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      failed(__FILE__, __LINE__, "expected " #condition);                                                              \
    }                                                                                                                  \
  } while (false)

// The statement must throw the given type with a message that holds the given text.
#define CHECK_THROWS(statement, type, text)                                                                            \
  do {                                                                                                                 \
    bool thrown_ = false;                                                                                              \
    try {                                                                                                              \
      statement;                                                                                                       \
    } catch (const type& error_) {                                                                                     \
      thrown_ = true;                                                                                                  \
      if (std::string(error_.what()).find(text) == std::string::npos) {                                                \
        failed(__FILE__, __LINE__, std::string("message '") + error_.what() + "' lacks '" + (text) + "'");             \
      }                                                                                                                \
    } catch (const std::exception& error_) {                                                                           \
      thrown_ = true;                                                                                                  \
      failed(__FILE__, __LINE__, std::string("another exception: ") + error_.what());                                  \
    }                                                                                                                  \
    if (!thrown_) {                                                                                                    \
      failed(__FILE__, __LINE__, "no exception from " #statement);                                                     \
    }                                                                                                                  \
  } while (false)

bool same_bits(const cf32* a, const cf32* b, std::size_t count)
{
  return std::memcmp(a, b, count * sizeof(cf32)) == 0;
}

std::vector<cf32> stored_values(const grid_sink& grid, const grid_shape& shape)
{
  std::vector<cf32> values(static_cast<std::size_t>(shape.rx_ports) * shape.plane());
  for (unsigned rx = 0; rx != shape.rx_ports; ++rx) {
    for (unsigned symbol = 0; symbol != shape.symbols; ++symbol) {
      grid.get(rx, symbol, values.data() + (static_cast<std::size_t>(rx) * shape.symbols + symbol) * shape.subcarriers,
               shape.subcarriers);
    }
  }
  return values;
}

grid_shape make_shape(unsigned rx, unsigned tx, unsigned symbols, unsigned subcarriers)
{
  grid_shape shape;
  shape.rx_ports    = rx;
  shape.tx_ports    = tx;
  shape.symbols     = symbols;
  shape.subcarriers = subcarriers;
  return shape;
}

// A small deterministic generator: values in [-1, 1) with 24 significant bits.
struct sequence {
  std::uint32_t state;
  explicit sequence(std::uint32_t seed) : state(seed) {}
  float next()
  {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(static_cast<double>(state >> 8) / 8388608.0 - 1.0);
  }
  cf32 next_complex()
  {
    const float re = next();
    const float im = next();
    return cf32(re, im);
  }
};

// A double of a grid whose storage differs from the documented rule: it cuts the low bits off.
class truncating_grid final : public grid_sink
{
public:
  truncating_grid(unsigned planes_, unsigned symbols_, unsigned subcarriers_) :
    nof_planes(planes_), nof_symbols(symbols_), nof_subcarriers(subcarriers_), cells(planes_ * symbols_ * subcarriers_)
  {
  }
  unsigned    planes() const override { return nof_planes; }
  unsigned    symbols() const override { return nof_symbols; }
  unsigned    subcarriers() const override { return nof_subcarriers; }
  void        clear() override { std::fill(cells.begin(), cells.end(), cf32()); }
  std::string storage() const override { return "truncated"; }
  void        put(unsigned plane, unsigned symbol, const cf32* row, unsigned count) override
  {
    for (unsigned k = 0; k != count; ++k) {
      float         parts[2] = {row[k].real(), row[k].imag()};
      std::uint32_t raw[2];
      std::memcpy(raw, parts, sizeof(raw));
      raw[0] &= 0xffff0000U;
      raw[1] &= 0xffff0000U;
      std::memcpy(parts, raw, sizeof(raw));
      cells[(plane * nof_symbols + symbol) * nof_subcarriers + k] = cf32(parts[0], parts[1]);
    }
  }
  void get(unsigned plane, unsigned symbol, cf32* row, unsigned count) const override
  {
    std::copy_n(cells.begin() + (plane * nof_symbols + symbol) * nof_subcarriers, count, row);
  }

private:
  unsigned          nof_planes, nof_symbols, nof_subcarriers;
  std::vector<cf32> cells;
};

// ---- Files for the session cases. ----
struct workspace {
  std::string directory;
  workspace()
  {
    const char* base = std::getenv("TMPDIR");
    std::string name = std::string(base != nullptr && *base != 0 ? base : "/tmp") + "/replay-core-XXXXXX";
    std::vector<char> buffer(name.begin(), name.end());
    buffer.push_back(0);
    if (::mkdtemp(buffer.data()) == nullptr) {
      throw std::runtime_error("cannot create a scratch directory");
    }
    directory = buffer.data();
  }
  ~workspace()
  {
    for (const std::string& name : files) {
      ::unlink(name.c_str());
    }
    ::rmdir(directory.c_str());
  }
  std::string write(const std::string& name, const std::string& descr, const std::vector<std::size_t>& shape,
                    const void* data, std::size_t bytes)
  {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (";
    for (std::size_t dimension : shape) {
      header += std::to_string(dimension) + ",";
    }
    header += "), }";
    while ((10 + header.size() + 1) % 64 != 0) {
      header += " ";
    }
    header += "\n";
    const std::string path = directory + "/" + name;
    std::ofstream     stream(path, std::ios::binary);
    const char        prefix[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
    const char        length[2] = {static_cast<char>(header.size() & 0xff), static_cast<char>(header.size() >> 8)};
    stream.write(prefix, 8);
    stream.write(length, 2);
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    stream.close();
    files.push_back(path);
    return path;
  }
  file_ref complex_file(const std::string& name, const std::vector<std::size_t>& shape, const std::vector<cf32>& values)
  {
    file_ref ref;
    ref.path   = write(name, "<c8", shape, values.data(), values.size() * sizeof(cf32));
    ref.sha256 = array_file(ref.path).file_sha256();
    return ref;
  }
  file_ref index_file(const std::string& name, const std::vector<std::pair<int, int>>& rows)
  {
    std::vector<std::int32_t> flat;
    for (const auto& row : rows) {
      flat.push_back(row.first);
      flat.push_back(row.second);
    }
    file_ref ref;
    ref.path   = write(name, "<i4", {rows.size(), 2}, flat.data(), flat.size() * sizeof(std::int32_t));
    ref.sha256 = array_file(ref.path).file_sha256();
    return ref;
  }
  std::vector<std::string> files;
};

// A two-grant fixture on a carrier grid of 2 symbols by 6 subcarriers with 2 receive ports. Grant
// "wide" occupies subcarriers 0 to 3 of both symbols, grant "narrow" subcarriers 4 and 5 of symbol 1.
// H holds its own coordinates: real part 1 + receive port, imaginary part (symbol * 8 + subcarrier) / 64.
// The realization has two opportunities: the second one holds minus half the channel of the first
// and other noise, and is scheduled with the narrow grant.
struct fixture {
  workspace         files;
  grid_shape        shape = make_shape(2, 1, 2, 6);
  std::vector<cf32> x_wide, x_narrow, h, w;
  run_spec          spec;

  fixture()
  {
    const std::size_t plane = shape.plane();
    x_wide.assign(plane, cf32());
    x_narrow.assign(plane, cf32());
    std::vector<std::pair<int, int>> wide, narrow;
    for (int symbol = 0; symbol != 2; ++symbol) {
      for (int k = 0; k != 4; ++k) {
        x_wide[symbol * 6 + k] = cf32(0.5F + 0.25F * static_cast<float>(k), -0.75F + 0.5F * static_cast<float>(symbol));
        wide.emplace_back(symbol, k);
      }
    }
    x_narrow[6 + 4] = cf32(1.0F, -1.0F);
    x_narrow[6 + 5] = cf32(-0.5F, 0.25F);
    narrow          = {{1, 4}, {1, 5}};
    h.assign(4 * plane, cf32());
    for (unsigned rx = 0; rx != 2; ++rx) {
      for (unsigned flat = 0; flat != plane; ++flat) {
        h[rx * plane + flat] = cf32(1.0F + static_cast<float>(rx),
                                    static_cast<float>((flat / 6) * 8 + flat % 6) / 64.0F);
        h[(2 + rx) * plane + flat] = -0.5F * h[rx * plane + flat];
      }
    }
    sequence noise(7);
    w.assign(4 * plane, cf32());
    for (cf32& value : w) {
      value = noise.next_complex() * 0.125F;
    }

    spec.run_id      = "unit";
    spec.rx_ports    = 2;
    spec.tx_ports    = 1;
    spec.symbols     = 2;
    spec.subcarriers = 6;
    spec.gain.present = true;
    spec.gain.value   = 2.0;
    spec.gain.source  = "override";
    spec.gain.reason  = "unit fixture without an entry";
    grant_spec first;
    first.grant_id      = "wide";
    first.record_sha256 = "record-wide";
    first.x             = files.complex_file("x_wide.npy", {1, 2, 6}, x_wide);
    first.active        = {files.index_file("wide_index.npy", wide)};
    grant_spec second;
    second.grant_id      = "narrow";
    second.record_sha256 = "record-narrow";
    second.x             = files.complex_file("x_narrow.npy", {1, 2, 6}, x_narrow);
    second.active        = {files.index_file("narrow_index.npy", narrow)};
    spec.grants          = {first, second};
    source_spec source;
    source.realization_id = "fx-0001";
    source.h              = files.complex_file("h.npy", {2, 2, 1, 2, 6}, h);
    source.w              = files.complex_file("w.npy", {2, 2, 2, 6}, w);
    spec.sources          = {source};
    opportunity_spec opportunity;
    opportunity.opportunity_id = "fx-0001.0000";
    opportunity.realization_id = "fx-0001";
    opportunity.index          = 0;
    opportunity.grant_id       = "wide";
    opportunity.slot           = 7;
    opportunity_spec later     = opportunity;
    later.opportunity_id       = "fx-0001.0001";
    later.index                = 1;
    later.grant_id             = "narrow";
    later.slot                 = 8;
    spec.opportunities         = {opportunity, later};
  }
};

// ---------------------------------------------------------------------------------------------- cases

void case_sha256()
{
  CHECK(sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const std::string two_blocks = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  CHECK(sha256_hex(two_blocks.data(), two_blocks.size()) ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  const std::string million(1000000, 'a');
  CHECK(sha256_hex(million.data(), million.size()) ==
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  // Any split of the input gives the same digest, also at the block boundaries.
  for (std::size_t first : {1U, 55U, 56U, 63U, 64U, 65U, 119U, 120U, 128U, 999U}) {
    sha256 hasher;
    hasher.update(million.data(), first);
    hasher.update(million.data() + first, 7);
    hasher.update(million.data() + first + 7, million.size() - first - 7);
    CHECK(hasher.hex() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  }
  for (std::size_t size : {55U, 56U, 57U, 63U, 64U, 65U}) {
    sha256 whole, parts;
    whole.update(million.data(), size);
    for (std::size_t i = 0; i != size; ++i) {
      parts.update(million.data() + i, 1);
    }
    CHECK(whole.hex() == parts.hex());
  }
}

void case_storage()
{
  // Values with at most eight significant bits are kept.
  for (float value : {0.0F, 1.0F, -1.0F, 0.31640625F, -0.94921875F, 6.5F, 14.0F, 28.0F, 1.0F + 1.0F / 128.0F}) {
    CHECK(bf16_value(value) == value);
  }
  // Nearest, ties to even: the grid keeps 7 fraction bits.
  const float step = 1.0F / 128.0F;
  CHECK(bf16_value(1.0F + 0.5F * step) == 1.0F);                 // tie, the even neighbour is below
  CHECK(bf16_value(1.0F + 1.5F * step) == 1.0F + 2.0F * step);   // tie, the even neighbour is above
  CHECK(bf16_value(1.0F + 0.5F * step + 1e-6F) == 1.0F + step);  // just above the tie
  CHECK(bf16_value(1.0F + 0.5F * step - 1e-6F) == 1.0F);         // just below the tie
  CHECK(bf16_value(-(1.0F + 1.5F * step)) == -(1.0F + 2.0F * step));
  CHECK(bf16_value(2.0F - 0.25F * step) == 2.0F);                // rounds up into the next binade
  // The error stays within 2^-8 of the magnitude at any scale, far below the 16-bit float range as well.
  sequence values(3);
  for (double scale : {1.0, 1e-2, 3.4e-4, 2.6e-6, 7.8e-3, 2942.4}) {
    for (int i = 0; i != 2000; ++i) {
      const float value = static_cast<float>(scale * values.next());
      CHECK(std::fabs(bf16_value(value) - value) <= std::ldexp(std::fabs(value), -8));
    }
  }
}

// The two-transmit, two-symbol vector of the Sprint 0.3 replay fixture and its four answers.
void case_s03_vector()
{
  const grid_shape shape = make_shape(2, 2, 2, 1);
  const cf32       x[]   = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
  const cf32       h[]   = {{1, 0}, {1, 0}, {0, 1}, {0, -1}, {0.5F, 0}, {1, 0}, {2, 0}, {3, 0}};
  const cf32       w[]   = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
  const cf32       answers[] = {{1, 3}, {2, -4}, {6.5F, 0}, {14, 0}};
  for (double gain : {1.0, 2.0}) {
    model_grid        grid(2, 2, 1);
    std::vector<cf32> written(4);
    apply(shape, x, h, w, gain, grid, written.data());
    const std::vector<cf32> kept = stored_values(grid, shape);
    for (unsigned i = 0; i != 4; ++i) {
      const cf32 expected(static_cast<float>(gain) * answers[i].real(), static_cast<float>(gain) * answers[i].imag());
      CHECK(written[i] == expected);
      CHECK(kept[i] == expected);
    }
  }
}

void case_unity()
{
  // SISO: a unity channel with zero noise and g = 1 returns X bit for bit, zeros included.
  const grid_shape  siso = make_shape(1, 1, 3, 8);
  std::vector<cf32> x(siso.plane());
  sequence          values(11);
  for (std::size_t i = 0; i != x.size(); ++i) {
    x[i] = (i % 5 == 4) ? cf32() : cf32(bf16_value(values.next()), bf16_value(values.next()));
  }
  const std::vector<cf32> unity(siso.plane(), cf32(1.0F, 0.0F));
  const std::vector<cf32> silence(siso.plane());
  model_grid              grid(1, 3, 8);
  std::vector<cf32>       written(siso.plane());
  apply(siso, x.data(), unity.data(), silence.data(), 1.0, grid, written.data());
  CHECK(same_bits(written.data(), x.data(), x.size()));
  CHECK(same_bits(stored_values(grid, siso).data(), x.data(), x.size()));

  // A square identity channel: receive plane i holds transmit port i bit for bit.
  const grid_shape  square = make_shape(2, 2, 3, 8);
  std::vector<cf32> x2(2 * square.plane());
  for (cf32& value : x2) {
    value = cf32(bf16_value(values.next()), bf16_value(values.next()));
  }
  std::vector<cf32> identity(4 * square.plane());
  for (std::size_t flat = 0; flat != square.plane(); ++flat) {
    identity[0 * square.plane() + flat] = cf32(1.0F, 0.0F); // rx 0, tx 0
    identity[3 * square.plane() + flat] = cf32(1.0F, 0.0F); // rx 1, tx 1
  }
  const std::vector<cf32> silence2(2 * square.plane());
  model_grid              grid2(2, 3, 8);
  apply(square, x2.data(), identity.data(), silence2.data(), 1.0, grid2);
  CHECK(same_bits(stored_values(grid2, square).data(), x2.data(), x2.size()));
}

void case_zero_channel()
{
  // Zero H gives g W for any X, with the gain on the noise.
  const grid_shape  shape = make_shape(2, 2, 2, 5);
  sequence          values(23);
  std::vector<cf32> x(2 * shape.plane()), w(2 * shape.plane());
  for (cf32& value : x) {
    value = values.next_complex();
  }
  for (cf32& value : w) {
    value = values.next_complex() * 1.0747184e-05F;
  }
  const std::vector<cf32> nothing(4 * shape.plane());
  for (double gain : {1.0, 2.0, 2942.4243526773034}) {
    model_grid        grid(2, 2, 5);
    std::vector<cf32> written(w.size());
    apply(shape, x.data(), nothing.data(), w.data(), gain, grid, written.data());
    const std::vector<cf32> kept = stored_values(grid, shape);
    for (std::size_t i = 0; i != w.size(); ++i) {
      const cf32 expected(static_cast<float>(gain * static_cast<double>(w[i].real())),
                          static_cast<float>(gain * static_cast<double>(w[i].imag())));
      CHECK(written[i] == expected);
      CHECK(kept[i] == cf32(bf16_value(expected.real()), bf16_value(expected.imag())));
    }
  }
}

void case_cancellation()
{
  // Opposite transmit terms cancel and leave the noise.
  const grid_shape  shape = make_shape(1, 2, 2, 4);
  sequence          values(31);
  std::vector<cf32> x(2 * shape.plane()), h(2 * shape.plane()), w(shape.plane());
  for (std::size_t flat = 0; flat != shape.plane(); ++flat) {
    x[flat] = x[shape.plane() + flat] = values.next_complex();
    h[flat]                           = values.next_complex();
    h[shape.plane() + flat]           = -h[flat];
    w[flat]                           = values.next_complex() * 0.01F;
  }
  model_grid        grid(1, 2, 4);
  std::vector<cf32> written(shape.plane());
  apply(shape, x.data(), h.data(), w.data(), 1.0, grid, written.data());
  CHECK(same_bits(written.data(), w.data(), w.size()));
}

void case_permutation()
{
  // A permutation of the receive planes changes the delivered grid and its digest.
  const grid_shape  shape = make_shape(4, 1, 2, 6);
  sequence          values(41);
  std::vector<cf32> x(shape.plane()), h(4 * shape.plane()), w(4 * shape.plane());
  for (cf32& value : x) {
    value = values.next_complex();
  }
  for (cf32& value : h) {
    value = values.next_complex();
  }
  model_grid ordered(4, 2, 6), permuted(4, 2, 6);
  apply(shape, x.data(), h.data(), w.data(), 1.0, ordered);
  std::vector<cf32> swapped = h;
  std::swap_ranges(swapped.begin(), swapped.begin() + static_cast<std::ptrdiff_t>(shape.plane()),
                   swapped.begin() + static_cast<std::ptrdiff_t>(2 * shape.plane()));
  apply(shape, x.data(), swapped.data(), w.data(), 1.0, permuted);
  const std::vector<cf32> a = stored_values(ordered, shape);
  const std::vector<cf32> b = stored_values(permuted, shape);
  CHECK(!same_bits(a.data(), b.data(), a.size()));
  CHECK(readback_sha256(ordered, shape) != readback_sha256(permuted, shape));
  // Plane 0 of the permuted grid is plane 2 of the ordered one, and plane 1 is untouched.
  CHECK(same_bits(b.data(), a.data() + 2 * shape.plane(), shape.plane()));
  CHECK(same_bits(b.data() + shape.plane(), a.data() + shape.plane(), shape.plane()));
  // The digest covers every plane: one changed element of the last plane changes it.
  const std::string before = readback_sha256(ordered, shape);
  cf32              row[6];
  ordered.get(3, 1, row, 6);
  row[5] = cf32(row[5].real() + 1.0F, row[5].imag());
  ordered.put(3, 1, row, 6);
  CHECK(readback_sha256(ordered, shape) != before);
}

void case_unit_scale()
{
  // A physical channel with the gain: the values handed to the grid agree with an independent
  // complex-double reference within the unit-scale tolerance, which applies after the gain, and
  // the stored values are the reference rounded to the storage format.
  const grid_shape  shape = make_shape(4, 2, 3, 16);
  const double      gain  = 2942.4243526773034;
  sequence          values(53);
  std::vector<cf32> x(2 * shape.plane()), h(8 * shape.plane()), w(4 * shape.plane());
  for (cf32& value : x) {
    value = values.next_complex();
  }
  for (cf32& value : h) {
    value = values.next_complex() * 3.4e-4F;
  }
  for (cf32& value : w) {
    value = values.next_complex() * 1.07e-05F;
  }
  model_grid        grid(4, 3, 16);
  std::vector<cf32> written(w.size());
  apply(shape, x.data(), h.data(), w.data(), gain, grid, written.data());
  const std::vector<cf32> kept  = stored_values(grid, shape);
  double                  worst = 0.0;
  for (unsigned rx = 0; rx != 4; ++rx) {
    for (std::size_t flat = 0; flat != shape.plane(); ++flat) {
      std::complex<double> sum(0.0, 0.0);
      for (unsigned tx = 0; tx != 2; ++tx) {
        sum += std::complex<double>(h[(rx * 2 + tx) * shape.plane() + flat]) *
               std::complex<double>(x[tx * shape.plane() + flat]);
      }
      const std::complex<double> reference = gain * (sum + std::complex<double>(w[rx * shape.plane() + flat]));
      const cf32                 value     = written[rx * shape.plane() + flat];
      const double               error     = std::abs(std::complex<double>(value) - reference);
      CHECK(error <= 1e-6 + 1e-5 * std::abs(reference));
      worst = std::max(worst, error / std::abs(reference));
      // Double precision throughout: the value is the reference converted to single precision, bit for bit.
      CHECK(value == cf32(static_cast<float>(reference.real()), static_cast<float>(reference.imag())));
      const cf32 rounded(bf16_value(static_cast<float>(reference.real())), bf16_value(static_cast<float>(reference.imag())));
      CHECK(kept[rx * shape.plane() + flat] == rounded);
    }
  }
  CHECK(worst < 1e-6); // single-precision conversion only
}

// IEEE binary16, the format the pinned CUDA receiver keeps its estimates in: step of a value.
double half_step(double value)
{
  const double magnitude = std::fabs(value);
  if (magnitude < std::ldexp(1.0, -14)) {
    return std::ldexp(1.0, -24); // subnormal range: a fixed step
  }
  return std::ldexp(1.0, static_cast<int>(std::floor(std::log2(magnitude))) - 10);
}

// The reason for the gain. The stored channel is physical: the return of the reference pair has
// an amplitude of 2.6e-06 and the noise a sigma of 1.07e-05. Both lie below the normal range of
// a 16-bit float, which ends at 6.1e-05, so the return would keep about five significant bits
// in the receiver's estimates. With g = 1 / sqrt(reference_power) both are in the normal range.
void case_gain_reason()
{
  const double gain          = 2942.4243526773034;
  const double echo          = 2.6468e-06;
  const double sigma         = 1.0747184230211497e-05;
  const double smallest      = std::ldexp(1.0, -14);
  CHECK(std::fabs(smallest - 6.1035e-05) < 1e-9);
  CHECK(echo < smallest && sigma < smallest);
  CHECK(half_step(echo) / echo > 0.02);                 // about 44 steps: five to six bits
  CHECK(std::log2(echo / half_step(echo)) < 5.6);
  CHECK(gain * echo > smallest && gain * sigma > smallest);
  CHECK(half_step(gain * echo) / (gain * echo) < 0.001); // eleven bits
  CHECK(std::log2(gain * echo / half_step(gain * echo)) > 10.0);
  CHECK(std::fabs(gain * sigma - 0.0316228) < 1e-6);     // 30 dB below unit signal power
  // The grid's own format has the exponent range of a single: the factor does not change its rounding.
  const float value = 2.6468e-06F;
  CHECK(std::fabs(bf16_value(value) - value) / value <= 1.0 / 256.0);
  const float scaled = static_cast<float>(gain * static_cast<double>(value));
  CHECK(std::fabs(bf16_value(scaled) - scaled) / scaled <= 1.0 / 256.0);
}

run_spec entry_run(double gain, double power)
{
  run_spec spec;
  spec.gain.present             = true;
  spec.gain.value               = gain;
  spec.gain.source              = "entry";
  spec.gain.has_reference_power = true;
  spec.gain.reference_power     = power;
  for (const char* name : {"rz-aaaaaaaaaaaa", "rz-bbbbbbbbbbbb"}) {
    source_spec source;
    source.realization_id        = name;
    source.from_entry            = true;
    source.entry_grid_gain       = gain;
    source.entry_reference_power = power;
    spec.sources.push_back(source);
  }
  return spec;
}

void case_gain_rules()
{
  const double power = 1.1550196887810671e-07;
  const double gain  = 1.0 / std::sqrt(power);
  CHECK(std::fabs(gain - 2942.4243526773034) < 1e-9);
  check_gain(entry_run(gain, power)); // the accepted pair

  run_spec spec = entry_run(gain, power);
  spec.gain.present = false;
  CHECK_THROWS(check_gain(spec), input_error, "names no grid gain");
  for (double bad : {0.0, -gain, -0.0}) {
    spec = entry_run(bad, power);
    CHECK_THROWS(check_gain(spec), input_error, "must be positive");
  }
  for (double bad : {std::nan(""), HUGE_VAL, -HUGE_VAL}) {
    spec = entry_run(bad, power);
    CHECK_THROWS(check_gain(spec), input_error, "not finite");
  }
  // The rule g = 1 / sqrt(reference_power), within 1e-6.
  check_gain(entry_run(gain * (1.0 + 5e-7), power)); // inside for the run, and the sources carry the same value
  spec = entry_run(gain * (1.0 + 2e-6), power);
  CHECK_THROWS(check_gain(spec), input_error, "differs from 1 / sqrt(reference_power)");
  spec = entry_run(gain * (1.0 - 2e-6), power);
  CHECK_THROWS(check_gain(spec), input_error, "differs from 1 / sqrt(reference_power)");
  spec = entry_run(gain, power);
  spec.gain.has_reference_power = false;
  CHECK_THROWS(check_gain(spec), input_error, "reference power");
  // Two gains in one run: one realization carries another entry's gain, however close.
  spec = entry_run(gain, power);
  spec.sources[1].entry_grid_gain = std::nextafter(gain, 2.0 * gain);
  CHECK_THROWS(check_gain(spec), input_error, "two gains in one run");
  spec = entry_run(gain, power);
  spec.sources[1].entry_grid_gain = 1140.799;
  CHECK_THROWS(check_gain(spec), input_error, "two gains in one run");
  // An override is for a fixture that has no entry, and it is recorded with its reason.
  spec = entry_run(gain, power);
  spec.gain.source = "override";
  spec.gain.reason = "trying to replace the entry's gain";
  CHECK_THROWS(check_gain(spec), input_error, "only for a fixture that has no entry");
  run_spec fixture_run;
  fixture_run.gain.present = true;
  fixture_run.gain.value   = 1.0;
  fixture_run.gain.source  = "override";
  fixture_run.sources.resize(1);
  CHECK_THROWS(check_gain(fixture_run), input_error, "records its reason");
  fixture_run.gain.reason = "unity check";
  check_gain(fixture_run);
  // An entry gain on a run with a fixture realization, and an unknown source.
  spec = entry_run(gain, power);
  spec.sources[0].from_entry = false;
  CHECK_THROWS(check_gain(spec), input_error, "not both");
  spec = entry_run(gain, power);
  spec.gain.source = "measured";
  CHECK_THROWS(check_gain(spec), input_error, "'entry' or 'override'");
  // The kernel itself refuses a gain that is not finite and positive.
  const grid_shape shape = make_shape(1, 1, 1, 1);
  const cf32       one(1.0F, 0.0F);
  model_grid       grid(1, 1, 1);
  for (double bad : {0.0, -1.0, std::nan(""), HUGE_VAL}) {
    CHECK_THROWS(apply(shape, &one, &one, &one, bad, grid), input_error, "finite and positive");
  }
}

void case_grid_states()
{
  fixture       setup;
  const session run(setup.spec);
  model_grid    first(2, 2, 6), second(2, 2, 6);
  o1_pool       pool({&first, &second});
  CHECK(pool.state(0) == grid_state::free && pool.generation(0) == 0);

  // filling: the grid is not ready, so no consumer can read it.
  ticket key;
  CHECK(pool.try_acquire(key));
  CHECK(key.grid == 0 && key.generation == 1 && pool.state(0) == grid_state::filling);
  CHECK_THROWS(pool.view(key, "record-wide"), state_error, "is filling, not ready");
  CHECK_THROWS(pool.release(key), state_error, "is filling, not ready");
  run.fill(0, pool.writer(key));
  pool.commit(key, run.describe(0, run.scheduled(0), pool.writer(key)));
  CHECK(pool.state(0) == grid_state::ready);
  CHECK_THROWS(pool.writer(key), state_error, "is ready, not filling");

  // ready: the consumer of this allocation reads it, a consumer of another allocation is refused.
  const o1_view view = pool.view(key, "record-wide");
  CHECK(view.descriptor().opportunity_id == "fx-0001.0000" && view.descriptor().grant_id == "wide");
  CHECK(view.descriptor().active_re == 8 && view.descriptor().grid_gain == 2.0 && view.descriptor().slot == 7);
  CHECK(view.descriptor().readback_sha256 == view.digest() && view.digest().size() == 64);
  CHECK_THROWS(pool.view(key, "record-narrow"), state_error, "changed allocation");
  cf32 row[6];
  view.get(1, 1, row, 6);
  CHECK_THROWS(view.get(2, 0, row, 6), state_error, "leaves the carrier grid");

  // A second grid is filled while the first is retained: the first keeps its values.
  ticket next;
  CHECK(pool.try_acquire(next));
  CHECK(next.grid == 1 && next.generation == 1);
  pool.writer(next).clear();
  run.fill(0, "narrow", pool.writer(next));
  pool.commit(next, run.describe(0, run.bound("narrow"), pool.writer(next)));
  CHECK(view.digest() == view.descriptor().readback_sha256);
  CHECK(pool.view(next, "record-narrow").digest() != view.digest());
  // No grid is free: the request is refused at once.
  ticket none;
  CHECK(!pool.try_acquire(none));

  // release, exactly once. The view of a released grid is dead.
  pool.release(key);
  CHECK(pool.state(0) == grid_state::free);
  CHECK_THROWS(pool.release(key), state_error, "is free, not ready");
  CHECK_THROWS(view.get(0, 0, row, 6), state_error, "is free, not ready");
  CHECK_THROWS(view.digest(), state_error, "is free, not ready");

  // The grid is filled again: every ticket and view of the earlier fill is stale.
  ticket again;
  CHECK(pool.try_acquire(again));
  CHECK(again.grid == 0 && again.generation == 2);
  pool.writer(again).clear();
  run.fill(0, pool.writer(again));
  pool.commit(again, run.describe(0, run.scheduled(0), pool.writer(again)));
  CHECK_THROWS(pool.view(key, "record-wide"), state_error, "stale generation 1");
  CHECK_THROWS(view.get(0, 0, row, 6), state_error, "stale generation 1");
  CHECK_THROWS(pool.release(key), state_error, "stale generation 1");
  CHECK_THROWS(pool.writer(key), state_error, "stale generation 1");
  CHECK(pool.view(again, "record-wide").digest() == pool.view(again, "record-wide").descriptor().readback_sha256);
  ticket unknown;
  unknown.grid = 5;
  CHECK_THROWS(pool.release(unknown), state_error, "no grid 5");

  // A failed fill returns the grid to the pool, cleared.
  pool.release(again);
  pool.release(next);
  ticket broken;
  CHECK(pool.try_acquire(broken));
  run.fill(0, pool.writer(broken));
  pool.abort(broken);
  CHECK(pool.state(broken.grid) == grid_state::free);
  const std::vector<cf32> cleared = stored_values(first, setup.shape);
  const std::vector<cf32> zeros(cleared.size());
  CHECK(same_bits(cleared.data(), zeros.data(), zeros.size()));
  std::vector<grid_sink*> no_grids;
  CHECK_THROWS(o1_pool empty(no_grids), input_error, "at least one grid");
}

void case_session()
{
  fixture       setup;
  const session run(setup.spec);
  CHECK(run.opportunities() == 2 && run.files().size() == 6);
  CHECK(run.scheduled(0).grant_id == "wide" && run.scheduled(0).active.size() == 8);
  CHECK(run.scheduled(1).grant_id == "narrow" && run.scheduled(1).active.size() == 2);
  CHECK(run.bound("narrow").active == std::vector<std::uint32_t>({10, 11}));

  // The session applies what apply() applies, through the pool as well.
  const std::size_t plane = setup.shape.plane();
  model_grid        direct(2, 2, 6), through(2, 2, 6);
  std::vector<cf32> written_direct(2 * plane), written(2 * plane);
  apply(setup.shape, setup.x_wide.data(), setup.h.data(), setup.w.data(), 2.0, direct, written_direct.data());
  o1_pool pool({&through});
  ticket  key;
  CHECK(run.deliver(0, pool, key, written.data()));
  CHECK(same_bits(written.data(), written_direct.data(), written.size()));
  const o1_view view = pool.view(key, run.scheduled(0).record_sha256);
  CHECK(view.digest() == readback_sha256(direct, setup.shape));
  ticket busy;
  CHECK(!run.deliver(0, pool, busy));

  // The second opportunity reads its own channel and noise and the grid of its own grant.
  model_grid        direct_later(2, 2, 6), through_later(2, 2, 6);
  std::vector<cf32> written_later(2 * plane);
  apply(setup.shape, setup.x_narrow.data(), setup.h.data() + 2 * plane, setup.w.data() + 2 * plane, 2.0, direct_later);
  run.fill(1, through_later, written_later.data());
  CHECK(readback_sha256(through_later, setup.shape) == readback_sha256(direct_later, setup.shape));
  CHECK(readback_sha256(through_later, setup.shape) != view.digest());
  const o1_descriptor described = run.describe(1, run.scheduled(1), through_later);
  CHECK(described.opportunity_id == "fx-0001.0001" && described.grant_id == "narrow" && described.slot == 8);
  CHECK(described.active_re == 2 && described.readback_sha256.empty());

  // What storage did: values with more than eight bits changed, each one as the model says.
  const storage_stats stats = measure(view, run.scheduled(0), written.data());
  CHECK(stats.planes.size() == 2 && stats.components == 4 * plane);
  CHECK(stats.components_changed > 0 && stats.model_mismatches == 0);
  CHECK(stats.max_relative_error > 0.0 && stats.max_relative_error <= 1.0 / 256.0);
  CHECK(stats.error_power > 0.0 && stats.error_power < 1e-4 * stats.written_power);
  CHECK(stats.planes[0].active > stats.planes[0].all && stats.planes[1].active > stats.planes[0].active);
  // The receiver grid is larger than the carrier, as the pinned one is. Whatever it held before,
  // a delivered grid is zero outside the carrier and holds the opportunity inside it.
  model_grid large(2, 3, 9);
  const std::vector<cf32> stale(9, cf32(7.0F, -7.0F));
  for (unsigned rx = 0; rx != 2; ++rx) {
    for (unsigned symbol = 0; symbol != 3; ++symbol) {
      large.put(rx, symbol, stale.data(), 9);
    }
  }
  o1_pool roomy({&large});
  ticket  roomy_key;
  CHECK(run.deliver(0, roomy, roomy_key));
  CHECK(roomy.view(roomy_key, run.scheduled(0).record_sha256).digest() == readback_sha256(direct, setup.shape));
  cf32 wide_row[9];
  for (unsigned rx = 0; rx != 2; ++rx) {
    for (unsigned symbol = 0; symbol != 3; ++symbol) {
      large.get(rx, symbol, wide_row, 9);
      for (unsigned k = (symbol < 2 ? 6 : 0); k != 9; ++k) {
        CHECK(wide_row[k] == cf32(0.0F, 0.0F));
      }
    }
  }

  // A grid that stores by another rule is measured as such, element by element.
  truncating_grid   cut(2, 2, 6);
  o1_pool           other({&cut});
  ticket            cut_key;
  std::vector<cf32> cut_written(2 * plane);
  CHECK(run.deliver(0, other, cut_key, cut_written.data()));
  const storage_stats cut_stats = measure(other.view(cut_key, run.scheduled(0).record_sha256), run.scheduled(0),
                                          cut_written.data());
  CHECK(cut_stats.model_mismatches > 0 && cut_stats.model_mismatches <= cut_stats.components_changed);
  CHECK(other.view(cut_key, run.scheduled(0).record_sha256).descriptor().storage == "truncated");

  // g^2 times the mean |H|^2 on the occupied elements: the number the gain rule makes one.
  double expected = 0.0;
  for (unsigned rx = 0; rx != 2; ++rx) {
    for (std::uint32_t flat : run.scheduled(0).active) {
      expected += std::norm(std::complex<double>(setup.h[rx * plane + flat]));
    }
  }
  CHECK(std::fabs(run.unit_power(0) - 4.0 * expected / 16.0) < 1e-12);
}

void case_changed_grant()
{
  // The grant of an opportunity changes: X and the occupied elements come from the new grant, and
  // H is read at the coordinates the new grant transmits on. H holds its own coordinates.
  fixture       setup;
  const session run(setup.spec);
  const std::size_t plane = setup.shape.plane();
  model_grid        wide(2, 2, 6), narrow(2, 2, 6);
  std::vector<cf32> written(2 * plane);
  run.fill(0, wide);
  run.fill(0, "narrow", narrow, written.data());
  CHECK(readback_sha256(wide, setup.shape) != readback_sha256(narrow, setup.shape));
  const double gain = 2.0;
  for (unsigned rx = 0; rx != 2; ++rx) {
    for (std::size_t flat = 0; flat != plane; ++flat) {
      const std::complex<double> channel(1.0 + rx, static_cast<double>((flat / 6) * 8 + flat % 6) / 64.0);
      const std::complex<double> reference =
          gain * (channel * std::complex<double>(setup.x_narrow[flat]) + std::complex<double>(setup.w[rx * plane + flat]));
      const cf32 expected(static_cast<float>(reference.real()), static_cast<float>(reference.imag()));
      CHECK(written[rx * plane + flat] == expected);
    }
  }
  // Outside the two elements of the new grant only the noise is left.
  for (std::size_t flat = 0; flat != plane; ++flat) {
    if (flat != 10 && flat != 11) {
      const cf32 noise(static_cast<float>(gain * static_cast<double>(setup.w[flat].real())),
                       static_cast<float>(gain * static_cast<double>(setup.w[flat].imag())));
      CHECK(written[flat] == noise);
    }
  }
  // A grant that the run does not bind is refused.
  model_grid other(2, 2, 6);
  CHECK_THROWS(run.fill(0, "unbound", other), input_error, "does not bind the grant unbound");
}

void case_refusals()
{
  fixture setup;
  { const session accepted(setup.spec); }

  run_spec bad = setup.spec;
  bad.grants[0].x.sha256 = std::string(64, '0');
  CHECK_THROWS(session refused(bad), input_error, "its record binds 0000");
  bad = setup.spec;
  bad.grants[0].x.sha256.clear();
  CHECK_THROWS(session refused(bad), input_error, "no bound transmit grid");
  bad = setup.spec;
  bad.opportunities[0].grant_id.clear();
  CHECK_THROWS(session refused(bad), input_error, "has no bound grid");
  bad = setup.spec;
  bad.opportunities[0].grant_id = "elsewhere";
  CHECK_THROWS(session refused(bad), input_error, "which the run does not bind");
  bad = setup.spec;
  bad.sources[0].h.sha256 = std::string(64, 'f');
  CHECK_THROWS(session refused(bad), input_error, "the channel of realization fx-0001 has the hash");
  bad = setup.spec;
  bad.opportunities[0].index = 2;
  CHECK_THROWS(session refused(bad), input_error, "reads index 2 of a realization with 2 opportunities");
  bad = setup.spec;
  bad.opportunities.push_back(bad.opportunities[0]);
  CHECK_THROWS(session refused(bad), input_error, "is listed twice");
  bad = setup.spec;
  bad.gain.present = false;
  CHECK_THROWS(session refused(bad), input_error, "names no grid gain");

  // A changed allocation: another carrier grid, or occupied elements that are not the grid's.
  bad = setup.spec;
  bad.subcarriers = 12;
  CHECK_THROWS(session refused(bad), input_error, "changed allocation");
  bad = setup.spec;
  bad.grants[0].x = setup.spec.grants[1].x;
  CHECK_THROWS(session refused(bad), input_error, "which the grant");
  bad = setup.spec;
  bad.grants[1].active.push_back(bad.grants[1].active[0]);
  CHECK_THROWS(session refused(bad), input_error, "list one element twice");
  bad = setup.spec;
  bad.grants[1].active = {setup.files.index_file("outside.npy", {{1, 4}, {1, 6}})};
  CHECK_THROWS(session refused(bad), input_error, "outside the carrier grid");

  // Damaged files.
  std::vector<cf32> poisoned = setup.h;
  poisoned[3]                = cf32(std::nanf(""), 0.0F);
  bad                        = setup.spec;
  bad.sources[0].h           = setup.files.complex_file("h_nan.npy", {2, 2, 1, 2, 6}, poisoned);
  CHECK_THROWS(session refused(bad), input_error, "nonfinite");
  bad              = setup.spec;
  bad.sources[0].w = setup.files.complex_file("w_axes.npy", {2, 2, 6, 2}, setup.w);
  CHECK_THROWS(session refused(bad), input_error, "the noise of realization fx-0001 is <c8 [2, 2, 6, 2]");
  bad                   = setup.spec;
  bad.sources[0].h.path = setup.files.directory + "/absent.npy";
  CHECK_THROWS(session refused(bad), input_error, "cannot read");
  const std::string short_file =
      setup.files.write("short.npy", "<c8", {2, 2, 1, 2, 6}, setup.h.data(), (setup.h.size() - 1) * sizeof(cf32));
  CHECK_THROWS(array_file truncated(short_file), input_error, "bytes of data");
  const float       reals[4] = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::string real_file = setup.files.write("real.npy", "<f4", {4}, reals, sizeof(reals));
  CHECK_THROWS(array_file wrong_type(real_file), input_error, "expected little-endian complex64 or int32");

  // A result that does not fit a single is refused, and the grid goes back to the pool.
  std::vector<cf32> huge = setup.h;
  huge[0]                = cf32(3.0e38F, 0.0F);
  bad                    = setup.spec;
  bad.sources[0].h       = setup.files.complex_file("h_huge.npy", {2, 2, 1, 2, 6}, huge);
  const session overflow(bad);
  model_grid    grid(2, 2, 6);
  o1_pool       pool({&grid});
  ticket        key;
  CHECK_THROWS(overflow.deliver(0, pool, key), input_error, "nonfinite value at receive port 0, symbol 0, subcarrier 0");
  CHECK(pool.state(0) == grid_state::free);
  // A grid that is too small for the carrier.
  model_grid small(1, 2, 6);
  const session run(setup.spec);
  CHECK_THROWS(run.fill(0, small), input_error, "too small");
}

// A fixture that another language can generate bit for bit: X, then H, then W from one sequence,
// W scaled by 2^-6. Every value has at most 24 significant bits, so it is exact in single precision.
int print_digest(int argc, char** argv)
{
  if (argc != 8) {
    std::cerr << "replay_core_test --digest SEED RX TX SYMBOLS SUBCARRIERS GAIN" << std::endl;
    return 2;
  }
  const grid_shape shape = make_shape(static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)),
                                      static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)),
                                      static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)),
                                      static_cast<unsigned>(std::strtoul(argv[6], nullptr, 10)));
  const double     gain  = std::strtod(argv[7], nullptr);
  sequence         values(static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10)));
  std::vector<cf32> x(shape.tx_ports * shape.plane());
  std::vector<cf32> h(static_cast<std::size_t>(shape.rx_ports) * shape.tx_ports * shape.plane());
  std::vector<cf32> w(shape.rx_ports * shape.plane());
  for (cf32& value : x) {
    value = values.next_complex();
  }
  for (cf32& value : h) {
    value = values.next_complex();
  }
  for (cf32& value : w) {
    value = values.next_complex() * 0.015625F;
  }
  try {
    model_grid        grid(shape.rx_ports, shape.symbols, shape.subcarriers);
    std::vector<cf32> written(w.size());
    apply(shape, x.data(), h.data(), w.data(), gain, grid, written.data());
    std::cout << "written " << sha256_hex(written.data(), written.size() * sizeof(cf32)) << "\n"
              << "stored " << readback_sha256(grid, shape) << std::endl;
  } catch (const input_error& error) {
    std::cerr << "replay_core_test: input error: " << error.what() << std::endl;
    return 2;
  }
  return 0;
}

struct named_case {
  const char*           name;
  std::function<void()> body;
};

const named_case CASES[] = {
    {"sha256", case_sha256},
    {"storage", case_storage},
    {"s03_vector", case_s03_vector},
    {"unity", case_unity},
    {"zero_channel", case_zero_channel},
    {"cancellation", case_cancellation},
    {"permutation", case_permutation},
    {"unit_scale", case_unit_scale},
    {"gain_reason", case_gain_reason},
    {"gain_rules", case_gain_rules},
    {"grid_states", case_grid_states},
    {"session", case_session},
    {"changed_grant", case_changed_grant},
    {"refusals", case_refusals},
};

} // namespace

int main(int argc, char** argv)
{
  const std::string selected = argc > 1 ? argv[1] : "";
  if (selected == "--digest") {
    return print_digest(argc, argv);
  }
  if (selected == "--list") {
    for (const named_case& item : CASES) {
      std::cout << item.name << "\n";
    }
    return 0;
  }
  unsigned ran = 0;
  for (const named_case& item : CASES) {
    if (!selected.empty() && selected != item.name) {
      continue;
    }
    const int before = failures;
    try {
      item.body();
    } catch (const std::exception& error) {
      failed(__FILE__, __LINE__, std::string(item.name) + " threw: " + error.what());
    }
    std::cout << (failures == before ? "PASS  " : "FAIL  ") << item.name << std::endl;
    ++ran;
  }
  if (ran == 0) {
    std::cerr << "replay_core_test: no case named '" << selected << "'" << std::endl;
    return 2;
  }
  std::cout << ran << " case(s), " << failures << " failed check(s)" << std::endl;
  return failures == 0 ? 0 : 1;
}
