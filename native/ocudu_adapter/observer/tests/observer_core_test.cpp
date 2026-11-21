// Unit cases of the observer core (Sprint 3.1, selector pilot_ls).
//
// The target links observer_core and replay_core and nothing else: no OCUDU library, no test
// framework. The grids are the storage model of the channel application core.
//
//     observer_core_test            run every case
//     observer_core_test NAME       run one case (CTest registers each)
//     observer_core_test --list     print the case names
//     observer_core_test --vectors  read lines of four hexadecimal float32 patterns (yr yi xr xi)
//                                   from the standard input and print hr, hi and the finite flag
//                                   of the estimate, for a reference in another language
//     observer_core_test --key TEXT print the opaque join key of a text
//     observer_core_test --slot-start SCS_KHZ ABSOLUTE_SLOT
//                                   print the start of a slot in T_c
//
// ASCII only.

#include "observer_core.h"
#include "replay_core.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace isac_observer;
using isac_replay::grid_shape;
using isac_replay::model_grid;
using isac_replay::o1_descriptor;
using isac_replay::o1_pool;
using isac_replay::o1_view;
using isac_replay::sha256_hex;
using isac_replay::ticket;

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

constexpr unsigned SYMBOLS     = 14;
constexpr unsigned SUBCARRIERS = 24;
constexpr unsigned SLOT        = 7;
const char* const  RECORD      = "record-a";

grid_shape shape_of(unsigned planes)
{
  grid_shape shape;
  shape.rx_ports    = planes;
  shape.tx_ports    = 1;
  shape.symbols     = SYMBOLS;
  shape.subcarriers = SUBCARRIERS;
  return shape;
}

// The reference symbol of pilot n: one of the four points (+-0.5, +-0.5).
cf32 reference_of(std::size_t n)
{
  return cf32((n % 2) != 0 ? -0.5F : 0.5F, (n % 3) != 0 ? 0.5F : -0.5F);
}

// A source on symbols 2 and 11 with a comb of step two that starts at subcarrier 4: eight pilots.
source_spec comb_source(unsigned planes = 2, const std::string& record = RECORD)
{
  source_spec spec;
  spec.grant_id            = "grant-" + record;
  spec.grant_record_sha256 = record;
  spec.source              = "pusch_dmrs";
  spec.channel             = "PUSCH";
  spec.link_direction      = "uplink";
  spec.processor           = "pusch";
  spec.ports               = {0};
  spec.grid_symbols        = SYMBOLS;
  spec.grid_subcarriers    = SUBCARRIERS;
  spec.scs_khz             = 30;
  spec.slot                = SLOT;
  spec.slot_symbols        = {2, 11};
  // Useful starts of symbols 2 and 11 of a slot at 30 kHz, in T_c from the slot start.
  spec.symbol_useful_start_tc = {145920, 777216};
  spec.tc_per_second          = TC_PER_SECOND;
  spec.amplitude              = 0.70710678118654757;
  for (unsigned symbol : {2U, 11U}) {
    for (unsigned k = 4; k != 12; k += 2) {
      pilot item;
      item.symbol     = symbol;
      item.subcarrier = k;
      spec.index.push_back(item);
    }
  }
  for (std::size_t n = 0; n != spec.index.size(); ++n) {
    spec.reference.push_back(reference_of(n));
  }
  for (unsigned plane = 0; plane != planes; ++plane) {
    spec.antenna_map.push_back(plane);
  }
  return spec;
}

// Y of one resource element of a grid that carries a channel per plane: H * X on the pilots of
// the source, and a value that depends on the data seed everywhere else.
using channel_fn = std::function<cf32(unsigned plane, unsigned symbol, unsigned subcarrier)>;

ticket make_ready(o1_pool& pool, const source_spec& spec, const channel_fn& channel, float data_seed,
                  const std::string& opportunity = "op.0000", unsigned slot = SLOT, float pilot_scale = 1.0F)
{
  ticket key;
  if (!pool.try_acquire(key)) {
    throw std::runtime_error("the case has no free grid");
  }
  isac_replay::grid_sink& sink   = pool.writer(key);
  const unsigned          planes = sink.planes();
  sink.clear();
  std::vector<cf32> row(SUBCARRIERS);
  for (unsigned plane = 0; plane != planes; ++plane) {
    for (unsigned symbol = 0; symbol != SYMBOLS; ++symbol) {
      for (unsigned k = 0; k != SUBCARRIERS; ++k) {
        row[k] = cf32(data_seed + 0.125F * static_cast<float>(k), data_seed - 0.25F * static_cast<float>(symbol));
      }
      for (std::size_t n = 0; n != spec.index.size(); ++n) {
        if (spec.index[n].symbol == symbol) {
          row[spec.index[n].subcarrier] = channel(plane, symbol, spec.index[n].subcarrier) * spec.reference[n] * pilot_scale;
        }
      }
      sink.put(plane, symbol, row.data(), SUBCARRIERS);
    }
  }
  o1_descriptor descriptor;
  descriptor.run_id              = "unit";
  descriptor.opportunity_id      = opportunity;
  descriptor.grant_id            = spec.grant_id;
  descriptor.grant_record_sha256 = spec.grant_record_sha256;
  descriptor.sfn                 = 0;
  descriptor.slot                = slot;
  descriptor.shape               = shape_of(planes);
  descriptor.grid_gain           = 1.0;
  descriptor.storage             = sink.storage();
  pool.commit(key, descriptor);
  return key;
}

slot_context context(const std::string& opportunity = "op.0000", unsigned sfn = 0, unsigned slot = SLOT,
                     const std::string& episode = "rz-unit")
{
  slot_context item;
  item.opportunity_id = opportunity;
  item.episode_id     = episode;
  item.sfn            = sfn;
  item.slot           = slot;
  item.cell           = 1;
  item.ue             = 0x1234;
  return item;
}

cf32 flat(unsigned, unsigned, unsigned)
{
  return cf32(1.0F, 0.0F);
}

bool same_bits(cf32 a, cf32 b)
{
  return std::memcmp(&a, &b, sizeof(cf32)) == 0;
}

// ---------------------------------------------------------------------------------------------- cases

void case_kernel()
{
  // The vector of the sprint card: a pilot of 1+j and a noiseless Y of -1+3j give 1+2j.
  bool ok = false;
  CHECK(same_bits(pilot_ls(cf32(-1.0F, 3.0F), cf32(1.0F, 1.0F), &ok), cf32(1.0F, 2.0F)));
  CHECK(ok);
  // Y = H X returns H for a few exact values, a measured zero included.
  for (const cf32 h : {cf32(1.0F, 0.0F), cf32(0.0F, -1.0F), cf32(-0.75F, 0.5F), cf32(0.0F, 0.0F)}) {
    for (std::size_t n = 0; n != 6; ++n) {
      CHECK(same_bits(pilot_ls(h * reference_of(n), reference_of(n), &ok) + cf32(0.0F, 0.0F), h + cf32(0.0F, 0.0F)));
      CHECK(ok);
    }
  }
  // A result that does not fit single precision is reported and not returned as a number.
  pilot_ls(cf32(3.0e38F, 0.0F), cf32(1.0e-3F, 0.0F), &ok);
  CHECK(!ok);
  // The estimate of the stored pilot against the reference of two CDM groups without data: the
  // stored value is one per component, the reference 0.998815.
  const float reference = static_cast<float>(1.4125375446227544 / std::sqrt(2.0));
  const cf32  unity     = pilot_ls(cf32(-1.0F, 1.0F), cf32(-reference, reference), &ok);
  CHECK(ok && unity.real() == static_cast<float>(1.0 / static_cast<double>(reference)) && unity.imag() == 0.0F);
  CHECK(std::fabs(unity.real() - 1.00119F) < 5e-6F);
}

void case_amplitude_scale()
{
  // A halved pilot amplitude with a correspondingly scaled Y gives the same estimate, bit for bit.
  model_grid  first(2, SYMBOLS, SUBCARRIERS), second(2, SYMBOLS, SUBCARRIERS);
  o1_pool     pool({&first, &second});
  source_spec full = comb_source();
  source_spec half = comb_source(2, "record-half");
  for (cf32& value : half.reference) {
    value *= 0.5F;
  }
  half.amplitude *= 0.5;
  const channel_fn channel = [](unsigned plane, unsigned symbol, unsigned k) {
    return cf32(0.25F * static_cast<float>(k) - 1.0F, 0.5F * static_cast<float>(plane) + 0.125F * static_cast<float>(symbol));
  };
  observer     tap("unit", 1);
  tap.bind(full);
  tap.bind(half);
  const ticket a = make_ready(pool, full, channel, 3.0F, "op.full");
  const ticket b = make_ready(pool, half, channel, 3.0F, "op.half");
  const observation one = tap.observe(pool.view(a, full.grant_record_sha256), context("op.full"));
  const observation two = tap.observe(pool.view(b, half.grant_record_sha256), context("op.half"));
  CHECK(one.values.size() == 16 && one.payload_sha256 == two.payload_sha256);
  CHECK(one.ingress_sha256 != two.ingress_sha256);
  CHECK(one.scale == 2.0 * two.scale);
  // And the estimate is the channel that was applied.
  const pilot_source source(full);
  for (std::size_t plane = 0; plane != 2; ++plane) {
    for (std::size_t row = 0; row != 2; ++row) {
      for (std::size_t column = 0; column != 4; ++column) {
        const cf32 wanted = channel(static_cast<unsigned>(plane), source.symbols()[row], source.subcarriers()[column]);
        CHECK(same_bits(one.values[(plane * 2 + row) * 4 + column], wanted));
      }
    }
  }
}

void case_data_invariance()
{
  // Altering data-only resource elements changes the grid and leaves O2 as it was.
  model_grid        first(2, SYMBOLS, SUBCARRIERS), second(2, SYMBOLS, SUBCARRIERS);
  o1_pool           pool({&first, &second});
  const source_spec spec = comb_source();
  const channel_fn  channel = [](unsigned plane, unsigned, unsigned k) {
    return cf32(1.0F - 0.5F * static_cast<float>(plane), 0.125F * static_cast<float>(k));
  };
  observer tap("unit", 1);
  tap.bind(spec);
  const ticket      a   = make_ready(pool, spec, channel, 1.0F, "op.0000");
  const ticket      b   = make_ready(pool, spec, channel, -2.5F, "op.0001");
  const observation one = tap.observe(pool.view(a, RECORD), context("op.0000"));
  const observation two = tap.observe(pool.view(b, RECORD), context("op.0001"));
  CHECK(one.ingress_sha256 != two.ingress_sha256);
  CHECK(one.payload_sha256 == two.payload_sha256 && one.mask == two.mask && one.values == two.values);
  // A pilot that changes does change it.
  pool.release(b);
  const ticket      c     = make_ready(pool, spec, [](unsigned, unsigned, unsigned) { return cf32(0.5F, 0.0F); }, 1.0F, "op.0002");
  const observation three = tap.observe(pool.view(c, RECORD), context("op.0002"));
  CHECK(three.payload_sha256 != one.payload_sha256);
}

void case_pilot_hole()
{
  // The record leaves out the pilot at symbol 11, subcarrier 8: the element is a hole.
  source_spec spec = comb_source();
  spec.index.erase(spec.index.begin() + 6);
  spec.reference.erase(spec.reference.begin() + 6);
  const pilot_source source(spec);
  CHECK(source.holes() == 1 && source.plane_elements() == 8 && source.pilot_at(1, 2) == -1 && source.pilot_at(1, 3) == 6);

  model_grid first(2, SYMBOLS, SUBCARRIERS), second(2, SYMBOLS, SUBCARRIERS);
  o1_pool    pool({&first, &second});
  // The channel is zero on symbol 2, subcarrier 6 of plane 1: a measured zero, not a hole.
  const channel_fn channel = [](unsigned plane, unsigned symbol, unsigned k) {
    return (plane == 1 && symbol == 2 && k == 6) ? cf32(0.0F, 0.0F) : cf32(0.5F, -0.25F);
  };
  observer tap("unit", 1);
  tap.bind(spec);
  const ticket      key  = make_ready(pool, spec, channel, 2.0F);
  const observation made = tap.observe(pool.view(key, RECORD), context());
  CHECK(made.values.size() == 16 && made.mask.size() == 2 && made.holes == 2 && made.valid == 14);
  for (std::size_t plane = 0; plane != 2; ++plane) {
    const std::size_t hole = (plane * 2 + 1) * 4 + 2;
    CHECK(!made.is_valid(hole) && made.values[hole] == cf32(0.0F, 0.0F));
  }
  const std::size_t zero = (1 * 2 + 0) * 4 + 1;
  CHECK(made.is_valid(zero) && made.values[zero] == cf32(0.0F, 0.0F));
  CHECK(made.is_valid(0) && same_bits(made.values[0], cf32(0.5F, -0.25F)));
  // Bit n of the mask is bit (n mod 8) of byte (n / 8): the holes are elements 6 and 14.
  CHECK(made.mask[0] == 0xbf && made.mask[1] == 0xbf);
  CHECK(made.mask_sha256 == sha256_hex(made.mask.data(), made.mask.size()));
}

// A source that was edited must be refused with a message that holds the given text.
void refused(const std::function<void(source_spec&)>& edit, const char* text, int line)
{
  source_spec spec = comb_source();
  edit(spec);
  try {
    pilot_source source(spec);
    failed(__FILE__, line, std::string("the source was accepted, expected a refusal with '") + text + "'");
  } catch (const input_error& error) {
    if (std::string(error.what()).find(text) == std::string::npos) {
      failed(__FILE__, line, std::string("message '") + error.what() + "' lacks '" + text + "'");
    }
  }
}

void case_refusals()
{
  refused([](source_spec& spec) { spec.reference[3] = cf32(0.0F, 0.0F); }, "a configured pilot with zero reference power",
          __LINE__);
  refused([](source_spec& spec) { spec.index[1] = spec.index[0]; }, "duplicate pilot", __LINE__);
  refused([](source_spec& spec) { std::swap(spec.index[1], spec.index[2]); }, "out of order", __LINE__);
  refused([](source_spec& spec) { spec.index[7].subcarrier = SUBCARRIERS; }, "outside the grid", __LINE__);
  refused(
      [](source_spec& spec) {
        spec.ports.push_back(1);
        spec.reference.insert(spec.reference.end(), spec.reference.begin(), spec.reference.end());
      },
      "estimates one port", __LINE__);
  refused([](source_spec& spec) { spec.reference.pop_back(); }, "reference symbols for", __LINE__);
  refused([](source_spec& spec) { spec.reference[0] = cf32(std::nanf(""), 0.5F); }, "not finite", __LINE__);
  refused(
      [](source_spec& spec) {
        spec.index.clear();
        spec.reference.clear();
      },
      "has no pilot", __LINE__);
  refused([](source_spec& spec) { spec.slot_symbols[1] = 12; }, "other reference symbols", __LINE__);
  refused([](source_spec& spec) { spec.symbol_useful_start_tc.pop_back(); }, "no start time", __LINE__);
  refused([](source_spec& spec) { spec.amplitude = 0.0; }, "amplitude", __LINE__);
  refused([](source_spec& spec) { spec.antenna_map[1] = spec.antenna_map[0]; }, "names an array element twice", __LINE__);
  refused([](source_spec& spec) { spec.antenna_map.clear(); }, "no antenna map", __LINE__);
  refused([](source_spec& spec) { spec.scs_khz = 20; }, "subcarrier spacing", __LINE__);
  refused([](source_spec& spec) { spec.slot = 20; }, "outside the frame", __LINE__);
  refused([](source_spec& spec) { spec.grant_record_sha256.clear(); }, "no grant record", __LINE__);
  CHECK_THROWS(observer("", 1), input_error, "identifier of its run");
  CHECK_THROWS(observer("unit", 0), input_error, "at least 1");
}

void case_antenna_map()
{
  // Two antennas with two phases. Plane 0 carries +j and plane 1 carries -1.
  model_grid       first(2, SYMBOLS, SUBCARRIERS), second(2, SYMBOLS, SUBCARRIERS);
  o1_pool          pool({&first, &second});
  const channel_fn channel = [](unsigned plane, unsigned, unsigned) {
    return plane == 0 ? cf32(0.0F, 1.0F) : cf32(-1.0F, 0.0F);
  };
  source_spec straight = comb_source();
  source_spec reversed = comb_source(2, "record-reversed");
  reversed.antenna_map = {1, 0};
  observer tap("unit", 1);
  tap.bind(straight);
  tap.bind(reversed);
  const ticket      a   = make_ready(pool, straight, channel, 1.0F, "op.straight");
  const ticket      b   = make_ready(pool, reversed, channel, 1.0F, "op.reversed");
  const observation one = tap.observe(pool.view(a, straight.grant_record_sha256), context("op.straight"));
  const observation two = tap.observe(pool.view(b, reversed.grant_record_sha256), context("op.reversed"));
  // The producer never reorders: both payloads hold the planes in the order of the grid.
  CHECK(one.values == two.values);
  CHECK(same_bits(one.values[0], cf32(0.0F, 1.0F)) && same_bits(one.values[8], cf32(-1.0F, 0.0F)));
  CHECK((one.antenna_map == std::vector<unsigned>{0, 1}) && (two.antenna_map == std::vector<unsigned>{1, 0}));
  // The map travels with the data: the phase of array element 0 is read from the plane it names.
  const pilot_source& first_source  = tap.source_of(straight.grant_record_sha256);
  const pilot_source& second_source = tap.source_of(reversed.grant_record_sha256);
  CHECK(first_source.plane_of(0) == 0 && first_source.plane_of(1) == 1);
  CHECK(second_source.plane_of(0) == 1 && second_source.plane_of(1) == 0);
  CHECK(same_bits(one.values[first_source.plane_of(0) * 8], cf32(0.0F, 1.0F)));
  CHECK(same_bits(two.values[second_source.plane_of(0) * 8], cf32(-1.0F, 0.0F)));
  CHECK_THROWS(first_source.plane_of(2), input_error, "names no receive plane");
}

void case_coordinates()
{
  const source_spec  spec = comb_source(2);
  const pilot_source source(spec);
  CHECK((source.symbols() == std::vector<unsigned>{2, 11}));
  CHECK((source.subcarriers() == std::vector<unsigned>{4, 6, 8, 10}));
  CHECK(source.first_subcarrier() == 4 && source.subcarrier_step() == 2 && source.holes() == 0);
  // The digest of the axes, computed here from the bytes: 2, 11, 4, 6, 8, 10 as little-endian uint32.
  const std::uint8_t bytes[] = {2, 0, 0, 0, 11, 0, 0, 0, 4, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 10, 0, 0, 0};
  CHECK(source.coordinates_sha256() == sha256_hex(bytes, sizeof(bytes)));

  model_grid first(2, SYMBOLS, SUBCARRIERS), second(2, SYMBOLS, SUBCARRIERS);
  o1_pool    pool({&first, &second});
  // The channel encodes the coordinate, so a wrong element is a wrong value.
  const channel_fn channel = [](unsigned plane, unsigned symbol, unsigned k) {
    return cf32(static_cast<float>(symbol) + 0.125F * static_cast<float>(plane), static_cast<float>(k));
  };
  observer tap("unit", 1);
  tap.bind(spec);
  const ticket      key  = make_ready(pool, spec, channel, 0.5F);
  const observation made = tap.observe(pool.view(key, RECORD), context());
  CHECK((made.shape == std::vector<std::size_t>{2, 2, 4}) && (made.strides == std::vector<std::size_t>{8, 4, 1}));
  CHECK(made.first_subcarrier == 4 && made.subcarrier_step == 2 && made.subcarrier_count == 4);
  CHECK((made.symbols == std::vector<unsigned>{2, 11}) && made.rx_count == 2 && made.layer_count == 1);
  CHECK(made.coordinates_sha256 == source.coordinates_sha256());
  for (std::size_t plane = 0; plane != 2; ++plane) {
    for (std::size_t row = 0; row != 2; ++row) {
      for (std::size_t column = 0; column != 4; ++column) {
        const std::size_t element = plane * made.strides[0] + row * made.strides[1] + column * made.strides[2];
        const cf32 wanted(static_cast<float>(made.symbols[row]) + 0.125F * static_cast<float>(plane),
                          static_cast<float>(made.first_subcarrier + made.subcarrier_step * column));
        CHECK(same_bits(made.values[element], wanted) && made.is_valid(element));
      }
    }
  }
  CHECK(made.valid == 16 && made.holes == 0 && made.payload_bytes() == 128);
  // An axis that is not regular reports no step.
  source_spec ragged = comb_source();
  ragged.index.erase(ragged.index.begin() + 1); // symbol 2 loses subcarrier 6, symbol 11 keeps it
  ragged.reference.erase(ragged.reference.begin() + 1);
  ragged.index.erase(ragged.index.begin() + 4); // symbol 11 loses subcarrier 6 too
  ragged.reference.erase(ragged.reference.begin() + 4);
  const pilot_source irregular(ragged);
  CHECK((irregular.subcarriers() == std::vector<unsigned>{4, 8, 10}) && irregular.subcarrier_step() == 0);
}

void case_record()
{
  model_grid        first(2, SYMBOLS, SUBCARRIERS), second(2, SYMBOLS, SUBCARRIERS);
  o1_pool           pool({&first, &second});
  const source_spec spec = comb_source();
  observer          tap("run-7", 3);
  tap.bind(spec);
  const ticket      a   = make_ready(pool, spec, flat, 1.0F, "op.0000");
  const o1_view     view = pool.view(a, RECORD);
  const observation one  = tap.observe(view, context("op.0000", 1023, SLOT));
  // Identity: the stage, the producer, the source, the layout and the precision that was used.
  CHECK(one.stage == "O2" && one.producer == "observer" && one.source == "pusch_dmrs" && one.port == 0);
  CHECK((one.producer_steps == std::vector<std::string>{"pilot_ls"}));
  CHECK(one.layout == "ocudu_o2_comb_c64" && one.dtype == "<c8" && one.scale == spec.amplitude);
  // Join keys: opaque integers, and the text they were made from in the host block.
  CHECK(one.run_id == "run-7" && one.epoch == 3 && one.sequence == 0);
  CHECK(one.opportunity == opaque_key("op.0000") && one.episode == opaque_key("rz-unit") && one.grant == opaque_key(RECORD));
  CHECK(opaque_key("abc") == 0xba7816bf8f01cfeaULL);
  CHECK(one.opportunity_id == "op.0000" && one.grant_id == spec.grant_id && one.grant_record_sha256 == RECORD);
  CHECK(one.cell == 1 && one.ue == 0x1234 && one.sfn == 1023 && one.slot == SLOT);
  // The ingress digest is the digest of the grid the processors read.
  CHECK(one.ingress_sha256 == view.digest() && one.ingress_sha256 == view.descriptor().readback_sha256);
  CHECK(one.grid.grid == a.grid && one.grid.generation == a.generation);
  CHECK(one.payload_sha256 == sha256_hex(one.values.data(), one.values.size() * sizeof(cf32)));
  // Time: 20 slots per frame at 30 kHz, 983,040 T_c per slot.
  CHECK(one.absolute_slot == 1023ULL * 20 + SLOT);
  CHECK(one.t_first_sample == one.absolute_slot * 983040ULL + 145920ULL);
  CHECK(one.t_last_sample == one.absolute_slot * 983040ULL + 777216ULL + 65536ULL);
  CHECK(one.t_ready_ns > 0 && one.service_us >= 0.0);

  // The next opportunity of the episode lies in frame 0: the frame number wrapped, the slot count did not.
  const ticket      b   = make_ready(pool, spec, flat, 1.0F, "op.0001");
  const observation two = tap.observe(pool.view(b, RECORD), context("op.0001", 0, SLOT));
  CHECK(two.sequence == 1 && two.epoch == 3 && two.absolute_slot == 1024ULL * 20 + SLOT);
  CHECK(two.absolute_slot > one.absolute_slot && two.t_first_sample > one.t_last_sample);
  // Another episode has its own slot count. A repeated slot number in a new epoch is told apart by the epoch.
  pool.release(a);
  const ticket      c     = make_ready(pool, spec, flat, 1.0F, "op.0002");
  const observation three = tap.observe(pool.view(c, RECORD), context("op.0002", 0, SLOT, "rz-other"));
  CHECK(three.sequence == 2 && three.absolute_slot == SLOT && three.episode != two.episode);
  CHECK(tap.observations().size() == 3 && tap.epoch() == 3);
  observer restarted("run-7", 4);
  restarted.bind(spec);
  const observation again = restarted.observe(pool.view(c, RECORD), context("op.0002", 0, SLOT, "rz-other"));
  CHECK(again.epoch == 4 && again.sequence == 0 && again.absolute_slot == three.absolute_slot);
}

void case_ownership()
{
  model_grid        first(2, SYMBOLS, SUBCARRIERS), second(2, SYMBOLS, SUBCARRIERS);
  o1_pool           pool({&first, &second});
  const source_spec spec = comb_source();
  observer          tap("unit", 1);
  tap.bind(spec);
  const ticket      a     = make_ready(pool, spec, flat, 1.0F, "op.0000");
  const o1_view     view  = pool.view(a, RECORD);
  const std::string before = view.digest();
  const observation kept  = tap.observe(view, context("op.0000"));
  // The observer only reads: the grid has its digest after the tap.
  CHECK(view.digest() == before && pool.state(a.grid) == isac_replay::grid_state::ready);
  // The observation owns its payload: it survives the release and the next fill of the grid.
  const std::vector<cf32> values = kept.values;
  pool.release(a);
  ticket again;
  CHECK(pool.try_acquire(again) && again.grid == a.grid);
  pool.abort(again);
  const ticket b = make_ready(pool, spec, [](unsigned, unsigned, unsigned) { return cf32(0.25F, 0.75F); }, 9.0F, "op.0001");
  CHECK(b.grid == a.grid && b.generation != a.generation);
  CHECK(kept.values == values && tap.observations().front().values == values);
  CHECK(kept.payload_sha256 == sha256_hex(values.data(), values.size() * sizeof(cf32)));
  // A view of the earlier fill is stale: nothing is read and nothing is recorded.
  CHECK_THROWS(tap.observe(view, context("op.late")), state_error, "");
  CHECK(tap.observations().size() == 1);
}

void case_binding()
{
  model_grid        first(2, SYMBOLS, SUBCARRIERS), second(2, SYMBOLS, SUBCARRIERS);
  model_grid        single(1, SYMBOLS, SUBCARRIERS);
  o1_pool           pool({&first, &second});
  o1_pool           narrow({&single});
  const source_spec spec = comb_source();
  observer          tap("unit", 1);
  tap.bind(spec);
  CHECK(tap.bound(RECORD) && !tap.bound("record-b") && tap.sources() == 1);
  CHECK_THROWS(tap.bind(spec), input_error, "two sensing sources");
  CHECK_THROWS(tap.source_of("record-b"), input_error, "no sensing source is bound");
  // A grid of another grant record has no source: the allocation changed.
  const source_spec other = comb_source(2, "record-b");
  const ticket      a     = make_ready(pool, other, flat, 1.0F, "op.0000");
  CHECK_THROWS(tap.observe(pool.view(a, "record-b"), context("op.0000")), input_error, "changed allocation");
  // The reference depends on the slot index: a grid of another slot is refused.
  const ticket b = make_ready(pool, spec, flat, 1.0F, "op.0001", SLOT + 1);
  CHECK_THROWS(tap.observe(pool.view(b, RECORD), context("op.0001", 0, SLOT + 1)), input_error, "depends on the slot index");
  CHECK_THROWS(tap.observe(pool.view(b, RECORD), context("op.0001", 0, SLOT)), input_error, "depends on the slot index");
  // A grid with another number of planes than the antenna map names is refused.
  const ticket c = make_ready(narrow, spec, flat, 1.0F, "op.0002");
  CHECK_THROWS(tap.observe(narrow.view(c, RECORD), context("op.0002")), input_error, "receive planes");
  CHECK_THROWS(tap.observe(pool.view(a, "record-b"), context("op.0000", 1024)), input_error, "");
  CHECK(tap.observations().empty());
}

void case_slot_times()
{
  CHECK(slots_per_frame(15) == 10 && slots_per_frame(30) == 20 && slots_per_frame(60) == 40 && slots_per_frame(120) == 80);
  CHECK(useful_tc(15) == 131072 && useful_tc(30) == 65536 && useful_tc(120) == 16384);
  // One subframe is 1,966,080 T_c at every spacing, and a frame is ten of them.
  for (const unsigned scs : {15U, 30U, 60U, 120U}) {
    const unsigned per = slots_per_frame(scs) / 10;
    CHECK(slot_start_tc(scs, 0) == 0 && slot_start_tc(scs, per) == 1966080ULL);
    CHECK(slot_start_tc(scs, slots_per_frame(scs)) == 19660800ULL);
    for (unsigned slot = 1; slot != 2 * per; ++slot) {
      CHECK(slot_start_tc(scs, slot) > slot_start_tc(scs, slot - 1));
    }
  }
  CHECK(slot_start_tc(30, 7) == 7ULL * 983040);
  // At 60 kHz only the first and the third slot of a subframe carry the longer prefix.
  CHECK(slot_start_tc(60, 1) == 14ULL * (32768 + 2304) + 1024 && slot_start_tc(60, 2) == 983040ULL);
  CHECK_THROWS(slot_start_tc(45, 0), input_error, "subcarrier spacing");
}

// ---------------------------------------------------------------------------------------------- reference modes

int print_vectors()
{
  char line[256];
  while (std::fgets(line, sizeof(line), stdin) != nullptr) {
    unsigned raw[4] = {0, 0, 0, 0};
    if (std::sscanf(line, "%x %x %x %x", &raw[0], &raw[1], &raw[2], &raw[3]) != 4) {
      std::cerr << "observer_core_test: a vector is four hexadecimal float32 patterns" << std::endl;
      return 2;
    }
    float value[4];
    for (unsigned n = 0; n != 4; ++n) {
      const std::uint32_t bits = raw[n];
      std::memcpy(&value[n], &bits, sizeof(float));
    }
    bool          ok       = false;
    const cf32    estimate = pilot_ls(cf32(value[0], value[1]), cf32(value[2], value[3]), &ok);
    const float   parts[2] = {estimate.real(), estimate.imag()};
    std::uint32_t bits[2];
    std::memcpy(bits, parts, sizeof(bits));
    std::printf("%08x %08x %d\n", static_cast<unsigned>(bits[0]), static_cast<unsigned>(bits[1]), ok ? 1 : 0);
  }
  return 0;
}

struct named_case {
  const char*           name;
  std::function<void()> body;
};

const named_case CASES[] = {
    {"kernel", case_kernel},
    {"amplitude_scale", case_amplitude_scale},
    {"data_invariance", case_data_invariance},
    {"pilot_hole", case_pilot_hole},
    {"refusals", case_refusals},
    {"antenna_map", case_antenna_map},
    {"coordinates", case_coordinates},
    {"record", case_record},
    {"ownership", case_ownership},
    {"binding", case_binding},
    {"slot_times", case_slot_times},
};

} // namespace

int main(int argc, char** argv)
{
  const std::string selected = argc > 1 ? argv[1] : "";
  if (selected == "--vectors") {
    return print_vectors();
  }
  if (selected == "--key" && argc == 3) {
    std::printf("%016llx\n", static_cast<unsigned long long>(opaque_key(argv[2])));
    return 0;
  }
  if (selected == "--slot-start" && argc == 4) {
    try {
      std::printf("%llu\n", static_cast<unsigned long long>(slot_start_tc(static_cast<unsigned>(std::atoi(argv[2])),
                                                                         std::strtoull(argv[3], nullptr, 10))));
    } catch (const std::exception& error) {
      std::cerr << "observer_core_test: " << error.what() << std::endl;
      return 2;
    }
    return 0;
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
    std::cerr << "observer_core_test: no case named '" << selected << "'" << std::endl;
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
