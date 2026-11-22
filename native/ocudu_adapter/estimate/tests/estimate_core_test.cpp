// Unit cases of the capture of the receiver's estimate (Sprint 5.2, selector native_estimate).
//
// The target links estimate_core, obs_publisher, observer_core, replay_core and obs_ring and
// nothing else: no OCUDU library, no CUDA and no test framework. The device is a fake
// asynchronous producer: one buffer that it overwrites with every launch, and copies that
// complete when the case says so. The consumer is the one of the ring library, in this process.
//
//     estimate_core_test             run every case
//     estimate_core_test NAME        run one case (CTest registers each)
//     estimate_core_test --list      print the case names
//     estimate_core_test --halves    read float bit patterns from standard input, one per line in
//                                    hexadecimal, and print the half of each
//     estimate_core_test --element D P R S  DMRS PRBS PORTS   print the position and the subcarrier
//
// alloc_count.cpp of transport/tests replaces the global allocation functions, so that a case
// can count allocations. ASCII only.

#include "estimate_core.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Allocations of this process so far, counted by the replaced allocation functions of alloc_count.cpp.
std::uint64_t allocation_count();

using namespace isac_estimate;
using isac_observer::bound_slot;
using isac_replay::hash_count;
using isac_replay::sha256_count;
using isac_replay::sha256_hex;

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

// One slot of a small cell: two DM-RS symbols, three resource blocks from block 5, two receive ports.
constexpr unsigned      DMRS_MASK = (1U << 2) | (1U << 11);
constexpr unsigned      START_PRB = 5;
constexpr unsigned      PRBS      = 3;
constexpr unsigned      PORTS     = 2;
constexpr unsigned      RNTI      = 0x4601;
constexpr unsigned      SFN       = 12;
constexpr unsigned      SLOT      = 7;
constexpr std::size_t   ELEMENTS  = 2 * PRBS * PORTS * 12; // 144
constexpr std::size_t   BYTES     = ELEMENTS * 4;          // 576
const float             SCALING   = 1.41253757F;           // 10^(3/20) as the receiver holds it

expectation small_expectation()
{
  expectation expected;
  expected.rnti             = RNTI;
  expected.sfn              = SFN;
  expected.slot             = SLOT;
  expected.start_prb        = START_PRB;
  expected.nof_prb          = PRBS;
  expected.rx_ports         = PORTS;
  expected.dmrs_type        = 1;
  expected.dmrs_symbol_mask = DMRS_MASK;
  return expected;
}

bound_slot small_slot(const std::string& opportunity, std::uint64_t absolute_slot = 247)
{
  bound_slot slot;
  slot.opportunity_id      = opportunity;
  slot.episode_id          = "rz-unit";
  slot.grant_record_sha256 = "record-a";
  slot.episode             = isac_observer::opaque_key(slot.episode_id);
  slot.opportunity         = isac_observer::opaque_key(opportunity);
  slot.grant               = isac_observer::opaque_key(slot.grant_record_sha256);
  slot.cell                = 1;
  slot.ue                  = RNTI;
  slot.sfn                 = SFN;
  slot.slot                = SLOT;
  slot.absolute_slot       = absolute_slot;
  slot.t_first_sample      = 1000 + absolute_slot;
  slot.t_last_sample       = 9000 + absolute_slot;
  return slot;
}

/// A fake asynchronous producer. It owns one "device" buffer, as the kernel handle of the
/// receiver does, and writes a new estimate into it with every launch. A copy that was submitted
/// is carried out when the case completes it. Copies on the producer's own stream are carried
/// out before a later launch overwrites the buffer, as a stream orders them. Copies on another
/// stream are not: they read the buffer as it is when they complete.
class fake_producer final : public copier
{
public:
  explicit fake_producer(bool own_stream_) : own_stream(own_stream_), device(BYTES, 0) {}

  /// Writes one estimate into the device buffer: every element is value, in the element's own
  /// half precision. Returns what the receiver reports of it.
  producer_view launch(cf32 value)
  {
    if (own_stream) {
      carry_out(); // the stream runs the copy that was enqueued before this launch first
    }
    for (std::size_t element = 0; element != ELEMENTS; ++element) {
      const std::uint16_t parts[2] = {float_to_half(value.real()), float_to_half(value.imag())};
      std::memcpy(device.data() + 4 * element, parts, 4);
    }
    producer_view view;
    view.estimates               = device.data();
    view.bytes                   = BYTES;
    view.stream                  = this;
    view.launch                  = ++launches;
    view.rnti                    = RNTI;
    view.sfn                     = SFN;
    view.slot                    = SLOT;
    view.numerology              = 1;
    view.start_prb               = START_PRB;
    view.nof_prb                 = PRBS;
    view.dmrs_symbol_mask        = DMRS_MASK;
    view.nof_dmrs_symbols        = 2;
    view.rx_ports                = PORTS;
    view.layers                  = 1;
    view.dmrs_scaling            = SCALING;
    view.frequency_smoothing     = true;
    view.resident                = true;
    return view;
  }

  /// The device finishes the copy that is in flight.
  void complete() { carry_out(); }
  void break_copy() { broken = true; }
  void refuse_submit(bool refuse) { refusing = refuse; }
  std::uint8_t* buffer() { return device.data(); }

  bool submit(const producer_view& view, void* destination) noexcept override
  {
    ++submits;
    if (refusing) {
      return false;
    }
    pending_source = static_cast<const std::uint8_t*>(view.estimates);
    pending_target = static_cast<std::uint8_t*>(destination);
    pending_bytes  = view.bytes;
    done           = false;
    return true;
  }
  copy_state poll() noexcept override
  {
    ++polls;
    return broken ? copy_state::failed : done ? copy_state::complete : copy_state::pending;
  }
  copy_state drain() noexcept override
  {
    ++drains;
    carry_out();
    return broken ? copy_state::failed : copy_state::complete;
  }
  bool        on_producer_stream() const noexcept override { return own_stream; }
  const char* stream_name() const noexcept override { return own_stream ? COPY_PRODUCER : COPY_SIDE; }

  unsigned submits  = 0;
  unsigned polls    = 0;
  unsigned drains   = 0;
  unsigned launches = 0;

private:
  void carry_out()
  {
    if (pending_target != nullptr && !done) {
      std::memcpy(pending_target, pending_source, pending_bytes);
      done = true;
    }
  }

  bool                      own_stream;
  bool                      broken   = false;
  bool                      refusing = false;
  bool                      done     = true;
  std::vector<std::uint8_t> device;
  const std::uint8_t*       pending_source = nullptr;
  std::uint8_t*             pending_target = nullptr;
  std::size_t               pending_bytes  = 0;
};

std::string scratch_directory()
{
  const char* base    = std::getenv("TMPDIR");
  std::string pattern = std::string(base != nullptr && *base != '\0' ? base : "/tmp") + "/estimate-core-test-XXXXXX";
  if (::mkdtemp(&pattern[0]) == nullptr) {
    throw std::runtime_error("cannot create a scratch directory");
  }
  return pattern;
}

/// One ring sized for the small estimate, with or without a consumer attached.
struct ring_fixture {
  std::string                    directory = scratch_directory();
  std::unique_ptr<obs::producer> writer;
  std::unique_ptr<obs::consumer> reader;

  explicit ring_fixture(std::uint32_t slots, bool attach = true, std::uint32_t payload_bytes = BYTES)
  {
    obs::ring_config config;
    config.socket_path      = directory + "/o3.sock";
    config.slot_count       = slots;
    config.host_capacity    = sizeof(isac_transport::staged_host);
    config.mask_capacity    = static_cast<std::uint32_t>((ELEMENTS + 7) / 8);
    config.payload_capacity = payload_bytes;
    writer.reset(new obs::producer(config));
    if (!attach) {
      return;
    }
    std::atomic<bool> finished{false};
    std::thread       worker([&]() {
      obs::consumer_config settings;
      settings.socket_path = config.socket_path;
      reader.reset(new obs::consumer(settings));
      finished = true;
    });
    while (!finished) {
      writer->service();
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    worker.join();
  }
  ~ring_fixture()
  {
    reader.reset();
    writer.reset();
    ::rmdir(directory.c_str());
  }
};

capture_config settings(evidence mode, std::vector<std::uint8_t>& staging, bool slot_landing = false,
                        std::int64_t limit_ns = 2000000)
{
  capture_config config;
  config.mode             = mode;
  config.slot_landing     = slot_landing;
  config.staging          = staging.data();
  config.staging_bytes    = staging.size();
  config.resolve_limit_ns = limit_ns;
  return config;
}

bool all_equal(const std::uint8_t* payload, cf32 value)
{
  for (std::size_t element = 0; element != ELEMENTS; ++element) {
    if (value_at(payload, element) != value) {
      return false;
    }
  }
  return true;
}

const isac_replay::ticket GRID = {1, 42};

// ---------------------------------------------------------------------------------------------- cases

void case_half()
{
  // Known halves.
  CHECK(half_to_float(0x3c00) == 1.0F && half_to_float(0xc000) == -2.0F && half_to_float(0x3800) == 0.5F);
  CHECK(half_to_float(0x3c01) == 1.0F + 1.0F / 1024.0F && half_to_float(0x7bff) == 65504.0F);
  CHECK(half_to_float(0x0400) == std::ldexp(1.0F, -14) && half_to_float(0x0001) == std::ldexp(1.0F, -24));
  CHECK(half_to_float(0x0000) == 0.0F && std::signbit(half_to_float(0x8000)) && half_to_float(0x8000) == 0.0F);
  CHECK(std::isinf(half_to_float(0x7c00)) && std::isinf(half_to_float(0xfc00)) && std::isnan(half_to_float(0x7e00)));
  CHECK(half_is_finite(0x7bff) && !half_is_finite(0x7c00) && !half_is_finite(0xfe01));
  // Every half is a float, and that float is that half again.
  unsigned same = 0;
  for (unsigned bits = 0; bits != 0x10000U; ++bits) {
    const std::uint16_t half = static_cast<std::uint16_t>(bits);
    same += (!half_is_finite(half) && (half & 0x3ffU) != 0) || float_to_half(half_to_float(half)) == half ? 1 : 0;
  }
  CHECK(same == 0x10000U);
  // Round to nearest, ties to even. A unity channel reads as 1.00119 in single precision and as
  // 1 + 2^-10 in the estimate of the receiver.
  CHECK(float_to_half(1.0011865F) == 0x3c01 && float_to_half(1.0F) == 0x3c00 && float_to_half(0.5F) == 0x3800);
  CHECK(float_to_half(1.0F + std::ldexp(1.0F, -11)) == 0x3c00);        // halfway, the even one below
  CHECK(float_to_half(1.0F + 3.0F * std::ldexp(1.0F, -11)) == 0x3c02); // halfway, the even one above
  CHECK(float_to_half(1.0F + std::ldexp(1.0F, -11) + std::ldexp(1.0F, -20)) == 0x3c01);
  CHECK(float_to_half(65519.0F) == 0x7bff && float_to_half(65520.0F) == 0x7c00 && float_to_half(-1.0e9F) == 0xfc00);
  CHECK(float_to_half(std::ldexp(1.0F, -25)) == 0x0000 && float_to_half(std::ldexp(1.5F, -25)) == 0x0001);
  CHECK(float_to_half(std::ldexp(1.0F, -24)) == 0x0001 && float_to_half(1.0e-9F) == 0x0000);
  CHECK(float_to_half(std::ldexp(1.0F, -14) - std::ldexp(1.0F, -25)) == 0x0400); // rounds up into the normal range
  CHECK(float_to_half(std::nanf("")) == 0x7e00 && float_to_half(-0.0F) == 0x8000);
  // A pair converts to complex64 within the rounding of a half: 2^-11 of each part in the normal range.
  const float         parts[2]  = {0.73912F, -1.2907e-3F};
  const std::uint16_t halves[2] = {float_to_half(parts[0]), float_to_half(parts[1])};
  const cf32          value     = value_at(halves, 0);
  CHECK(std::fabs(value.real() - parts[0]) <= std::ldexp(std::fabs(parts[0]), -11));
  CHECK(std::fabs(value.imag() - parts[1]) <= std::ldexp(std::fabs(parts[1]), -11));
  const std::uint16_t exact[2] = {float_to_half(1.0F), float_to_half(0.5F)};
  CHECK(value_at(exact, 0) == cf32(1.0F, 0.5F));
}

void case_layout()
{
  // The demonstration: two DM-RS symbols, 51 resource blocks, four receive ports.
  layout demo;
  demo.dmrs_symbols = 2;
  demo.prbs         = 51;
  demo.rx_ports     = 4;
  CHECK(demo.elements() == 4896 && demo.payload_bytes() == 19584 && demo.mask_bytes() == 612);
  // Element [1][3][2][5] is DM-RS symbol 1, receive port 2 and subcarrier 41 of the allocation.
  const std::size_t element = demo.element(1, 3, 2, 5);
  const coordinate  where   = coordinate_of(demo, element);
  CHECK(element == 2621);
  CHECK(where.dmrs_symbol == 1 && where.prb == 3 && where.rx_port == 2 && where.subcarrier_in_prb == 5 &&
        where.subcarrier == 41);
  CHECK(demo.element(0, 0, 0, 0) == 0 && demo.element(1, 50, 3, 11) == 4895);
  // The last axis is the fastest, then the receive port, then the block.
  CHECK(demo.element(0, 0, 0, 1) == 1 && demo.element(0, 0, 1, 0) == 12 && demo.element(0, 1, 0, 0) == 48 &&
        demo.element(1, 0, 0, 0) == 2448);
  CHECK_THROWS(demo.element(2, 0, 0, 0), input_error, "outside a payload of 2 by 51 by 4 by 12");
  CHECK_THROWS(demo.element(0, 51, 0, 0), input_error, "outside a payload");
  CHECK_THROWS(demo.element(0, 0, 4, 0), input_error, "outside a payload");
  CHECK_THROWS(demo.element(0, 0, 0, 12), input_error, "outside a payload");
  CHECK_THROWS(coordinate_of(demo, 4896), input_error, "outside a payload of 4896 elements");
  // The expectation of a decode gives the same rectangle.
  expectation expected      = small_expectation();
  expected.nof_prb          = 51;
  expected.rx_ports         = 4;
  const layout rectangle    = expected.shape();
  CHECK(rectangle.dmrs_symbols == 2 && rectangle.prbs == 51 && rectangle.rx_ports == 4 && rectangle.elements() == 4896);
}

void case_zero_and_absent()
{
  // A known flat channel of 1 + 0.5j: every element is valid and none is zero.
  std::vector<std::uint8_t> payload(BYTES, 0);
  std::vector<std::uint8_t> mask((ELEMENTS + 7) / 8, 0xff);
  const std::uint16_t       flat[2] = {float_to_half(1.0F), float_to_half(0.5F)};
  for (std::size_t element = 0; element != ELEMENTS; ++element) {
    std::memcpy(payload.data() + 4 * element, flat, 4);
  }
  payload_counts counts = scan(payload.data(), ELEMENTS, mask.data());
  CHECK(counts.valid == ELEMENTS && counts.holes == 0 && counts.nonzero == ELEMENTS);
  CHECK(mask[0] == 0xff && mask[(ELEMENTS + 7) / 8 - 1] == 0xff && all_equal(payload.data(), cf32(1.0F, 0.5F)));
  // A zero-filled buffer is what a stub returns. Every element is a valid value and none is
  // nonzero: against a reference that is not zero the parity fails on the count.
  std::fill(payload.begin(), payload.end(), std::uint8_t(0));
  counts = scan(payload.data(), ELEMENTS, mask.data());
  CHECK(counts.valid == ELEMENTS && counts.holes == 0 && counts.nonzero == 0);
  // A true zero is a valid zero: one measured zero among nonzero elements keeps its bit.
  for (std::size_t element = 1; element != ELEMENTS; ++element) {
    std::memcpy(payload.data() + 4 * element, flat, 4);
  }
  counts = scan(payload.data(), ELEMENTS, mask.data());
  CHECK(counts.valid == ELEMENTS && counts.nonzero == ELEMENTS - 1 && (mask[0] & 1U) == 1 &&
        value_at(payload.data(), 0) == cf32(0.0F, 0.0F));
  // A negative zero is zero. A part that is not finite clears the bit of its element alone.
  const std::uint16_t minus_zero[2] = {0x8000, 0x0000};
  const std::uint16_t infinite[2]   = {0x3c00, 0x7c00};
  const std::uint16_t not_number[2] = {0x7e00, 0x3c00};
  std::memcpy(payload.data() + 4 * 3, minus_zero, 4);
  std::memcpy(payload.data() + 4 * 9, infinite, 4);
  std::memcpy(payload.data() + 4 * 10, not_number, 4);
  counts = scan(payload.data(), ELEMENTS, mask.data());
  CHECK(counts.valid == ELEMENTS - 2 && counts.holes == 2 && counts.nonzero == ELEMENTS - 4);
  CHECK(mask[0] == 0xff && mask[1] == static_cast<std::uint8_t>(0xff & ~0x06));
  // A count that is no multiple of eight leaves the bits beyond it cleared.
  std::vector<std::uint8_t> short_mask(2, 0xff);
  counts = scan(payload.data(), 9, short_mask.data());
  CHECK(counts.valid == 9 && short_mask[0] == 0xff && short_mask[1] == 0x01);
}

void case_refusals()
{
  fake_producer             device(true);
  const expectation         expected = small_expectation();
  const producer_view       good     = device.launch(cf32(1.0F, 0.5F));
  std::vector<std::uint8_t> staging(BYTES);
  CHECK(std::string(refusal(expected, good)).empty());
  const auto changed = [&](const std::function<void(producer_view&)>& edit) {
    producer_view view = good;
    edit(view);
    return std::string(refusal(expected, view));
  };
  CHECK(changed([](producer_view& view) { view.estimates = nullptr; }) == "null_estimate");
  CHECK(changed([](producer_view& view) { view.layers = 2; }) == "layers");
  CHECK(changed([](producer_view& view) { view.contiguous = false; }) == "not_contiguous");
  CHECK(changed([](producer_view& view) { view.frequency_interpolation = false; }) == "not_interpolated");
  CHECK(changed([](producer_view& view) { view.rnti = 0x4602; }) == "other_pdu");
  CHECK(changed([](producer_view& view) { view.sfn = 13; }) == "other_pdu");
  CHECK(changed([](producer_view& view) { view.slot = 17; }) == "other_pdu");
  CHECK(changed([](producer_view& view) { view.start_prb = 6; }) == "other_allocation");
  CHECK(changed([](producer_view& view) { view.nof_prb = 4; }) == "other_allocation");
  CHECK(changed([](producer_view& view) { view.dmrs_symbol_mask = 1U << 2; }) == "other_allocation");
  CHECK(changed([](producer_view& view) { view.rx_ports = 4; }) == "other_allocation");
  CHECK(changed([](producer_view& view) { view.dmrs_type = 2; }) == "other_allocation");
  CHECK(changed([](producer_view& view) { view.bytes = BYTES - 4; }) == "size");
  // Without smoothing the estimate is still the expected one: the record names the steps.
  CHECK(changed([](producer_view& view) { view.frequency_smoothing = false; }).empty());

  // The capture: what it needs before the run.
  CHECK_THROWS(capture(device, nullptr, settings(evidence::timed, staging)), input_error, "the capture has no ring");
  capture_config none = settings(evidence::audit, staging);
  none.staging        = nullptr;
  CHECK_THROWS(capture(device, nullptr, none), input_error, "has no staging");
  capture_config late = settings(evidence::audit, staging);
  late.resolve_limit_ns = -1;
  CHECK_THROWS(capture(device, nullptr, late), input_error, "is not negative");

  capture     plain(device, nullptr, settings(evidence::audit, staging));
  expectation wide = expected;
  wide.nof_prb     = 4;
  CHECK_THROWS(plain.prepare(small_slot("op.wide"), wide, {0, 1}, 0, 0), input_error, "the staging cannot carry");
  CHECK_THROWS(plain.prepare(small_slot("op.map"), expected, {0}, 0, 0), input_error, "names 1 array elements for 2");
  CHECK_THROWS(plain.prepare(small_slot("op.twice"), expected, {3, 3}, 0, 0), input_error, "names an array element twice");
  expectation empty = expected;
  empty.dmrs_symbol_mask = 0;
  CHECK_THROWS(plain.prepare(small_slot("op.empty"), empty, {0, 1}, 0, 0), input_error, "has no element");
  expectation many = expected;
  many.rx_ports    = 17;
  std::vector<std::uint8_t> large(2 * PRBS * 17 * 12 * 4);
  capture                   roomy(device, nullptr, settings(evidence::audit, large));
  CHECK_THROWS(roomy.prepare(small_slot("op.many"), many, std::vector<unsigned>(17, 0), 0, 0), input_error,
               "at most 16 receive planes");
  expectation high      = expected;
  high.dmrs_symbol_mask = (1ULL << 33) | (1ULL << 2);
  CHECK_THROWS(plain.prepare(small_slot("op.high"), high, {0, 1}, 0, 0), input_error, "DM-RS symbols 0 to 31");
  ring_fixture narrow(2, false, BYTES - 64);
  capture      tight(device, narrow.writer.get(), settings(evidence::timed, staging));
  CHECK_THROWS(tight.prepare(small_slot("op.ring"), expected, {0, 1}, 0, 0), input_error, "the ring cannot carry");
  CHECK(plain.prepared() == 0 && tight.prepared() == 0);

  // The order of the three steps.
  const std::size_t handle = plain.prepare(small_slot("op.0000"), expected, {5, 3}, 0, 0);
  CHECK_THROWS(plain.resolve(), state_error, "no capture is open");
  CHECK_THROWS(plain.expect(handle + 1, GRID), state_error, "no prepared decode 1");
  plain.expect(handle, GRID);
  CHECK_THROWS(plain.expect(handle, GRID), state_error, "a capture is open");
  CHECK(plain.resolve().result == outcome::absent);
  CHECK(std::string(to_string(outcome::published)) == "published" && std::string(to_string(outcome::failed)) == "failed");
}

void case_async_completion()
{
  // The producer's stream orders the copy before the next launch. Two estimates: 1 + 0.5j, then -2 + 0.25j.
  fake_producer             device(true);
  ring_fixture              ring(4);
  std::vector<std::uint8_t> staging(BYTES);
  capture                   taker(device, ring.writer.get(), settings(evidence::timed, staging, false, 0));
  const std::size_t         first  = taker.prepare(small_slot("op.0000"), small_expectation(), {5, 3}, 0, 0);
  const std::size_t         second = taker.prepare(small_slot("op.0001", 248), small_expectation(), {5, 3}, 0, 1);
  taker.reserve(8);

  // A capture is attempted before the copy completed. Nothing is published, and the consumer sees nothing.
  taker.expect(first, GRID);
  taker.on_estimate(device.launch(cf32(1.0F, 0.5F)));
  const capture_line early = taker.resolve();
  CHECK(early.result == outcome::failed && std::string(early.reason) == "timeout" && !early.landed && early.seen &&
        early.admitted && early.polls == 1 && device.submits == 1);
  CHECK(taker.in_flight() && ring.writer->published() == 0 && ring.writer->counts().cancelled == 0);
  obs::record_view view;
  CHECK(ring.reader->next(view, std::chrono::microseconds(200)) == obs::read_result::none);
  // The slot of that copy stays acquired, and while the copy can still write its destination
  // the next capture admits nothing, copies nothing and waits for nothing.
  taker.expect(second, GRID);
  taker.on_estimate(device.launch(cf32(-2.0F, 0.25F)));
  const capture_line held = taker.resolve();
  CHECK(held.result == outcome::failed && std::string(held.reason) == "busy" && held.seen && !held.admitted &&
        !held.landed && held.polls == 0 && device.submits == 1 && device.drains == 0);
  CHECK(taker.in_flight() && ring.writer->published() == 0 && ring.writer->counts().cancelled == 0 &&
        ring.writer->counts().dropped == 0);

  // The stream has run the old copy by now. The next capture finds it ended, gives its slot back
  // and proceeds. The same estimate again, and this time the copy completes before the capture
  // is resolved.
  taker.expect(first, GRID);
  CHECK(!taker.in_flight() && ring.writer->counts().cancelled == 1 && device.drains == 0);
  taker.on_estimate(device.launch(cf32(1.0F, 0.5F)));
  device.complete();
  const capture_line done = taker.resolve();
  CHECK(done.result == outcome::published && std::string(done.reason).empty() && done.landed && done.sequence == 0 &&
        done.counts.valid == ELEMENTS && done.counts.nonzero == ELEMENTS && done.producer.launch == 3);
  CHECK(std::string(done.copy_stream) == COPY_PRODUCER && std::string(done.landing) == LANDING_STAGING);
  CHECK(done.t_expect_ns <= done.t_tap_ns && done.t_tap_ns <= done.t_submit_ns && done.t_submit_ns <= done.t_resolve_ns &&
        done.t_resolve_ns <= done.t_complete_ns && done.t_complete_ns <= done.t_publish_ns &&
        done.t_publish_ns <= done.t_done_ns);
  // The producer overwrites its buffer at once: the next PDU. The published record keeps its values.
  taker.expect(second, GRID);
  taker.on_estimate(device.launch(cf32(-2.0F, 0.25F)));
  CHECK(value_at(device.buffer(), 0) == cf32(-2.0F, 0.25F));
  CHECK(ring.reader->next(view, std::chrono::milliseconds(200)) == obs::read_result::record);
  CHECK(view.sequence == 0 && all_equal(view.payload, cf32(1.0F, 0.5F)) && ring.reader->intact(view));
  // The second estimate is published when its own copy completed, with its own values.
  device.complete();
  const capture_line next = taker.resolve();
  CHECK(next.result == outcome::published && next.sequence == 1 && next.producer.launch == 4);
  CHECK(ring.reader->next(view, std::chrono::milliseconds(200)) == obs::read_result::record);
  CHECK(view.sequence == 1 && all_equal(view.payload, cf32(-2.0F, 0.25F)) &&
        view.header->opportunity == isac_observer::opaque_key("op.0001") && view.header->absolute_slot == 248);
  CHECK(taker.account().size() == 4 && taker.unexpected() == 0 && taker.overruns() == 0);
  // After the last decode the owner may wait for a copy that ran out of time: settle() does.
  taker.expect(first, GRID);
  taker.on_estimate(device.launch(cf32(1.0F, 0.5F)));
  CHECK(taker.resolve().result == outcome::failed && taker.in_flight());
  taker.settle();
  CHECK(!taker.in_flight() && device.drains == 1 && ring.writer->counts().cancelled == 2 &&
        ring.writer->published() == 2);

  // A copy that the device reports as failed is not published either.
  fake_producer             broken(true);
  capture                   other(broken, ring.writer.get(), settings(evidence::timed, staging));
  const std::size_t         handle = other.prepare(small_slot("op.0002", 249), small_expectation(), {5, 3}, 0, 2);
  other.expect(handle, GRID);
  other.on_estimate(broken.launch(cf32(1.0F, 0.5F)));
  broken.break_copy();
  const capture_line lost = other.resolve();
  CHECK(lost.result == outcome::failed && std::string(lost.reason) == "copy" && !lost.landed &&
        ring.writer->published() == 2);
  // And one that the device did not take is a failure of the submission.
  fake_producer     deaf(true);
  capture           unheard(deaf, ring.writer.get(), settings(evidence::timed, staging));
  const std::size_t quiet = unheard.prepare(small_slot("op.0003", 250), small_expectation(), {5, 3}, 0, 3);
  deaf.refuse_submit(true);
  unheard.expect(quiet, GRID);
  unheard.on_estimate(deaf.launch(cf32(1.0F, 0.5F)));
  const capture_line refused = unheard.resolve();
  CHECK(refused.result == outcome::failed && std::string(refused.reason) == "submit" && deaf.polls == 0 &&
        ring.writer->published() == 2);
}

void case_generation()
{
  std::vector<std::uint8_t> staging(BYTES);
  // A copy on a stream of its own is ordered after its producer by an event, and by nothing
  // before the next launch. The producer launches again before the copy completed: the copy
  // holds the values of the later estimate, and it must not be published as the earlier one.
  {
    fake_producer     device(false);
    ring_fixture      ring(4);
    capture           taker(device, ring.writer.get(), settings(evidence::timed, staging));
    const std::size_t handle = taker.prepare(small_slot("op.0000"), small_expectation(), {5, 3}, 0, 0);
    taker.reserve(4);
    taker.expect(handle, GRID);
    taker.on_estimate(device.launch(cf32(1.0F, 0.5F)));
    taker.on_estimate(device.launch(cf32(-2.0F, 0.25F))); // the next PDU of the same demodulator
    device.complete();
    CHECK(all_equal(staging.data(), cf32(-2.0F, 0.25F))); // what a blind publication would have carried
    const capture_line torn = taker.resolve();
    CHECK(torn.result == outcome::failed && std::string(torn.reason) == "overtaken" && !torn.landed &&
          torn.producer.launch == 1 && std::string(torn.copy_stream) == COPY_SIDE);
    CHECK(taker.overruns() == 1 && ring.writer->published() == 0 && ring.writer->counts().cancelled == 1);
    // Resolved before the next launch, the same copy is whole and is published.
    taker.expect(handle, GRID);
    taker.on_estimate(device.launch(cf32(1.0F, 0.5F)));
    device.complete();
    const capture_line whole = taker.resolve();
    taker.on_estimate(device.launch(cf32(-2.0F, 0.25F))); // nobody expects it: counted, nothing happens
    CHECK(whole.result == outcome::published && taker.unexpected() == 1 && taker.overruns() == 1);
    obs::record_view view;
    CHECK(ring.reader->next(view, std::chrono::milliseconds(200)) == obs::read_result::record);
    CHECK(all_equal(view.payload, cf32(1.0F, 0.5F)) && view.sequence == 0);
  }
  // On the producer's own stream the copy runs before the later launch: the earlier estimate is
  // published whole, and the later one is counted, because one capture is open at a time.
  {
    fake_producer     device(true);
    ring_fixture      ring(4);
    capture           taker(device, ring.writer.get(), settings(evidence::timed, staging));
    const std::size_t handle = taker.prepare(small_slot("op.0000"), small_expectation(), {5, 3}, 0, 0);
    taker.reserve(4);
    taker.expect(handle, GRID);
    taker.on_estimate(device.launch(cf32(1.0F, 0.5F)));
    taker.on_estimate(device.launch(cf32(-2.0F, 0.25F)));
    device.complete();
    const capture_line kept = taker.resolve();
    CHECK(kept.result == outcome::published && kept.producer.launch == 1 && taker.overruns() == 1);
    obs::record_view view;
    CHECK(ring.reader->next(view, std::chrono::milliseconds(200)) == obs::read_result::record);
    CHECK(all_equal(view.payload, cf32(1.0F, 0.5F)));
  }
  // An estimate of another PDU is refused and not copied.
  {
    fake_producer     device(true);
    ring_fixture      ring(4);
    capture           taker(device, ring.writer.get(), settings(evidence::timed, staging));
    const std::size_t handle = taker.prepare(small_slot("op.0000"), small_expectation(), {5, 3}, 0, 0);
    taker.expect(handle, GRID);
    producer_view other = device.launch(cf32(1.0F, 0.5F));
    other.slot          = 17;
    taker.on_estimate(other);
    const capture_line line = taker.resolve();
    CHECK(line.result == outcome::refused && std::string(line.reason) == "other_pdu" && line.seen && !line.landed &&
          device.submits == 0 && ring.writer->published() == 0 && ring.writer->counts().cancelled == 1);
  }
}

void case_timed()
{
  // The copy lands in the slot that was acquired for it. Nothing is hashed or allocated, and the
  // ring makes no system call between the admission and the publication store.
  fake_producer             device(true);
  ring_fixture              ring(4);
  std::vector<std::uint8_t> staging(BYTES, 0xee);
  capture                   taker(device, ring.writer.get(), settings(evidence::timed, staging, true));
  const std::size_t         handle = taker.prepare(small_slot("op.0000"), small_expectation(), {5, 3}, 0, 9);
  taker.reserve(4);
  const producer_view view = device.launch(cf32(1.0F, 0.5F));

  const hash_count    hashed    = sha256_count();
  const std::uint64_t allocated = allocation_count();
  const std::uint64_t calls     = obs::sys::calls();
  taker.expect(handle, GRID);
  taker.on_estimate(view);
  device.complete();
  const capture_line line = taker.resolve();
  CHECK(allocation_count() == allocated);
  CHECK(obs::sys::calls() == calls && line.window_calls == 0);
  const hash_count after = sha256_count();
  CHECK(after.digests == hashed.digests && after.updates == hashed.updates && after.bytes == hashed.bytes);

  CHECK(line.result == outcome::published && std::string(line.landing) == LANDING_SLOT && line.landed &&
        line.bytes == 256 + 40 + 18 + BYTES && line.depth == 0 && line.epoch == ring.writer->epoch());
  CHECK(staging[0] == 0xee && staging[BYTES - 1] == 0xee); // the staging was not touched
  obs::record_view record;
  CHECK(ring.reader->next(record, std::chrono::milliseconds(200)) == obs::read_result::record);
  const obs::record_header& header = *record.header;
  CHECK(header.stage == obs::STAGE_O3 && header.source == obs::SOURCE_PUSCH_DMRS &&
        header.layout == obs::LAYOUT_OCUDU_O3_FP16 && header.dtype == obs::DTYPE_CFP16);
  CHECK(header.producer_steps == (obs::STEP_PILOT_LS | obs::STEP_FREQ_INTERP | obs::STEP_SMOOTHING) &&
        header.scale == static_cast<double>(SCALING));
  CHECK(header.rank == 4 && header.shape[0] == 2 && header.shape[1] == PRBS && header.shape[2] == PORTS &&
        header.shape[3] == 12 && header.strides[0] == PRBS * PORTS * 12 && header.strides[1] == PORTS * 12 &&
        header.strides[2] == 12 && header.strides[3] == 1);
  CHECK(header.first_subcarrier == 12 * START_PRB && header.subcarrier_step == 1 && header.subcarrier_count == 12 * PRBS &&
        header.dmrs_symbols == DMRS_MASK && header.rx_count == PORTS && header.layer_count == 1 &&
        header.antenna_map[0] == 5 && header.antenna_map[1] == 3 && header.port == 0);
  CHECK(header.payload_bytes == BYTES && header.mask_bytes == 18 && header.host_bytes == 40 && header.valid == ELEMENTS &&
        header.evidence == obs::EVIDENCE_TIMED && header.digest_kind == obs::DIGEST_NONE && header.digest[0] == 0 &&
        header.time_unit == obs::TIME_TC);
  CHECK(header.episode == isac_observer::opaque_key("rz-unit") && header.opportunity == isac_observer::opaque_key("op.0000") &&
        header.absolute_slot == 247 && header.cell == 1 && header.ue == RNTI && header.t_first_sample == 1247 &&
        header.t_last_sample == 9247 && header.t_ready_ns == line.t_complete_ns && header.t_publish_ns == line.t_publish_ns);
  isac_transport::staged_host host;
  std::memcpy(&host, record.host, sizeof(host));
  CHECK(host.kind == isac_transport::HOST_STAGED_HARNESS && host.opportunity_sequence == 9 && host.grid_index == 1 &&
        host.grid_generation == 42 && host.sfn == SFN && host.slot == SLOT && host.t_read_ns == line.t_tap_ns);
  // The bytes of the record are the bytes of the device buffer, and every validity bit is set.
  // The consumer maps the segment at its own address: the producer names the same slot through its own mapping.
  CHECK(std::memcmp(record.payload, device.buffer(), BYTES) == 0 && taker.payload() != staging.data() &&
        std::memcmp(taker.payload(), record.payload, BYTES) == 0);
  CHECK(record.mask[0] == 0xff && record.mask[17] == 0xff && std::memcmp(taker.mask(), record.mask, 18) == 0);

  // Without a page-locked ring the copy lands in the staging and is copied into the slot.
  capture           staged(device, ring.writer.get(), settings(evidence::timed, staging, false));
  const std::size_t other = staged.prepare(small_slot("op.0001", 248), small_expectation(), {5, 3}, 0, 10);
  staged.reserve(2);
  staged.expect(other, GRID);
  staged.on_estimate(device.launch(cf32(-2.0F, 0.25F)));
  device.complete();
  const capture_line second = staged.resolve();
  CHECK(second.result == outcome::published && std::string(second.landing) == LANDING_STAGING && second.sequence == 1);
  CHECK(all_equal(staging.data(), cf32(-2.0F, 0.25F)));
  CHECK(ring.reader->next(record, std::chrono::milliseconds(200)) == obs::read_result::record);
  CHECK(std::memcmp(record.payload, device.buffer(), BYTES) == 0 && record.header->valid == ELEMENTS);
}

void case_audit()
{
  fake_producer             device(true);
  std::vector<std::uint8_t> staging(BYTES);
  // Without a ring the estimate is captured for the archive.
  capture           plain(device, nullptr, settings(evidence::audit, staging));
  const std::size_t handle = plain.prepare(small_slot("op.0000"), small_expectation(), {5, 3}, 0, 0);
  plain.expect(handle, GRID);
  plain.on_estimate(device.launch(cf32(1.0F, 0.5F)));
  device.complete();
  const capture_line kept = plain.resolve();
  CHECK(kept.result == outcome::captured && kept.landed && !kept.admitted && plain.payload() == staging.data() &&
        std::memcmp(plain.payload(), device.buffer(), BYTES) == 0 && plain.mask()[0] == 0xff &&
        kept.counts.valid == ELEMENTS);
  // With a ring the audited record carries the SHA-256 of its payload.
  ring_fixture      ring(2);
  capture           audited(device, ring.writer.get(), settings(evidence::audit, staging, true));
  const std::size_t first = audited.prepare(small_slot("op.0001", 248), small_expectation(), {5, 3}, 0, 1);
  audited.reserve(4);
  const hash_count before = sha256_count();
  audited.expect(first, GRID);
  audited.on_estimate(device.launch(cf32(1.0F, 0.5F)));
  device.complete();
  const capture_line sent = audited.resolve();
  CHECK(sent.result == outcome::published && std::string(sent.landing) == LANDING_STAGING);
  CHECK(sha256_count().digests == before.digests + 1 && sha256_count().bytes == before.bytes + BYTES);
  obs::record_view record;
  CHECK(ring.reader->next(record, std::chrono::milliseconds(200)) == obs::read_result::record);
  const std::string digest = sha256_hex(record.payload, BYTES);
  char              text[65];
  for (unsigned n = 0; n != 32; ++n) {
    std::snprintf(text + 2 * n, 3, "%02x", record.header->digest[n]);
  }
  CHECK(record.header->digest_kind == obs::DIGEST_SHA256 && record.header->evidence == obs::EVIDENCE_AUDIT &&
        digest == text);
  // No credit: the ring drops the record, and the audited payload is complete all the same.
  audited.expect(first, GRID);
  audited.on_estimate(device.launch(cf32(-2.0F, 0.25F)));
  device.complete();
  CHECK(audited.resolve().result == outcome::published); // the second slot
  audited.expect(first, GRID);
  audited.on_estimate(device.launch(cf32(0.5F, -0.5F)));
  device.complete();
  const capture_line dropped = audited.resolve();
  CHECK(dropped.result == outcome::dropped && std::string(dropped.reason) == "no_credit" && dropped.landed &&
        !dropped.admitted && all_equal(audited.payload(), cf32(0.5F, -0.5F)) && ring.writer->counts().dropped == 1);
}

void case_drop()
{
  // A consumer that is delayed: it releases nothing. The ring of two slots admits two estimates.
  // The third finds no credit: nothing is copied, nothing in the ring is overwritten, and the
  // capture returns at once.
  fake_producer             device(true);
  ring_fixture              ring(2);
  std::vector<std::uint8_t> staging(BYTES);
  capture                   taker(device, ring.writer.get(), settings(evidence::timed, staging, true));
  const std::size_t         handle = taker.prepare(small_slot("op.0000"), small_expectation(), {5, 3}, 0, 0);
  taker.reserve(8);
  const cf32 values[3] = {cf32(1.0F, 0.5F), cf32(-2.0F, 0.25F), cf32(0.5F, -0.5F)};
  for (unsigned n = 0; n != 3; ++n) {
    taker.expect(handle, GRID);
    taker.on_estimate(device.launch(values[n]));
    device.complete();
    const capture_line line = taker.resolve();
    CHECK((line.result == outcome::published) == (n < 2));
    if (n == 2) {
      CHECK(line.result == outcome::dropped && std::string(line.reason) == "no_credit" && line.seen && !line.admitted &&
            !line.landed && std::string(line.landing).empty() && line.producer.launch == 3);
    }
  }
  CHECK(device.submits == 2 && ring.writer->published() == 2);
  const obs::producer_counts counts = ring.writer->counts();
  CHECK(counts.admitted == 2 && counts.published == 2 && counts.dropped == 1 && counts.cancelled == 0 &&
        counts.pending == 2);
  // The two records are the first two estimates, in order.
  obs::record_view record;
  for (unsigned n = 0; n != 2; ++n) {
    CHECK(ring.reader->next(record, std::chrono::milliseconds(200)) == obs::read_result::record);
    CHECK(record.sequence == n && all_equal(record.payload, values[n]));
  }
  // Once the consumer released one record, the next estimate is admitted again.
  ring.reader->release(1);
  taker.expect(handle, GRID);
  taker.on_estimate(device.launch(values[2]));
  device.complete();
  CHECK(taker.resolve().result == outcome::published && ring.writer->published() == 3);
  // Every line of the account is one of the four ends, and the counts of the ring are those of the lines.
  std::size_t published = 0, dropped = 0, admitted = 0;
  for (const capture_line& line : taker.account()) {
    published += line.result == outcome::published ? 1 : 0;
    dropped += line.result == outcome::dropped ? 1 : 0;
    admitted += line.admitted ? 1 : 0;
  }
  CHECK(taker.account().size() == 4 && published == 3 && dropped == 1 && admitted == 3);
}

void case_absent()
{
  // The decode ran on the host path: the receiver reports no estimate. The slot is given back
  // and the next estimate takes the same sequence.
  fake_producer             device(true);
  ring_fixture              ring(4);
  std::vector<std::uint8_t> staging(BYTES);
  capture                   taker(device, ring.writer.get(), settings(evidence::timed, staging, true));
  const std::size_t         handle = taker.prepare(small_slot("op.0000"), small_expectation(), {5, 3}, 0, 0);
  taker.reserve(4);
  taker.expect(handle, GRID);
  const capture_line none = taker.resolve();
  CHECK(none.result == outcome::absent && std::string(none.reason) == "no_estimate" && !none.seen && none.admitted &&
        !none.landed && none.polls == 0 && device.submits == 0);
  CHECK(ring.writer->counts().cancelled == 1 && ring.writer->published() == 0 && taker.payload() == nullptr);
  // An estimate that arrives after its capture was resolved fills nothing.
  taker.on_estimate(device.launch(cf32(1.0F, 0.5F)));
  CHECK(taker.unexpected() == 1 && device.submits == 0);
  taker.expect(handle, GRID);
  taker.on_estimate(device.launch(cf32(1.0F, 0.5F)));
  device.complete();
  const capture_line sent = taker.resolve();
  CHECK(sent.result == outcome::published && sent.sequence == 0);
  obs::record_view record;
  CHECK(ring.reader->next(record, std::chrono::milliseconds(200)) == obs::read_result::record && record.sequence == 0);
}

void case_threads()
{
  // The tap is called by the thread that demodulates. The owner expects and resolves on its own.
  fake_producer             device(true);
  ring_fixture              ring(8);
  std::vector<std::uint8_t> staging(BYTES);
  capture                   taker(device, ring.writer.get(), settings(evidence::timed, staging, true, 50000000));
  const std::size_t         handle = taker.prepare(small_slot("op.0000"), small_expectation(), {5, 3}, 0, 0);
  taker.reserve(64);
  unsigned published = 0;
  for (unsigned n = 0; n != 40; ++n) {
    taker.expect(handle, GRID);
    const producer_view view = device.launch(cf32(static_cast<float>(n), 0.5F));
    std::thread         radio([&]() {
      taker.on_estimate(view);
      device.complete();
    });
    radio.join();
    const capture_line line = taker.resolve();
    published += line.result == outcome::published ? 1 : 0;
    obs::record_view record;
    CHECK(ring.reader->next(record, std::chrono::milliseconds(500)) == obs::read_result::record);
    CHECK(record.sequence == n && all_equal(record.payload, cf32(static_cast<float>(n), 0.5F)));
    ring.reader->release(record.sequence + 1);
  }
  CHECK(published == 40 && taker.unexpected() == 0 && taker.overruns() == 0 && ring.writer->counts().dropped == 0);
}

struct named_case {
  const char*           name;
  std::function<void()> body;
};

const named_case CASES[] = {
    {"half", case_half},
    {"layout", case_layout},
    {"zero_and_absent", case_zero_and_absent},
    {"refusals", case_refusals},
    {"async_completion", case_async_completion},
    {"generation", case_generation},
    {"timed", case_timed},
    {"audit", case_audit},
    {"drop", case_drop},
    {"absent", case_absent},
    {"threads", case_threads},
};

int print_halves()
{
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    const std::uint32_t word = static_cast<std::uint32_t>(std::strtoul(line.c_str(), nullptr, 16));
    float               value;
    std::memcpy(&value, &word, sizeof(value));
    const std::uint16_t half = float_to_half(value);
    const float         back = half_to_float(half);
    std::uint32_t       wide;
    std::memcpy(&wide, &back, sizeof(wide));
    std::printf("%04x %08x\n", half, wide);
  }
  return 0;
}

} // namespace

int main(int argc, char** argv)
{
  const std::string selected = argc > 1 ? argv[1] : "";
  if (selected == "--halves") {
    return print_halves();
  }
  if (selected == "--element" && argc == 9) {
    try {
      layout shape;
      shape.dmrs_symbols        = static_cast<unsigned>(std::atoi(argv[6]));
      shape.prbs                = static_cast<unsigned>(std::atoi(argv[7]));
      shape.rx_ports            = static_cast<unsigned>(std::atoi(argv[8]));
      const std::size_t element = shape.element(static_cast<unsigned>(std::atoi(argv[2])), static_cast<unsigned>(std::atoi(argv[3])),
                                                static_cast<unsigned>(std::atoi(argv[4])), static_cast<unsigned>(std::atoi(argv[5])));
      std::printf("%zu %u\n", element, coordinate_of(shape, element).subcarrier);
    } catch (const std::exception& error) {
      std::cerr << "estimate_core_test: " << error.what() << std::endl;
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
    std::cerr << "estimate_core_test: no case named '" << selected << "'" << std::endl;
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
