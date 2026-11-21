// The observation record obs/v1: the byte-level definition (Sprint 3.2, selector snapshot_ring).
//
// One record is a header of one fixed size, then an opaque host block, the validity bits and the
// payload. Every field is little-endian and has a fixed width. The header holds no pointer and no
// text, so it is copied, never formatted. The same bytes travel on every carrier: a slot of the
// ring obs-ring/v1 (ring.h), a frame of Sprint 7, a file of an archive.
//
//     [ record_header, 256 bytes ][ host block, host_bytes ][ validity bits, mask_bytes ][ payload, payload_bytes ]
//
// A carrier may pad between the parts. The ring places each part at a fixed offset of a slot, so
// that a producer writes its estimate where the consumer reads it and nothing is copied twice.
//
// Nothing in the record is truth: no channel, no noise, no transport block and no label. The join
// keys are opaque integers. The host block belongs to the host that produced the record, and a
// sensing core never reads it.
//
// The reference reader ocudu_cuda_isac.sensing.obs holds the same table. RECORD_DEFINITION is the
// text both hash, and a reader built from another definition is refused when it attaches.
//
// This file includes the C++ standard library only: no OCUDU, no CUDA. ASCII only.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace obs {

constexpr std::uint32_t RECORD_SCHEMA       = 1; ///< obs/v1.
constexpr std::uint32_t RECORD_HEADER_BYTES = 256;
constexpr unsigned      MAX_RANK            = 4;
constexpr unsigned      MAX_PLANES          = 16;

/// Where in the receiver the payload was taken.
enum stage_id : std::uint8_t { STAGE_O1 = 1, STAGE_O2 = 2, STAGE_O3 = 3 };
/// The registered sensing source.
enum source_id : std::uint8_t { SOURCE_PUSCH_DMRS = 1 };
/// The element order of the payload, as its producer holds it (docs/interfaces/OBSERVATION.md).
enum layout_id : std::uint8_t {
  LAYOUT_OCUDU_O3_FP16     = 1,
  LAYOUT_OCUDU_O2_COMB_C64 = 2,
  LAYOUT_OCUDU_O1_CBF16    = 3,
  LAYOUT_AERIAL_HEST_C64   = 4,
  LAYOUT_AERIAL_IQ_FP16    = 5
};
/// The element type: complex of two float32, of two float16, of two bf16.
enum dtype_id : std::uint8_t { DTYPE_C64 = 1, DTYPE_CFP16 = 2, DTYPE_CBF16 = 3 };
/// Unit of t_first_sample and t_last_sample: T_c from slot 0 of the epoch, or TAI nanoseconds.
enum time_unit_id : std::uint8_t { TIME_TC = 1, TIME_TAI_NS = 2 };
/// A timed run publishes no digest. The evaluator digests the archived payload after the run.
enum digest_id : std::uint8_t { DIGEST_NONE = 0, DIGEST_SHA256 = 1 };
/// The evidence mode of the run that produced the record.
enum evidence_id : std::uint8_t { EVIDENCE_AUDIT = 1, EVIDENCE_TIMED = 2 };
/// Bits of producer_steps.
enum step_bit : std::uint32_t { STEP_PILOT_LS = 1, STEP_FREQ_INTERP = 2, STEP_SMOOTHING = 4, STEP_TIME_INTERP = 8 };

struct record_header {
  // Version. A reader refuses a version it does not implement.
  std::uint32_t schema;       ///< RECORD_SCHEMA.
  std::uint32_t header_bytes; ///< RECORD_HEADER_BYTES.
  // Join keys. Opaque integers, used only for joins.
  std::uint64_t epoch;       ///< Set by the carrier. Rises with every consumer that attaches.
  std::uint64_t sequence;    ///< Set by the carrier. Rises by one per published record of an epoch.
  std::uint64_t episode;
  std::uint64_t opportunity;
  std::uint64_t grant;
  // Radio identity.
  std::uint64_t absolute_slot; ///< Counts from the epoch start and does not wrap with the frame number.
  std::uint32_t cell;
  std::uint32_t ue;
  // What it is.
  std::uint8_t  stage;          ///< stage_id.
  std::uint8_t  source;         ///< source_id.
  std::uint8_t  layout;         ///< layout_id.
  std::uint8_t  dtype;          ///< dtype_id.
  std::uint32_t producer_steps; ///< Bit set of step_bit.
  double        scale;          ///< The reference amplitude the producer used.
  // Shape, in elements.
  std::uint32_t rank;
  std::uint32_t shape[MAX_RANK];
  std::uint32_t strides[MAX_RANK];
  // Allocation, in carrier coordinates.
  std::uint32_t first_subcarrier;
  std::uint32_t subcarrier_step; ///< Of the payload's subcarrier axis. 0 when the axis is not regular.
  std::uint32_t subcarrier_count;
  std::uint32_t dmrs_symbols; ///< Bit set of symbol indices of the slot.
  std::uint16_t rx_count;
  std::uint16_t layer_count;
  std::uint16_t antenna_map[MAX_PLANES]; ///< antenna_map[i] is the array element behind receive plane i.
  // Sizes of what follows the header.
  std::uint32_t mask_bytes;    ///< One validity bit per payload element, bit n % 8 of byte n / 8. 0: none.
  std::uint32_t payload_bytes;
  std::uint32_t host_bytes;
  std::uint8_t  time_unit;   ///< time_unit_id.
  std::uint8_t  digest_kind; ///< digest_id.
  std::uint8_t  evidence;    ///< evidence_id.
  std::uint8_t  port;        ///< Reference-signal port of the source.
  // Time.
  std::uint64_t t_first_sample; ///< Radio time of the first contributing sample.
  std::uint64_t t_last_sample;  ///< Radio time of the end of the last contributing sample.
  std::int64_t  t_ready_ns;     ///< Host monotonic time of the producer's completion event.
  std::int64_t  t_publish_ns;   ///< Set by the carrier: host monotonic time of the publication.
  // Payload identity.
  std::uint8_t  digest[32]; ///< SHA-256 of the payload when digest_kind says so, else zero.
  std::uint32_t valid;      ///< Set validity bits.
  std::uint32_t reserved;   ///< Zero.
};

static_assert(std::is_trivially_copyable<record_header>::value, "the record header is copied byte by byte");
static_assert(sizeof(record_header) == RECORD_HEADER_BYTES, "the record header has one fixed size per version");
static_assert(offsetof(record_header, epoch) == 8 && offsetof(record_header, absolute_slot) == 48 &&
                  offsetof(record_header, cell) == 56 && offsetof(record_header, stage) == 64 &&
                  offsetof(record_header, producer_steps) == 68 && offsetof(record_header, scale) == 72 &&
                  offsetof(record_header, rank) == 80 && offsetof(record_header, shape) == 84 &&
                  offsetof(record_header, strides) == 100 && offsetof(record_header, first_subcarrier) == 116 &&
                  offsetof(record_header, dmrs_symbols) == 128 && offsetof(record_header, rx_count) == 132 &&
                  offsetof(record_header, antenna_map) == 136 && offsetof(record_header, mask_bytes) == 168 &&
                  offsetof(record_header, host_bytes) == 176 && offsetof(record_header, time_unit) == 180 &&
                  offsetof(record_header, t_first_sample) == 184 && offsetof(record_header, t_ready_ns) == 200 &&
                  offsetof(record_header, t_publish_ns) == 208 && offsetof(record_header, digest) == 216 &&
                  offsetof(record_header, valid) == 248,
              "the offsets are those of RECORD_DEFINITION");

/// The definition both readers hash: every field as name:offset:bytes, in the order of the header.
constexpr const char RECORD_DEFINITION[] =
    "obs/v1 little-endian 256;"
    "schema:0:4;header_bytes:4:4;epoch:8:8;sequence:16:8;episode:24:8;opportunity:32:8;grant:40:8;"
    "absolute_slot:48:8;cell:56:4;ue:60:4;stage:64:1;source:65:1;layout:66:1;dtype:67:1;producer_steps:68:4;"
    "scale:72:8;rank:80:4;shape:84:16;strides:100:16;first_subcarrier:116:4;subcarrier_step:120:4;"
    "subcarrier_count:124:4;dmrs_symbols:128:4;rx_count:132:2;layer_count:134:2;antenna_map:136:32;"
    "mask_bytes:168:4;payload_bytes:172:4;host_bytes:176:4;time_unit:180:1;digest_kind:181:1;evidence:182:1;"
    "port:183:1;t_first_sample:184:8;t_last_sample:192:8;t_ready_ns:200:8;t_publish_ns:208:8;digest:216:32;"
    "valid:248:4;reserved:252:4;";

/// FNV-1a over a text, 64 bits. It names a definition. It is computed when a ring is created or
/// attached and never per record: a record carries no checksum of its own.
constexpr std::uint64_t fnv1a(const char* text, std::uint64_t value = 14695981039346656037ULL)
{
  for (; *text != '\0'; ++text) {
    value = (value ^ static_cast<std::uint8_t>(*text)) * 1099511628211ULL;
  }
  return value;
}

} // namespace obs
