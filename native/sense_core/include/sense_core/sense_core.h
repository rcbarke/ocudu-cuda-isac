/* The native sensing core: its C API (Sprint 3.6, selector sense_core).
 *
 * The core takes observation views and returns decisions. A view is one obs/v1 record as an
 * adapter holds it: the 256 bytes of its header, the address of its payload and of its validity
 * bits, and where that memory lives. The core never sees a grid reader, a ring, an E3 message or
 * a file of a host. docs/interfaces/OBSERVATION.md fixes the view, docs/architecture/RUNTIME.md
 * the contracts of the core.
 *
 * A plan is one configured chain of registry entries on one device: an ingest unit, the units of
 * the cuSense-like front end, an engine and a detector, with every workspace allocated when the
 * plan is created. One plan serves one sensing branch of one consumer.
 *
 *     sense_plan_create      load a frozen artifact, check every hash, allocate, bind the engine
 *     sense_plan_warm_up     run the chain on the template, so that no record waits for a first use
 *     sense_plan_offer       hand over the next view, in the order of its carrier. No device call:
 *                            the header is checked, the window is tracked, the view is held in place
 *     sense_plan_submit      the window is complete: ingest, kernels and engine, once, on one stream
 *     sense_plan_poll        whether that submission has completed. Never blocks
 *     sense_plan_collect     the decision of the window. The publication time is taken here
 *     sense_plan_skip        give up the complete window without device work: it was overtaken
 *     sense_plan_close_open  give up the window that is filling: its carrier went silent
 *
 * Statuses of an answer are those of the reference window stream, ocudu_cuda_isac.sensing.cusense,
 * with two that only a live core has: "stale", a record of the window was rewritten while it was
 * read, and "stale_skipped", the window was overtaken by a newer complete one before it ran.
 *
 * The header compiles as C99 and as C++20. Handles are opaque. Every function returns a
 * sense_status, and sense_last_error gives the text of the last refusal of the calling thread.
 * ASCII only.
 */

#ifndef SENSE_CORE_SENSE_CORE_H
#define SENSE_CORE_SENSE_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SENSE_CORE_API_VERSION 1
#define SENSE_RECORD_HEADER_BYTES 256
#define SENSE_MAX_WINDOW_RECORDS 8
#define SENSE_MAX_PLANES 16
#define SENSE_TEXT_BYTES 320
#define SENSE_NAME_BYTES 48
#define SENSE_HASH_BYTES 65 /* 64 hexadecimal digits and the terminator */

typedef enum sense_status {
  SENSE_OK         = 0,
  SENSE_E_ARGUMENT = 1, /* a null handle, an option outside its range */
  SENSE_E_ARTIFACT = 2, /* the artifact is incomplete, was altered or is not the frozen one */
  SENSE_E_REGISTRY = 3, /* a stage names an entry this core has no unit for */
  SENSE_E_ENGINE   = 4, /* the engine is missing, was built elsewhere or does not fit the artifact */
  SENSE_E_DEVICE   = 5, /* the device or its runtime refused a call */
  SENSE_E_STATE    = 6, /* the call does not fit the state of the plan */
  SENSE_E_IO       = 7  /* a file cannot be read */
} sense_status;

/* Where the payload of a view lives. */
typedef enum sense_memory {
  SENSE_MEMORY_HOST_SHARED = 1, /* a slot of the ring, an E3 shared-memory row */
  SENSE_MEMORY_HOST_PINNED = 2, /* a received frame in page-locked memory */
  SENSE_MEMORY_DEVICE      = 3  /* a device buffer of the adapter's own process */
} sense_memory;

/* One observation, as the adapter holds it. The adapter keeps the three addresses valid until
 * the core releases the view: at once for a record that takes part in no window, and after the
 * ingest of its window otherwise. Views are released in the order they were offered. */
typedef struct sense_view {
  const void* record;  /* the obs/v1 header, SENSE_RECORD_HEADER_BYTES bytes, little-endian */
  const void* payload; /* payload_bytes of the header, in the layout the header names */
  const void* mask;    /* mask_bytes of validity bits, in payload order */
  int32_t     memory;  /* sense_memory */
  int32_t     device;  /* the CUDA device of a device payload, otherwise -1 */
  void*       stream;  /* the stream that last wrote a device payload, otherwise null */
  int64_t     t_notify_ns; /* host monotonic: the adapter learned of the record */
  /* Asked once, after the ingest copy of the view: nonzero when the memory still holds this
   * record. Null: the memory cannot change. On the ring it is the generation check of the slot. */
  int (*intact)(void* owner);
  /* Invoked once when the core is done with the memory of the view. May be null: an adapter of
   * a carrier with cumulative credits asks sense_plan_held instead. */
  void (*release)(void* owner);
  void* owner;
} sense_view;

/* The clock of a plan: host monotonic nanoseconds. Null selects CLOCK_MONOTONIC. A unit test
 * passes its own. */
typedef int64_t (*sense_clock)(void* owner);

typedef struct sense_options {
  uint32_t    api_version;     /* SENSE_CORE_API_VERSION */
  const char* device;          /* "cuda", "cuda:N" or "host": the twin of the kernels on the CPU */
  const char* engine;          /* "tensorrt", "libtorch" or "host_reference" */
  const char* engine_path;     /* tensorrt: the engine directory. libtorch: the module directory */
  const char* expect_sha256;   /* the hash a freeze record names for the artifact, or null */
  uint32_t    keep_mean;       /* nonzero: the window mean returns to the host with a decision */
  uint32_t    use_graph;       /* nonzero: launch the engine as a captured CUDA graph where possible */
  /* Budgets in nanoseconds, 0 for none. A stage that overruns its budget is counted, and the
   * decision is published with the outcome "late". */
  int64_t budget_ingest_ns;
  int64_t budget_kernels_ns;
  int64_t budget_engine_ns;
  int64_t budget_window_ns; /* from the notification of the last record to the decision */
  sense_clock clock;
  void*       clock_owner;
  /* The CUDA stream of the submissions of the plan, or null: the plan creates one. An adapter in
   * the process of a receiver passes a stream of its own context here. */
  void* stream;
} sense_options;

typedef struct sense_plan sense_plan;

/* What a plan was built from. Texts are owned by the plan. */
typedef struct sense_plan_info {
  uint32_t    rx, subcarriers, dmrs_symbols; /* [A, K, S] of a window */
  uint32_t    window_records;                /* slots of one window */
  uint32_t    rows, cols;                    /* of the location map */
  uint32_t    period_slots, scs_khz;
  double      threshold_logit;
  const char* model_id;
  const char* artifact_sha256;
  const char* spec_sha256;
  const char* device;         /* "host" or "cuda:N" */
  const char* device_name;    /* of the GPU, or the machine of the host twin */
  const char* engine_backend; /* "tensorrt", "libtorch" or "host_reference" */
  const char* engine_detail;  /* versions and precision of the backend, as one line */
  const char* engine_sha256;  /* of the engine file, or of the weights for host_reference */
  const char* core_sources_sha256;
  const char* plan_json;      /* the plan as a JSON object: every stage with its entry and unit hash */
  const char* entries_json;   /* the entries by stage, as a report of a sensor names them */
  const char* plan_sha256;    /* of plan_json */
  int32_t     engine_on_stream; /* nonzero: the engine runs on the stream of the submission */
  int32_t     engine_graph;     /* nonzero: the engine is launched as a captured graph */
  int32_t     timing_valid;     /* nonzero: device cuda with the tensorrt backend */
} sense_plan_info;

typedef enum sense_answer_kind {
  SENSE_ANSWER_REJECTED = 1,      /* the record is no input of the specification */
  SENSE_ANSWER_FILLING  = 2,      /* insufficient_history: the window is filling */
  SENSE_ANSWER_GAP      = 3,      /* a window closed without one of its slots */
  SENSE_ANSWER_INVALID  = 4,      /* the slots were not estimated on one array or by one chain */
  SENSE_ANSWER_READY    = 5,      /* the window is complete and waits for sense_plan_submit */
  SENSE_ANSWER_STALE_SKIPPED = 6  /* a complete window was overtaken by a newer one */
} sense_answer_kind;

typedef struct sense_contributor {
  uint64_t sequence, opportunity, absolute_slot;
  uint32_t first_position, positions;
} sense_contributor;

/* One answer of the window stream. The reference writes one line per answer. */
typedef struct sense_answer {
  int32_t  kind;        /* sense_answer_kind */
  int32_t  has_window;  /* the fields epoch, episode and period are set */
  int32_t  has_trigger; /* the fields of the triggering record are set */
  uint64_t epoch, episode, period;
  uint64_t trigger_epoch, trigger_sequence, trigger_opportunity, trigger_slot;
  int64_t  t_ready_ns, t_publish_ns, t_notify_ns;
  uint32_t have, need;
  uint32_t missing_slots[SENSE_MAX_WINDOW_RECORDS]; /* GAP: slots of the period that never arrived */
  uint32_t missing;
  uint32_t contributors; /* INVALID and STALE_SKIPPED: the records of the window */
  sense_contributor contributing[SENSE_MAX_WINDOW_RECORDS];
  uint64_t t_first_sample, t_last_sample;
  char     reason[SENSE_TEXT_BYTES];
} sense_answer;

typedef enum sense_decision_status {
  SENSE_DECISION_OK      = 1, /* scored */
  SENSE_DECISION_INVALID = 2, /* too few valid elements, or a score that is not finite */
  SENSE_DECISION_STALE   = 3  /* a record was rewritten while it was read: no score */
} sense_decision_status;

/* The decision of one complete window. The tensors belong to the plan and hold until the next
 * sense_plan_submit or sense_plan_skip. */
typedef struct sense_decision {
  int32_t  status; /* sense_decision_status */
  int32_t  late;   /* a stage or the window overran its budget */
  uint32_t overrun_ingest, overrun_kernels, overrun_engine, overrun_window; /* 0 or 1 each */
  uint64_t epoch, episode, period;
  uint64_t trigger_sequence, trigger_opportunity, trigger_slot;
  uint32_t contributors;
  sense_contributor contributing[SENSE_MAX_WINDOW_RECORDS];
  uint64_t t_first_sample, t_last_sample;
  /* Scored decisions only. */
  float    logit;
  double   probability;
  int32_t  present;
  uint32_t row, col;
  double   position_x_m, position_y_m;
  float    peak;
  /* Validity of the window. */
  uint32_t elements, valid, records_min, records_max;
  /* Host monotonic times. t_ingest, t_kernels and t_engine are those of the completion event,
   * moved back by what the device measured between its marks. */
  int64_t t_ready_ns, t_publish_ns, t_notify_ns, t_submit_ns, t_ingest_ns, t_kernels_ns, t_engine_ns, t_decision_ns;
  /* Tensors, in host memory. */
  const float*   input;  /* [A, K] */
  const float*   map;    /* [rows, cols] */
  const float*   mean;   /* [A, K, S] pairs {real, imaginary}, or null without keep_mean */
  const uint8_t* count;  /* [A, K, S] records behind each element */
  const float*   difference; /* [A, K] before the z-score */
  /* What the records of the window said of themselves: a line of the window names them. */
  uint32_t port, producer_steps; /* of the first record; the steps as bits of the header */
  uint32_t evidence;             /* bit 0: a record of an audited run, bit 1: of a timed run, bit 2: another */
  double   scale;
  char reason[SENSE_TEXT_BYTES];
} sense_decision;

/* The output of the optional stage map_mean_kalman for one scored decision. */
typedef struct sense_smoothed {
  int32_t  active; /* 0: the artifact names the entry none, or the decision carries no score */
  uint32_t maps, row, col;
  double   x_m, y_m, probability;
  double   kalman_x_m, kalman_y_m, kalman_vx_m_s, kalman_vy_m_s;
} sense_smoothed;

typedef struct sense_counters {
  uint64_t records, passed, resets;
  uint64_t rejected, filling, gap, invalid, ready, stale, stale_skipped, stale_skipped_records;
  uint64_t submissions, decisions, scored, present, late;
  uint64_t overrun_ingest, overrun_kernels, overrun_engine, overrun_window;
  uint64_t device_calls;          /* calls into the device since the plan was created */
  uint64_t device_calls_open;     /* of those, calls made while a window was open: must stay 0 */
  uint64_t device_allocations;    /* device and page-locked allocations: all at configure time */
  uint64_t views_held;
} sense_counters;

/* The text of the last refusal of this thread. Never null. */
const char* sense_last_error(void);

/* Versions and what this build holds, as a JSON object. The text is static. */
const char* sense_core_build_json(void);

/* The registry of this build as a JSON array: stage, entry, the hash of the unit and whether
 * the unit runs on the device. */
const char* sense_core_registry_json(void);

/* The tolerances this build declares against the reference, as a JSON object. */
const char* sense_core_tolerance_json(void);

sense_status sense_plan_create(const char* artifact_dir, const sense_options* options, sense_plan** plan);
void         sense_plan_destroy(sense_plan* plan);
sense_status sense_plan_info_get(const sense_plan* plan, sense_plan_info* info);

/* Runs the whole chain on a window that equals the template, `rounds` times. Returns the time. */
sense_status sense_plan_warm_up(sense_plan* plan, uint32_t rounds, int64_t* elapsed_ns);

/* Hands over the next view. Writes up to two answers, oldest first, and their number. */
sense_status sense_plan_offer(sense_plan* plan, const sense_view* view, sense_answer answers[2], uint32_t* count);

/* Nonzero while a complete window waits for sense_plan_submit. */
int sense_plan_ready(const sense_plan* plan);

/* Views the plan still holds, and the carrier sequence of the oldest of them. A carrier with
 * cumulative credits releases every record below that sequence. */
uint32_t sense_plan_held(const sense_plan* plan, uint64_t* oldest_sequence);

/* The t_ready_ns of the last record of the complete window, for a release policy. */
int64_t sense_plan_ready_time(const sense_plan* plan);

sense_status sense_plan_submit(sense_plan* plan);
int          sense_plan_poll(sense_plan* plan);
sense_status sense_plan_collect(sense_plan* plan, sense_decision* decision);
sense_status sense_plan_skip(sense_plan* plan, sense_answer* answer);

/* Gives up the window that is filling: it is closed as a gap, with the slots that never arrived,
 * and its views are released. A sensor calls it when its carrier was silent for longer than a
 * window can stay open, so that the credits of a window that can no longer complete return to
 * the producer. The answer is the one the first record of a later period would have caused.
 * SENSE_E_STATE when no window is filling. */
sense_status sense_plan_close_open(sense_plan* plan, sense_answer* answer);

/* Ends the history: an open window is closed as the reference closes it at the end of a stream.
 * Nothing is answered for it. Returns the slots it held and the slots a window needs. */
sense_status sense_plan_finish(sense_plan* plan, uint32_t* have, uint32_t* need);

sense_status sense_plan_counters(const sense_plan* plan, sense_counters* counters);

/* The optional stage map_mean_kalman of the artifact, on the host and after the publication
 * time. Call it once per scored decision, in the order of the decisions. */
sense_status sense_plan_smooth(sense_plan* plan, const sense_decision* decision, sense_smoothed* smoothed);

/* JSON lines of the reference schemas. `line` is the number of the answer in its stream. The
 * digests are passed by the caller, which takes them after the last record, or null. */
sense_status sense_answer_window_json(const sense_plan* plan, const sense_answer* answer, uint64_t line, char* json,
                                      size_t bytes);
sense_status sense_answer_decision_json(const sense_plan* plan, const sense_answer* answer, uint64_t line, char* json,
                                        size_t bytes);
sense_status sense_decision_window_json(const sense_plan* plan, const sense_decision* decision, uint64_t line,
                                        uint64_t index, const char* mean_sha256, const char* count_sha256, char* json,
                                        size_t bytes);
sense_status sense_decision_json(const sense_plan* plan, const sense_decision* decision, uint64_t line, uint64_t index,
                                 const char* input_sha256, const char* map_sha256, const sense_smoothed* smoothed,
                                 char* json, size_t bytes);

/* Ingest alone, for a unit test and for a parity check: gathers one record into [A, K_r, S].
 * values holds pairs {real, imaginary} and valid one byte per element. Runs on the device of
 * the plan. */
sense_status sense_plan_ingest(sense_plan* plan, const sense_view* view, float* values, uint8_t* valid, size_t elements,
                               uint32_t shape[3], uint32_t* first_position);

/* SHA-256 of a byte range as 64 hexadecimal digits. Never called by the core while it reads. */
void sense_sha256_hex(const void* data, size_t bytes, char out[SENSE_HASH_BYTES]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SENSE_CORE_SENSE_CORE_H */
