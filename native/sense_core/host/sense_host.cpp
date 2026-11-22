// sense_host: the sensor process on DeepSig-OCUDU hosts (Sprint 3.6, selector sense_core).
//
// It is the only consumer of one observation ring. It loads one frozen artifact, builds one plan
// of the native core for the cuSense-like branch, warms it up, and only then attaches. It reads
// the records of a window in place, submits ingest, kernels and engine once per window on one
// stream, and writes decisions.jsonl. It holds no Python interpreter and links no library of a
// receiver.
//
//     sense_host --socket PATH --artifact DIR --output-dir DIR [options]      a live ring
//     sense_host --archive DIR --artifact DIR --output-dir DIR [options]      the records of a
//                                                                             consumer archive
//     sense_host --build | --registry | --tolerance                           what this build is
//
// Nothing is allocated, formatted, hashed or written while records arrive. After the producer
// said that no record follows, it takes the digests of what it kept and writes the output of a
// cuSense sensor, as ocudu_cuda_isac.sensing.cusense.open_output verifies it:
//
//     decisions.jsonl, inputs.f32, maps.f32, cusense.json
//     windows/windows.jsonl, windows/windows.json
//     windows/records/       with --keep-records: every record it read, and consumer.json
//
// Every line carries the runtime entry "native". A run on the host twin, or on another backend
// than tensorrt, says so in cusense.json and is no timed run.
//
// It links sense_core and obs_ring. ASCII only.

#include "core/json.h"
#include "obs/ring.h"
#include "sense_core/sense_core.h"
#include "sensor_loop.h"
#include "sources.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <memory>
#include <sched.h>
#include <string>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <vector>

namespace {

using sense::json::put_double;
using sense::json::put_int;
using sense::json::put_string;
using sense::archive_source;
using sense::record_keeper;
using sense::ring_source;

void usage()
{
  std::puts(
      "Usage: sense_host --socket PATH --artifact DIR --output-dir DIR [options]\n"
      "       sense_host --archive DIR --artifact DIR --output-dir DIR [options]\n"
      "       sense_host --build | --registry | --tolerance\n"
      "  --socket PATH           control channel of the producer's ring: the sensor is its consumer\n"
      "  --archive DIR           the records of a consumer archive, in order, instead of a ring\n"
      "  --artifact DIR          directory of a frozen cusense-artifact/v1 artifact\n"
      "  --output-dir DIR        existing, empty directory for the output of the sensor\n"
      "  --expect-sha256 HEX     the hash a freeze record names for the artifact\n"
      "  --device NAME           cuda, cuda:N or host: the twin of the kernels on the CPU (default: cuda\n"
      "                          where this build holds the device)\n"
      "  --engine NAME           tensorrt, libtorch or host_reference (default: tensorrt on the device\n"
      "                          where this build holds it)\n"
      "  --engine-dir DIR        the engine directory of the backend: model.plan or model.ts with engine.json\n"
      "  --no-graph              enqueue the engine directly and capture no CUDA graph\n"
      "  --wait MODE             spin (default): poll the published count. wait: sleep on the eventfd\n"
      "  --connect-ms N          how long to try to attach (default 10000)\n"
      "  --idle-ms N             give up when no record arrived for this long (default 30000)\n"
      "  --open-ms N             close a window that is filling as a gap when no record arrived for this\n"
      "                          long, so that its credits return to the producer (default 20, 0: never)\n"
      "  --release POLICY        immediate (default) or after_radio\n"
      "  --radio-allowance-us N  after_radio: submit this long after t_ready of the last record of a window\n"
      "  --budget-us I,K,E,W     budgets of the ingest, the kernels, the engine and the window, 0 for none\n"
      "  --evidence MODE         timed (default): the window mean stays on the device. audit: it returns\n"
      "                          with every decision and its digest is written\n"
      "  --lines N               answers and decisions to keep (default 16384)\n"
      "  --windows N             scored windows whose tensors are kept (default 4096)\n"
      "  --keep-records          keep every record that was read, as obs_tail does (--capacity, --host-bytes)\n"
      "  --capacity N            records to keep (default 4096)\n"
      "  --host-bytes N          the largest host block of a record to keep (default 64)\n"
      "  --ready-file PATH       written when the plan is warm, before the sensor attaches\n"
      "  --live FILE             append each decision here after its publication time, without digests\n"
      "  --warm-rounds N         runs of the chain before the first record (default 3)\n"
      "  --hold-us N             test switch: work N microseconds on each window before it is released\n"
      "  --cpu N                 run on this CPU");
}

std::int64_t monotonic_ns()
{
  return obs::monotonic_ns();
}

bool write_file(const std::string& path, const void* data, std::size_t bytes)
{
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return false;
  }
  const bool ok = bytes == 0 || std::fwrite(data, 1, bytes, file) == bytes;
  return std::fclose(file) == 0 && ok;
}

bool write_text(const std::string& path, const std::string& text)
{
  return write_file(path, text.data(), text.size());
}

bool empty_directory(const std::string& path)
{
  DIR* listing = ::opendir(path.c_str());
  if (listing == nullptr) {
    return false;
  }
  bool empty = true;
  while (const dirent* found = ::readdir(listing)) {
    if (std::strcmp(found->d_name, ".") != 0 && std::strcmp(found->d_name, "..") != 0) {
      empty = false;
    }
  }
  ::closedir(listing);
  return empty;
}

void put_member(std::string& out, const char* key, const std::string& value)
{
  out += '"';
  out += key;
  out += "\": ";
  put_string(out, value);
}

void put_member(std::string& out, const char* key, std::int64_t value)
{
  out += '"';
  out += key;
  out += "\": ";
  put_int(out, value);
}

void put_raw(std::string& out, const char* key, const std::string& value)
{
  out += '"';
  out += key;
  out += "\": ";
  out += value;
}

} // namespace

int main(int argc, char** argv)
{
  std::string socket_path, archive_dir, artifact_dir, output_dir, expect, device, engine, engine_dir, wait_name = "spin";
  std::string release_name = "immediate", evidence = "timed", ready_file, live_file, budget_text;
  long        connect_ms = 10000, idle_ms = 30000, allowance_us = 0, lines = 16384, windows = 4096, capacity = 4096;
  long        host_bytes = 64, warm_rounds = 3, hold_us = 0, cpu = -1, open_ms = 20;
  bool        no_graph = false, keep_records = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg  = argv[i];
    const bool        more = i + 1 < argc;
    if (arg == "--help") {
      usage();
      return 0;
    } else if (arg == "--build") {
      std::puts(sense_core_build_json());
      return 0;
    } else if (arg == "--registry") {
      std::puts(sense_core_registry_json());
      return 0;
    } else if (arg == "--tolerance") {
      std::puts(sense_core_tolerance_json());
      return 0;
    } else if (arg == "--socket" && more) {
      socket_path = argv[++i];
    } else if (arg == "--archive" && more) {
      archive_dir = argv[++i];
    } else if (arg == "--artifact" && more) {
      artifact_dir = argv[++i];
    } else if (arg == "--output-dir" && more) {
      output_dir = argv[++i];
    } else if (arg == "--expect-sha256" && more) {
      expect = argv[++i];
    } else if (arg == "--device" && more) {
      device = argv[++i];
    } else if (arg == "--engine" && more) {
      engine = argv[++i];
    } else if (arg == "--engine-dir" && more) {
      engine_dir = argv[++i];
    } else if (arg == "--no-graph") {
      no_graph = true;
    } else if (arg == "--wait" && more) {
      wait_name = argv[++i];
    } else if (arg == "--connect-ms" && more) {
      connect_ms = std::atol(argv[++i]);
    } else if (arg == "--idle-ms" && more) {
      idle_ms = std::atol(argv[++i]);
    } else if (arg == "--open-ms" && more) {
      open_ms = std::atol(argv[++i]);
    } else if (arg == "--release" && more) {
      release_name = argv[++i];
    } else if (arg == "--radio-allowance-us" && more) {
      allowance_us = std::atol(argv[++i]);
    } else if (arg == "--budget-us" && more) {
      budget_text = argv[++i];
    } else if (arg == "--evidence" && more) {
      evidence = argv[++i];
    } else if (arg == "--lines" && more) {
      lines = std::atol(argv[++i]);
    } else if (arg == "--windows" && more) {
      windows = std::atol(argv[++i]);
    } else if (arg == "--keep-records") {
      keep_records = true;
    } else if (arg == "--capacity" && more) {
      capacity = std::atol(argv[++i]);
    } else if (arg == "--host-bytes" && more) {
      host_bytes = std::atol(argv[++i]);
    } else if (arg == "--ready-file" && more) {
      ready_file = argv[++i];
    } else if (arg == "--live" && more) {
      live_file = argv[++i];
    } else if (arg == "--warm-rounds" && more) {
      warm_rounds = std::atol(argv[++i]);
    } else if (arg == "--hold-us" && more) {
      hold_us = std::atol(argv[++i]);
    } else if (arg == "--cpu" && more) {
      cpu = std::atol(argv[++i]);
    } else {
      usage();
      std::fprintf(stderr, "sense_host: unknown or incomplete option '%s'\n", arg.c_str());
      return 2;
    }
  }
  long budgets[4] = {0, 0, 0, 0};
  if (!budget_text.empty() &&
      std::sscanf(budget_text.c_str(), "%ld,%ld,%ld,%ld", &budgets[0], &budgets[1], &budgets[2], &budgets[3]) != 4) {
    std::fprintf(stderr, "sense_host: --budget-us takes four numbers: ingest,kernels,engine,window\n");
    return 2;
  }
  if (socket_path.empty() == archive_dir.empty() || artifact_dir.empty() || output_dir.empty() ||
      (wait_name != "spin" && wait_name != "wait") || (release_name != "immediate" && release_name != "after_radio") ||
      (evidence != "audit" && evidence != "timed") || connect_ms < 0 || idle_ms < 1 || allowance_us < 0 || lines < 1 ||
      windows < 1 || capacity < 1 || host_bytes < 0 || warm_rounds < 0 || hold_us < 0 || open_ms < 0 || budgets[0] < 0 || budgets[1] < 0 ||
      budgets[2] < 0 || budgets[3] < 0) {
    std::fprintf(stderr, "sense_host: one of --socket and --archive is required, with --artifact and --output-dir; "
                         "--wait is spin or wait, --release immediate or after_radio, --evidence audit or timed, and "
                         "no count is negative\n");
    return 2;
  }
  if (!empty_directory(output_dir)) {
    std::fprintf(stderr, "sense_host: the output directory %s does not exist or is not empty\n", output_dir.c_str());
    return 2;
  }
  if (keep_records && socket_path.empty()) {
    std::fprintf(stderr, "sense_host: --keep-records keeps what a ring delivered: an archive holds its records already\n");
    return 2;
  }
  if (cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<unsigned>(cpu), &set);
    if (::sched_setaffinity(0, sizeof(set), &set) != 0) {
      std::fprintf(stderr, "sense_host: cannot run on CPU %ld\n", cpu);
      return 2;
    }
  }

  // The plan: the artifact with every hash checked, the workspaces and the engine. Loaded once.
  sense_options options;
  std::memset(&options, 0, sizeof(options));
  options.api_version       = SENSE_CORE_API_VERSION;
  options.device            = device.empty() ? nullptr : device.c_str();
  options.engine            = engine.empty() ? nullptr : engine.c_str();
  options.engine_path       = engine_dir.empty() ? nullptr : engine_dir.c_str();
  options.expect_sha256     = expect.empty() ? nullptr : expect.c_str();
  options.keep_mean         = evidence == "audit" ? 1 : 0;
  options.use_graph         = no_graph ? 0 : 1;
  options.budget_ingest_ns  = budgets[0] * 1000;
  options.budget_kernels_ns = budgets[1] * 1000;
  options.budget_engine_ns  = budgets[2] * 1000;
  options.budget_window_ns  = budgets[3] * 1000;
  sense_plan* plan          = nullptr;
  if (sense_plan_create(artifact_dir.c_str(), &options, &plan) != SENSE_OK) {
    std::fprintf(stderr, "sense_host: %s\n", sense_last_error());
    return 2;
  }
  std::int64_t warm_ns = 0;
  if (sense_plan_warm_up(plan, static_cast<std::uint32_t>(warm_rounds), &warm_ns) != SENSE_OK) {
    std::fprintf(stderr, "sense_host: the warm-up failed: %s\n", sense_last_error());
    sense_plan_destroy(plan);
    return 1;
  }
  sense_plan_info info;
  sense_plan_info_get(plan, &info);

  int live_fd = -1;
  if (!live_file.empty()) {
    live_fd = ::open(live_file.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_APPEND, 0644);
    if (live_fd < 0) {
      std::fprintf(stderr, "sense_host: cannot create %s: a live file is never overwritten\n", live_file.c_str());
      sense_plan_destroy(plan);
      return 2;
    }
  }
  sense::loop_options loop;
  loop.release            = release_name == "after_radio" ? sense::loop_options::release_policy::after_radio
                                                          : sense::loop_options::release_policy::immediate;
  loop.radio_allowance_ns = allowance_us * 1000;
  loop.idle_ns            = idle_ms * 1000000LL;
  loop.open_ns            = open_ms * 1000000LL;
  loop.hold_ns            = hold_us * 1000;
  loop.lines              = static_cast<std::size_t>(lines);
  loop.windows            = static_cast<std::size_t>(windows);
  loop.keep_mean          = evidence == "audit";
  loop.live_fd            = live_fd;

  int status = 0;
  try {
    // Everything is allocated and touched before the sensor attaches, so that it reads from the
    // first record on.
    sense::sensor_loop sensor(plan, loop);
    const std::size_t  elements   = static_cast<std::size_t>(info.rx) * info.subcarriers * info.dmrs_symbols;
    const std::size_t  record_max = obs::RECORD_HEADER_BYTES + static_cast<std::size_t>(host_bytes) + (elements + 7) / 8 +
                                    8 * elements;
    std::unique_ptr<record_keeper> keeper;
    if (keep_records) {
      keeper.reset(new record_keeper(static_cast<std::size_t>(capacity), record_max));
    }
    if (!ready_file.empty()) {
      std::string ready = "{";
      put_member(ready, "device", info.device);
      ready += ", ";
      put_member(ready, "engine_backend", info.engine_backend);
      ready += ", ";
      put_member(ready, "artifact_sha256", info.artifact_sha256);
      ready += ", \"warm_up_ms\": ";
      put_double(ready, static_cast<double>(warm_ns) / 1.0e6);
      ready += ", ";
      put_member(ready, "t_ready_ns", monotonic_ns());
      ready += "}\n";
      if (!write_text(ready_file, ready)) {
        throw std::runtime_error("cannot write the ready file " + ready_file);
      }
    }

    std::string  source_json;
    sense_status ran = SENSE_OK;
    std::string  consumer_json;
    if (!archive_dir.empty()) {
      archive_source records(archive_dir);
      ran         = sensor.run(records);
      source_json = "{\"kind\": \"archive\", ";
      put_member(source_json, "directory", archive_dir);
      source_json += ", \"reader\": \"native\", ";
      put_member(source_json, "records", static_cast<std::int64_t>(records.records()));
    } else {
      obs::consumer_config config;
      config.socket_path   = socket_path;
      config.mode          = wait_name == "wait" ? obs::wait_mode::wait : obs::wait_mode::spin;
      config.connect_limit = std::chrono::milliseconds(connect_ms);
      obs::consumer              reader(config);
      const obs::segment_header& ring        = reader.header();
      const std::int64_t         attached_ns = monotonic_ns();
      if (keeper && ring.record_header_bytes + ring.host_capacity + ring.mask_capacity +
                            static_cast<std::size_t>(ring.payload_capacity) > record_max) {
        throw std::runtime_error("a record of this ring holds more bytes than the sensor allocated to keep one. Name its "
                                 "host block with --host-bytes");
      }
      ring_source records(reader, keeper.get());
      ran = sensor.run(records);
      reader.release(reader.position()); // nothing is held any more
      const obs::consumer_counts counts = reader.counts();
      source_json = "{\"kind\": \"ring\", ";
      put_member(source_json, "socket", socket_path);
      source_json += ", \"reader\": \"native\", ";
      put_member(source_json, "wait", wait_name);
      source_json += ", ";
      put_member(source_json, "epoch", static_cast<std::int64_t>(reader.epoch()));
      source_json += ", ";
      put_member(source_json, "records", static_cast<std::int64_t>(counts.read));
      source_json += ", ";
      put_member(source_json, "stale", static_cast<std::int64_t>(counts.stale));
      source_json += ", ";
      put_member(source_json, "released", static_cast<std::int64_t>(counts.released));
      source_json += ", ";
      put_member(source_json, "skipped", static_cast<std::int64_t>(counts.skipped));
      source_json += ", ";
      put_member(source_json, "sleeps", static_cast<std::int64_t>(counts.sleeps));
      source_json += ", ";
      put_member(source_json, "slot_count", ring.slot_count);
      source_json += ", ";
      put_member(source_json, "system_calls", static_cast<std::int64_t>(obs::sys::total()));
      source_json += ", ";
      put_member(source_json, "ended", sensor.counters().closed ? "closed" : "idle");
      if (keeper) {
        // consumer.json of the kept records, as obs_tail writes it.
        char line[512];
        consumer_json = "{\n  \"schema\": \"obs-consumer/v1\",\n  \"reader\": \"native\",\n";
        std::snprintf(line, sizeof(line),
                      "  \"definition\": \"%016llx\",\n  \"wait\": \"%s\",\n  \"epoch\": %llu,\n  \"pid\": %ld,\n"
                      "  \"cpu\": %ld,\n  \"hold_us\": %ld,\n  \"window\": %u,\n  \"slot_count\": %u,\n  \"slot_bytes\": %u,\n"
                      "  \"attached_ns\": %lld,\n  \"ended\": \"%s\",\n",
                      static_cast<unsigned long long>(obs::DEFINITION), wait_name.c_str(),
                      static_cast<unsigned long long>(reader.epoch()), static_cast<long>(::getpid()), cpu, hold_us,
                      info.window_records, ring.slot_count, ring.slot_bytes, static_cast<long long>(attached_ns),
                      sensor.counters().closed ? "closed" : "idle");
        consumer_json += line;
        std::snprintf(line, sizeof(line),
                      "  \"counts\": {\"read\": %llu, \"stale\": %llu, \"released\": %llu, \"skipped\": %llu, \"sleeps\": %llu, "
                      "\"kept\": %zu, \"not_kept\": %zu, \"system_calls\": %llu},\n  \"records\": [\n",
                      static_cast<unsigned long long>(counts.read), static_cast<unsigned long long>(counts.stale),
                      static_cast<unsigned long long>(counts.released), static_cast<unsigned long long>(counts.skipped),
                      static_cast<unsigned long long>(counts.sleeps), keeper->count, keeper->not_kept,
                      static_cast<unsigned long long>(obs::sys::total()));
        consumer_json += line;
        for (std::size_t n = 0; n != keeper->count; ++n) {
          const record_keeper::kept& item = keeper->records[n];
          std::snprintf(line, sizeof(line),
                        "    {\"epoch\": %llu, \"sequence\": %llu, \"opportunity\": \"%016llx\", \"depth\": %u, \"bytes\": %u, "
                        "\"t_publish_ns\": %lld, \"t_read_ns\": %lld, \"delay_ns\": %lld, \"intact\": %s, \"file\": %s}%s\n",
                        static_cast<unsigned long long>(item.epoch), static_cast<unsigned long long>(item.sequence),
                        static_cast<unsigned long long>(item.opportunity), item.depth, item.bytes,
                        static_cast<long long>(item.t_publish_ns), static_cast<long long>(item.t_read_ns),
                        static_cast<long long>(item.t_read_ns - item.t_publish_ns), item.intact ? "true" : "false",
                        item.intact ? "true" : "null", n + 1 == keeper->count ? "" : ",");
          consumer_json += line;
        }
        consumer_json += "  ]\n}\n";
      }
    }
    if (ran != SENSE_OK) {
      throw std::runtime_error(std::string("the plan refused a call: ") + sense_last_error());
    }
    std::uint32_t open_have = 0, open_need = 0;
    sense_plan_finish(plan, &open_have, &open_need);
    const sense::loop_counters& done = sensor.counters();
    if (done.lines_not_kept != 0 || done.windows_not_kept != 0) {
      throw std::runtime_error(std::to_string(done.lines_not_kept) + " lines and " + std::to_string(done.windows_not_kept) +
                               " scored windows were not kept: raise --lines and --windows");
    }

    // After the last record: digests, lines and files.
    std::vector<std::string> window_lines, decision_lines;
    sensor.seal(window_lines, decision_lines);
    std::uint64_t by_status[7] = {0, 0, 0, 0, 0, 0, 0}; // ok, insufficient_history, gap, invalid, rejected, stale, stale_skipped
    std::uint64_t present = 0;
    for (std::size_t number = 0; number != window_lines.size(); ++number) {
      const sense::kept_line& line = sensor.lines()[number];
      if (line.is_decision) {
        const int kind = line.decision.status;
        ++by_status[kind == SENSE_DECISION_OK ? 0 : kind == SENSE_DECISION_INVALID ? 3 : 5];
        present += kind == SENSE_DECISION_OK && line.decision.present != 0 ? 1U : 0U;
      } else {
        const int kind = line.answer.kind;
        ++by_status[kind == SENSE_ANSWER_FILLING ? 1 : kind == SENSE_ANSWER_GAP ? 2 : kind == SENSE_ANSWER_INVALID ? 3
                    : kind == SENSE_ANSWER_REJECTED ? 4 : 6];
      }
    }
    sense_counters counters;
    sense_plan_counters(plan, &counters);
    const std::string windows_dir = output_dir + "/windows";
    if (::mkdir(windows_dir.c_str(), 0755) != 0) {
      throw std::runtime_error("cannot create " + windows_dir);
    }
    std::string text;
    for (const std::string& line : window_lines) {
      text += line;
      text += '\n';
    }
    if (!write_text(windows_dir + "/windows.jsonl", text)) {
      throw std::runtime_error("cannot write windows.jsonl");
    }
    text.clear();
    for (const std::string& line : decision_lines) {
      text += line;
      text += '\n';
    }
    if (!write_text(output_dir + "/decisions.jsonl", text)) {
      throw std::runtime_error("cannot write decisions.jsonl");
    }
    if (!write_file(output_dir + "/inputs.f32", sensor.input_of(0), sizeof(float) * sensor.input_size() * sensor.scored()) ||
        !write_file(output_dir + "/maps.f32", sensor.map_of(0), sizeof(float) * sensor.map_size() * sensor.scored())) {
      throw std::runtime_error("cannot write the tensors of the scored windows");
    }
    if (keeper) {
      const std::string records_dir = windows_dir + "/records";
      if (::mkdir(records_dir.c_str(), 0755) != 0) {
        throw std::runtime_error("cannot create " + records_dir);
      }
      for (std::size_t n = 0; n != keeper->count; ++n) {
        const record_keeper::kept& item = keeper->records[n];
        char                       name[64];
        std::snprintf(name, sizeof(name), "/%04llu-%08llu.obs", static_cast<unsigned long long>(item.epoch),
                      static_cast<unsigned long long>(item.sequence));
        if (item.intact && !write_file(records_dir + name, keeper->arena.data() + n * keeper->record_max, item.bytes)) {
          throw std::runtime_error("cannot write " + records_dir + name);
        }
      }
      if (!write_text(records_dir + "/consumer.json", consumer_json)) {
        throw std::runtime_error("cannot write consumer.json");
      }
    }

    // The counts both reports share.
    std::string status_counts;
    static const char* const names[7] = {"ok", "insufficient_history", "gap", "invalid", "rejected", "stale", "stale_skipped"};
    for (unsigned n = 0; n != 7; ++n) {
      status_counts += n != 0 ? ", " : "";
      put_member(status_counts, names[n], static_cast<std::int64_t>(by_status[n]));
    }
    struct utsname machine;
    ::uname(&machine);
    std::string runtime = "{\"entry\": \"native\", ";
    put_member(runtime, "device", info.device);
    runtime += ", ";
    put_member(runtime, "device_name", info.device_name);
    runtime += ", ";
    put_member(runtime, "engine_backend", info.engine_backend);
    runtime += ", ";
    put_member(runtime, "engine_detail", info.engine_detail);
    runtime += ", ";
    put_member(runtime, "engine_sha256", info.engine_sha256);
    runtime += ", \"engine_graph\": ";
    runtime += info.engine_graph != 0 ? "true" : "false";
    runtime += ", \"timing_valid\": ";
    runtime += info.timing_valid != 0 ? "true" : "false";
    runtime += ", ";
    put_member(runtime, "machine", machine.machine);
    runtime += ", ";
    put_raw(runtime, "core", sense_core_build_json());
    runtime += ", ";
    put_raw(runtime, "plan", info.plan_json);
    runtime += ", ";
    put_raw(runtime, "tolerance", sense_core_tolerance_json());
    runtime += "}";
    source_json += ", \"warm_up_ms\": {\"plan\": ";
    put_double(source_json, static_cast<double>(warm_ns) / 1.0e6);
    source_json += ", \"rounds\": ";
    put_int(source_json, warm_rounds);
    source_json += "}, ";
    put_member(source_json, "release", release_name);
    source_json += ", ";
    put_member(source_json, "radio_allowance_us", allowance_us);
    source_json += ", ";
    put_member(source_json, "evidence", evidence);
    source_json += ", ";
    put_member(source_json, "waited_radio", static_cast<std::int64_t>(done.waited_radio));
    source_json += ", ";
    put_member(source_json, "open_ms", open_ms);
    source_json += ", ";
    put_member(source_json, "closed_open", static_cast<std::int64_t>(done.closed_open));
    source_json += "}";

    // windows.json: the report of the window stream.
    const sense::json::value manifest = sense::json::parse([&] {
      std::FILE*  file = std::fopen((artifact_dir + "/model.json").c_str(), "rb");
      std::string all;
      char        chunk[65536];
      std::size_t got;
      while (file != nullptr && (got = std::fread(chunk, 1, sizeof(chunk), file)) != 0) {
        all.append(chunk, got);
      }
      if (file != nullptr) {
        std::fclose(file);
      }
      return all;
    }());
    const std::string spec_json     = sense::json::canonical(manifest.object_at("input").object_at("spec"));
    const std::string manifest_json = sense::json::canonical(manifest);
    std::string report = "{\"schema\": \"cusense-windows-report/v1\", ";
    put_raw(report, "spec", spec_json);
    report += ", ";
    put_member(report, "spec_sha256", info.spec_sha256);
    report += ", ";
    put_raw(report, "runtime", runtime);
    report += ", ";
    put_raw(report, "source", source_json);
    report += ", \"counts\": {";
    put_member(report, "records", static_cast<std::int64_t>(counters.records));
    report += ", ";
    put_member(report, "passed", static_cast<std::int64_t>(counters.passed));
    report += ", ";
    put_member(report, "resets", static_cast<std::int64_t>(counters.resets));
    report += ", " + status_counts + "}, ";
    put_member(report, "lines", static_cast<std::int64_t>(window_lines.size()));
    report += ", ";
    put_member(report, "windows", static_cast<std::int64_t>(by_status[0]));
    report += ", \"shape\": [";
    put_int(report, info.rx);
    report += ", ";
    put_int(report, info.subcarriers);
    report += ", ";
    put_int(report, info.dmrs_symbols);
    report += "], \"open_at_end\": {";
    put_member(report, "have", open_have);
    report += ", ";
    put_member(report, "need", open_need);
    report += "}}\n";
    if (!write_text(windows_dir + "/windows.json", report)) {
      throw std::runtime_error("cannot write windows.json");
    }

    // cusense.json: the report of the sensor.
    report = "{\"schema\": \"cusense-report/v1\", \"artifact\": {";
    put_member(report, "directory", artifact_dir);
    report += ", ";
    put_raw(report, "manifest", manifest_json);
    report += ", \"model\": {";
    put_member(report, "model_id", info.model_id);
    report += ", ";
    put_member(report, "entry", manifest.object_at("entries").string_at("model"));
    report += ", ";
    put_member(report, "artifact_sha256", info.artifact_sha256);
    report += ", ";
    put_member(report, "weights_sha256", manifest.object_at("weights").string_at("sha256"));
    report += ", ";
    put_member(report, "template_sha256", manifest.object_at("template").string_at("sha256"));
    report += ", ";
    put_member(report, "normalizer_sha256", manifest.object_at("normalizer").string_at("sha256"));
    report += ", ";
    put_member(report, "grid_sha256", manifest.object_at("grid").string_at("sha256"));
    report += ", ";
    put_member(report, "threshold_sha256", manifest.object_at("threshold").string_at("sha256"));
    report += "}}, ";
    put_raw(report, "entries", info.entries_json);
    report += ", ";
    put_raw(report, "runtime", runtime);
    report += ", \"loads\": 1, \"windows\": \"windows\", ";
    put_raw(report, "source", source_json);
    report += ", \"counts\": {" + status_counts + ", ";
    put_member(report, "decisions", static_cast<std::int64_t>(decision_lines.size()));
    report += ", ";
    put_member(report, "scored", static_cast<std::int64_t>(by_status[0]));
    report += ", ";
    put_member(report, "present", static_cast<std::int64_t>(present));
    report += ", ";
    put_member(report, "absent", static_cast<std::int64_t>(by_status[0] - present));
    report += ", ";
    put_member(report, "submissions", static_cast<std::int64_t>(counters.submissions));
    report += ", ";
    put_member(report, "stale_skipped_records", static_cast<std::int64_t>(counters.stale_skipped_records));
    report += ", ";
    put_member(report, "late", static_cast<std::int64_t>(counters.late));
    report += ", \"overrun\": {";
    put_member(report, "ingest", static_cast<std::int64_t>(counters.overrun_ingest));
    report += ", ";
    put_member(report, "kernels", static_cast<std::int64_t>(counters.overrun_kernels));
    report += ", ";
    put_member(report, "engine", static_cast<std::int64_t>(counters.overrun_engine));
    report += ", ";
    put_member(report, "window", static_cast<std::int64_t>(counters.overrun_window));
    report += "}, ";
    put_member(report, "device_calls", static_cast<std::int64_t>(counters.device_calls));
    report += ", ";
    put_member(report, "device_calls_in_an_open_window", static_cast<std::int64_t>(counters.device_calls_open));
    report += ", ";
    put_member(report, "device_allocations", static_cast<std::int64_t>(counters.device_allocations));
    report += ", ";
    put_member(report, "stale_reads", static_cast<std::int64_t>(done.stale_reads));
    report += "}, \"budget_us\": {";
    put_member(report, "ingest", budgets[0]);
    report += ", ";
    put_member(report, "kernels", budgets[1]);
    report += ", ";
    put_member(report, "engine", budgets[2]);
    report += ", ";
    put_member(report, "window", budgets[3]);
    report += "}, \"files\": {\"inputs.f32\": {\"shape\": [";
    put_int(report, static_cast<std::int64_t>(sensor.scored()));
    report += ", ";
    put_int(report, info.rx);
    report += ", ";
    put_int(report, info.subcarriers);
    report += "], \"dtype\": \"<f4\"}, \"maps.f32\": {\"shape\": [";
    put_int(report, static_cast<std::int64_t>(sensor.scored()));
    report += ", ";
    put_int(report, info.rows);
    report += ", ";
    put_int(report, info.cols);
    report += "], \"dtype\": \"<f4\"}}}\n";
    if (!write_text(output_dir + "/cusense.json", report)) {
      throw std::runtime_error("cannot write cusense.json");
    }
    std::printf("sense_host: %zu decisions on %s with the %s backend, %llu scored: %llu present, %llu absent; %llu of a "
                "filling window, %llu gaps, %llu invalid, %llu rejected, %llu stale, %llu skipped; the model %s %.12s was "
                "loaded 1 time(s)\n",
                decision_lines.size(), info.device, info.engine_backend, static_cast<unsigned long long>(by_status[0]),
                static_cast<unsigned long long>(present), static_cast<unsigned long long>(by_status[0] - present),
                static_cast<unsigned long long>(by_status[1]), static_cast<unsigned long long>(by_status[2]),
                static_cast<unsigned long long>(by_status[3]), static_cast<unsigned long long>(by_status[4]),
                static_cast<unsigned long long>(by_status[5]), static_cast<unsigned long long>(by_status[6]),
                info.model_id, info.artifact_sha256);
  } catch (const obs::ring_error& error) {
    std::fprintf(stderr, "sense_host: %s\n", error.what());
    status = 2;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "sense_host: %s\n", error.what());
    status = 1;
  }
  if (live_fd >= 0) {
    ::close(live_fd);
  }
  sense_plan_destroy(plan);
  return status;
}
