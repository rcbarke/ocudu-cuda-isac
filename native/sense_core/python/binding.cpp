// The Python binding of the sensing core, imported as ocudu_cuda_isac.sensing.runtime (Sprint 3.6,
// selector sense_core).
//
// It exposes the C API of include/sense_core/sense_core.h to Python: plans, views and decisions,
// for unit tests, parity checks and offline scoring. No timed run imports it: the sensor process
// is sense_host, which holds no interpreter.
//
// A record is handed over as one bytes object, as a file of an archive holds it: the header, the
// host block, the validity bits and the payload. The binding keeps that object alive for as long
// as the plan holds its view, which is how a ring keeps a slot until it is released.
//
// pybind11 and NumPy. ASCII only.

#include "sense_core/sense_core.h"
#include <cstring>
#include <map>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct sense_error : std::runtime_error {
  sense_error(sense_status status_, const std::string& text) : std::runtime_error(text), status(status_) {}
  sense_status status;
};

void check(sense_status status)
{
  if (status != SENSE_OK) {
    throw sense_error(status, sense_last_error());
  }
}

const char* kind_name(int kind)
{
  switch (kind) {
    case SENSE_ANSWER_REJECTED: return "rejected";
    case SENSE_ANSWER_FILLING: return "insufficient_history";
    case SENSE_ANSWER_GAP: return "gap";
    case SENSE_ANSWER_INVALID: return "invalid";
    case SENSE_ANSWER_READY: return "ready";
    case SENSE_ANSWER_STALE_SKIPPED: return "stale_skipped";
    default: return "unknown";
  }
}

const char* status_name(int status)
{
  switch (status) {
    case SENSE_DECISION_OK: return "ok";
    case SENSE_DECISION_INVALID: return "invalid";
    case SENSE_DECISION_STALE: return "stale";
    default: return "unknown";
  }
}

py::list contributors(const sense_contributor* items, std::uint32_t count)
{
  py::list made;
  for (std::uint32_t n = 0; n != count; ++n) {
    py::dict item;
    item["sequence"]       = items[n].sequence;
    item["opportunity"]    = items[n].opportunity;
    item["absolute_slot"]  = items[n].absolute_slot;
    item["first_position"] = items[n].first_position;
    item["positions"]      = items[n].positions;
    made.append(item);
  }
  return made;
}

py::dict answer_dict(const sense_answer& answer)
{
  py::dict made;
  made["kind"]   = kind_name(answer.kind);
  made["reason"] = std::string(answer.reason);
  made["have"]   = answer.have;
  made["need"]   = answer.need;
  if (answer.has_window != 0) {
    made["epoch"]   = answer.epoch;
    made["episode"] = answer.episode;
    made["period"]  = answer.period;
  }
  if (answer.has_trigger != 0) {
    made["trigger_sequence"]    = answer.trigger_sequence;
    made["trigger_opportunity"] = answer.trigger_opportunity;
    made["trigger_slot"]        = answer.trigger_slot;
    made["t_ready_ns"]          = answer.t_ready_ns;
    made["t_publish_ns"]        = answer.t_publish_ns;
    made["t_notify_ns"]         = answer.t_notify_ns;
  }
  py::list missing;
  for (std::uint32_t n = 0; n != answer.missing; ++n) {
    missing.append(answer.missing_slots[n]);
  }
  made["missing"]      = missing;
  made["contributing"] = contributors(answer.contributing, answer.contributors);
  return made;
}

/// The clock of a test: a Python callable that returns nanoseconds.
std::int64_t python_clock(void* owner)
{
  py::gil_scoped_acquire held;
  return (*static_cast<py::object*>(owner))().cast<std::int64_t>();
}

class plan
{
public:
  plan(const std::string& artifact, const std::string& device, const std::string& engine, const std::string& engine_dir,
       const std::string& expect, bool keep_mean, bool use_graph, const std::vector<std::int64_t>& budgets_ns, py::object clock)
  {
    if (budgets_ns.size() != 4) {
      throw std::invalid_argument("budgets_ns names four budgets: ingest, kernels, engine, window");
    }
    sense_options options;
    std::memset(&options, 0, sizeof(options));
    options.api_version       = SENSE_CORE_API_VERSION;
    options.device            = device.empty() ? nullptr : device.c_str();
    options.engine            = engine.empty() ? nullptr : engine.c_str();
    options.engine_path       = engine_dir.empty() ? nullptr : engine_dir.c_str();
    options.expect_sha256     = expect.empty() ? nullptr : expect.c_str();
    options.keep_mean         = keep_mean ? 1 : 0;
    options.use_graph         = use_graph ? 1 : 0;
    options.budget_ingest_ns  = budgets_ns[0];
    options.budget_kernels_ns = budgets_ns[1];
    options.budget_engine_ns  = budgets_ns[2];
    options.budget_window_ns  = budgets_ns[3];
    if (!clock.is_none()) {
      ticks.reset(new py::object(clock));
      options.clock       = &python_clock;
      options.clock_owner = ticks.get();
    }
    check(sense_plan_create(artifact.c_str(), &options, &handle));
    check(sense_plan_info_get(handle, &facts));
    std::memset(&last, 0, sizeof(last));
  }

  ~plan()
  {
    sense_plan_destroy(handle); // releases what it still holds, through release() below
  }
  plan(const plan&)            = delete;
  plan& operator=(const plan&) = delete;

  py::dict info() const
  {
    sense_plan_info found;
    check(sense_plan_info_get(handle, &found));
    py::dict made;
    made["rx"]               = found.rx;
    made["subcarriers"]      = found.subcarriers;
    made["dmrs_symbols"]     = found.dmrs_symbols;
    made["window_records"]   = found.window_records;
    made["rows"]             = found.rows;
    made["cols"]             = found.cols;
    made["period_slots"]     = found.period_slots;
    made["scs_khz"]          = found.scs_khz;
    made["threshold_logit"]  = found.threshold_logit;
    made["model_id"]         = std::string(found.model_id);
    made["artifact_sha256"]  = std::string(found.artifact_sha256);
    made["spec_sha256"]      = std::string(found.spec_sha256);
    made["device"]           = std::string(found.device);
    made["device_name"]      = std::string(found.device_name);
    made["engine_backend"]   = std::string(found.engine_backend);
    made["engine_detail"]    = std::string(found.engine_detail);
    made["engine_sha256"]    = std::string(found.engine_sha256);
    made["core_sources_sha256"] = std::string(found.core_sources_sha256);
    made["plan_json"]        = std::string(found.plan_json);
    made["entries_json"]     = std::string(found.entries_json);
    made["plan_sha256"]      = std::string(found.plan_sha256);
    made["engine_on_stream"] = found.engine_on_stream != 0;
    made["engine_graph"]     = found.engine_graph != 0;
    made["timing_valid"]     = found.timing_valid != 0;
    return made;
  }

  std::int64_t warm_up(std::uint32_t rounds)
  {
    std::int64_t elapsed = 0;
    check(sense_plan_warm_up(handle, rounds, &elapsed));
    return elapsed;
  }

  /// Splits a record into its parts. Returns false with the reason when it is not whole.
  static void parts(const std::string& record, const char*& mask, const char*& payload)
  {
    if (record.size() < SENSE_RECORD_HEADER_BYTES) {
      throw std::invalid_argument("a record holds a header of 256 bytes");
    }
    std::uint32_t mask_bytes, payload_bytes, host_bytes;
    std::memcpy(&mask_bytes, record.data() + 168, 4);
    std::memcpy(&payload_bytes, record.data() + 172, 4);
    std::memcpy(&host_bytes, record.data() + 176, 4);
    if (record.size() != static_cast<std::size_t>(SENSE_RECORD_HEADER_BYTES) + host_bytes + mask_bytes + payload_bytes) {
      throw std::invalid_argument("the record does not hold the parts its header names");
    }
    mask    = record.data() + SENSE_RECORD_HEADER_BYTES + host_bytes;
    payload = mask + mask_bytes;
  }

  py::list offer(py::bytes record, std::int64_t t_notify_ns, bool intact)
  {
    // The object stays alive, and its bytes in place, until the plan releases the view.
    std::unique_ptr<holder> kept(new holder{this, next_id++, record, std::string(record), intact});
    const char *            mask = nullptr, *payload = nullptr;
    parts(kept->bytes, mask, payload);
    sense_view view;
    std::memset(&view, 0, sizeof(view));
    view.record      = kept->bytes.data();
    view.payload     = payload;
    view.mask        = mask;
    view.memory      = SENSE_MEMORY_HOST_SHARED;
    view.device      = -1;
    view.t_notify_ns = t_notify_ns;
    view.intact      = &plan::intact_of;
    view.release     = &plan::release;
    view.owner       = kept.get();
    holder* raw      = kept.get();
    held[raw->id]    = std::move(kept);
    std::uint32_t count = 0;
    check(sense_plan_offer(handle, &view, answers, &count));
    py::list made;
    for (std::uint32_t n = 0; n != count; ++n) {
      made.append(answer_dict(answers[n]));
    }
    answer_count = count;
    return made;
  }

  bool ready() const { return sense_plan_ready(handle) != 0; }

  py::tuple held_views() const
  {
    std::uint64_t       oldest = 0;
    const std::uint32_t count  = sense_plan_held(handle, &oldest);
    return py::make_tuple(count, count != 0 ? py::object(py::int_(oldest)) : py::object(py::none()));
  }

  void submit() { check(sense_plan_submit(handle)); }
  bool poll() { return sense_plan_poll(handle) != 0; }

  py::dict collect()
  {
    check(sense_plan_collect(handle, &last));
    have_last = true;
    return decision_dict();
  }

  py::dict run()
  {
    submit();
    while (!poll()) {
    }
    return collect();
  }

  py::dict skip()
  {
    sense_answer answer;
    check(sense_plan_skip(handle, &answer));
    return answer_dict(answer);
  }

  py::tuple finish()
  {
    std::uint32_t have = 0, need = 0;
    check(sense_plan_finish(handle, &have, &need));
    return py::make_tuple(have, need);
  }

  py::dict counters() const
  {
    sense_counters found;
    check(sense_plan_counters(handle, &found));
    py::dict made;
    made["records"]               = found.records;
    made["passed"]                = found.passed;
    made["resets"]                = found.resets;
    made["rejected"]              = found.rejected;
    made["insufficient_history"]  = found.filling;
    made["gap"]                   = found.gap;
    made["invalid"]               = found.invalid;
    made["ready"]                 = found.ready;
    made["stale"]                 = found.stale;
    made["stale_skipped"]         = found.stale_skipped;
    made["stale_skipped_records"] = found.stale_skipped_records;
    made["submissions"]           = found.submissions;
    made["decisions"]             = found.decisions;
    made["scored"]                = found.scored;
    made["present"]               = found.present;
    made["late"]                  = found.late;
    made["overrun_ingest"]        = found.overrun_ingest;
    made["overrun_kernels"]       = found.overrun_kernels;
    made["overrun_engine"]        = found.overrun_engine;
    made["overrun_window"]        = found.overrun_window;
    made["device_calls"]          = found.device_calls;
    made["device_calls_open"]     = found.device_calls_open;
    made["device_allocations"]    = found.device_allocations;
    made["views_held"]            = found.views_held;
    return made;
  }

  /// The optional stage on the decision that collect() returned last. None when it is not active.
  py::object smooth()
  {
    require_last();
    check(sense_plan_smooth(handle, &last, &smoothed));
    if (smoothed.active == 0) {
      return py::none();
    }
    py::dict made;
    made["maps"]                = smoothed.maps;
    made["cell"]                = py::make_tuple(smoothed.row, smoothed.col);
    made["position_m"]          = py::make_tuple(smoothed.x_m, smoothed.y_m);
    made["probability"]         = smoothed.probability;
    made["kalman_position_m"]   = py::make_tuple(smoothed.kalman_x_m, smoothed.kalman_y_m);
    made["kalman_velocity_m_s"] = py::make_tuple(smoothed.kalman_vx_m_s, smoothed.kalman_vy_m_s);
    return made;
  }

  std::string decision_json(std::uint64_t line, std::uint64_t index, const std::string& input_sha256,
                            const std::string& map_sha256)
  {
    require_last();
    std::vector<char> text(32768);
    check(sense_decision_json(handle, &last, line, index, input_sha256.empty() ? nullptr : input_sha256.c_str(),
                              map_sha256.empty() ? nullptr : map_sha256.c_str(), &smoothed, text.data(), text.size()));
    return text.data();
  }

  std::string window_json(std::uint64_t line, std::uint64_t index, const std::string& mean_sha256,
                          const std::string& count_sha256)
  {
    require_last();
    std::vector<char> text(32768);
    check(sense_decision_window_json(handle, &last, line, index, mean_sha256.empty() ? nullptr : mean_sha256.c_str(),
                                     count_sha256.empty() ? nullptr : count_sha256.c_str(), text.data(), text.size()));
    return text.data();
  }

  /// The two lines of answer n of the last offer(): (window line, decision line).
  py::tuple answer_json(std::uint32_t n, std::uint64_t line)
  {
    if (n >= answer_count || answers[n].kind == SENSE_ANSWER_READY) {
      throw std::invalid_argument("the last offer gave no such answer without a score");
    }
    std::vector<char> text(16384);
    check(sense_answer_window_json(handle, &answers[n], line, text.data(), text.size()));
    const std::string window = text.data();
    check(sense_answer_decision_json(handle, &answers[n], line, text.data(), text.size()));
    return py::make_tuple(window, std::string(text.data()));
  }

  /// One record through its ingest unit alone: (values complex64 [A, K, S], valid uint8 [A, K, S], first subcarrier).
  py::tuple ingest(py::bytes record)
  {
    const std::string bytes = record;
    const char *      mask = nullptr, *payload = nullptr;
    parts(bytes, mask, payload);
    sense_view view;
    std::memset(&view, 0, sizeof(view));
    view.record  = bytes.data();
    view.payload = payload;
    view.mask    = mask;
    view.memory  = SENSE_MEMORY_HOST_SHARED;
    view.device  = -1;
    std::uint32_t rx, positions, symbols_bits;
    std::uint16_t planes;
    std::memcpy(&positions, bytes.data() + 124, 4);
    std::memcpy(&symbols_bits, bytes.data() + 128, 4);
    std::memcpy(&planes, bytes.data() + 132, 2);
    rx                    = planes;
    std::uint32_t symbols = 0;
    for (unsigned bit = 0; bit != 14; ++bit) {
      symbols += (symbols_bits >> bit) & 1U;
    }
    const std::size_t          elements = static_cast<std::size_t>(rx) * positions * symbols;
    std::vector<float>         values(2 * elements + 2);
    std::vector<std::uint8_t>  valid(elements + 1);
    std::uint32_t              shape[3] = {0, 0, 0}, first = 0;
    check(sense_plan_ingest(handle, &view, values.data(), valid.data(), elements, shape, &first));
    py::array_t<std::complex<float>> cube({static_cast<py::ssize_t>(shape[0]), static_cast<py::ssize_t>(shape[1]),
                                           static_cast<py::ssize_t>(shape[2])});
    py::array_t<std::uint8_t> bits({static_cast<py::ssize_t>(shape[0]), static_cast<py::ssize_t>(shape[1]),
                                    static_cast<py::ssize_t>(shape[2])});
    std::memcpy(cube.mutable_data(), values.data(), 2 * sizeof(float) * elements);
    std::memcpy(bits.mutable_data(), valid.data(), elements);
    return py::make_tuple(cube, bits, first);
  }

private:
  struct holder {
    plan*       owner;
    std::uint64_t id;
    py::bytes   object;
    std::string bytes;
    bool        intact;
  };

  static int intact_of(void* owner) { return static_cast<holder*>(owner)->intact ? 1 : 0; }

  static void release(void* owner)
  {
    holder* kept = static_cast<holder*>(owner);
    kept->owner->held.erase(kept->id);
  }

  void require_last() const
  {
    if (!have_last) {
      throw std::invalid_argument("no decision was collected yet");
    }
  }

  py::dict decision_dict() const
  {
    py::dict made;
    made["status"]  = status_name(last.status);
    made["reason"]  = std::string(last.reason);
    made["late"]    = last.late != 0;
    made["overrun"] = py::make_tuple(last.overrun_ingest != 0, last.overrun_kernels != 0, last.overrun_engine != 0,
                                     last.overrun_window != 0);
    made["epoch"]   = last.epoch;
    made["episode"] = last.episode;
    made["period"]  = last.period;
    made["trigger_sequence"]    = last.trigger_sequence;
    made["trigger_opportunity"] = last.trigger_opportunity;
    made["trigger_slot"]        = last.trigger_slot;
    made["contributing"]        = contributors(last.contributing, last.contributors);
    made["t_first_sample"]      = last.t_first_sample;
    made["t_last_sample"]       = last.t_last_sample;
    made["elements"]            = last.elements;
    made["valid"]               = last.valid;
    made["records_min"]         = last.records_min;
    made["records_max"]         = last.records_max;
    for (const auto& mark : {std::make_pair("t_ready_ns", last.t_ready_ns), std::make_pair("t_publish_ns", last.t_publish_ns),
                             std::make_pair("t_notify_ns", last.t_notify_ns), std::make_pair("t_submit_ns", last.t_submit_ns),
                             std::make_pair("t_ingest_ns", last.t_ingest_ns), std::make_pair("t_kernels_ns", last.t_kernels_ns),
                             std::make_pair("t_engine_ns", last.t_engine_ns),
                             std::make_pair("t_decision_ns", last.t_decision_ns)}) {
      made[mark.first] = mark.second;
    }
    const py::ssize_t a = facts.rx, k = facts.subcarriers, s = facts.dmrs_symbols;
    if (last.status != SENSE_DECISION_STALE && last.input != nullptr) {
      py::array_t<float>        input({a, k}), difference({a, k});
      py::array_t<float>        map({static_cast<py::ssize_t>(facts.rows), static_cast<py::ssize_t>(facts.cols)});
      py::array_t<std::uint8_t> count({a, k, s});
      std::memcpy(input.mutable_data(), last.input, sizeof(float) * static_cast<std::size_t>(a * k));
      std::memcpy(difference.mutable_data(), last.difference, sizeof(float) * static_cast<std::size_t>(a * k));
      std::memcpy(map.mutable_data(), last.map, sizeof(float) * static_cast<std::size_t>(facts.rows) * facts.cols);
      std::memcpy(count.mutable_data(), last.count, static_cast<std::size_t>(a * k * s));
      made["input"]      = input;
      made["difference"] = difference;
      made["map"]        = map;
      made["count"]      = count;
      if (last.mean != nullptr) {
        py::array_t<std::complex<float>> mean({a, k, s});
        std::memcpy(mean.mutable_data(), last.mean, 2 * sizeof(float) * static_cast<std::size_t>(a * k * s));
        made["mean"] = mean;
      } else {
        made["mean"] = py::none();
      }
    }
    if (last.status == SENSE_DECISION_OK) {
      made["logit"]       = static_cast<double>(last.logit);
      made["probability"] = last.probability;
      made["present"]     = last.present != 0;
      made["cell"]        = py::make_tuple(last.row, last.col);
      made["position_m"]  = py::make_tuple(last.position_x_m, last.position_y_m);
      made["peak"]        = static_cast<double>(last.peak);
    }
    return made;
  }

  sense_plan*                                    handle = nullptr;
  sense_plan_info                                facts;
  std::unique_ptr<py::object>                    ticks;
  std::map<std::uint64_t, std::unique_ptr<holder>> held;
  std::uint64_t                                  next_id = 0;
  sense_answer                                   answers[2];
  std::uint32_t                                  answer_count = 0;
  sense_decision                                 last;
  sense_smoothed                                 smoothed{};
  bool                                           have_last = false;
};

} // namespace

PYBIND11_MODULE(_sense_core, module)
{
  module.doc() = "The native sensing core of ocudu-cuda-isac: plans, views and decisions. Import it as "
                 "ocudu_cuda_isac.sensing.runtime.";
  static py::exception<sense_error> refused(module, "SenseError", PyExc_ValueError);
  py::register_exception_translator([](std::exception_ptr pointer) {
    try {
      if (pointer) {
        std::rethrow_exception(pointer);
      }
    } catch (const sense_error& fault) {
      py::set_error(refused, fault.what());
    }
  });
  module.attr("API_VERSION") = SENSE_CORE_API_VERSION;
  module.def("build", [] { return std::string(sense_core_build_json()); }, "What this build of the core is, as JSON.");
  module.def("registry", [] { return std::string(sense_core_registry_json()); }, "The registry of this build, as JSON.");
  module.def("tolerance", [] { return std::string(sense_core_tolerance_json()); },
             "The tolerances this build declares against the reference, as JSON.");
  module.def("sha256", [](py::bytes data) {
    const std::string bytes = data;
    char              text[SENSE_HASH_BYTES];
    sense_sha256_hex(bytes.data(), bytes.size(), text);
    return std::string(text);
  });
  py::class_<plan>(module, "Plan")
      .def(py::init<const std::string&, const std::string&, const std::string&, const std::string&, const std::string&,
                    bool, bool, const std::vector<std::int64_t>&, py::object>(),
           py::arg("artifact"), py::arg("device") = "", py::arg("engine") = "", py::arg("engine_dir") = "",
           py::arg("expect_sha256") = "", py::arg("keep_mean") = true, py::arg("use_graph") = true,
           py::arg("budgets_ns") = std::vector<std::int64_t>{0, 0, 0, 0}, py::arg("clock") = py::none())
      .def("info", &plan::info)
      .def("warm_up", &plan::warm_up, py::arg("rounds") = 1)
      .def("offer", &plan::offer, py::arg("record"), py::arg("t_notify_ns") = 0, py::arg("intact") = true)
      .def("ready", &plan::ready)
      .def("held", &plan::held_views)
      .def("submit", &plan::submit)
      .def("poll", &plan::poll)
      .def("collect", &plan::collect)
      .def("run", &plan::run)
      .def("skip", &plan::skip)
      .def("finish", &plan::finish)
      .def("counters", &plan::counters)
      .def("smooth", &plan::smooth)
      .def("decision_json", &plan::decision_json, py::arg("line"), py::arg("index"), py::arg("input_sha256") = "",
           py::arg("map_sha256") = "")
      .def("window_json", &plan::window_json, py::arg("line"), py::arg("index"), py::arg("mean_sha256") = "",
           py::arg("count_sha256") = "")
      .def("answer_json", &plan::answer_json, py::arg("answer"), py::arg("line"))
      .def("ingest", &plan::ingest, py::arg("record"));
}
