// The loop of the sensor process (Sprint 3.6, selector sense_core). ASCII only.

#include "sensor_loop.h"
#include <cstring>
#include <time.h>
#include <unistd.h>

namespace sense {

namespace {

std::int64_t monotonic_ns()
{
  timespec stamp;
  ::clock_gettime(CLOCK_MONOTONIC, &stamp);
  return static_cast<std::int64_t>(stamp.tv_sec) * 1000000000LL + stamp.tv_nsec;
}

inline void relax()
{
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

} // namespace

sensor_loop::sensor_loop(sense_plan* plan_, const loop_options& options_) : plan(plan_), options(options_)
{
  sense_plan_info_get(plan, &info);
  input_values = static_cast<std::size_t>(info.rx) * info.subcarriers;
  map_values   = static_cast<std::size_t>(info.rows) * info.cols;
  count_values = input_values * info.dmrs_symbols;
  mean_values  = 2 * count_values;
  // Everything the loop will keep, allocated and touched before the first record.
  kept.assign(options.lines, kept_line());
  inputs.assign(options.windows * input_values, 0.0f);
  maps.assign(options.windows * map_values, 0.0f);
  if (options.keep_mean) {
    means.assign(options.windows * mean_values, 0.0f);
    window_counts.assign(options.windows * count_values, 0);
  }
}

std::int64_t sensor_loop::now() const
{
  return options.clock != nullptr ? options.clock(options.clock_owner) : monotonic_ns();
}

void sensor_loop::keep(const sense_answer& answer)
{
  ++counts.lines;
  if (kept_lines == kept.size()) {
    ++counts.lines_not_kept;
    return;
  }
  kept_line& line  = kept[kept_lines++];
  line.is_decision = false;
  line.answer      = answer;
  line.tensors     = -1;
}

void sensor_loop::keep(const sense_decision& decision, const sense_smoothed& smoothed)
{
  ++counts.lines;
  if (kept_lines == kept.size()) {
    ++counts.lines_not_kept;
    return;
  }
  kept_line& line  = kept[kept_lines++];
  line.is_decision = true;
  line.decision    = decision;
  line.smoothed    = smoothed;
  line.tensors     = -1;
  if (decision.status == SENSE_DECISION_OK) {
    if (kept_windows == options.windows) {
      ++counts.windows_not_kept;
    } else {
      const std::size_t n = kept_windows++;
      line.tensors        = static_cast<std::int64_t>(n);
      std::memcpy(inputs.data() + n * input_values, decision.input, sizeof(float) * input_values);
      std::memcpy(maps.data() + n * map_values, decision.map, sizeof(float) * map_values);
      if (options.keep_mean && decision.mean != nullptr) {
        std::memcpy(means.data() + n * mean_values, decision.mean, sizeof(float) * mean_values);
        std::memcpy(window_counts.data() + n * count_values, decision.count, count_values);
      }
    }
  }
  // The tensors of the plan are reused by the next window: a kept decision names none.
  line.decision.input = line.decision.map = line.decision.mean = line.decision.difference = nullptr;
  line.decision.count = nullptr;
}

void sensor_loop::release(record_source& source)
{
  std::uint64_t       oldest = 0;
  const std::uint32_t held   = sense_plan_held(plan, &oldest);
  source.release(held != 0 ? oldest : source.position());
}

sense_status sensor_loop::decide(record_source& source)
{
  if (options.release == loop_options::release_policy::after_radio) {
    // The radio keeps the device for the calibrated rest of its last uplink slot. No device
    // call is made while this waits.
    const std::int64_t target = sense_plan_ready_time(plan) + options.radio_allowance_ns;
    if (now() < target) {
      ++counts.waited_radio;
      while (now() < target) {
        relax();
      }
    }
  }
  sense_status status = sense_plan_submit(plan);
  if (status != SENSE_OK) {
    return status;
  }
  ++counts.submissions;
  while (sense_plan_poll(plan) == 0) {
    relax(); // the completion event is asked for its state: nothing here waits in the runtime
  }
  sense_decision decision;
  status = sense_plan_collect(plan, &decision);
  if (status != SENSE_OK) {
    return status;
  }
  // Everything below runs after the publication time of the decision.
  const std::int64_t published = decision.t_decision_ns;
  sense_smoothed     smoothed;
  sense_plan_smooth(plan, &decision, &smoothed);
  if (options.live_fd >= 0) {
    char text[8192];
    if (sense_decision_json(plan, &decision, counts.lines, kept_windows, nullptr, nullptr, &smoothed, text, sizeof(text) - 1) ==
        SENSE_OK) {
      const std::size_t length = std::strlen(text);
      text[length]             = '\n';
      const ssize_t written    = ::write(options.live_fd, text, length + 1);
      (void)written; // a page that fell behind must not stop the sensor
    }
  }
  keep(decision, smoothed);
  if (options.hold_ns > 0) {
    while (now() < published + options.hold_ns) {
      relax();
    }
  }
  release(source);
  return SENSE_OK;
}

sense_status sensor_loop::run(record_source& source)
{
  std::int64_t last = now();
  for (;;) {
    sense_view view;
    std::memset(&view, 0, sizeof(view));
    const record_source::result got = source.next(view, options.poll_ns);
    if (got == record_source::result::closed) {
      counts.closed = true;
      break;
    }
    if (got == record_source::result::none) {
      const std::int64_t silent = now() - last;
      if (silent > options.idle_ns) {
        counts.idle = true;
        break;
      }
      if (options.open_ns > 0 && silent >= options.open_ns && sense_plan_ready(plan) == 0 &&
          sense_plan_held(plan, nullptr) != 0) {
        // The last slot of the window that is filling cannot arrive any more. It closes as the
        // first record of a later period would close it, and its credits return now.
        sense_answer answer;
        if (sense_plan_close_open(plan, &answer) == SENSE_OK) {
          ++counts.closed_open;
          keep(answer);
        }
        release(source);
      }
      continue;
    }
    if (got == record_source::result::stale) {
      // The slot was rewritten before it was read. No decision uses that record: its window
      // closes as a gap.
      ++counts.stale_reads;
      release(source);
      last = now();
      continue;
    }
    const std::int64_t seen = now();
    last                    = seen;
    view.t_notify_ns        = seen;
    if (counts.read++ == 0) {
      counts.first_record_ns = seen;
    }
    counts.last_record_ns = seen;
    sense_answer  answers[2];
    std::uint32_t count = 0;
    sense_status  status = sense_plan_offer(plan, &view, answers, &count);
    if (status != SENSE_OK) {
      return status;
    }
    for (std::uint32_t n = 0; n != count; ++n) {
      if (answers[n].kind != SENSE_ANSWER_READY) {
        keep(answers[n]);
      }
    }
    if (sense_plan_ready(plan) == 0) {
      release(source);
      continue;
    }
    if (source.live() && source.pending() > 0) {
      // More records are published already. They are read before anything is submitted: when
      // they complete a newer window, this one is skipped and never queued behind it.
      release(source);
      continue;
    }
    status = decide(source);
    if (status != SENSE_OK) {
      return status;
    }
  }
  if (sense_plan_ready(plan) != 0) {
    const sense_status status = decide(source); // the newest complete window, when the carrier ended
    if (status != SENSE_OK) {
      return status;
    }
  }
  release(source);
  return SENSE_OK;
}

void sensor_loop::seal(std::vector<std::string>& window_lines, std::vector<std::string>& decision_lines) const
{
  std::vector<char> text(32768);
  char              input_hash[SENSE_HASH_BYTES], map_hash[SENSE_HASH_BYTES], mean_hash[SENSE_HASH_BYTES],
      count_hash[SENSE_HASH_BYTES];
  window_lines.clear();
  decision_lines.clear();
  for (std::size_t number = 0; number != kept_lines; ++number) {
    const kept_line& line = kept[number];
    if (!line.is_decision) {
      sense_answer_window_json(plan, &line.answer, number, text.data(), text.size());
      window_lines.emplace_back(text.data());
      sense_answer_decision_json(plan, &line.answer, number, text.data(), text.size());
      decision_lines.emplace_back(text.data());
      continue;
    }
    const bool        scored = line.tensors >= 0;
    const std::size_t n      = scored ? static_cast<std::size_t>(line.tensors) : 0;
    const bool        means_kept = scored && options.keep_mean;
    if (scored) {
      sense_sha256_hex(input_of(n), sizeof(float) * input_values, input_hash);
      sense_sha256_hex(map_of(n), sizeof(float) * map_values, map_hash);
    }
    if (means_kept) {
      sense_sha256_hex(mean_of(n), sizeof(float) * mean_values, mean_hash);
      sense_sha256_hex(count_of(n), count_values, count_hash);
    }
    sense_decision_window_json(plan, &line.decision, number, n, means_kept ? mean_hash : nullptr,
                               means_kept ? count_hash : nullptr, text.data(), text.size());
    window_lines.emplace_back(text.data());
    sense_decision_json(plan, &line.decision, number, n, scored ? input_hash : nullptr, scored ? map_hash : nullptr,
                        &line.smoothed, text.data(), text.size());
    decision_lines.emplace_back(text.data());
  }
}

} // namespace sense
