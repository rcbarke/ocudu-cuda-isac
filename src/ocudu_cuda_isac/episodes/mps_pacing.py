"""Paced receiver and two CUDA clients under MPS: the plan, the readers and the rules (Sprint 4.1, selector mps_pacing).

The native tool pilot_ls reads a fourth plan version from this sprint on. pilot-ls-plan/v4 is the
plan of Sprint 5.2 with one more object:

  pacing    null, or the offered schedule of a paced run: the clock, the delay between the end of
            the preparation and the start of the schedule, the processing allowance of one job,
            the latest start of a job after its release, and one release offset per opportunity
            of the replay, in its order, in microseconds from the start of the schedule.

A paced run is a timed run in which the staged harness behaves as a scripted receiver. The grid
of an opportunity is filled ahead of its slot, the tap and the decodes start at the absolute
release time, a release the harness could not start within its latest start is skipped, kept
under its key and counted, and no release time ever moves. The rule is one function of the
host-free pacing core (native/ocudu_adapter/pacing/pacing_core.h) and is repeated here by
Pacer, so that a unit case drives it with a clock of its own.

This module is the offline side of the ticket. It

  * derives the offered schedule of a replay from the frame and slot of its opportunities;
  * writes and checks a plan of version 4 around a receiver replay plan of Sprint 2.4;
  * reads a native output of such a plan: everything a plan of version 3 writes with the readers
    of the earlier sprints, and pacing/ with one line per release, checked against the rule,
    against the offered schedule and against the radio records;
  * states what an MPS inventory must show before an arm may be labeled MPS, and refuses the
    label with the client list in the message;
  * reduces the radio side of an arm to counts and service-time tails, compares two arms by a
    rule that was declared before either was measured, and counts the decodes that ran on the
    device while the sensor's submission ran on it.

The launcher that starts the processes of an arm is scripts/run_m1.py. Nothing here starts a
process, reads a GPU or runs in a slot loop. Standard library only.
"""

import hashlib
import json
import math
from pathlib import Path

from . import grid_apply as replay
from . import native_estimate as ne
from . import replay_pusch as rp
from . import snapshot_ring as sr
from .grid_apply import GridApplyError
from .native_estimate import NativeEstimatePlanError
from .replay_pusch import ReplayPuschError
from .snapshot_ring import SnapshotRingError
from ..replay import ContractError, load_json
from ..sensing import pilot_ls as o2_reference

PLAN_SCHEMA = "pilot-ls-plan/v4"
NATIVE_SCHEMA = "pilot-ls-native/v4"
RELEASE_SCHEMA = "pace-release/v1"
PACING_SCHEMA = "pace-report/v1"
INVENTORY_SCHEMA = "mps-inventory/v1"
RULE_SCHEMA = "tail-shift-rule/v1"
PACING_DIR = "pacing"
RELEASES_FILE = "releases.jsonl"
CLOCK = "host_monotonic"
PLAN_FIELDS = ne.PLAN_FIELDS + ("pacing",)
PACING_FIELDS = ("clock", "start_delay_us", "allowance_us", "latest_start_us", "release_us")
RELEASE_FIELDS = ("schema", "run_id", "sequence", "opportunity_id", "sfn", "slot", "offset_us", "outcome", "overrun",
                  "t_release_ns", "t_free_ns", "t_ready_ns", "t_start_ns", "t_done_ns", "fill_us", "decodes")
DECODE_FIELDS = ("backend", "key", "outcome", "skip", "t_start_ns", "t_callback_ns", "service_us")
COUNT_FIELDS = ("offered", "taken", "on_time", "late", "skipped", "overruns", "max_lateness_ns", "max_excess_ns")
RELEASE_OUTCOMES = ("on_time", "late", "skipped")
SKIPS = ("missed_release", "no_free_grid", "backend_stopped")
MAX_OFFSET_US = 3_600_000_000
# The arms of the ticket. Each names what runs beside the receiver, and whether the receiver runs.
ARMS = ("dsp_only", "capture_only", "ai_only", "concurrent", "concurrent_reference")
TIMED_ARMS = ("dsp_only", "capture_only", "ai_only", "concurrent")
# How the device is shared. An arm carries one of these labels, and "mps" only after its proof.
SHARING = ("mps", "time_slicing")
# The contention controls, in the order they are applied, one at a time, to the sensor client only.
CONTROLS = ("mps_default", "mps_priority", "mps_thread_pct", "release_after_radio")
RULE_FIELDS = ("schema", "status", "quantile", "min_samples", "allowed_shift_us", "allowed_shift_fraction")
QUANTILES = {"p50": 0.50, "p95": 0.95, "p99": 0.99, "p999": 0.999}


class MpsPacingError(ValueError):
    """A plan, a native output, an inventory or a comparison violates the documented interface."""


# ---------------------------------------------------------------- offered schedule
def opportunities_of(replay_plan_path):
    """The opportunities of a receiver replay plan, in its order: what a schedule is derived from."""
    try:
        channel = replay.load_plan(rp.load_plan(replay_plan_path)["channel_plan"]["path"])
    except (ReplayPuschError, GridApplyError, OSError, KeyError, TypeError) as error:
        raise MpsPacingError(f"the replay plan names no readable channel plan: {error}") from None
    return channel["opportunities"]


def release_offsets(opportunities, slot_us=500, slots_per_frame=20, gap_slots=10):
    """The release offset of every opportunity, in microseconds from the start of the schedule.

    An opportunity follows the one before it by the distance of their slots when both belong to
    one realization and the slot rises. Anything else, as the first opportunity of another
    realization, comes gap_slots later: one TDD period by default. This is the pace at which
    sense_feed publishes a consumer archive, so that an arm with a receiver and an arm that
    replays buffered records offer the same trace.
    """
    if type(slot_us) is not int or type(slots_per_frame) is not int or type(gap_slots) is not int \
            or slot_us < 1 or slots_per_frame < 1 or gap_slots < 1:
        raise MpsPacingError("a schedule needs a slot length, the slots of a frame and a gap, each a positive integer")
    offsets, before, at = [], None, 0
    for item in opportunities:
        try:
            now = (item["realization_id"], item["sfn"] * slots_per_frame + item["slot"])
        except (KeyError, TypeError) as error:
            raise MpsPacingError(f"an opportunity names no realization, frame or slot: {error!r}") from None
        if before is not None:
            follows = now[0] == before[0] and now[1] > before[1]
            at += (now[1] - before[1] if follows else gap_slots) * slot_us
        offsets.append(at)
        before = now
    return offsets


def pacing_spec(release_us, allowance_us, latest_start_us, start_delay_us=20000):
    """The pacing object of a plan."""
    return check_pacing({"clock": CLOCK, "start_delay_us": start_delay_us, "allowance_us": allowance_us,
                         "latest_start_us": latest_start_us, "release_us": list(release_us)})


def check_pacing(pacing, releases=None):
    """Apply the rules of an offered schedule. releases, when given, is the number of opportunities it must name."""
    if not isinstance(pacing, dict) or set(pacing) != set(PACING_FIELDS):
        raise MpsPacingError(f"pacing requires exactly {', '.join(PACING_FIELDS)}, or is null")
    if pacing["clock"] != CLOCK:
        raise MpsPacingError(f'pacing.clock is "{CLOCK}"')
    for name in ("start_delay_us", "allowance_us", "latest_start_us"):
        if type(pacing[name]) is not int or not 0 <= pacing[name] <= 60_000_000:
            raise MpsPacingError(f"pacing.{name} must be an integer from 0 to 60000000")
    if pacing["allowance_us"] < 1:
        raise MpsPacingError("the processing allowance of a paced run is positive")
    if pacing["latest_start_us"] > pacing["allowance_us"]:
        raise MpsPacingError("the latest start of a paced run lies between zero and the processing allowance")
    offsets = pacing["release_us"]
    if not isinstance(offsets, list) or not offsets:
        raise MpsPacingError("a paced run names at least one release")
    before = -1
    for position, offset in enumerate(offsets):
        if type(offset) is not int or not 0 <= offset <= MAX_OFFSET_US or offset <= before:
            raise MpsPacingError(f"release {position} of a paced run does not lie after the release before it: the "
                                 f"offsets are integers from 0 to {MAX_OFFSET_US} and rise")
        before = offset
    if releases is not None and len(offsets) != releases:
        raise MpsPacingError(f"pacing.release_us names {len(offsets)} releases, and the replay holds {releases} "
                             "opportunities: one release each")
    return pacing


# ---------------------------------------------------------------- plan
def _third_version(plan):
    """The part of a plan that is a plan of Sprint 5.2."""
    inner = {key: plan[key] for key in ne.PLAN_FIELDS}
    inner["schema"] = ne.PLAN_SCHEMA
    return inner


def check_plan(plan):
    """Apply every rule of a plan of version 4 that needs no file."""
    if not isinstance(plan, dict) or set(plan) != set(PLAN_FIELDS):
        raise MpsPacingError(f"a plan requires exactly {', '.join(PLAN_FIELDS)}")
    if plan["schema"] != PLAN_SCHEMA:
        raise MpsPacingError(f"unsupported plan schema {plan['schema']!r}")
    try:
        ne.check_plan(_third_version(plan))
    except NativeEstimatePlanError as error:
        raise MpsPacingError(str(error)) from None
    if plan["pacing"] is None:
        return plan
    check_pacing(plan["pacing"])
    if plan["evidence"]["mode"] != "timed":
        raise MpsPacingError("a paced run is a timed run: the plan names an offered schedule and the evidence mode audit")
    return plan


def check_files(plan):
    """Verify everything a plan binds, as Sprint 5.2 does, and the length of its schedule."""
    check_plan(plan)
    try:
        replay_plan, channel, sources = ne.check_files(_third_version(plan))
    except NativeEstimatePlanError as error:
        raise MpsPacingError(str(error)) from None
    if plan["pacing"] is not None:
        check_pacing(plan["pacing"], len(channel["opportunities"]))
    return replay_plan, channel, sources


def build_plan(run_id, replay_plan_path, pacing, ring=None, estimate=None, tap="pilot_ls", antenna_map=None, epoch=1,
               source="pusch_dmrs"):
    """A plan of version 4 around a replay plan file that was written before. The evidence mode is timed."""
    try:
        plan = sr.build_plan(run_id, replay_plan_path, "timed", ring, tap, antenna_map, epoch, source)
    except SnapshotRingError as error:
        raise MpsPacingError(str(error)) from None
    plan.update(schema=PLAN_SCHEMA, estimate=estimate, pacing=pacing)
    check_files(plan)
    return plan


def write_plan(plan, path):
    """Check a plan and write it. The file must not exist."""
    path = Path(path)
    if path.exists():
        raise MpsPacingError(f"{path} already exists. A plan is never overwritten.")
    check_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return path


def load_plan(path):
    try:
        return check_plan(load_json(Path(path)))
    except (ContractError, OSError) as error:
        raise MpsPacingError(f"cannot read the plan: {error}") from None


# ---------------------------------------------------------------- the release rule
class Pacer:
    """The release rule of the pacing core, in another language. It reads no clock: every call is told the time.

        release(n) = start + offset(n)                        never shifted
        free at t <= release(n) + latest_start                job n runs, at release(n) or, when its
                                                              input was ready later, when it is ready
        free at t >  release(n) + latest_start                a skipped release, kept and counted
        done later than release(n) + allowance                an overrun, counted

    Times are integers in one unit, nanoseconds in the tool.
    """

    def __init__(self, release, allowance, latest_start):
        if not release or allowance <= 0 or not 0 <= latest_start <= allowance \
                or any(b <= a for a, b in zip(release, release[1:])) or release[0] < 0:
            raise MpsPacingError("a schedule names rising offsets, a positive allowance and a latest start inside it")
        self.offsets, self.allowance, self.latest_start = list(release), allowance, latest_start
        self.marks, self.next, self.started = [], 0, None

    def begin(self, start):
        if self.started is not None:
            raise MpsPacingError("the offered schedule of a paced run is fixed once")
        self.started = start
        self.marks = [{"release": start + offset, "free": None, "ready": None, "start": None, "done": None,
                       "outcome": "pending", "overrun": False} for offset in self.offsets]

    def _at(self, n, action):
        if self.started is None or not 0 <= n < len(self.marks):
            raise MpsPacingError(f"{action}: the paced run has not begun, or holds no release {n}")
        return self.marks[n]

    def take(self, n, now):
        mark = self._at(n, "take")
        if n != self.next:
            raise MpsPacingError(f"take: release {n} is taken out of order, release {self.next} is next")
        self.next += 1
        mark["free"] = now
        if now > mark["release"] + self.latest_start:
            mark["outcome"] = "skipped"
            return False
        return True

    def give_up(self, n, now):
        mark = self._at(n, "give_up")
        if mark["outcome"] != "pending" or mark["free"] is None or mark["ready"] is not None:
            raise MpsPacingError(f"give_up: release {n} was not taken, or was prepared before")
        mark.update(free=now, outcome="skipped")

    def ready(self, n, now):
        mark = self._at(n, "ready")
        if mark["outcome"] != "pending" or mark["free"] is None or mark["ready"] is not None:
            raise MpsPacingError(f"ready: release {n} was not taken, or was prepared before")
        mark["ready"] = now

    def release_time(self, n):
        return self._at(n, "release_time")["release"]

    def start(self, n, now):
        mark = self._at(n, "start")
        if mark["ready"] is None or mark["start"] is not None:
            raise MpsPacingError(f"start: release {n} was not prepared, or was started before")
        mark.update(start=now, outcome="late" if mark["ready"] > mark["release"] else "on_time")

    def done(self, n, now):
        mark = self._at(n, "done")
        if mark["start"] is None or mark["done"] is not None:
            raise MpsPacingError(f"done: release {n} was not started, or was done before")
        mark.update(done=now, overrun=now > mark["release"] + self.allowance)

    def counts(self):
        return count_marks(self.marks, self.allowance)


def count_marks(marks, allowance):
    """The counts of a list of marks, as the pacing core reports them."""
    total = dict.fromkeys(COUNT_FIELDS, 0)
    total["offered"] = len(marks)
    for mark in marks:
        if mark["outcome"] == "pending":
            continue
        total["taken"] += 1
        total[mark["outcome"]] += 1
        if mark["outcome"] != "skipped":
            total["max_lateness_ns"] = max(total["max_lateness_ns"], mark["start"] - mark["release"])
        if mark["overrun"]:
            total["overruns"] += 1
            total["max_excess_ns"] = max(total["max_excess_ns"], mark["done"] - mark["release"] - allowance)
    return total


def drive(pacer, start, work, prepare=0):
    """A worker that takes work on each job it runs, with its input ready prepare after it is free.

    work is a number or a function of the job index. Returns the time at which the worker is free
    at the end. The unit target of the pacing core has the same function.
    """
    now = start
    pacer.begin(start)
    for n in range(len(pacer.offsets)):
        if not pacer.take(n, now):
            continue
        now += prepare
        pacer.ready(n, now)
        now = max(now, pacer.release_time(n))
        pacer.start(n, now)
        now += work(n) if callable(work) else work
        pacer.done(n, now)
    return now


# ---------------------------------------------------------------- native output
class Output:
    """A verified native output of a plan of version 4."""

    def __init__(self, directory, native, releases):
        self.directory, self.native, self.releases = Path(directory), native, releases
        self.report, self.records, self.run_id, self.mode = native.report, native.records, native.run_id, native.mode
        self.pacing = self.report["pacing"]
        self.base, self.estimate, self.captures = native.base, native.estimate, native.captures

    def paced(self):
        return self.pacing is not None

    def ran(self):
        """The releases that were handed to the receiver."""
        return [line for line in self.releases if line["outcome"] != "skipped"]


def release_violations(line, pacing):
    """What is wrong with one pace-release/v1 line on its own, against the schedule of its report."""
    if not isinstance(line, dict) or set(line) != set(RELEASE_FIELDS):
        return ["the line does not hold exactly the fields of a pace-release/v1 line"]
    found = []
    outcome = line["outcome"]
    if line["schema"] != RELEASE_SCHEMA or outcome not in RELEASE_OUTCOMES:
        return ["another schema or an unknown outcome"]
    allowance, latest = pacing["allowance_us"] * 1000, pacing["latest_start_us"] * 1000
    release, free = line["t_release_ns"], line["t_free_ns"]
    if type(release) is not int or type(free) is not int \
            or release != pacing["t_start_ns"] + line["offset_us"] * 1000:
        return found + ["the release time is not the start of the schedule plus the offset of the release"]
    decodes = line["decodes"]
    if not isinstance(decodes, list) or not decodes \
            or any(not isinstance(item, dict) or set(item) != set(DECODE_FIELDS) for item in decodes):
        return found + ["a release names its decodes, each with exactly the fields of a decode"]
    times = [line[name] for name in ("t_ready_ns", "t_start_ns", "t_done_ns")]
    if outcome == "skipped":
        if any(time is not None for time in times) or line["overrun"] or line["fill_us"] is not None:
            found.append("a skipped release names no time of a job and no overrun")
        if any(item["outcome"] != "not_dispatched" or item["skip"] not in SKIPS or item["t_start_ns"] is not None
               for item in decodes):
            found.append("every decode of a skipped release is not dispatched, with the reason of the skip")
        missed = any(item["skip"] == "missed_release" for item in decodes)
        if missed and not free > release + latest:
            found.append("a release was skipped although the receiver was free within its latest start")
        return found
    ready, start, done = times
    if any(type(time) is not int for time in times):
        return found + ["a release that ran names when its input was ready, when it started and when it was done"]
    if free > release + latest:
        found.append("a release ran although the receiver was free only after its latest start")
    if not free <= ready <= start <= done:
        found.append("the times of a release are not in order: free, ready, start, done")
    if start < release:
        found.append("a job started before its offered release time")
    if (outcome == "late") != (ready > release):
        found.append("a release is late exactly when its input was ready after its release time")
    if line["overrun"] != (done > release + allowance):
        found.append("a release is an overrun exactly when it was done later than its release plus the allowance")
    for item in decodes:
        began, called = item["t_start_ns"], item["t_callback_ns"]
        if item["outcome"] == "not_dispatched":
            if item["skip"] != "backend_stopped" or began is not None:
                found.append("a decode that was not started inside a release that ran names a stopped backend")
            continue
        if item["skip"] is not None or type(began) is not int or not start <= began <= done:
            found.append("a decode starts inside the job of its release")
        elif called is not None and (not began <= called <= done or item["service_us"] is None
                                     or abs(item["service_us"] - (called - began) / 1000.0) > 1e-6):
            found.append("the callback of a decode lies inside the job of its release, and its service time is the "
                         "distance from its start to its callback")
    return found


def open_output(directory):
    """Verify a native output of a plan of version 4 and return it.

    Everything a plan of version 3 writes is verified as Sprint 5.2 verifies it. With a schedule,
    pacing/ holds one line per opportunity, in the order of the opportunities, consistent with
    the release rule on its own times, with the offered schedule of the report and with the
    radio records: a skipped release has only decodes that were not dispatched, and the service
    time of a decode is the one its radio record names. The counts of the report are those of
    the lines. pacing/ holds nothing else.
    """
    directory = Path(directory)
    try:
        native = ne.open_output(directory, schema=NATIVE_SCHEMA)
        report = native.report
        pacing = report["pacing"]
    except NativeEstimatePlanError as error:
        raise MpsPacingError(str(error)) from None
    except (KeyError, TypeError) as error:
        raise MpsPacingError(f"{directory} is not a complete native output of a plan of version 4: {error!r}") from None
    folder = directory / PACING_DIR
    if pacing is None:
        if folder.exists():
            raise MpsPacingError("the report names no schedule, and the output holds pacing/")
        return Output(directory, native, [])
    try:
        if pacing["schema"] != PACING_SCHEMA or pacing["clock"] != CLOCK or set(pacing["counts"]) != set(COUNT_FIELDS) \
                or type(pacing["t_start_ns"]) is not int or pacing["file"] != f"{PACING_DIR}/{RELEASES_FILE}":
            raise MpsPacingError("the report names a schedule that is not a pace-report/v1 object on the host's "
                                 "monotonic clock")
        lines = [json.loads(line) for line in (folder / RELEASES_FILE).read_text(encoding="ascii").splitlines()
                 if line.strip()]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise MpsPacingError(f"{folder} holds no readable account of the releases: {error!r}") from None
    if native.mode != "timed":
        raise MpsPacingError("a paced run is a timed run, and the report names the evidence mode audit")
    entries = report["opportunities"]
    if len(lines) != len(entries) or pacing["counts"]["offered"] != len(lines):
        raise MpsPacingError(f"the account holds {len(lines)} releases for {len(entries)} opportunities")
    radio = {record["key"]: record for record in native.records}
    marks, before = [], None
    for position, (line, item) in enumerate(zip(lines, entries)):
        problems = release_violations(line, pacing)
        if problems:
            raise MpsPacingError(f"release {position}: {problems[0]}")
        if line["run_id"] != report["run_id"] or line["sequence"] != item["sequence"] \
                or line["opportunity_id"] != item["opportunity_id"] \
                or (line["sfn"], line["slot"]) != (item["sfn"], item["slot"]):
            raise MpsPacingError(f"release {position} does not belong to opportunity {item['sequence']} of the report")
        if before is not None and line["offset_us"] <= before:
            raise MpsPacingError(f"release {position} does not lie after the release before it")
        before = line["offset_us"]
        skipped = line["outcome"] == "skipped"
        if skipped != (item.get("grid_index") is None):
            raise MpsPacingError(f"release {position} is skipped exactly when its opportunity was filled with no grid")
        if len(line["decodes"]) != len(report["backends"]):
            raise MpsPacingError(f"release {position} does not name one decode per backend")
        for decode in line["decodes"]:
            record = radio.get(decode["key"])
            if record is None or record["opportunity_sequence"] != item["sequence"] \
                    or record["backend"] != decode["backend"] or record["outcome"] != decode["outcome"]:
                raise MpsPacingError(f"release {position} names a decode that is not the radio record of its key")
            if record["timing"]["service_us"] != decode["service_us"]:
                raise MpsPacingError(f"release {position} names another service time than the radio record of its "
                                     "decode")
        marks.append({"release": line["t_release_ns"], "start": line["t_start_ns"], "done": line["t_done_ns"],
                      "outcome": line["outcome"], "overrun": line["overrun"]})
    counted = count_marks(marks, pacing["allowance_us"] * 1000)
    if counted != pacing["counts"]:
        raise MpsPacingError("the counts of the report are not those of its releases")
    if pacing["first_offset_us"] != lines[0]["offset_us"] or pacing["last_offset_us"] != lines[-1]["offset_us"]:
        raise MpsPacingError("the report names another first or last offset than its releases")
    extra = sorted(path.name for path in folder.iterdir() if path.name != RELEASES_FILE or not path.is_file())
    if extra:
        raise MpsPacingError(f"pacing/ holds {extra[0]}, which the report does not name")
    return Output(directory, native, lines)


def offered_matches(output, plan):
    """What differs between the releases of an output and the schedule of the plan it ran. An empty list: nothing."""
    pacing = plan["pacing"]
    found = []
    if output.pacing is None:
        return ["the output names no schedule"]
    if [line["offset_us"] for line in output.releases] != pacing["release_us"]:
        found.append("the offsets of the releases are not those of the plan")
    for name in ("start_delay_us", "allowance_us", "latest_start_us"):
        if output.pacing[name] != pacing[name]:
            found.append(f"the report names another {name} than the plan")
    return found


# ---------------------------------------------------------------- reduction
def nearest_rank(values, share):
    """The nearest-rank quantile of a list: the smallest value with at least share of the list at or below it."""
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(1, math.ceil(share * len(ordered))) - 1]


def statistics(values):
    """Count, mean, median, tails and extremes of a list of durations in microseconds."""
    values = [float(value) for value in values]
    made = {"count": len(values), "mean_us": None, "min_us": None, "max_us": None}
    made.update({f"{name}_us": None for name in QUANTILES})
    if values:
        made.update(mean_us=sum(values) / len(values), min_us=min(values), max_us=max(values))
        made.update({f"{name}_us": nearest_rank(values, share) for name, share in QUANTILES.items()})
    return made


def service_times(output, backend="cuda"):
    """The PUSCH service times of one backend, start of the decode to its callback, without the first use."""
    return [record["timing"]["service_us"] for record in output.records
            if record["backend"] == backend and record["timing"]["service_us"] is not None
            and not record["timing"]["first_use"]]


def radio_summary(output, backend="cuda"):
    """The radio side of one paced arm: counts that stay in their denominators, and the service-time tails."""
    records = [record for record in output.records if record["backend"] == backend]
    outcomes = {}
    for record in records:
        outcomes[record["outcome"]] = outcomes.get(record["outcome"], 0) + 1
    ran = output.ran()
    allowance = None if output.pacing is None else output.pacing["allowance_us"] * 1000
    within = sum(1 for line in ran if not line["overrun"])
    made = {"backend": backend, "planned": len(records), "outcomes": outcomes,
            "decoded": outcomes.get("decoded", 0), "service": statistics(service_times(output, backend)),
            "resident": sum((record["path"] or {}).get("executed") == "resident" for record in records),
            "pacing": None}
    if output.pacing is not None:
        made["pacing"] = {
            "counts": output.pacing["counts"], "allowance_us": output.pacing["allowance_us"],
            "latest_start_us": output.pacing["latest_start_us"],
            "within_allowance": within, "offered": len(output.releases),
            "lateness": statistics([(line["t_start_ns"] - line["t_release_ns"]) / 1000.0 for line in ran]),
            "busy": statistics([(line["t_done_ns"] - line["t_start_ns"]) / 1000.0 for line in ran]),
            "release_to_done": statistics([(line["t_done_ns"] - line["t_release_ns"]) / 1000.0 for line in ran]),
            "fill": statistics([line["fill_us"] for line in ran]),
            "slack_ahead": statistics([(line["t_release_ns"] - line["t_ready_ns"]) / 1000.0 for line in ran]),
            "allowance_ns": allowance}
    return made


def check_rule(rule):
    """The rule by which the concurrent arm may move the radio's tail. Declared before either arm is measured."""
    if not isinstance(rule, dict) or set(rule) != set(RULE_FIELDS) or rule["schema"] != RULE_SCHEMA:
        raise MpsPacingError(f"a rule requires exactly {', '.join(RULE_FIELDS)} and the schema {RULE_SCHEMA}")
    if rule["quantile"] not in QUANTILES or type(rule["min_samples"]) is not int or rule["min_samples"] < 1:
        raise MpsPacingError("a rule names a quantile of " + ", ".join(QUANTILES) + " and a positive sample count")
    for name in ("allowed_shift_us", "allowed_shift_fraction"):
        if isinstance(rule[name], bool) or not isinstance(rule[name], (int, float)) or rule[name] < 0:
            raise MpsPacingError(f"{name} of a rule is a number that is not negative")
    if rule["status"] not in ("proposed", "calibrated"):
        raise MpsPacingError('the status of a rule is "proposed" or "calibrated"')
    needed = math.ceil(1.0 / (1.0 - QUANTILES[rule["quantile"]]))
    if rule["min_samples"] < needed:
        raise MpsPacingError(f"{rule['min_samples']} samples do not resolve the quantile {rule['quantile']}: "
                             f"at least {needed} are needed")
    return rule


def rule_sha256(rule):
    return hashlib.sha256(json.dumps(check_rule(rule), sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def tail_shift(base, other, rule):
    """How far the tail of other lies above the tail of base, against the rule.

    base and other are lists of service times in microseconds of the same offered trace. The
    allowed shift is the larger of the absolute term and the fraction of the base tail. A list
    with fewer samples than the rule needs gives no verdict: exceeded is None.
    """
    check_rule(rule)
    name, share = rule["quantile"], QUANTILES[rule["quantile"]]
    made = {"quantile": name, "base": {"count": len(base), "value_us": nearest_rank(base, share)},
            "other": {"count": len(other), "value_us": nearest_rank(other, share)},
            "shift_us": None, "allowed_us": None, "exceeded": None, "reason": None, "rule_sha256": rule_sha256(rule)}
    if min(len(base), len(other)) < rule["min_samples"]:
        made["reason"] = (f"{min(len(base), len(other))} samples, and the rule needs {rule['min_samples']} for the "
                          f"quantile {name}")
        return made
    allowed = max(rule["allowed_shift_us"], rule["allowed_shift_fraction"] * made["base"]["value_us"])
    shift = made["other"]["value_us"] - made["base"]["value_us"]
    made.update(shift_us=shift, allowed_us=allowed, exceeded=shift > allowed)
    return made


def select_control(shifts, order=CONTROLS):
    """The contention control in effect: the first of the order whose tail shift does not exceed the rule.

    shifts maps a control to its tail_shift result, to None for a planned cell that this host
    lacks, or is missing for a control that was never reached. Returns (control, reason). The
    control is None when none of those that ran restored the tail.
    """
    for name in order:
        if name not in shifts:
            break
        found = shifts[name]
        if found is None:
            continue
        if found["exceeded"] is None:
            return None, f"{name} gives no verdict: {found['reason']}"
        if not found["exceeded"]:
            return name, (f"{name} keeps the {found['quantile']} of the radio within {found['allowed_us']:.1f} us of "
                          f"the capture-only arm: {found['shift_us']:+.1f} us")
    return None, "no control that ran restored the tail of the radio"


# ---------------------------------------------------------------- MPS inventory
def parse_pids(text):
    """The process identifiers that a command of the MPS control interface printed, one per line."""
    found = []
    for line in str(text).splitlines():
        word = line.strip()
        if word.isdigit():
            found.append(int(word))
    return found


def inventory_problems(inventory):
    """Why an inventory proves no MPS service on one device. An empty list: one server on one verified device."""
    if not isinstance(inventory, dict) or inventory.get("schema") != INVENTORY_SCHEMA:
        return ["no mps-inventory/v1 record"]
    found = []
    servers = inventory.get("servers") or []
    if not inventory.get("daemon", {}).get("alive"):
        found.append("the MPS control daemon does not answer")
    if len(servers) != 1:
        found.append(f"{len(servers)} MPS servers are listed, and an arm runs on exactly one")
    devices = inventory.get("devices") or []
    if len(devices) != 1 or not devices[0].get("uuid"):
        found.append(f"an unverified server/device pair: the daemon sees {len(devices)} devices, and the pair is "
                     "verified for exactly one device named by its UUID")
    return found


def verify_clients(inventory, needed, extra=0):
    """Refuse the label MPS unless every needed process is a client of the one server on the one device.

    needed maps a role to its process identifier. Returns the server and the device. The message
    of a refusal carries the client list. Two roles with one identifier are refused, as is a
    client list that names an identifier twice, and more clients than the needed ones plus extra.
    """
    problems = inventory_problems(inventory)
    server = (inventory.get("servers") or [None])[0] if isinstance(inventory, dict) else None
    clients = list((inventory.get("clients") or {}).get(str(server), [])) if isinstance(inventory, dict) else []
    listed = ", ".join(str(pid) for pid in clients) or "none"
    if len(set(needed.values())) != len(needed):
        problems.append("duplicate client IDs: two roles of the arm name one process: "
                        + ", ".join(f"{role} {pid}" for role, pid in sorted(needed.items())))
    if len(set(clients)) != len(clients):
        problems.append("duplicate client IDs: the client list names a process twice")
    for role, pid in sorted(needed.items()):
        if pid not in clients:
            problems.append(f"the {role} process {pid} is not a client of the MPS server {server}: it shares the "
                            "device by time slicing")
    others = [pid for pid in clients if pid not in needed.values()]
    if len(others) > extra:
        problems.append(f"{len(others)} further client(s) of the server, {', '.join(str(pid) for pid in others)}, "
                        f"and the arm allows {extra}")
    if problems:
        raise MpsPacingError("the arm is not labeled MPS: " + "; ".join(problems) + f". Client list of the server "
                             f"{server}: {listed}")
    return {"server": server, "device": inventory["devices"][0], "clients": clients,
            "roles": {role: pid for role, pid in sorted(needed.items())}}


# ---------------------------------------------------------------- concurrency on the device
def overlap(output, decisions, backend="cuda"):
    """Whether both clients worked on the device over a common interval, from their own marks.

    A radio interval runs from the start of a decode to its callback. A sensor interval runs
    from the submission of a window to the completion of its engine. Both are on the host's
    monotonic clock. Returns the counts and the common interval.
    """
    radio = sorted((decode["t_start_ns"], decode["t_callback_ns"]) for line in output.releases
                   for decode in line["decodes"]
                   if decode["backend"] == backend and decode["t_callback_ns"] is not None)
    sensor = sorted((line["marks"]["t_submit_ns"], line["marks"]["t_engine_ns"]) for line in decisions
                    if isinstance(line.get("marks"), dict))
    made = {"radio_intervals": len(radio), "sensor_intervals": len(sensor), "sensor_overlapping": 0,
            "radio_overlapping": 0, "common_interval_ns": None, "sensor_inside_run": 0}
    if not radio or not sensor:
        return made
    first, last = max(radio[0][0], sensor[0][0]), min(radio[-1][1], sensor[-1][1])
    made["common_interval_ns"] = [first, last] if last > first else None
    made["sensor_inside_run"] = sum(1 for begin, end in sensor if begin >= radio[0][0] and end <= radio[-1][1])
    hit, position = set(), 0
    for begin, end in sensor:
        while position < len(radio) and radio[position][1] < begin:
            position += 1
        index, touched = position, False
        while index < len(radio) and radio[index][0] <= end:
            if radio[index][1] >= begin:
                touched = True
                hit.add(index)
            index += 1
        made["sensor_overlapping"] += touched
    made["radio_overlapping"] = len(hit)
    return made


def arrivals(output):
    """What the receiver side of an arm offered and delivered: the counts that must survive a comparison."""
    made = {"opportunities": len(output.report["opportunities"]), "radio_records": len(output.records),
            "o2": None, "o3": None}
    if output.base.ring is not None:
        state = sr.account(output.base)
        made["o2"] = {key: state[key] for key in ("eligible", "published", "dropped", "released", "pending")}
    if output.native.ring is not None:
        state = ne.account(output.native)
        made["o3"] = {key: state.get(key) for key in ("eligible", "published", "dropped", "released", "pending")}
    return made


def window_latency(output, decisions):
    """The latency of every scored window of a sensor against the offered schedule of the receiver beside it.

    The paced receiver fixes the map between the radio clock and the host clock: the slot of an
    opportunity begins at its release time. A window is joined to the releases of the
    opportunities it names. Three durations per window, in microseconds: end_to_end, from the
    release of its first slot to its decision; from_last_release, from the release of its last
    slot; after_radio, from the moment the receiver was done with its last slot. A window that
    names a skipped release, or an opportunity the run does not hold, is counted as unjoined.
    """
    by_key = {o2_reference.opaque_key(line["opportunity_id"]): line for line in output.releases}
    made = {"end_to_end": [], "from_last_release": [], "after_radio": [], "unjoined": 0,
            "clock_map": "the slot of an opportunity begins at its offered release time on the host's monotonic clock"}
    for line in decisions:
        named = [by_key.get(item["opportunity"]) for item in line["contributing"]]
        if not named or any(found is None or found["outcome"] == "skipped" for found in named) \
                or line.get("t_decision_ns") is None:
            made["unjoined"] += 1
            continue
        decided = line["t_decision_ns"]
        made["end_to_end"].append((decided - named[0]["t_release_ns"]) / 1000.0)
        made["from_last_release"].append((decided - named[-1]["t_release_ns"]) / 1000.0)
        made["after_radio"].append((decided - named[-1]["t_done_ns"]) / 1000.0)
    return {key: statistics(value) if isinstance(value, list) else value for key, value in made.items()}


def decision_agreement(first, second, logit_tolerance):
    """The scored windows of two sensor outputs on the same offered trace, joined by episode and period.

    Returns the count of joined windows, how many decide alike, the largest logit difference and
    every flip with its margins. The receiver's own estimate is not bit-reproducible between two
    decodes of one grid, so the scores of two arms are compared by tolerance and never by digest.
    """
    def keyed(lines):
        return {(line["window"]["episode"], line["window"]["period"]): line for line in lines}

    mine, theirs = keyed(first), keyed(second)
    joined = sorted(set(mine) & set(theirs))
    made = {"first": len(mine), "second": len(theirs), "joined": len(joined), "agree": 0, "beyond_tolerance": 0,
            "largest_logit_difference": 0.0, "logit_tolerance": logit_tolerance, "flips": []}
    for key in joined:
        a, b = mine[key], theirs[key]
        difference = abs(a["score"]["logit"] - b["score"]["logit"])
        made["largest_logit_difference"] = max(made["largest_logit_difference"], difference)
        made["beyond_tolerance"] += difference > logit_tolerance
        if a["decision"]["present"] == b["decision"]["present"]:
            made["agree"] += 1
        else:
            made["flips"].append({"episode": key[0], "period": key[1], "logits": [a["score"]["logit"], b["score"]["logit"]],
                                  "threshold_logit": a["decision"]["threshold_logit"]})
    return made
