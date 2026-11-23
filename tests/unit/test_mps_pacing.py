"""Sprint 4.1 unit gate (selector mps_pacing): python tests/unit/test_mps_pacing.py

Two independent CUDA clients under MPS, paced. This gate checks, without an OCUDU build, without
a CUDA device and without an MPS daemon:

  * the offered schedule of a replay and the rules of a plan of version 4;
  * the release rule with a clock of the test: a schedule released every 5 ms and a worker that
    takes 12 ms. No release time moves and no missed job is deleted;
  * the host-free pacing core of native/ocudu_adapter/pacing/, built on its own: the same rule,
    release by release, and a paced book that allocates nothing per decode;
  * the MPS inventory: duplicate client IDs and an unverified server/device pair are refused, and
    an arm labeled MPS whose client list lacks a process ID is refused with the list in the message;
  * the launcher on a fake clock and a fake process inventory: the sensor is ready and a client
    before the receiver exists, an observer is stopped and restarted without a signal to the
    producer, and no arm reconfigures the receiver for the sensor;
  * the rule by which a concurrent arm may move the radio's tail, and the order of the controls;
  * the tree: the paced loop of the tool, the build and the strings both sides share.
"""

import copy
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ocudu_cuda_isac.episodes import mps_pacing as mp  # noqa: E402
from ocudu_cuda_isac.episodes import native_estimate as ne  # noqa: E402
from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
from ocudu_cuda_isac.episodes.mps_pacing import MpsPacingError  # noqa: E402
import run_m1  # noqa: E402

ADAPTER = ROOT / "native/ocudu_adapter"
PACING = ADAPTER / "pacing"
CASES = ("schedule_rules", "releases_never_shift", "late_start", "book_decoded", "book_outcomes", "book_refusals",
         "book_no_allocation", "spin_until")
MS, US = 1_000_000, 1_000
PIPE = "/run/unit/mps/pipe"


def plan_of(pacing=None, mode="timed", tap="pilot_ls"):
    """A plan of version 4 that binds no file: enough for every rule that needs none."""
    digest = "a" * 64
    bound = {"path": "/nowhere/file", "sha256": digest}
    tapped = tap != "none"
    ring = sr.ring_spec(3, 3, "spin", stream="o2") if tapped else None
    estimate = ne.estimate_spec(ring=sr.ring_spec(3, 3, "spin", stream="o3")) if tapped else None
    return {"schema": mp.PLAN_SCHEMA, "run_id": "unit", "access": "offline_stimulus_and_evaluator", "replay_plan": bound,
            "observer": {"tap": tap, "stage": "O2" if tapped else None, "source": "pusch_dmrs" if tapped else None,
                         "layout": "ocudu_o2_comb_c64" if tapped else None, "epoch": 1},
            "sources": [{"grant_id": "g", "record_sha256": digest, "source": bound, "pilot_index": bound,
                         "pilot_ref": bound, "antenna_map": [0, 1]}] if tapped else [],
            "evidence": {"mode": mode}, "ring": ring, "estimate": estimate, "pacing": pacing}


def burst(realization, periods, sfn=0):
    """The uplink slots 7, 8 and 9 of some TDD periods of ten slots, as opportunities of one realization."""
    return [{"realization_id": realization, "sfn": sfn + (10 * period + slot) // 20, "slot": (10 * period + slot) % 20}
            for period in range(periods) for slot in (7, 8, 9)]


class ScheduleTests(unittest.TestCase):
    def test_a_release_follows_the_one_before_it_by_the_distance_of_their_slots(self):
        # Three uplink slots per period of 5 ms, 0.5 ms apart, and the next period 3.5 ms later.
        offsets = mp.release_offsets(burst("a", 3))
        self.assertEqual(offsets, [0, 500, 1000, 5000, 5500, 6000, 10000, 10500, 11000])
        # Another realization starts one period after the last slot of the one before, whatever its frame.
        both = mp.release_offsets(burst("a", 2) + burst("b", 2))
        self.assertEqual(both, [0, 500, 1000, 5000, 5500, 6000, 11000, 11500, 12000, 16000, 16500, 17000])
        # A slot that does not rise inside one realization is a gap as well, and the slot length is the caller's.
        again = mp.release_offsets(burst("a", 1) + burst("a", 1), slot_us=250, gap_slots=4)
        self.assertEqual(again, [0, 250, 500, 1500, 1750, 2000])
        self.assertEqual(mp.release_offsets([]), [])
        with self.assertRaisesRegex(MpsPacingError, "positive integer"):
            mp.release_offsets(burst("a", 1), slot_us=0)
        with self.assertRaisesRegex(MpsPacingError, "names no realization"):
            mp.release_offsets([{"sfn": 0, "slot": 7}])

    def test_an_offered_schedule_names_rising_offsets_an_allowance_and_a_latest_start(self):
        pacing = mp.pacing_spec([0, 500, 1000], 500, 250)
        self.assertEqual(pacing, {"clock": "host_monotonic", "start_delay_us": 20000, "allowance_us": 500,
                                  "latest_start_us": 250, "release_us": [0, 500, 1000]})
        mp.check_pacing(pacing, 3)
        for change, message in (
                (lambda p: p.update(clock="tai"), "pacing.clock"),
                (lambda p: p.update(allowance_us=0), "allowance of a paced run is positive"),
                (lambda p: p.update(latest_start_us=501), "latest start"),
                (lambda p: p.update(latest_start_us=-1), "latest_start_us must be an integer"),
                (lambda p: p.update(start_delay_us=1.5), "start_delay_us must be an integer"),
                (lambda p: p.update(release_us=[]), "at least one release"),
                (lambda p: p.update(release_us=[0, 500, 500]), "release 2"),
                (lambda p: p.update(release_us=[0, 1.5, 3]), "release 1"),
                (lambda p: p.update(release_us=[-1, 0, 1]), "release 0"),
                (lambda p: p.pop("clock"), "requires exactly"),
                (lambda p: p.update(wait="spin"), "requires exactly")):
            broken = copy.deepcopy(pacing)
            change(broken)
            with self.assertRaisesRegex(MpsPacingError, message):
                mp.check_pacing(broken)
        with self.assertRaisesRegex(MpsPacingError, "names 3 releases, and the replay holds 4"):
            mp.check_pacing(pacing, 4)

    def test_a_plan_of_version_4_names_a_schedule_or_null_and_a_paced_run_is_timed(self):
        pacing = mp.pacing_spec([0, 500, 1000], 500, 250)
        for plan in (plan_of(), plan_of(pacing), plan_of(pacing, tap="none")):
            self.assertIs(mp.check_plan(plan), plan)
        # With pacing null it is the plan of Sprint 5.2 under another schema name.
        third = {key: value for key, value in plan_of().items() if key != "pacing"}
        third["schema"] = ne.PLAN_SCHEMA
        ne.check_plan(third)
        with self.assertRaisesRegex(MpsPacingError, "a paced run is a timed run"):
            mp.check_plan(plan_of(pacing, tap="none", mode="audit"))
        with self.assertRaisesRegex(MpsPacingError, "unsupported plan schema"):
            mp.check_plan(dict(plan_of(pacing), schema=ne.PLAN_SCHEMA))
        with self.assertRaisesRegex(MpsPacingError, "requires exactly"):
            mp.check_plan(third)
        # The rules of the earlier versions still hold: a ring without a tap publishes nothing.
        broken = plan_of(pacing, tap="none")
        broken["ring"] = sr.ring_spec(3, 3, "spin")
        with self.assertRaisesRegex(MpsPacingError, "publishes nothing"):
            mp.check_plan(broken)
        with self.assertRaisesRegex(MpsPacingError, "release 1"):
            mp.check_plan(plan_of(dict(pacing, release_us=[5, 5])))


class PacerTests(unittest.TestCase):
    def test_a_worker_of_12_ms_cannot_shift_releases_of_5_ms_or_delete_missed_jobs(self):
        pacer = mp.Pacer([n * 5 * MS for n in range(7)], 6 * MS, 1 * MS)
        start = 1000 * MS
        mp.drive(pacer, start, 12 * MS)
        # Seven marks, each on its own offset: nothing was shifted and nothing was deleted.
        self.assertEqual([mark["release"] for mark in pacer.marks], [start + n * 5 * MS for n in range(7)])
        self.assertEqual([mark["outcome"] for mark in pacer.marks],
                         ["on_time", "skipped", "skipped", "on_time", "skipped", "skipped", "on_time"])
        # The job after the worked-through releases starts on its own release time, 15 ms.
        self.assertEqual([mark["start"] for mark in pacer.marks if mark["start"] is not None],
                         [start, start + 15 * MS, start + 30 * MS])
        self.assertEqual(pacer.marks[1]["free"], start + 12 * MS)
        self.assertEqual(pacer.counts(), {"offered": 7, "taken": 7, "on_time": 3, "late": 0, "skipped": 4, "overruns": 3,
                                          "max_lateness_ns": 0, "max_excess_ns": 6 * MS})
        fast = mp.Pacer([n * 5 * MS for n in range(7)], 6 * MS, 1 * MS)
        mp.drive(fast, start, 3 * MS)
        self.assertEqual((fast.counts()["on_time"], fast.counts()["skipped"], fast.counts()["overruns"]), (7, 0, 0))

    def test_an_input_that_is_ready_after_its_release_starts_late_and_the_boundary_is_inclusive(self):
        pacer = mp.Pacer([n * 500 * US for n in range(4)], 600 * US, 100 * US)
        mp.drive(pacer, 0, 400 * US, 150 * US)
        self.assertEqual([mark["outcome"] for mark in pacer.marks], ["late", "late", "late", "skipped"])
        self.assertEqual([mark["start"] for mark in pacer.marks[:3]], [150 * US, 700 * US, 1250 * US])
        self.assertEqual([mark["overrun"] for mark in pacer.marks], [False, False, True, False])
        self.assertEqual((pacer.counts()["max_lateness_ns"], pacer.counts()["max_excess_ns"]), (250 * US, 50 * US))
        edge = mp.Pacer([0, 500 * US], 600 * US, 100 * US)
        edge.begin(0)
        self.assertTrue(edge.take(0, 100 * US))
        edge.ready(0, 100 * US)
        edge.start(0, 100 * US)
        edge.done(0, 600 * US)
        self.assertFalse(edge.marks[0]["overrun"])
        past = mp.Pacer([0, 500 * US], 600 * US, 100 * US)
        past.begin(0)
        self.assertFalse(past.take(0, 100 * US + 1))
        self.assertTrue(past.take(1, 100 * US + 1))

    def test_the_schedule_is_fixed_once_and_jobs_are_taken_in_order(self):
        pacer = mp.Pacer([0, 5 * MS, 10 * MS], 2 * MS, 1 * MS)
        with self.assertRaisesRegex(MpsPacingError, "has not begun"):
            pacer.take(0, 0)
        pacer.begin(100)
        for call, message in ((lambda: pacer.begin(200), "fixed once"), (lambda: pacer.take(1, 100), "out of order"),
                              (lambda: pacer.start(0, 100), "not prepared"), (lambda: pacer.take(3, 0), "no release 3")):
            with self.assertRaisesRegex(MpsPacingError, message):
                call()
        self.assertTrue(pacer.take(0, 100))
        pacer.give_up(0, 150)
        self.assertEqual((pacer.marks[0]["outcome"], pacer.marks[0]["free"]), ("skipped", 150))
        for call in (lambda: pacer.give_up(0, 160), lambda: pacer.ready(0, 160)):
            with self.assertRaisesRegex(MpsPacingError, "not taken"):
                call()
        for schedule in (([], 1, 0), ([0, 0], 1, 0), ([0, 1], 0, 0), ([0, 1], 5, 6), ([-1, 1], 5, 1)):
            with self.assertRaisesRegex(MpsPacingError, "rising offsets"):
                mp.Pacer(*schedule)


class RuleTests(unittest.TestCase):
    RULE = {"schema": mp.RULE_SCHEMA, "status": "proposed", "quantile": "p99", "min_samples": 100,
            "allowed_shift_us": 25.0, "allowed_shift_fraction": 0.10}

    def test_nearest_rank_and_statistics(self):
        values = [100, 200, 300, 400, 500]
        self.assertEqual((mp.nearest_rank(values, 0.5), mp.nearest_rank(values, 0.95), mp.nearest_rank([], 0.5)),
                         (300, 500, None))
        self.assertEqual(mp.nearest_rank(list(range(1, 1001)), 0.99), 990)
        made = mp.statistics(values)
        self.assertEqual((made["count"], made["mean_us"], made["p50_us"], made["p99_us"], made["max_us"]),
                         (5, 300.0, 300.0, 500.0, 500.0))
        self.assertEqual(mp.statistics([])["p99_us"], None)

    def test_a_tail_shift_is_judged_by_a_rule_declared_before_and_needs_its_samples(self):
        base = [200.0] * 990 + [300.0] * 10
        same = mp.tail_shift(base, base, self.RULE)
        self.assertEqual((same["shift_us"], same["allowed_us"], same["exceeded"]), (0.0, 25.0, False))
        # The allowed shift is the larger of the absolute term and the fraction of the base tail.
        moved = mp.tail_shift(base, [200.0] * 980 + [224.0] * 20, self.RULE)
        self.assertEqual((moved["shift_us"], moved["exceeded"]), (24.0, False))
        beyond = mp.tail_shift(base, [200.0] * 980 + [226.0] * 20, self.RULE)
        self.assertEqual((beyond["shift_us"], beyond["exceeded"], beyond["rule_sha256"]),
                         (26.0, True, mp.rule_sha256(self.RULE)))
        wide = mp.tail_shift([1000.0] * 200, [1090.0] * 200, self.RULE)
        self.assertEqual((wide["allowed_us"], wide["exceeded"]), (100.0, False))
        few = mp.tail_shift(base[:99], base, self.RULE)
        self.assertEqual((few["exceeded"], few["shift_us"]), (None, None))
        self.assertIn("99 samples, and the rule needs 100", few["reason"])
        for change, message in ((lambda r: r.update(quantile="p90"), "names a quantile"),
                                (lambda r: r.update(min_samples=99), "do not resolve the quantile p99"),
                                (lambda r: r.update(allowed_shift_us=-1), "not negative"),
                                (lambda r: r.update(status="measured"), "status of a rule"),
                                (lambda r: r.pop("status"), "requires exactly")):
            broken = dict(self.RULE)
            change(broken)
            with self.assertRaisesRegex(MpsPacingError, message):
                mp.check_rule(broken)

    def test_the_controls_are_applied_in_order_and_the_first_that_restores_the_tail_stands(self):
        good = {"exceeded": False, "quantile": "p99", "allowed_us": 25.0, "shift_us": 3.0, "reason": None}
        bad = dict(good, exceeded=True, shift_us=80.0)
        self.assertEqual(mp.CONTROLS, ("mps_default", "mps_priority", "mps_thread_pct", "release_after_radio"))
        self.assertEqual(mp.select_control({"mps_default": good})[0], "mps_default")
        self.assertEqual(mp.select_control({"mps_default": bad, "mps_priority": good})[0], "mps_priority")
        # A control this host lacks is a planned cell: it is passed over, and the next one is tried.
        self.assertEqual(mp.select_control({"mps_default": bad, "mps_priority": None, "mps_thread_pct": bad,
                                            "release_after_radio": good})[0], "release_after_radio")
        self.assertEqual(mp.select_control({"mps_default": bad, "mps_priority": bad})[0], None)
        none, why = mp.select_control({"mps_default": dict(good, exceeded=None, reason="12 samples")})
        self.assertEqual((none, why), (None, "mps_default gives no verdict: 12 samples"))
        # Only the sensor is changed by a control, and never the queue.
        self.assertEqual(run_m1.control_settings("mps_default"), ({}, "immediate", 0))
        self.assertEqual(run_m1.control_settings("mps_priority"), ({"CUDA_MPS_CLIENT_PRIORITY": "1"}, "immediate", 0))
        self.assertEqual(run_m1.control_settings("mps_thread_pct", thread_pct=40),
                         ({"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "40"}, "immediate", 0))
        self.assertEqual(run_m1.control_settings("release_after_radio", radio_allowance_us=120), ({}, "after_radio", 120))
        with self.assertRaisesRegex(run_m1.LaunchError, "unknown contention control"):
            run_m1.control_settings("deeper_ring")


def inventory(clients, servers=(900,), devices=1, alive=True):
    return {"schema": mp.INVENTORY_SCHEMA, "t_monotonic_ns": 1, "daemon": {"alive": alive}, "servers": list(servers),
            "clients": {str(server): list(clients) for server in servers},
            "devices": [{"index": n, "uuid": f"GPU-{n:04d}", "name": "NVIDIA GB10"} for n in range(devices)]}


class InventoryTests(unittest.TestCase):
    def test_both_clients_of_the_one_server_on_the_one_device_prove_the_label(self):
        proof = mp.verify_clients(inventory([4711, 4712]), {"producer": 4712, "sensor": 4711})
        self.assertEqual(proof, {"server": 900, "device": {"index": 0, "uuid": "GPU-0000", "name": "NVIDIA GB10"},
                                 "clients": [4711, 4712], "roles": {"producer": 4712, "sensor": 4711}})
        self.assertEqual(mp.parse_pids("4711\n 4712 \nServer 900 not found\n\n"), [4711, 4712])

    def test_an_arm_whose_client_list_lacks_a_process_is_refused_with_the_list(self):
        with self.assertRaises(MpsPacingError) as refused:
            mp.verify_clients(inventory([4712, 5000]), {"producer": 4712, "sensor": 4711}, extra=1)
        text = str(refused.exception)
        self.assertIn("the arm is not labeled MPS", text)
        self.assertIn("the sensor process 4711 is not a client of the MPS server 900: it shares the device by time slicing",
                      text)
        self.assertIn("Client list of the server 900: 4712, 5000", text)
        with self.assertRaisesRegex(MpsPacingError, "Client list of the server 900: none"):
            mp.verify_clients(inventory([]), {"producer": 4712})

    def test_duplicate_client_ids_and_an_unverified_server_device_pair_are_refused(self):
        with self.assertRaisesRegex(MpsPacingError, "duplicate client IDs: two roles of the arm name one process"):
            mp.verify_clients(inventory([4711]), {"producer": 4711, "sensor": 4711})
        with self.assertRaisesRegex(MpsPacingError, "duplicate client IDs: the client list names a process twice"):
            mp.verify_clients(inventory([4711, 4711, 4712]), {"producer": 4712, "sensor": 4711})
        with self.assertRaisesRegex(MpsPacingError, "an unverified server/device pair: the daemon sees 2 devices"):
            mp.verify_clients(inventory([4711, 4712], devices=2), {"producer": 4712, "sensor": 4711})
        with self.assertRaisesRegex(MpsPacingError, "an unverified server/device pair: the daemon sees 0 devices"):
            mp.verify_clients(inventory([4711, 4712], devices=0), {"producer": 4712, "sensor": 4711})
        with self.assertRaisesRegex(MpsPacingError, "2 MPS servers are listed"):
            mp.verify_clients(inventory([4711, 4712], servers=(900, 901)), {"producer": 4712, "sensor": 4711})
        with self.assertRaisesRegex(MpsPacingError, "control daemon does not answer"):
            mp.verify_clients(inventory([4711, 4712], alive=False), {"producer": 4712, "sensor": 4711})
        # A third client is not the arm that was specified: the timed arms hold two CUDA clients.
        with self.assertRaisesRegex(MpsPacingError, "1 further client\\(s\\) of the server, 7000, and the arm allows 0"):
            mp.verify_clients(inventory([4711, 4712, 7000]), {"producer": 4712, "sensor": 4711})
        self.assertEqual(mp.inventory_problems({"schema": "other"}), ["no mps-inventory/v1 record"])


# ---------------------------------------------------------------- a fake clock and a fake process inventory
class FakeProcess:
    def __init__(self, host, pid, command, env, cpus, behaviour):
        self.host, self.pid, self.command, self.env, self.cpus = host, pid, command, env, cpus
        self.started = host.now
        self.end = None if behaviour.get("life_ms") is None else host.now + behaviour["life_ms"] * MS
        self.status, self.killed = behaviour.get("status", 0), False
        self.ready_at = None if behaviour.get("ready_ms") is None else host.now + behaviour["ready_ms"] * MS
        self.client_at = None if behaviour.get("client_ms") is None else host.now + behaviour["client_ms"] * MS
        self.ready_file = command[command.index("--ready-file") + 1] if "--ready-file" in command else None
        # A paced receiver announces the start of its schedule when its preparation is done.
        self.schedule_at = None if behaviour.get("schedule_ms") is None else host.now + behaviour["schedule_ms"] * MS

    def poll(self):
        return self.status if self.end is not None and self.host.now >= self.end else None

    def kill(self):
        if self.poll() is None:
            self.end, self.status, self.killed = self.host.now, -9, True

    def wait(self, timeout=None):
        return self.poll()

    def is_client(self):
        """A client of the daemon when it was started with the daemon's pipe directory and has a context."""
        return self.poll() is None and self.client_at is not None and self.host.now >= self.client_at \
            and self.env.get("CUDA_MPS_PIPE_DIRECTORY") == PIPE


class FakeHost(run_m1.Host):
    """A clock that moves only when the launcher sleeps, and processes that live as a script says."""

    def __init__(self, behaviours, devices=1, daemon=True):
        self.now, self.behaviours, self.devices, self.daemon = 5_000 * MS, behaviours, devices, daemon
        self.processes, self.asked, self.next_pid = [], [], 4711

    def now_ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += int(seconds * 1e9)

    def cpus(self):
        return list(range(20))

    def affinity(self, process):
        return list(process.cpus)

    def schedule_start(self, log):
        for process in self.processes:
            if Path(process.command[0]).name == "pilot_ls" and process.schedule_at is not None \
                    and self.now >= process.schedule_at - 50 * MS:
                return process.schedule_at
        return None

    def spawn(self, command, log, env, cpus):
        process = FakeProcess(self, self.next_pid, list(command), dict(env), list(cpus),
                              self.behaviours[Path(command[0]).name])
        self.next_pid += 1
        self.processes.append(process)
        return process

    def exists(self, path):
        return any(process.ready_file == str(path) and process.ready_at is not None and self.now >= process.ready_at
                   for process in self.processes)

    def maps(self, pid):
        process = [item for item in self.processes if item.pid == pid][0]
        return ["/usr/lib/libpython3.12.so.1.0"] if Path(process.command[0]).name == "python" else ["/usr/lib/libc.so.6"]

    def state(self):
        return {"host": "unit", "monotonic_ns": self.now}

    def run(self, command, text="", env=None, timeout=30):
        name = Path(command[0]).name
        if name == "nvidia-smi":
            if any("--query-gpu=index,uuid,name" in word for word in command):
                return 0, "".join(f"{n}, GPU-{n:04d}, NVIDIA GB10\n" for n in range(self.devices))
            return 0, ""
        if name != run_m1.MPS_CONTROL:
            return 127, "no such tool"
        question = text.strip()
        self.asked.append((self.now, question))
        if not self.daemon or (env or {}).get("CUDA_MPS_PIPE_DIRECTORY") != PIPE:
            return 1, "Cannot find MPS control daemon process"
        clients = [process.pid for process in self.processes if process.is_client()]
        if question == "get_server_list":
            return 0, "900\n" if clients else ""
        if question == "get_client_list 900":
            return 0, "".join(f"{pid}\n" for pid in clients)
        if question in ("get_default_active_thread_percentage", "get_active_thread_percentage 900"):
            return 0, "100.0\n"
        if question == "get_default_client_priority":
            return 0, "0\n"
        return 1, "Unrecognized command"


def concurrent_spec(fault=None, sensor_env=None, producer_env=None, sharing="mps", extra=()):
    producer = run_m1.process_spec("producer", "pilot_ls", ["/build/pilot_ls", "--plan", "plan.json"], True, "5-9,15-19",
                                   producer_env)
    sensor = run_m1.process_spec("sensor", "sense_host", ["/build/sense_host", "--ready-file", "/run/sensor.ready"], True,
                                 "4", sensor_env, "/run/sensor.ready")
    tail = run_m1.process_spec("o2", "obs_tail", ["/build/obs_tail", "--socket", "o2.sock"], False, "3")
    return run_m1.arm_spec("concurrent", "unit", producer, [sensor, tail] + list(extra), sharing, fault,
                           {"ready_s": 5, "run_s": 60, "grace_s": 5}, 100)


BEHAVIOUR = {"pilot_ls": {"life_ms": 3000, "client_ms": 250, "schedule_ms": 400}, "sense_host": {"life_ms": 3400, "ready_ms": 700,
                                                                            "client_ms": 300},
             "obs_tail": {"life_ms": 3350}, "sense_feed": {"life_ms": 2000}, "python": {"life_ms": 3400, "ready_ms": 900,
                                                                                         "client_ms": 500}}


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.directory = Path(self.scratch.name)

    def launch(self, spec, behaviours=None, **host):
        self.host = FakeHost(behaviours or BEHAVIOUR, **host)
        self.mps = run_m1.Mps(self.host, PIPE, "/run/unit/mps/log", base_env={})
        return run_m1.launch(spec, self.directory, self.host, self.mps if spec["sharing"] == "mps" else None,
                             base_env={"PATH": "/usr/bin", "CUDA_MPS_PIPE_DIRECTORY": "/somewhere/else"})

    def test_the_sensor_is_ready_and_a_client_before_the_receiver_exists(self):
        record = self.launch(concurrent_spec())
        self.assertEqual(record["problems"], [])
        self.assertTrue(run_m1.clean(record))
        order = [(item["event"], item["role"]) for item in record["order"]]
        self.assertEqual(order[:4], [("started", "sensor"), ("ready", "sensor"), ("started", "o2"), ("started", "producer")])
        by_role = {entry["role"]: entry for entry in record["processes"]}
        self.assertLessEqual(by_role["sensor"]["ready_ns"], by_role["producer"]["started_ns"])
        sharing = record["sharing"]
        self.assertEqual((sharing["label"], sharing["verified"], sharing["context_before_receiver"]), ("mps", True, True))
        self.assertEqual(sharing["proof"]["roles"], {"producer": by_role["producer"]["pid"], "sensor": by_role["sensor"]["pid"]})
        self.assertEqual((sharing["proof"]["server"], sharing["proof"]["device"]["uuid"]), (900, "GPU-0000"))
        # Every process got the daemon of the arm, whatever the launcher's own environment named.
        for process in self.host.processes:
            self.assertEqual(process.env["CUDA_MPS_PIPE_DIRECTORY"], PIPE)
        self.assertEqual([process.cpus for process in self.host.processes], [[4], [3], [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]])
        self.assertEqual((by_role["sensor"]["maps_python"], by_role["sensor"]["status"], by_role["producer"]["status"]),
                         (False, 0, 0))
        # The readings while the producer runs are light, and they stop with the one that proves the label.
        began = by_role["producer"]["started_ns"]
        readings = [found.get("light", False) for found in record["inventories"]]
        self.assertEqual((readings[0], set(readings[1:]), len(readings) >= 3), (False, {True}, True))
        questions = [question for _, question in self.host.asked]
        self.assertEqual((questions.count("get_default_client_priority"), questions.count("get_active_thread_percentage 900")),
                         (1, 1))
        proven = sharing["proof"]["t_monotonic_ns"]
        self.assertFalse([at for at, _ in self.host.asked if at > proven])
        self.assertGreaterEqual(proven, began + 250 * MS)
        self.assertLess(proven, began + 500 * MS)

    def test_an_arm_labeled_mps_whose_sensor_is_no_client_is_refused_with_the_client_list(self):
        behaviours = dict(BEHAVIOUR, sense_host={"life_ms": 3400, "ready_ms": 700, "client_ms": None})
        record = self.launch(concurrent_spec(), behaviours)
        sharing = record["sharing"]
        self.assertEqual((sharing["label"], sharing["verified"], sharing["context_before_receiver"]),
                         ("refused", False, False))
        pids = {entry["role"]: entry["pid"] for entry in record["processes"]}
        self.assertIn(f"the sensor process {pids['sensor']} is not a client of the MPS server 900", sharing["reason"])
        self.assertIn(f"Client list of the server 900: {pids['producer']}", sharing["reason"])
        self.assertFalse(run_m1.clean(record))
        # The same for a host with two devices: the pair of the server and its device is not verified.
        record = self.launch(concurrent_spec(), devices=2)
        self.assertEqual(record["sharing"]["label"], "refused")
        self.assertIn("an unverified server/device pair", record["sharing"]["reason"])

    def test_an_observer_is_stopped_and_restarted_without_stopping_the_producer(self):
        again = run_m1.process_spec("sensor-2", "sense_host", ["/build/sense_host", "--ready-file", "/run/again.ready"], True,
                                    "4", None, "/run/again.ready")
        fault = {"kind": "consumer_restart", "role": "sensor", "after_ms": 1000, "restart": again}
        record = self.launch(concurrent_spec(fault))
        by_role = {entry["role"]: entry for entry in record["processes"]}
        note = record["fault"]
        self.assertEqual((note["kind"], note["role"], note["producer_alive_after"]), ("consumer_restart", "sensor", True))
        self.assertEqual((by_role["sensor"]["ended_by"], by_role["sensor"]["status"]), ("fault", -9))
        # The fault is timed from the start of the schedule that the receiver announced, not from its start.
        self.assertEqual((note["clock"], note["clock_start_ns"]), ("schedule_start", by_role["producer"]["started_ns"] + 400 * MS))
        self.assertGreaterEqual(note["applied_ns"], note["clock_start_ns"] + 1000 * MS)
        # The producer was never signalled: it ended by itself, with its own status, after its whole life.
        producer = [process for process in self.host.processes if Path(process.command[0]).name == "pilot_ls"][0]
        self.assertFalse(producer.killed)
        self.assertEqual((by_role["producer"]["ended_by"], by_role["producer"]["status"]), ("itself", 0))
        self.assertGreaterEqual(by_role["producer"]["ended_ns"] - by_role["producer"]["started_ns"], 3000 * MS)
        self.assertEqual(note["restarted"]["pid"], by_role["sensor-2"]["pid"])
        self.assertLess(by_role["sensor-2"]["started_ns"], by_role["producer"]["ended_ns"])
        self.assertTrue(run_m1.clean(record, allow=("sensor",)))
        self.assertEqual(record["sharing"]["label"], "mps")
        # A consumer that only exits: the receiver goes on, and nothing takes over.
        record = self.launch(concurrent_spec({"kind": "consumer_exit", "role": "sensor", "after_ms": 1500, "restart": None}))
        self.assertEqual((record["fault"]["restarted"], record["fault"]["producer_alive_after"]), (None, True))
        self.assertEqual([entry["status"] for entry in record["processes"] if entry["role"] == "producer"], [0])
        # A receiver that prepares for eight seconds: the sensor is not ended while it prepares, so it is attached
        # and both are clients in one reading before the fault, which comes 700 ms into the schedule.
        slow = dict(BEHAVIOUR, pilot_ls={"life_ms": 10000, "client_ms": 1200, "schedule_ms": 8000},
                    sense_host={"life_ms": 11000, "ready_ms": 700, "client_ms": 300}, obs_tail={"life_ms": 10500})
        record = self.launch(concurrent_spec({"kind": "consumer_exit", "role": "sensor", "after_ms": 700, "restart": None}),
                             slow)
        by_role = {entry["role"]: entry for entry in record["processes"]}
        self.assertGreaterEqual(record["fault"]["applied_ns"], by_role["producer"]["started_ns"] + 8700 * MS)
        self.assertEqual((record["sharing"]["label"], sorted(record["sharing"]["proof"]["roles"])),
                         ("mps", ["producer", "sensor"]))
        self.assertLess(record["sharing"]["proof"]["t_monotonic_ns"], record["fault"]["applied_ns"])
        self.assertTrue(run_m1.clean(record, allow=("sensor",)))
        # A producer that announces no schedule: the fault is not applied, and the record says why.
        silent = dict(BEHAVIOUR, pilot_ls={"life_ms": 3000, "client_ms": 250})
        record = self.launch(concurrent_spec({"kind": "consumer_exit", "role": "sensor", "after_ms": 700, "restart": None}),
                             silent)
        self.assertIsNone(record["fault"])
        self.assertIn("it announced no start of its schedule", record["problems"][0])
        # A fault that would come after the end of the producer is a problem, not a silent pass.
        record = self.launch(concurrent_spec({"kind": "consumer_exit", "role": "sensor", "after_ms": 9000, "restart": None}))
        self.assertIn("was never applied", record["problems"][0])

    def test_a_sensor_that_is_not_ready_keeps_the_receiver_from_starting(self):
        behaviours = dict(BEHAVIOUR, sense_host={"life_ms": 300, "status": 3, "ready_ms": None})
        record = self.launch(concurrent_spec(), behaviours)
        self.assertIn("did not become ready", record["problems"][0])
        self.assertEqual([entry["role"] for entry in record["processes"]], ["sensor"])
        self.assertEqual(record["processes"][0]["status"], 3)
        self.assertFalse([process for process in self.host.processes if Path(process.command[0]).name == "pilot_ls"])
        # A producer that never ends is stopped at the end of its time, and the arm says so.
        behaviours = dict(BEHAVIOUR, pilot_ls={"life_ms": None, "client_ms": 250})
        record = self.launch(concurrent_spec(), behaviours)
        self.assertIn("did not end within 60 s and was stopped", record["problems"][0])
        self.assertEqual([entry["ended_by"] for entry in record["processes"] if entry["role"] == "producer"], ["timeout"])

    def test_time_slicing_is_its_own_labeled_control_and_never_carries_the_label_mps(self):
        record = self.launch(concurrent_spec(sharing="time_slicing"))
        self.assertEqual((record["sharing"]["label"], record["sharing"]["verified"]), ("time_slicing", False))
        for process in self.host.processes:
            self.assertNotIn("CUDA_MPS_PIPE_DIRECTORY", process.env)
        self.assertEqual((record["inventories"], self.host.asked), ([], []))
        with self.assertRaisesRegex(run_m1.LaunchError, "names the daemon it runs on"):
            run_m1.launch(concurrent_spec(), self.directory, self.host, None)

    def test_a_control_reaches_the_sensor_client_only_and_no_arm_reconfigures_the_receiver(self):
        record = self.launch(concurrent_spec(sensor_env={"CUDA_MPS_CLIENT_PRIORITY": "1"}))
        by_name = {Path(process.command[0]).name: process for process in self.host.processes}
        self.assertEqual(by_name["sense_host"].env["CUDA_MPS_CLIENT_PRIORITY"], "1")
        self.assertNotIn("CUDA_MPS_CLIENT_PRIORITY", by_name["pilot_ls"].env)
        self.assertEqual([entry["env"] for entry in record["processes"]], [{"CUDA_MPS_CLIENT_PRIORITY": "1"}, {}, {}])
        for name in run_m1.CLIENT_ENV:
            with self.assertRaisesRegex(run_m1.LaunchError, "no arm reconfigures the receiver for the sensor"):
                concurrent_spec(producer_env={name: "1"})
        with self.assertRaisesRegex(run_m1.LaunchError, "names no MPS directory of its own"):
            concurrent_spec(sensor_env={"CUDA_MPS_PIPE_DIRECTORY": "/tmp/other"})

    def test_an_arm_is_what_its_name_says(self):
        producer = run_m1.process_spec("producer", "pilot_ls", ["pilot_ls"], True)
        feed = run_m1.process_spec("producer", "sense_feed", ["sense_feed"], False)
        sensor = run_m1.process_spec("sensor", "sense_host", ["sense_host"], True, ready_file="/run/ready")
        tail = run_m1.process_spec("o3", "obs_tail", ["obs_tail"], False)
        for arm, made, consumers in (("dsp_only", producer, []), ("capture_only", producer, [tail]),
                                     ("ai_only", feed, [sensor]), ("concurrent", producer, [sensor, tail]),
                                     ("concurrent_reference", producer, [sensor])):
            run_m1.arm_spec(arm, arm, made, consumers)
        for arm, made, consumers, message in (
                ("dsp_only", producer, [tail], "runs the receiver alone"),
                ("capture_only", producer, [sensor], "runs no sensor"),
                ("capture_only", producer, [], "runs no sensor"),
                ("ai_only", producer, [sensor], "runs no receiver"),
                ("ai_only", feed, [sensor, tail], "has 1 consumer"),
                ("concurrent", feed, [sensor], "runs the receiver"),
                ("concurrent", producer, [tail], "runs a sensor"),
                ("concurrent", producer, [sensor, dict(sensor)], "have one role"),
                ("concurrent", producer, [dict(sensor, ready_file=None)], "names no ready file"),
                ("concurrent", dict(producer, role="receiver"), [sensor], "in the role producer"),
                ("concurrent", producer, [dict(sensor, kind="pilot_ls")], "a sensor or obs_tail"),
                ("concurrent", producer, [dict(sensor, cpus="9-5")], "no list of CPUs"),
                ("m1", producer, [sensor], "unknown arm")):
            with self.assertRaisesRegex(run_m1.LaunchError, message):
                run_m1.arm_spec(arm, arm, made, consumers)
        # A fault ends a consumer and never the producer.
        for fault, message in (({"kind": "consumer_exit", "role": "producer", "after_ms": 5, "restart": None}, "never ends the producer"),
                               ({"kind": "consumer_exit", "role": "sensor", "after_ms": 5, "restart": sensor}, "names none"),
                               ({"kind": "consumer_restart", "role": "sensor", "after_ms": 5, "restart": None}, "takes over"),
                               ({"kind": "consumer_restart", "role": "sensor", "after_ms": 5, "restart": sensor}, "role of its own"),
                               ({"kind": "producer_exit", "role": "sensor", "after_ms": 5, "restart": None}, "names its kind")):
            with self.assertRaisesRegex(run_m1.LaunchError, message):
                run_m1.arm_spec("concurrent", "unit", producer, [sensor], fault=fault)
        self.assertEqual(run_m1.parse_cpus("5-9,15-19, 3"), [3, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19])
        self.assertEqual((run_m1.parse_cpus(None), run_m1.parse_cpus("")), ([], []))

    def test_the_daemon_is_started_once_and_read_as_one_inventory(self):
        host = FakeHost(BEHAVIOUR)
        mps = run_m1.Mps(host, PIPE, "/run/unit/mps/log", base_env={})
        self.assertEqual(mps.start(), "running")
        found = mps.inventory()
        self.assertEqual((found["schema"], found["daemon"]["alive"], found["servers"], found["devices"]),
                         (mp.INVENTORY_SCHEMA, True, [], [{"index": 0, "uuid": "GPU-0000", "name": "NVIDIA GB10"}]))
        self.assertEqual(mps.env(), {"CUDA_MPS_PIPE_DIRECTORY": PIPE, "CUDA_MPS_LOG_DIRECTORY": "/run/unit/mps/log"})
        # No server exists before the first client: the pair is not proven by an empty daemon.
        self.assertIn("0 MPS servers are listed, and an arm runs on exactly one", mp.inventory_problems(found))
        pinned = run_m1.Mps(FakeHost(BEHAVIOUR, devices=2), PIPE, "/run/unit/mps/log", visible="GPU-0001", base_env={})
        self.assertEqual([device["index"] for device in pinned.devices()], [1])
        self.assertEqual(pinned.env()["CUDA_VISIBLE_DEVICES"], "GPU-0001")
        gone = run_m1.Mps(FakeHost(BEHAVIOUR, daemon=False), PIPE, "/run/unit/mps/log", base_env={})
        self.assertFalse(gone.inventory()["daemon"]["alive"])


class ReleaseLineTests(unittest.TestCase):
    PACING = {"allowance_us": 500, "latest_start_us": 250, "t_start_ns": 1_000_000_000}

    def line(self, **changes):
        made = {"schema": mp.RELEASE_SCHEMA, "run_id": "unit", "sequence": 0, "opportunity_id": "op-0", "sfn": 0, "slot": 7,
                "offset_us": 500, "outcome": "on_time", "overrun": False, "t_release_ns": 1_000_500_000,
                "t_free_ns": 1_000_300_000, "t_ready_ns": 1_000_380_000, "t_start_ns": 1_000_500_040,
                "t_done_ns": 1_000_760_000, "fill_us": 76.5,
                "decodes": [{"backend": "cuda", "key": "op-0/cuda", "outcome": "decoded", "skip": None,
                             "t_start_ns": 1_000_512_000, "t_callback_ns": 1_000_722_000, "service_us": 210.0}]}
        made.update(changes)
        return made

    def test_a_release_line_is_consistent_with_the_rule_on_its_own_times(self):
        self.assertEqual(mp.release_violations(self.line(), self.PACING), [])
        late = self.line(outcome="late", t_ready_ns=1_000_520_000, t_start_ns=1_000_520_010,
                         decodes=[dict(self.line()["decodes"][0], t_start_ns=1_000_522_000, t_callback_ns=1_000_732_000)])
        self.assertEqual(mp.release_violations(late, self.PACING), [])
        over = self.line(overrun=True, t_done_ns=1_001_000_001)
        self.assertEqual(mp.release_violations(over, self.PACING), [])
        skipped = self.line(outcome="skipped", t_free_ns=1_000_750_001, t_ready_ns=None, t_start_ns=None, t_done_ns=None,
                            fill_us=None, decodes=[{"backend": "cuda", "key": "op-0/cuda", "outcome": "not_dispatched",
                                                    "skip": "missed_release", "t_start_ns": None, "t_callback_ns": None,
                                                    "service_us": None}])
        self.assertEqual(mp.release_violations(skipped, self.PACING), [])
        for broken, message in (
                (self.line(t_release_ns=1_000_500_001), "start of the schedule plus the offset"),
                (self.line(t_start_ns=1_000_499_999), "started before its offered release time"),
                (self.line(outcome="late"), "late exactly when its input was ready after"),
                (self.line(overrun=True), "overrun exactly when"),
                (self.line(t_free_ns=1_000_750_001, t_ready_ns=1_000_750_002, t_start_ns=1_000_750_003,
                           outcome="late"), "free only after its latest start"),
                (dict(skipped, t_free_ns=1_000_750_000), "skipped although the receiver was free"),
                (dict(skipped, overrun=True), "names no time of a job"),
                (self.line(decodes=[dict(self.line()["decodes"][0], service_us=211.0)]), "its service time is the distance"),
                (self.line(decodes=[dict(self.line()["decodes"][0], t_start_ns=1_000_400_000)]), "starts inside the job"),
                (self.line(decodes=[]), "names its decodes"),
                (self.line(outcome="dropped"), "unknown outcome"),
                ({key: value for key, value in self.line().items() if key != "fill_us"}, "exactly the fields")):
            self.assertIn(message, " ".join(mp.release_violations(broken, self.PACING)))

    def test_both_clients_worked_on_the_device_over_a_common_interval(self):
        class Output:
            releases = [{"decodes": [{"backend": "cuda", "t_start_ns": 1000 + 500 * n, "t_callback_ns": 1200 + 500 * n}]}
                        for n in range(3)]
        inside = [{"marks": {"t_submit_ns": 1210, "t_engine_ns": 1330}}, {"marks": {"t_submit_ns": 1450, "t_engine_ns": 1560}},
                  {"marks": {"t_submit_ns": 2300, "t_engine_ns": 2400}}, {"status": "gap"}]
        found = mp.overlap(Output, inside)
        self.assertEqual((found["radio_intervals"], found["sensor_intervals"], found["sensor_overlapping"],
                          found["radio_overlapping"], found["sensor_inside_run"]), (3, 3, 1, 1, 2))
        self.assertEqual(found["common_interval_ns"], [1210, 2200])
        self.assertEqual(mp.overlap(Output, [])["common_interval_ns"], None)


class NativeCoreTests(unittest.TestCase):
    """The pacing core, built on its own: no OCUDU source, no OCUDU build and no CUDA."""

    @classmethod
    def setUpClass(cls):
        cls.missing = [tool for tool in ("cmake", "ctest", "c++") if shutil.which(tool) is None]
        if cls.missing:
            return
        cls.temporary = tempfile.TemporaryDirectory()
        cls.build = Path(cls.temporary.name) / "build"
        steps = (["cmake", "-S", str(PACING), "-B", str(cls.build)], ["cmake", "--build", str(cls.build), "-j2"])
        cls.log, cls.status = "", 0
        for step in steps:
            result = subprocess.run(step, capture_output=True, text=True, timeout=1200, check=False)
            cls.log += result.stdout + result.stderr
            cls.status = cls.status or result.returncode
        cls.test = cls.build / "pacing_core_test"

    @classmethod
    def tearDownClass(cls):
        if not cls.missing:
            cls.temporary.cleanup()

    def setUp(self):
        if self.missing:
            self.skipTest("the native core needs " + ", ".join(self.missing))
        if self.status:
            self.fail("the native core does not build:\n" + self.log[-3000:])

    def test_core_builds_without_a_warning_and_links_only_its_modules(self):
        self.assertNotRegex(self.log, r"warning:")
        link = (self.build / "CMakeFiles/pacing_core_test.dir/link.txt").read_text()
        self.assertEqual(re.findall(r"\S*?(lib\w+\.(?:a|so))\b", link),
                         ["libpacing_core.a", "libdispatch_core.a", "libreplay_core.a"])
        for word in ("cuda", "ocudu_phy", "obs_ring"):
            self.assertNotIn(word, link.lower().replace(str(ROOT).lower(), ""))
        allowed = {"pacing_core.h", "dispatch_core.h", "replay_core.h", "time.h"}
        for name in ("pacing_core.h", "pacing_core.cpp", "tests/pacing_core_test.cpp"):
            for include in re.findall(r'#include [<"]([^>"]+)[>"]', (PACING / name).read_text()):
                self.assertTrue(include in allowed or "/" not in include and "." not in include, include)

    def test_every_native_unit_case_passes(self):
        listed = subprocess.run([str(self.test), "--list"], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(tuple(listed), CASES)
        result = subprocess.run(["ctest", "--output-on-failure", "-R", "pacing_core_"], cwd=self.build,
                                capture_output=True, text=True, timeout=600, check=False)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn(f"100% tests passed, 0 tests failed out of {len(CASES)}", result.stdout)
        for name in CASES:
            self.assertRegex(result.stdout, rf"pacing_core_{name} \.+\s+Passed")
        unknown = subprocess.run([str(self.test), "no_such_case"], capture_output=True, text=True, check=False)
        self.assertEqual(unknown.returncode, 2)

    def test_native_core_and_reference_release_alike(self):
        generator = random.Random(20261122)
        cases = [(5 * MS, 12 * MS, 7, 6 * MS, 1 * MS), (500 * US, 300 * US, 12, 500 * US, 250 * US),
                 (500 * US, 499 * US, 9, 500 * US, 0), (500 * US, 501 * US, 9, 500 * US, 0)]
        cases += [(generator.randint(100, 5000) * US, generator.randint(50, 9000) * US, generator.randint(1, 40),
                   allowance, generator.randint(0, allowance // US) * US)
                  for allowance in (generator.randint(1, 6000) * US for _ in range(60))]
        skipped = 0
        for period, work, jobs, allowance, latest in cases:
            said = subprocess.run([str(self.test), "--trace", *map(str, (period, work, jobs, allowance, latest))],
                                  capture_output=True, text=True, check=True).stdout.splitlines()
            pacer = mp.Pacer([n * period for n in range(jobs)], allowance, latest)
            mp.drive(pacer, 0, work)
            wanted = [f"{mark['outcome']} {mark['release']} {mark['start'] or 0} {mark['done'] or 0} {int(mark['overrun'])}"
                      for mark in pacer.marks]
            self.assertEqual(said, wanted, (period, work, jobs, allowance, latest))
            skipped += pacer.counts()["skipped"]
        self.assertGreater(skipped, 20)  # the comparison covered skipped releases, not only quiet schedules


class TreeTests(unittest.TestCase):
    FILES = ("native/ocudu_adapter/pacing/CMakeLists.txt", "native/ocudu_adapter/pacing/pacing_core.h",
             "native/ocudu_adapter/pacing/pacing_core.cpp", "native/ocudu_adapter/pacing/tests/pacing_core_test.cpp",
             "native/ocudu_adapter/observer/pilot_ls.cpp", "native/ocudu_adapter/observer/observer_plan.h",
             "native/ocudu_adapter/observer/CMakeLists.txt", "native/ocudu_adapter/CMakeLists.txt",
             "native/ocudu_adapter/README.md", "scripts/run_m1.py", "scripts/accept_s4_1.sh",
             "src/ocudu_cuda_isac/episodes/mps_pacing.py", "src/ocudu_cuda_isac/episodes/native_estimate.py",
             "tests/unit/test_mps_pacing.py", "tests/unit/mps_pacing_double.py", "tests/unit/test_mps_pacing_driver.py",
             "tests/integration/mps_pacing.py", "tests/test_accept_s4_1.py", "configs/integration/mps_pacing.json",
             "docs/interfaces/MPS_PACING.md", "Makefile")

    def test_files_are_ascii_and_do_not_name_the_project(self):
        # The working name lives on line 1 of the top-level README and nowhere else.
        name = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").split(":")[0].strip()
        for relative in self.FILES:
            data = (ROOT / relative).read_bytes()
            self.assertTrue(data.isascii(), relative)
            if name and name.lower() != "ocudu-cuda-isac":
                self.assertNotIn(name.lower(), data.decode("ascii").lower(), relative)

    def test_the_paced_loop_releases_on_the_offered_time_and_keeps_the_harness_out_of_the_job(self):
        tool = (ADAPTER / "observer/pilot_ls.cpp").read_text()
        paced = tool[tool.index("  // ---- The paced run (Sprint 4.1)."):tool.index("  // ---- The drain:")]
        loop = paced[paced.index("    pace->begin("):]
        # Admission against the schedule, the fill ahead of the slot, the release, the tap, the decodes.
        order = ["pace->take(n, free_ns)", "replay.fill(sequence, sink);", "pace->ready(n, isac_pacing::monotonic_ns());",
                 "pace->start(n, isac_pacing::spin_until(pace->release_time(n)));", "out->publish(handle_of[sequence], ready);",
                 "taker->expect(estimate_handle[sequence], key);", "isac_pacing::paced_dispatch(*paced_jobs, backend",
                 "taker->resolve();", "pace->done(n, isac_pacing::monotonic_ns());", "retire_paced(kept[last]);"]
        positions = [loop.index(text) for text in order]
        self.assertEqual(positions, sorted(positions))
        # A skipped release keeps its key: it is recorded, never dropped from the run.
        self.assertIn("paced_jobs->skip(now.decodes[b].token, isac_pacing::skip_reason::missed_release", loop)
        # The loop hashes nothing, writes nothing, builds no record and never uses the ledger of Sprint 2.4.
        for word in ("sha256_hex", "sha256(", "sha256_count", "write_bytes", "json", "estimate_entry", "std::cout",
                     "book.", "dispatch(book", "std::mutex", "sleep", "new ", "make_unique", "push_back", "settle"):
            self.assertNotIn(word, loop, word)
        # Everything of a decode exists before the schedule is fixed: its place in the book and its notifier.
        before = paced[:paced.index("    pace->begin(")]
        self.assertLess(before.index("paced_jobs->prepare(name, bound.grant_id, bound.record_sha256"),
                        before.index("backends[b]->arrange(t, *paced_jobs);"))
        # The paced run says one thing, once: the start of its schedule, after the preparation and before the
        # schedule exists. The launcher reads that line, so both sides share its words.
        said = 'std::cout << "pilot_ls: schedule start " << schedule_start << " ns on the host monotonic clock, "'
        self.assertEqual((paced.count("std::cout"), before.count(said)), (1, 1))
        self.assertLess(before.index("backends[b]->arrange(t, *paced_jobs);"), before.index(said))
        # Apart from that line the whole paced section, its preparation too, is held to what the tree checks of
        # Sprint 3.2 and Sprint 5.2 require of the timed run: no digest, no file, no record, no output.
        line = paced[paced.index("    std::cout << "):paced.index("    pace->begin(")]
        self.assertEqual((line.count(";"), line.count("std::endl;")), (1, 1))
        for word in ("sha256_hex", "sha256(", "sha256_count", "digest()", "write_bytes", "json", ".dump(", "read_back(",
                     "estimate_entry", "std::cout", "make_directory", "watcher.observe("):
            self.assertNotIn(word, paced.replace(line, ""), word)
        self.assertEqual(run_m1.SCHEDULE_MARK.search(
            "cmd line\npilot_ls: schedule start 297698971293180 ns on the host monotonic clock, 960 releases\n")[1],
            "297698971293180")
        # The unpaced timed loop and the audited loop are the ones that were accepted: they keep the ledger.
        timed = tool[tool.index("  // ---- The timed run."):tool.index("  // ---- The paced run (Sprint 4.1).")]
        self.assertIn("if (timed && !paced) {", timed)
        self.assertIn("dispatch(book, backend, names[n * backends.size() + b], key", timed)
        self.assertNotIn("pace->", timed)
        # A paced plan is refused before a processor exists unless it is timed and whole.
        for text in ("a paced run is a timed run", "one release each", "--only selects opportunities"):
            self.assertLess(tool.index(text), tool.index("std::make_unique<processor_backend>"), text)
        core = (PACING / "pacing_core.cpp").read_text()
        admit = core[core.index("admission paced_book::admit("):core.index("const request& paced_book::describe(")]
        wait = core[core.index("bool paced_book::wait("):core.index("outcome paced_book::resolve(")]
        complete = core[core.index("void paced_book::complete("):core.index("bool paced_book::wait(")]
        for text in (wait, complete):
            for word in ("std::mutex", "condition_variable", "sleep", "new ", "push_back", "nanosleep"):
                self.assertNotIn(word, text, word)
        self.assertNotIn("new ", admit)
        self.assertNotIn("mutex", core + (PACING / "pacing_core.h").read_text().split("#pragma once")[1])

    def test_the_build_the_gate_and_both_sides_share_their_strings(self):
        adapter = (ADAPTER / "CMakeLists.txt").read_text()
        self.assertLess(adapter.index("add_subdirectory(receiver)"), adapter.index("add_subdirectory(pacing)"))
        self.assertLess(adapter.index("add_subdirectory(pacing)"), adapter.index("add_subdirectory(observer)"))
        observer = (ADAPTER / "observer/CMakeLists.txt").read_text()
        self.assertIn("target_link_libraries(pilot_ls PRIVATE pacing_core)", observer)
        self.assertIn("add_test(NAME pilot_ls_pacing COMMAND pilot_ls --pacing)", observer)
        cmake = (PACING / "CMakeLists.txt").read_text()
        self.assertEqual(tuple(re.search(r"set\(PACING_CORE_CASES ([^)]*)\)", cmake)[1].split()), CASES)
        # The pacing core leaves the accepted cores as they are: it adds a directory and edits neither.
        self.assertNotIn("pacing", (ADAPTER / "receiver/CMakeLists.txt").read_text())
        self.assertNotIn("pacing", (ADAPTER / "receiver/dispatch_core.h").read_text())
        self.assertIn("accept-s4.1:", (ROOT / "Makefile").read_text())
        tool = (ADAPTER / "observer/pilot_ls.cpp").read_text() + (ADAPTER / "observer/observer_plan.h").read_text()
        for text in (mp.PLAN_SCHEMA, mp.NATIVE_SCHEMA, mp.RELEASE_SCHEMA, mp.PACING_SCHEMA, mp.PACING_DIR, mp.RELEASES_FILE,
                     mp.CLOCK):
            self.assertIn(f'"{text}"', tool, text)
        for name in mp.RELEASE_FIELDS + mp.DECODE_FIELDS + mp.COUNT_FIELDS:
            self.assertIn(f'{{"{name}",', tool, name)
        for name in mp.PACING_FIELDS:
            self.assertIn(f'"{name}"', tool, name)
        core = (PACING / "pacing_core.cpp").read_text()
        for name in mp.RELEASE_OUTCOMES + mp.SKIPS:
            self.assertIn(f'return "{name}";', core, name)
        launcher = (ROOT / "scripts/run_m1.py").read_text()
        for word in ("shell=True", "os.system", "import torch", "import numpy"):
            self.assertNotIn(word, launcher, word)
        config = json.loads((ROOT / "configs/integration/mps_pacing.json").read_text())
        self.assertEqual((config["schema"], config["selector"]), ("integration-config/mps_pacing/v1", "mps_pacing"))
        mp.check_rule(config["contention"]["rule"])
        self.assertEqual(config["contention"]["rule"]["status"], "proposed")
        self.assertEqual(tuple(config["contention"]["order"]), mp.CONTROLS)
        self.assertEqual(config["rings"]["o3"]["slots"], 8)
        for ring in config["rings"].values():
            self.assertEqual((ring["wait"], ring["slots"]), ("spin", sr.ring_spec(ring["window_records"],
                                                                                 ring["burst_records"])["slots"]))
        self.assertLessEqual(config["pacing"]["latest_start_us"], config["pacing"]["allowance_us"])
        phy, others = set(run_m1.parse_cpus(config["cpus"]["receiver"])), set()
        for role in ("sensor", "o2", "tools"):
            others |= set(run_m1.parse_cpus(config["cpus"][role]))
        self.assertFalse(phy & others, "a sensor or a tool shares a CPU with the receiver")


if __name__ == "__main__":
    unittest.main()
