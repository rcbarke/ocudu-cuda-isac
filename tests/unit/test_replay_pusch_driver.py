"""Check the Sprint 2.4 integration driver against doubles of the native tools.

The stimulus is a small pair on the demonstration grant: two TDD periods, six opportunities, six
slot grants, four receive ports, with the stored grids of a Sprint 2.3 run beside it. The double
of the receiver tool (tests/unit/replay_pusch_double.py) fills every grid with the reference
arithmetic and decodes it with a model receiver, and the double of the fixture adapter realizes a
transmit grid per transport block, so every check of the driver has a known answer, and each
injected fault must fail the check that names it. NumPy is needed, as for the driver.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import schedule as schedules  # noqa: E402
from ocudu_cuda_isac.episodes.episode import TC_PER_SECOND  # noqa: E402
import grid_apply_double as apply_double  # noqa: E402
import replay_pusch_double as double  # noqa: E402

SCALE = 3.4e-4
CHECKS = 36
EXECUTABLE = '''#!{python}
import sys
sys.path.insert(0, {tests!r})
import replay_pusch_double
sys.exit(replay_pusch_double.{entry}())
'''


def load_driver():
    spec = importlib.util.spec_from_file_location("replay_pusch_driver", ROOT / "tests/integration/replay_pusch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def store_of(grant, directory):
    """A store entry with the transmit grid the fixture double realizes for the grant and its block."""
    grants.export(grant, directory, {"x": {"values": double.transmit_grid(grant), "provenance": {"double": True}}})
    return json.loads((directory / "manifest.json").read_text())["files"]["x.npy"]["sha256"]


def make_runs(root):
    """What the accepted Sprint 2.1, 2.2 and 2.3 runs leave: the grant stores, a matched pair, the stored grids."""
    base = grants.load(ROOT / "configs/grants/demo_51prb_4rx.json")
    schedule = schedules.build({"schema": schedules.SCHEMA, "schedule_id": "unit",
                                "tdd": {"pattern": "DDDDDDSUUU", "ul_slots": [7, 8, 9]}, "start_sfn": 0, "periods": 2,
                                "payload": "fixed"}, base)
    s21, s22, s23 = root / "s21/grant_fixture", root / "s22/episode_pair", root / "s23/grid_apply"
    waveforms = {}
    for grant_id, grant in schedule.grants.items():
        directory = (s21 if grant is base else s22 / "grant_fixture") / grant_id / "store"
        waveforms[grant_id] = {"x_sha256": store_of(grant, directory), "store": str(directory)}
    for name in ("table2_51prb_1rx", "two_prb_offset"):
        grant = grants.load(ROOT / "configs/grants" / f"{name}.json")
        store_of(grant, s21 / grant.grant_id / "store")
    h_shape, w_shape = episodes.shapes(schedule)
    opportunity, rx, symbol, k = np.meshgrid(*(np.arange(size) for size in (h_shape[0], h_shape[1], 14, h_shape[4])),
                                             indexing="ij")
    direct = SCALE * np.exp(1j * (0.4 * rx - 0.011 * k))
    echo = 0.0078 * SCALE * np.exp(1j * (0.9 * rx - 0.03 * k - 0.3 * symbol - 1.1 * opportunity))
    h_a = direct.astype("<c8")[:, :, None]
    h_b = (direct + echo).astype("<c8")[:, :, None]
    symbols, subcarriers = zip(*sorted(schedule.opportunities[0].grant.occupied()))
    power = float(np.mean(np.abs(h_a[0, :, 0][:, list(symbols), list(subcarriers)]) ** 2))
    sigma = math.sqrt(power / 1000.0)
    generator = np.random.Generator(np.random.PCG64(20261121))
    w = ((generator.standard_normal(w_shape) + 1j * generator.standard_normal(w_shape)) * sigma / math.sqrt(2)).astype("<c8")
    motion = {"kind": "constant_velocity", "position_m": [10.0, 25.0, 1.5], "velocity_m_s": [0.0, 2.0, 0.0]}
    geometry = {"ue_tx_m": [0.0, 0.0, 1.5], "gnb_rx_m": [20.0, 0.0, 3.0],
                "rx_element_offsets_m": [[0.0, 0.02 * n, 0.0] for n in range(4)]}
    description = {"geometry": geometry, "array": {}, "scene": {"kind": "free_space"},
                   "target": {"model": "constant_rcs_cuboid", "sigma_m2": 1.0, "motion": motion}, "solver": {},
                   "time_evolution": {}, "noise": {"snr_db": 30.0, "sigma": sigma, "seed": 20261121},
                   "calibration": {"reference_power": power, "grid_gain": 1.0 / math.sqrt(power)},
                   "software": {}, "provenance": {}}
    truth = []
    for item in schedule.opportunities:
        position, velocity = episodes.target_state(motion, item.slot_start_tc / TC_PER_SECOND)
        truth.append({"opportunity": item.index, "position_m": position, "velocity_m_s": velocity})
    entry = s22 / "reference_pair/store"
    episodes.export(entry, "unit-pair", schedule, description,
                    {"a": h_a.tobytes(), "b": h_b.tobytes(), "w": w.tobytes()}, truth, None, waveforms)
    # The grids that the accepted Sprint 2.3 run stored for the same entry.
    plan = replay.entry_plan("s23-pair-unit", [entry], [s22 / "grant_fixture"])
    (s23 / "reference_pair/native").mkdir(parents=True)
    replay.write_plan(plan, s23 / "reference_pair/plan.json")
    with contextlib.redirect_stdout(io.StringIO()):
        status = apply_double.main(["--plan", str(s23 / "reference_pair/plan.json"),
                                    "--output-dir", str(s23 / "reference_pair/native")])
    assert status == 0, status
    stored = [item["readback_sha256"] for item in replay.open_output(s23 / "reference_pair/native").report["opportunities"]]
    return {"gain": 1.0 / math.sqrt(power), "s22": root / "s22", "s23": root / "s23",
            "run_digest": hashlib.sha256("\n".join(stored).encode("ascii")).hexdigest(),
            "tb_sha256": hashlib.sha256(base.tb).hexdigest()}


@unittest.skipIf(np is None, "the integration driver needs NumPy")
class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.made = make_runs(cls.root)
        config = json.loads((ROOT / "configs/integration/replay_pusch.json").read_text())
        config["pair"]["expect"].update(opportunities=6, grid_gain=cls.made["gain"], s23_run_digest=cls.made["run_digest"])
        cls.config = cls.root / "config.json"
        cls.config.write_text(json.dumps(config))
        tests = str(Path(__file__).resolve().parent)
        cls.tools = {}
        for name, entry in (("replay_pusch", "main"), ("grant_fixture", "fixture_main")):
            path = cls.root / name
            path.write_text(EXECUTABLE.format(python=sys.executable, tests=tests, entry=entry))
            path.chmod(0o755)
            cls.tools[name] = path
        # One run without a fault, read by several tests.
        cls.reference = cls.run_once(cls.root / "reference")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def receipt_of(cls):
        return {"executables": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                for name, path in cls.tools.items()},
                "ocudu_build": "double", "pin": {"cuda": True}}

    @staticmethod
    def native(tool, plan_path, output, log, extra=()):
        """The tool double without a new interpreter: the same files, the same status, the same last line."""
        output.mkdir(parents=True)
        said = io.StringIO()
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
            try:
                status = double.main(["--plan", str(plan_path), "--output-dir", str(output), *extra])
            except Exception as error:  # a process that crashes exits with status 1
                said.write(f"{type(error).__name__}: {error}\n")
                status = 1
        log.write_text(f"{tool} --plan {plan_path} --output-dir {output}\n" + said.getvalue())
        try:
            return status, json.loads((output / "native.json").read_text())
        except (OSError, ValueError):
            return status, None

    @staticmethod
    def adapter(adapter, request, output, log):
        """The fixture double without a new interpreter."""
        output.mkdir(parents=True)
        status = double.fixture_main(["--grant", str(request / "grant.json"), "--tb", str(request / "tb.bin"),
                                      "--output-dir", str(output), "--dmrs-reference", "crb0", "--no-loopback"])
        log.write_text(f"{adapter} --grant {request / 'grant.json'} --no-loopback\n")
        return status, json.loads((output / "native.json").read_text())

    @classmethod
    def run_once(cls, out, fault=None, config=None, s22=None, s23="default", extra=(), environment=None, processes=False,
                 receipt=None):
        driver = cls.driver
        saved = (driver.native_adapter.verified_receipt, driver.run_native, driver.realize_grid)
        driver.native_adapter.verified_receipt = lambda build_dir: receipt or cls.receipt_of()
        if not processes:
            driver.run_native, driver.realize_grid = cls.native, cls.adapter
        changed = dict(environment or {})
        if fault:
            changed["REPLAY_PUSCH_FAULT"] = fault
        kept = {key: os.environ.get(key) for key in ("REPLAY_PUSCH_FAULT", "REPLAY_PUSCH_BUILD", "ADAPTER_FAULT")}
        for key in kept:
            os.environ.pop(key, None)
        os.environ.update(changed)
        arguments = ["--config", str(config or cls.config), "--s22-run", str(s22 or cls.made["s22"]),
                     "--output-dir", str(out), *extra]
        if s23 is not None:
            arguments += ["--s23-run", str(cls.made["s23"] if s23 == "default" else s23)]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as said:
                status = driver.main(arguments)
        finally:
            driver.native_adapter.verified_receipt, driver.run_native, driver.realize_grid = saved
            for key, value in kept.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
        grade = json.loads((out / "grade.json").read_text())
        failed = sorted(f"{item['case']} {item['check']}" for item in grade["results"] if not item["passed"])
        return {"status": status, "grade": grade, "failed": failed, "out": out, "said": said.getvalue(),
                "lines": (out / "grade.txt").read_text()}

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(scratch.cleanup)
        self.out = Path(scratch.name)

    def faulty(self, fault, **options):
        return self.run_once(self.out / fault, fault=fault, **options)

    # ---------------------------------------------------------------- the reference
    def test_the_reference_passes_every_check(self):
        run = self.reference
        grade, lines, out = run["grade"], run["lines"], run["out"]
        self.assertEqual((run["status"], run["failed"]), (0, []))
        self.assertEqual((grade["judgment"], grade["hardware_gate"], grade["cases"], grade["cases_passed"], grade["checks"]),
                         ("pass", "closed", 8, 8, CHECKS))
        self.assertIn(f"replay_pusch: 8 of 8 cases passed, {CHECKS} of {CHECKS} checks\n", lines)
        self.assertNotIn("rehearsal", run["said"])
        # Unity: the six slot grants and the two other shipped grants, with the values of the host path.
        self.assertIn("unity/slot-grants: 6 grids on 4 receive plane(s), cpu 6 of 6 decoded, cuda 6 of 6 decoded; "
                      "unity/shipped-1rx: 2 grids on 1 receive plane(s), cpu 2 of 2 decoded, cuda 2 of 2 decoded", lines)
        self.assertIn("PASS  S2.4 unity demo-51prb-4rx.s17: slot 17, 4 receive port(s), CRB 0 to 50", lines)
        self.assertIn("each backend returns the original 6787 bytes with a CRC pass; cpu on the host path: decoded, LDPC "
                      "iterations 1.0, EPRE 3.01 dB, SINR 96.99 dB; cuda on the resident path", lines)
        self.assertIn("PASS  S2.4 unity two-prb-offset: slot 17, 1 receive port(s), CRB 15 to 16", lines)
        # The direct path and the pair.
        self.assertIn("6 of 6 readback digests equal the complex128 reference rounded to the storage format", lines)
        self.assertIn("high-SNR direct path: cpu 6 of 6 decoded, cuda 6 of 6 decoded, each with a CRC pass and all 1217 "
                      "bytes equal to the original", lines)
        self.assertIn("12 opportunities of 2 realizations on 6 grants", lines)
        self.assertIn("12 grids filled, made ready and decoded by cpu and cuda: 24 radio records under 24 unique keys", lines)
        self.assertIn("12 of 12 equal the grids that the Sprint 2.3 run s23 stored", lines)
        self.assertIn("NOTE  S2.4 reference-pair against the accepted Sprint 2.3 run (not graded): digest over the 12 "
                      "decoded grids identical", lines)
        self.assertIn("the physical pair at 30 dB: cpu 12 of 12 decoded, cuda 12 of 12 decoded", lines)
        self.assertIn("target absent and target present decode alike on all 6 opportunities and both backends", lines)
        self.assertIn("decoded on the resident path, 1217 bytes starting", lines)
        # A transport block per opportunity.
        self.assertIn("6 distinct transport blocks for 6 opportunities from the generator sha256-counter/v1 with the seed "
                      "'s24-opportunity-block'", lines)
        self.assertIn("payload rule per_opportunity, recorded in the plan and in the report: 12 opportunities of the pair "
                      "on 6 grants", lines)
        self.assertIn("24 of 24 decodes return the block that the rule generates for their own opportunity", lines)
        self.assertIn("cpu returned 6 distinct blocks for 6 opportunities; cuda returned 6 distinct blocks", lines)
        # Bad input stays in the denominator.
        self.assertIn("12 of 12 decodes end as crc_fail by the decoder's own callback, cpu 6 of 6 and cuda 6 of 6", lines)
        self.assertIn("0 of 24 decodes of bad input report a CRC pass and 0 match the original bytes", lines)
        self.assertIn("PASS  S2.4 rerun a failed opportunity: fx-noise-only.0005 of bad-input/noise_only, alone", lines)
        self.assertIn("(cpu crc_fail, cuda crc_fail)", lines)
        self.assertIn("(cpu decoded, cuda decoded)", lines)
        self.assertIn("a transport block of another size, a transport block with another hash, a grant record that was "
                      "edited, a grant without a record, an opportunity in another slot than its grant, an opportunity "
                      "delivered twice, a backend that does not exist, an observer on the control run, a generated payload "
                      "on fixed blocks", lines)
        self.assertIn("a channel plan with another hash, a channel plan without a gain, a transmit grid with another hash",
                      lines)
        # The path of every decode of the seven runs: 48 on four ports and 2 on one port per backend.
        self.assertIn("cpu: 50 on the host path; cuda: 50 on the resident path; the adapter is built with CUDA", lines)
        self.assertIn("100 decodes planned in 7 runs: 100 records, each under a key that is unique in its run, each with "
                      "one terminal outcome: 76 decoded, 0 payload_mismatch, 24 crc_fail, 0 decoder_error, 0 timeout, "
                      "0 not_dispatched; 0 anomalies, 0 grids held after the run", lines)
        self.assertIn("NOTE  S2.4 backends PUSCH service time of cuda, not graded", lines)
        comparison = json.loads((out / "comparison.json").read_text())
        pair = comparison["reference-pair"]
        self.assertEqual(pair["ingress"]["digests_equal_to_reference"], 12)
        self.assertEqual(pair["ingress"]["digests_equal_to_s23"], 12)
        self.assertEqual(pair["ingress"]["run_digest"], self.made["run_digest"])
        self.assertEqual(pair["access"], {"ingress_violations": [], "radio_leaks": []})
        self.assertEqual(pair["totals"]["cuda"]["matched_bytes"], 12 * 1217)
        self.assertEqual(comparison["payload"]["blocks"], 6)
        self.assertEqual(comparison["backends"]["paths"], {"cpu": {"host": 50}, "cuda": {"resident": 50}})
        self.assertGreaterEqual(comparison["direct"]["storage_sqnr_db"], 50.0)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual((manifest["judgment"], manifest["selector"], manifest["host_rehearsal"]),
                         ("pass", "replay_pusch", False))
        self.assertEqual(manifest["axes"]["receiver_stage"], "none: the untapped control")
        self.assertEqual([item["name"] for item in manifest["runs"]],
                         ["unity/slot-grants", "unity/shipped-1rx", "direct", "reference-pair", "payload",
                          "bad-input/noise_only", "bad-input/phase_break"])
        self.assertEqual([item["payload"]["rule"] for item in manifest["runs"]],
                         ["fixed", "fixed", "fixed", "fixed", "per_opportunity", "fixed", "fixed"])
        # The receiver side of the archived run: descriptors and stored grids, records and decoded bytes.
        native = out / "reference_pair/run/native"
        self.assertEqual(sorted(path.name for path in (native / "ingress").iterdir()),
                         [f"{n:04d}.c64" for n in range(12)] + ["ingress.jsonl"])
        self.assertEqual(sorted(path.name for path in (native / "radio").iterdir()),
                         sorted([f"{n:04d}.{backend}.bin" for n in range(12) for backend in ("cpu", "cuda")]
                                + ["radio.jsonl"]))
        self.assertEqual(sorted(path.name for path in (out / "bad_input/noise_only/run/native/radio").iterdir()),
                         ["radio.jsonl"])
        # Enough input to run a failed grant again: the plan, and the files it binds by hash.
        plan = json.loads((out / "bad_input/noise_only/run/plan.json").read_text())
        self.assertTrue(Path(plan["channel_plan"]["path"]).is_file())
        self.assertTrue(all(Path(grant[key]["path"]).is_file() for grant in plan["grants"] for key in ("record", "tb")))
        with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()):
            self.driver.main(["--config", str(self.config), "--s22-run", str(self.made["s22"]),  # never overwritten
                              "--output-dir", str(out)])
        self.assertEqual(refused.exception.code, 2)

    def test_the_tools_are_started_as_separate_processes(self):
        run = self.run_once(self.out / "process", processes=True)
        self.assertEqual((run["status"], run["failed"], run["grade"]["checks"]), (0, [], CHECKS))
        self.assertIn(str(self.tools["replay_pusch"]), (run["out"] / "reference_pair/run/native.log").read_text())
        self.assertIn("replay_pusch: input error:", (run["out"] / "refusals/plan-0/native.log").read_text())
        stores = sorted(path.name for path in (run["out"] / "payload/stores").iterdir())
        self.assertEqual(stores[0], "demo-51prb-4rx.p000000")
        log = (run["out"] / "payload/stores" / stores[0] / "native.log").read_text()
        self.assertIn(str(self.tools["grant_fixture"]), log)
        # The transmit grid alone: the loopback of the adapter is not run, and the store says so.
        self.assertIn("--dmrs-reference crb0 --no-loopback", log)
        report = json.loads((run["out"] / "payload/stores" / stores[0] / "native/native.json").read_text())
        self.assertEqual(report["loopback"], {"executed": False})
        manifest = json.loads((run["out"] / "payload/stores" / stores[0] / "store/manifest.json").read_text())
        self.assertIn("not run", json.dumps(manifest))

    def test_no_sprint_2_3_run_is_said_and_another_one_fails(self):
        run = self.run_once(self.out / "alone", s23=None)
        self.assertEqual((run["status"], run["failed"]), (0, []))
        self.assertIn("no Sprint 2.3 run was named, so its stored grids were not compared", run["lines"])
        # A Sprint 2.3 run of another stimulus: one of its grids differs.
        other = self.out / "s23"
        shutil.copytree(self.made["s23"], other)
        native = other / "grid_apply/reference_pair/native"
        report = json.loads((native / "native.json").read_text())
        lines = (native / "ingress/ingress.jsonl").read_text().splitlines()
        descriptor = json.loads(lines[3])
        grid = bytearray((native / "ingress/0003.c64").read_bytes())
        grid[0] ^= 1
        (native / "ingress/0003.c64").write_bytes(bytes(grid))
        descriptor["grid"]["sha256"] = hashlib.sha256(bytes(grid)).hexdigest()
        report["opportunities"][3]["readback_sha256"] = descriptor["grid"]["sha256"]
        lines[3] = json.dumps(descriptor, sort_keys=True)
        (native / "ingress/ingress.jsonl").write_text("\n".join(lines) + "\n")
        (native / "native.json").write_text(json.dumps(report))
        run = self.run_once(self.out / "other", s23=other)
        self.assertEqual((run["status"], run["failed"]), (1, ["reference-pair ingress digests"]))
        self.assertIn("11 of 12 equal the grids that the Sprint 2.3 run s23 stored", run["lines"])
        run = self.run_once(self.out / "missing", s23=self.out / "no-such-run")
        self.assertEqual((run["status"], run["failed"]), (1, ["reference-pair ingress digests"]))
        self.assertIn("holds no comparable output", run["lines"])

    # ---------------------------------------------------------------- faults of the decoder
    def test_a_valid_crc_with_a_wrong_byte_fails_the_comparisons(self):
        run = self.faulty("wrong-byte")
        self.assertEqual(run["status"], 1)
        for name in ("unity demo-51prb-4rx", "unity table2-51prb-1rx", "direct decodes", "reference-pair decodes",
                     "reference-pair backends agree", "reference-pair second reader", "payload decodes",
                     "backends accounting"):
            self.assertIn(name, run["failed"])
        # The tool said so itself: the run is clean, and bad input is still only a CRC failure.
        for name in ("unity native run", "reference-pair native run", "bad-input noise only", "bad-input never acknowledged",
                     "backends executed path", "reference-pair ingress digests"):
            self.assertNotIn(name, run["failed"])
        self.assertIn("cpu 12 of 12 decoded, cuda 0 of 12 decoded", run["lines"])
        self.assertIn("38 decoded, 38 payload_mismatch, 24 crc_fail", run["lines"])
        # The decoded bytes are kept for the comparison although they are wrong.
        self.assertTrue((run["out"] / "reference_pair/run/native/radio/0003.cuda.bin").is_file())

    def test_a_forged_match_is_caught_by_the_evaluators_own_comparison(self):
        run = self.faulty("forged-match")
        self.assertEqual(run["status"], 1)
        for name in ("unity native run", "direct native run", "reference-pair native run", "payload native run",
                     "reference-pair decodes", "payload decodes", "reference-pair second reader"):
            self.assertIn(name, run["failed"])
        # The tool reports 12 decodes: the count of the evaluator is the one that is graded.
        self.assertIn("comparison with the original 1217 bytes agrees on all 12 records", run["lines"])
        self.assertNotIn("backends accounting", run["failed"])

    def test_bytes_of_another_opportunity_are_only_seen_with_a_block_per_opportunity(self):
        run = self.faulty("stale-bytes")
        self.assertEqual(run["status"], 1)
        self.assertEqual(run["failed"], ["backends accounting", "payload decodes", "payload members alike"])
        # With one block for all opportunities, the same fault passes every check of the pair.
        self.assertIn("PASS  S2.4 reference-pair decodes", run["lines"])
        self.assertIn("cpu 12 of 12 decoded, cuda 1 of 12 decoded; 13 of 24 decodes return the block that the rule "
                      "generates for their own opportunity", run["lines"])

    def test_a_forced_acknowledgement_fails_the_bad_input_case(self):
        run = self.faulty("forced-ack")
        self.assertEqual(run["status"], 1)
        self.assertEqual(run["failed"], ["backends accounting", "bad-input never acknowledged", "bad-input noise only",
                                         "bad-input phase break"])
        self.assertIn("24 of 24 decodes of bad input report a CRC pass and 0 match the original bytes", run["lines"])

    def test_a_fallback_cannot_close_the_gate(self):
        run = self.faulty("host-fallback")
        self.assertEqual((run["status"], run["failed"]), (1, ["backends executed path"]))
        self.assertIn("cuda: 50 on the host path; built with CUDA: True, device usable: True; 50 decodes did not execute "
                      "the requested path (the double was asked to fall back). A fallback cannot close this gate.",
                      run["lines"])
        self.assertEqual((run["grade"]["judgment"], run["grade"]["hardware_gate"]), ("fail", "open"))

    # ---------------------------------------------------------------- a build without CUDA
    def test_a_host_build_is_a_rehearsal_and_never_a_pass(self):
        host = {"REPLAY_PUSCH_BUILD": "host"}
        run = self.run_once(self.out / "gate", environment=host)
        self.assertEqual((run["status"], run["failed"]), (1, ["backends executed path"]))
        self.assertIn("built with CUDA: False, device usable: False; 50 decodes did not execute the requested path (the "
                      "adapter was built without CUDA)", run["lines"])
        run = self.run_once(self.out / "rehearsal", environment=host, extra=("--host-rehearsal",))
        self.assertEqual((run["status"], run["failed"]), (0, []))
        grade = run["grade"]
        self.assertEqual((grade["judgment"], grade["checks"]), ("rehearsal", CHECKS))
        self.assertEqual(grade["hardware_gate"], "open: the CUDA processor did not execute, this is a host rehearsal")
        self.assertIn("REHEARSAL on a build without CUDA, not a hardware result: cpu: 50 on the host path; cuda: 50 on "
                      "the host path", run["lines"])
        self.assertIn(f"{CHECKS} of {CHECKS} checks, host rehearsal: the hardware gate stays open", run["said"])
        manifest = json.loads((run["out"] / "manifest.json").read_text())
        self.assertEqual((manifest["judgment"], manifest["host_rehearsal"]), ("rehearsal", True))
        # A rehearsal on a build with CUDA is refused by its own check.
        run = self.run_once(self.out / "wrong", extra=("--host-rehearsal",))
        self.assertEqual((run["status"], run["failed"], run["grade"]["judgment"]), (1, ["backends executed path"], "fail"))
        self.assertIn("--host-rehearsal was given on a build with CUDA", run["lines"])

    # ---------------------------------------------------------------- faults of the grid and of the records
    def test_another_grid_fails_the_digests(self):
        run = self.faulty("changed-grid")
        self.assertEqual(run["status"], 1)
        self.assertEqual(run["failed"], ["direct grid precision", "reference-pair ingress digests",
                                         "reference-pair second reader"])
        self.assertIn("0 of 12 readback digests of the decoded grids equal the complex128 reference", run["lines"])
        self.assertIn("0 of 12 equal the grids that the Sprint 2.3 run s23 stored", run["lines"])
        self.assertIn("digest over the 12 decoded grids different", run["lines"])

    def test_record_and_access_faults_fail_their_checks(self):
        for fault, wanted, text in (
                ("other-grid", ["reference-pair native run", "unity native run"], "does not name the grid of its opportunity"),
                ("drop-record", ["reference-pair native run", "payload native run"], "radio records for 24 planned decodes"),
                ("leak", ["reference-pair native run"], "is not an o1-grid/v1 record"),
                ("leak-file", ["reference-pair access"], "the receiver side could read: file ingress/plan.json"),
                ("anomaly", ["reference-pair native run", "backends accounting"], "7 anomalies"),
                ("exit", ["reference-pair native run", "rerun a decoded opportunity", "backends executed path"],
                 "exited with status 3")):
            run = self.faulty(fault)
            with self.subTest(fault=fault):
                self.assertEqual(run["status"], 1)
                for name in wanted:
                    self.assertIn(name, run["failed"])
                self.assertIn(text, run["lines"])

    def test_a_decode_that_never_completes_fails_the_run_and_keeps_its_record(self):
        run = self.faulty("timeout")
        self.assertEqual(run["status"], 1)
        for name in ("unity native run", "direct native run", "reference-pair native run", "payload native run",
                     "bad-input native run", "backends accounting"):
            self.assertIn(name, run["failed"])
        self.assertIn("the native tool exited with status 1", run["lines"])
        # The archived output still holds one record per planned decode.
        records = [json.loads(line) for line in
                   (run["out"] / "reference_pair/run/native/radio/radio.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 24)
        self.assertEqual([record["outcome"] for record in records if record["backend"] == "cuda"],
                         ["decoded", "timeout"] + ["not_dispatched"] * 10)

    def test_a_tool_that_refuses_nothing_fails_the_refusals(self):
        run = self.faulty("accept-all")
        self.assertEqual(run["status"], 1)
        self.assertEqual(run["failed"], ["refusals channel rules", "refusals plan rules"])
        self.assertIn("NOT REFUSED AS REQUIRED: ", run["lines"])

    def test_a_fixture_adapter_that_does_not_realize_the_grant_fails_the_payload_case(self):
        for fault, text in (("pilot", "demo-51prb-4rx.p000000: "), ("other-pdu", "the fixture adapter did not build the "
                            "PDU of the grant"), ("exit", "the fixture adapter exited with status 3")):
            run = self.run_once(self.out / f"fixture-{fault}", environment={"ADAPTER_FAULT": fault})
            with self.subTest(fault=fault):
                self.assertEqual((run["status"], run["failed"]), (1, ["payload blocks"]))
                self.assertIn(text, run["lines"])
                self.assertIn("no transmit grid was realized", run["lines"])

    # ---------------------------------------------------------------- stimulus and setup
    def test_a_missing_stimulus_fails_every_case(self):
        run = self.run_once(self.out / "empty", s22=self.out / "no-such-run")
        self.assertEqual(run["status"], 1)
        self.assertEqual(run["failed"], sorted(f"{case} stimulus" for case in self.driver.CASES))
        self.assertEqual((run["grade"]["cases_passed"], run["grade"]["judgment"]), (0, "fail"))
        self.assertIn("the stimulus of the Sprint 2.2 run is unusable", run["lines"])
        # A run that lacks one of the other shipped grants fails the check of that grant only.
        saved = self.root / "s21/grant_fixture/two-prb-offset"
        saved.rename(self.root / "s21/held")
        try:
            run = self.run_once(self.out / "no-shipped-grant")
        finally:
            (self.root / "s21/held").rename(saved)
        self.assertEqual((run["status"], run["failed"]), (1, ["unity two-prb-offset"]))
        self.assertIn("no verified store entry of the grant two-prb-offset", run["lines"])

    def test_an_expectation_of_another_run_fails_the_plan_check(self):
        config = json.loads(self.config.read_text())
        config["pair"]["expect"].update(grid_gain=2942.4243526773034, opportunities=24)
        path = self.out / "other.json"
        path.write_text(json.dumps(config))
        run = self.run_once(self.out / "other", config=path)
        self.assertEqual((run["status"], run["failed"]), (1, ["reference-pair plan"]))
        config = json.loads(self.config.read_text())
        config["unity"]["expect"]["sinr_db"]["4"] = 90.0
        path.write_text(json.dumps(config))
        run = self.run_once(self.out / "sinr", config=path)
        # Only the host path is held to the values Sprint 2.1 recorded, and it decodes the slot grants on four ports.
        self.assertEqual(run["failed"], [f"unity demo-51prb-4rx{suffix}" for suffix in
                                         ("", ".s08", ".s09", ".s17", ".s18", ".s19")])

    def test_an_unusable_setup_grades_nothing(self):
        config = json.loads(self.config.read_text())
        changes = (lambda c: c.update(selector="grid_apply"), lambda c: c["tolerance"].pop("sinr_db"),
                   lambda c: c.update(backends=["cpu"]), lambda c: c.update(backends=["cuda", "cpu"]),
                   lambda c: c["receiver"].update(ldpc_iterations=10), lambda c: c["receiver"].update(early_stop=False),
                   lambda c: c["receiver"].update(timeout_ms=0), lambda c: c["receiver"].update(tap="o2"),
                   lambda c: c["pair"].pop("expect"), lambda c: c["unity"]["expect"].pop("sinr_db"),
                   lambda c: c.update(payload={}))
        for number, change in enumerate(changes):
            edited = json.loads(json.dumps(config))
            change(edited)
            path = self.out / f"config-{number}.json"
            path.write_text(json.dumps(edited))
            with self.subTest(number=number), self.assertRaises(SystemExit) as refused, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.run_once(self.out / "never", config=path)
            self.assertEqual(refused.exception.code, 2)
            self.assertFalse((self.out / "never").exists())
        # An adapter build from before this sprint holds no replay_pusch.
        receipt = self.receipt_of()
        del receipt["executables"]["replay_pusch"]
        with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()) as said:
            self.run_once(self.out / "never", receipt=receipt)
        self.assertEqual(refused.exception.code, 2)
        self.assertIn("the adapter build holds no replay_pusch. Rebuild it", said.getvalue())
        self.assertFalse((self.out / "never").exists())


if __name__ == "__main__":
    unittest.main()
