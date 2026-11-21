"""Check the Sprint 2 gate runner against tool doubles. No compiler, source, build, NumPy or GPU is needed."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s2.sh"
BLOCK = b"original transport block of the demonstration grant"
BLOCK_SHA256 = hashlib.sha256(BLOCK).hexdigest()
GRADE = """PASS  S2 verdicts four gates: S2.1 s21-a: ACCEPTED on this DGX Spark; S2.4 s24-a: ACCEPTED on this DGX Spark
PASS  S2 links each gate consumed its predecessor: S2.2 built its pair on the grant store of s21-a
PASS  S2 waveform grant record: the store entry of demo-51prb-4rx in the Sprint 2.1 run verifies
PASS  S2 receiver original bytes: cpu 48 of 48; cuda 48 of 48 decodes of the pair end as decoded
PASS  S2 receiver executed path: in the 7 native runs of the gate, cpu: 140 on the host path; cuda: 140 on the resident path
NOTE  S2 sources commits of the four gates, not graded: S2.1 e092623
sprint2: 9 of 9 stages passed, 5 of 5 checks"""
GRADE_JSON = {"judgment": "pass", "profile": "DGX Spark", "stages": 9, "stages_passed": 9, "checks": 5,
              "checks_passed": 5, "checks_failed": 0,
              "runs": {"s21": "s21-a", "s22": "s22-a", "s23": "s23-a", "s24": "s24-a"}}
UNIT = "Ran 340 tests in 150.000s\n\nOK"
CARD = "Ran 21 tests in 6.500s\n\nOK"
S24_SUMMARY = """Sprint 2.4 acceptance on spark-ecf8, 2026-11-21T06:00:00Z
PASS  committed tree
Sprint 2.4: ACCEPTED on this DGX Spark. The host and the CUDA PUSCH processor of the pinned source decode the replayed grids.
"""
# The interpreter double answers each gate from FAKE_* variables and logs the call. The audit double
# leaves its grade, its manifest, the archived block and the bytes decoded from the pair.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    "tests/unit/test_"*".py -v")
        name=${{1#tests/unit/test_}}
        if [ "${{name%.py}}" = "${{FAKE_CARD_FAILS:-}}" ]; then printf 'Ran 21 tests in 1.0s\\n\\nOK (skipped=3)\\n' >&2; exit 0; fi
        printf '%s\\n' "$FAKE_CARD" >&2 ;;
    "scripts/verify_sprint2.py --s24-run "*)
        for out; do :; done
        echo "Audit artifacts: $out"
        [ -z "${{FAKE_NO_EVIDENCE:-}}" ] || exit "${{FAKE_AUDIT:-0}}"
        mkdir -p "$out/store" "$out/radio"
        printf '%s' "$FAKE_BLOCK" > "$out/store/tb.bin"
        for name in 0000.cpu 0000.cuda 0001.cpu 0001.cuda; do printf '%s' "$FAKE_BLOCK" > "$out/radio/$name.bin"; done
        [ -z "${{FAKE_DECODED:-}}" ] || printf '%s' "$FAKE_DECODED" > "$out/radio/0001.cuda.bin"
        echo '{{}}' > "$out/radio/radio.jsonl"
        printf '%s\\n' "$FAKE_GRADE" > "$out/grade.txt"
        printf '%s\\n' "$FAKE_GRADE_JSON" > "$out/grade.json"
        [ -n "${{FAKE_NO_MANIFEST:-}}" ] || printf '{{"original_block": {{"path": "%s", "sha256": "%s", "bytes": 51}}, "pair_radio": "%s", "pair_decodes": %s}}\\n' \\
            "$out/store/tb.bin" "${{FAKE_BLOCK_SHA256}}" "$out/radio" "${{FAKE_PLANNED:-4}}" > "$out/manifest.json"
        exit "${{FAKE_AUDIT:-0}}" ;;
    *"check_pins.py --worktrees") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""
# A gate double for --rerun: it logs its arguments and leaves the verdict of its sprint.
GATE_DOUBLE = """#!/bin/sh
echo "accept_s2_{gate}.sh $*" >> "$FAKE_LOG"
mkdir -p "$2/replay_pusch"
[ "{gate}" != 4 ] || echo '{{}}' > "$2/replay_pusch/manifest.json"
if [ "${{FAKE_GATE_FAILS:-}}" = "{gate}" ]; then
    echo "FAIL  S2.{gate} integration grade: 1 of 3 cases"
    echo "Sprint 2.{gate}: NOT ACCEPTED on this host" | tee "$2/summary.txt"
    exit 1
fi
echo "Sprint 2.{gate}: ACCEPTED on this DGX Spark. The evidence of the gate is archived." | tee "$2/summary.txt"
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum"), "the gate runner is a Bash script")
class SprintTwoGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s2.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        python.chmod(0o755)
        for gate in (1, 2, 3, 4):
            (self.root / f"scripts/accept_s2_{gate}.sh").write_text(GATE_DOUBLE.format(gate=gate))
        self.s24 = self.gate_run("s24-spark-ecf8-20261121T060000Z")
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    def gate_run(self, name, summary=S24_SUMMARY, decodes=True):
        run = self.root / "results/runs" / name
        (run / "replay_pusch").mkdir(parents=True)
        if decodes:
            (run / "replay_pusch/manifest.json").write_text("{}")
        (run / "summary.txt").write_text(summary)
        return run

    def gate(self, *arguments, run_dir=None, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT=UNIT, FAKE_CARD=CARD, FAKE_GRADE=GRADE,
                   FAKE_GRADE_JSON=json.dumps(GRADE_JSON), FAKE_BLOCK=BLOCK.decode("ascii"),
                   FAKE_BLOCK_SHA256=BLOCK_SHA256)
        env.update({f"FAKE_{key.upper()}": value if isinstance(value, str) else json.dumps(value)
                    for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s2.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def summary(self):
        return (self.run_dir / "summary.txt").read_text()

    def test_the_chain_of_the_accepted_runs_closes_the_sprint(self):
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("FAIL", result.stdout)
        self.assertNotIn("SKIP", result.stdout)
        summary = self.summary()
        self.assertEqual(summary, result.stdout)
        for line in ("PASS  unit tests (340 ran, 0 skipped)",
                     "PASS  S2 unit gate, run directly: python tests/unit/test_grant_fixture.py (21 ran, 0 skipped)",
                     "PASS  S2 unit gate, run directly: python tests/unit/test_episode_pair.py (21 ran, 0 skipped)",
                     "PASS  S2 unit gate, run directly: python tests/unit/test_grid_apply.py (21 ran, 0 skipped)",
                     "PASS  S2 unit gate, run directly: python tests/unit/test_replay_pusch.py (21 ran, 0 skipped)",
                     "PASS  S2 last gate: the accepted Sprint 2.4 run s24-spark-ecf8-20261121T060000Z holds the decodes",
                     "PASS  S2 receiver original bytes: cpu 48 of 48; cuda 48 of 48 decodes of the pair end as decoded",
                     "NOTE  S2 sources commits of the four gates, not graded",
                     "PASS  S2 chain audit: 9 of 9 stages, 5 of 5 checks passed on the runs s21-a, s22-a, s23-a and s24-a",
                     f"PASS  S2 a second reader follows the block: tb.bin of the Sprint 2.1 store has the SHA-256 "
                     f"{BLOCK_SHA256[:16]} by sha256sum, and 4 of 4 files of decoded bytes of the pair have the same one, "
                     "one for each of the 4 decodes on both backends",
                     "PASS  S2 sources at the locked revisions",
                     "Sprint 2: ACCEPTED on this DGX Spark. A known waveform reaches the native receiver: the grant "
                     "record, the matched pair, the channel application and the host and CUDA decodes are one chain, and "
                     "both backends return the original transport block (runs s21-a, s22-a, s23-a and s24-a). Sprint 2 is "
                     "complete. Next: S3.1 pilot_ls."):
            self.assertIn(line, summary)
        calls = self.calls.read_text()
        self.assertIn(f"scripts/verify_sprint2.py --s24-run {self.s24} --output-dir {self.run_dir}/chain", calls)
        self.assertNotIn("accept_s2_1.sh", calls)
        self.assertEqual(summary.count("PASS  "), 14)
        recovered = (self.run_dir / "recovered.txt").read_text()
        self.assertIn(BLOCK_SHA256, recovered)
        self.assertIn("4 of 4 files", recovered)

    def test_the_newest_accepted_sprint_2_4_run_is_audited_and_named_runs_are_passed_on(self):
        self.gate_run("s24-spark-ecf8-20261121T090000Z", S24_SUMMARY.replace("PASS  committed", "FAIL  committed"))
        self.gate_run("s24-spark-ecf8-20261121T100000Z", "Sprint 2.4: REHEARSED on this host, NOT ACCEPTED.\n")
        self.gate_run("s24-spark-ecf8-20261121T110000Z", decodes=False)
        newest = self.gate_run("s24-spark-ecf8-20261121T070000Z")
        self.assertEqual(self.gate().returncode, 0)
        self.assertIn(f"--s24-run {newest} --output-dir", self.calls.read_text())
        # Runs that were moved are named and reach the audit as they are, with the grant stores.
        result = self.gate("--s24-run", str(self.s24), "--s22-run", "/moved/s22 run", "--s21-run", "/moved/s21",
                           "--grant-store", "/moved/stores", run_dir=self.root.parent / "evidence-named")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"--s24-run {self.s24} --s22-run /moved/s22 run --s21-run /moved/s21 --grant-store /moved/stores "
                      "--output-dir", self.calls.read_text())

    def test_without_an_accepted_sprint_2_4_run_nothing_is_audited(self):
        shutil.rmtree(self.s24)
        self.gate_run("s24-devhost-20261121T100000Z", "Sprint 2.4: REHEARSED on this host, NOT ACCEPTED.\n")
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2 last gate: no Sprint 2.4 ACCEPTED verdict with decodes in results/runs/s24-*", result.stdout)
        self.assertIn("Run 'make accept-s2.4' on this host first, or name its run with --s24-run. A rehearsal is not one.",
                      result.stdout)
        self.assertIn("SKIP  S2 chain audit and block recovery: no Sprint 2.4 run", result.stdout)
        self.assertIn("Sprint 2: NOT ACCEPTED on this host", result.stdout)
        self.assertNotIn("verify_sprint2.py", self.calls.read_text())

    def test_rerun_runs_the_four_gates_in_order_each_on_the_run_before(self):
        result = self.gate("--rerun")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [line for line in self.calls.read_text().splitlines() if line.startswith("accept_s2_")]
        self.assertEqual(len(calls), 4)
        runs = [line.split()[2] for line in calls]
        self.assertEqual([Path(run).name[:4] for run in runs], ["s21-", "s22-", "s23-", "s24-"])
        self.assertEqual(calls[0].split()[3:], [])
        self.assertEqual(calls[1].split()[3:], ["--s21-run", runs[0]])
        self.assertEqual(calls[2].split()[3:], ["--s22-run", runs[1]])
        self.assertEqual(calls[3].split()[3:], ["--s23-run", runs[2]])
        for gate in (1, 2, 3, 4):
            self.assertIn(f"PASS  S2 rerun: accept_s2_{gate}.sh, ACCEPTED on this DGX Spark (s2{gate}-", result.stdout)
        # The audit reads the run that was just made, not the older accepted one.
        audited = [line for line in self.calls.read_text().splitlines() if "verify_sprint2.py" in line][0]
        self.assertIn(f"--s24-run {self.root / runs[3]} --output-dir", audited)
        self.assertNotIn(str(self.s24), audited)
        self.assertIn("Sprint 2: ACCEPTED on this DGX Spark.", result.stdout)
        # A gate that fails stops the sequence, and nothing is audited.
        result = self.gate("--rerun", run_dir=self.root.parent / "evidence-stop", gate_fails="3")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2 rerun: accept_s2_3.sh did not accept (see rerun-s23.log", result.stdout)
        self.assertIn("      FAIL  S2.3 integration grade: 1 of 3 cases", result.stdout)
        self.assertNotIn("accept_s2_4.sh did", result.stdout)
        self.assertIn("SKIP  S2 chain audit and block recovery: the four gates did not all accept", result.stdout)
        self.assertIn("Sprint 2: NOT ACCEPTED on this host", result.stdout)
        self.assertEqual(self.calls.read_text().count("accept_s2_4.sh"), 1)
        # With --rerun no earlier run can be named.
        refused = self.gate("--rerun", "--s24-run", str(self.s24), run_dir=self.root.parent / "evidence-refused")
        self.assertEqual(refused.returncode, 2)
        self.assertIn("--rerun audits the runs it makes", refused.stderr)

    def test_failures_of_single_gates(self):
        failed = dict(GRADE_JSON, judgment="fail", stages_passed=8, checks_passed=4, checks_failed=1)
        cases = (
            (dict(unit="Ran 340 tests in 4.0s\n\nFAILED (failures=1)", unit_status="1"), "FAIL  unit tests (340 ran"),
            (dict(card="Ran 0 tests in 0.0s\n\nOK"), "FAIL  S2 unit gate, run directly: python "
                                                    "tests/unit/test_grant_fixture.py (none ran)"),
            (dict(card_fails="grid_apply"), "FAIL  S2 unit gate, run directly: python tests/unit/test_grid_apply.py "
                                            "(21 ran, 3 skipped: the native unit cases need cmake"),
            (dict(card_fails="replay_pusch"), "FAIL  S2 unit gate, run directly: python tests/unit/test_replay_pusch.py "
                                              "(21 ran, 3 skipped: the native unit cases need cmake"),
            (dict(grade=GRADE.replace("PASS  S2 receiver executed path", "FAIL  S2 receiver executed path"),
                  grade_json=failed, audit="1"), "FAIL  S2 chain audit: 8 of 9 stages, 4 of 5 checks passed, exit status 1"),
            (dict(no_evidence="1", audit="2"), "FAIL  S2 chain audit produced no graded checks"),
            (dict(grade_json=dict(GRADE_JSON, stages=0, stages_passed=0)), "FAIL  S2 chain audit: 0 of 0 stages"),
            (dict(decoded="other bytes"), "and 3 of 4 files of decoded bytes have it, for 4 planned decodes"),
            (dict(planned="6"), "and 4 of 4 files of decoded bytes have it, for 6 planned decodes"),
            (dict(block_sha256="0" * 64), "FAIL  S2 a second reader follows the block: tb.bin has the SHA-256"),
            (dict(no_manifest="1"), "FAIL  S2 a second reader follows the block: the audit left no manifest"),
            (dict(pins="1"), "FAIL  S2 sources at the locked revisions"),
        )
        for number, (settings, line) in enumerate(cases):
            run_dir = self.root.parent / f"evidence-{number}"
            result = self.gate(run_dir=run_dir, **settings)
            with self.subTest(line=line):
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(line, result.stdout)
                self.assertIn("Sprint 2: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 2: ACCEPTED", result.stdout)

    def test_a_skip_in_a_unit_gate_without_a_native_part_is_reported_and_does_not_fail(self):
        result = self.gate(card_fails="grant_fixture")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASS  S2 unit gate, run directly: python tests/unit/test_grant_fixture.py (21 ran, 3 skipped)",
                      result.stdout)

    def test_a_failed_check_of_the_audit_fails_the_gate_even_with_exit_status_zero(self):
        result = self.gate(grade=GRADE.replace("PASS  S2 links each gate", "FAIL  S2 links each gate"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2 links each gate consumed its predecessor", result.stdout)
        self.assertNotIn("Sprint 2: ACCEPTED", result.stdout)

    def test_a_development_host_is_named_in_the_verdict(self):
        result = self.gate(grade_json=dict(GRADE_JSON, profile="host"))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Sprint 2: ACCEPTED on this host.", result.stdout)

    def test_uncommitted_changes_fail_and_evidence_is_never_overwritten(self):
        if not shutil.which("git"):
            self.skipTest("git is not installed")
        commands = (["init", "-q"], ["add", "scripts"], ["-c", "user.name=t", "-c", "user.email=t@example.com",
                                                         "commit", "-q", "-m", "gate"])
        for command in commands:
            subprocess.run(["git", "-C", str(self.root), *command], check=True, capture_output=True)
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  committed tree (uncommitted changes are listed in git-status.txt)", result.stdout)
        again = self.gate()
        self.assertEqual(again.returncode, 2)
        self.assertIn("already exists. Evidence is never overwritten.", again.stderr)

    def test_usage(self):
        self.assertEqual(self.gate("--help").returncode, 0)
        for arguments in (("--unknown",), ("--s24-run",), ("--s21-run",), ("--grant-store",)):
            result = self.gate(*arguments, run_dir=self.root.parent / "evidence-usage")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Usage: bash scripts/accept_s2.sh", result.stderr)
        (self.bin / "python").unlink()
        missing = self.gate(run_dir=self.root.parent / "evidence-python")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("Run 'make env' first", missing.stderr)


if __name__ == "__main__":
    unittest.main()
