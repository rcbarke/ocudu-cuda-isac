"""Synthetic transcripts exercise the fail-closed transcript grader. No GPU and no native binary is needed.

The files in tests/audit/transcripts hold the text that the report code of the pinned test
(pusch_e2e_pipeline_test.cpp at 27e84cb3, compiled against its bundled fmt) prints for chosen
counters: a complete Table II run, the same run without GPU residency, a run of zero
iterations and a complete four-RX smoke. No decode produced them. The times are invented and
differ from the paper on purpose. They are parser fixtures and supply no empirical result.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import verify_pusch_log
from scripts.verify_pusch_log import grade_transcript

TRANSCRIPTS = Path(__file__).resolve().parent / "transcripts"
PAPER = (TRANSCRIPTS / "paper.log").read_text()
WITHOUT_GPU = (TRANSCRIPTS / "paper-without-gpu.log").read_text()
ZERO = (TRANSCRIPTS / "paper-zero-iterations.log").read_text()
SMOKE = (TRANSCRIPTS / "smoke.log").read_text()
GATES = ["workload", "cpu_decodes", "gpu_decodes", "decoded_bytes", "cuda_work", "service_times", "native_verdict"]
GPU_TOTAL = "GPU total:          256.0    264.9    274.0"
LDPC = "LDPC decode:           96.0    100.0    104.0    41.7%"
VERDICT = "PASS: Both paths decoded all 200 iterations, 0 byte mismatches"


def failing(transcript, profile="paper"):
    """Names of the gates one transcript fails."""
    return [name for name, passed in grade_transcript(transcript, profile)[2]["gates"].items() if not passed]


class PaperRowGradeTest(unittest.TestCase):
    def test_complete_transcript_passes_every_gate_and_records_the_comparison(self):
        failed, lines, record = grade_transcript(PAPER, "paper")
        self.assertFalse(failed, "\n".join(lines))
        self.assertEqual([line[:4] for line in lines], ["PASS"] * 7 + ["NOTE"] * 6)
        self.assertEqual(list(record["gates"]), GATES)
        for expected in (
                "PASS  S1.2 paper workload is the Table II 20 MHz one-layer row: 51 PRB, MCS 27 (qam256), 1 layer, "
                "1 RX port, TBS 54296 bits, 200 measured iterations after 5 warmup",
                "PASS  S1.2 paper CPU decodes: 200/200 CRC passes",
                "PASS  S1.2 paper GPU decodes: 200/200 CRC passes",
                "PASS  S1.2 paper decoded bytes agree: 0 CRC disagreements, 0 byte mismatches, 0 output guard failures",
                "PASS  S1.2 paper CUDA work completed on the device: resident pipeline active after warmup, resident "
                "timing 200/200, producer-owned grid 200/200, immutable snapshot 200/200, LDPC decode 100.0 us mean "
                "by CUDA events",
                "PASS  S1.2 paper service times measured over 200 iterations: CPU 304.0 us mean (293.0 to 315.0), "
                "GPU 264.9 us mean (256.0 to 274.0), speedup 1.15x",
                "PASS  S1.2 paper native test verdict: " + VERDICT,
                "NOTE  S1.2 paper correctness reproduced: 200/200 CPU and 200/200 GPU decodes with 0 differing "
                "bytes, as Table II reports",
                "NOTE  S1.2 paper performance, CPU mean: 304.0 us here; 334.0 us in the paper (-9.0 percent)",
                "NOTE  S1.2 paper performance, GPU mean: 264.9 us here; 183.9 us in the paper (+44.0 percent)",
                "NOTE  S1.2 paper performance, speedup: 1.15x here; 1.82x in the paper. The GPU path was faster "
                "than the CPU path on this run.",
                "NOTE  S1.2 paper CSI parity, SINR: CPU 29.87 dB, GPU 29.86 dB, delta -0.01 dB, GPU populated: Yes. "
                "The paper reports deltas no larger than 0.02 dB.",
                "NOTE  S1.2 paper settings that Table II does not state: 30 dB SINR, 5 warmup iterations, 1 RX port, "
                "DM-RS symbols 2 and 11"):
            self.assertIn(expected, lines)
        self.assertTrue(record["correctness_reproduced"] and record["service_times_measured"])
        self.assertEqual(record["workload"]["tbs_bits"], 54296)
        self.assertEqual((record["workload"]["codeblocks"], record["workload"]["modulation"]), (7, "256QAM"))
        self.assertIn("rx_device_grid=direct-managed, gpu_output=pinned", record["workload"]["config"])
        self.assertEqual(record["decodes"]["cpu_crc_pass"], 200)
        self.assertEqual(record["decodes"]["producer_owned_grid"], 200)
        self.assertEqual(record["service_time_us"]["gpu"], {"min": 256.0, "mean": 264.9, "max": 274.0})
        self.assertEqual(record["gpu_stages_us"]["LDPC decode"], {"min": 96.0, "mean": 100.0, "max": 104.0, "percent": 41.7})
        self.assertEqual(len(record["gpu_stages_us"]), len(verify_pusch_log.STAGES))
        self.assertEqual(record["ldpc_iterations"], {"cpu": 1.0, "gpu": 1.0})
        self.assertEqual(record["reference"]["gpu_mean_us"], 183.9)
        performance = record["performance"]
        self.assertFalse(performance["graded"])
        self.assertTrue(performance["gpu_faster_on_this_run"])
        self.assertAlmostEqual(performance["cpu_mean_delta_percent"], 100 * (304.0 / 334.0 - 1))

    def test_slower_gpu_still_passes_and_records_its_measured_comparison(self):
        slower = (PAPER.replace(GPU_TOTAL, "GPU total:          356.0    380.0    474.0")
                  .replace("Speedup:                  1.15x", "Speedup:                  0.80x"))
        failed, lines, record = grade_transcript(slower, "paper")
        self.assertFalse(failed, "\n".join(lines))
        self.assertIn("NOTE  S1.2 paper performance, speedup: 0.80x here; 1.82x in the paper. The GPU path was slower "
                      "than the CPU path on this run.", lines)
        self.assertIn("NOTE  S1.2 paper performance, GPU mean: 380.0 us here; 183.9 us in the paper (+106.6 percent)",
                      lines)
        self.assertFalse(record["performance"]["gpu_faster_on_this_run"])
        self.assertTrue(record["correctness_reproduced"])

    def test_each_defect_fails_its_own_gate_and_no_other(self):
        defects = [
            ("MCS 27 (table=qam256)", "MCS 20 (table=qam64)", "workload"),
            ("30.0 dB SINR", "25.0 dB SINR", "workload"),
            ("30.0 dB SINR", "nan dB SINR", "workload"),
            ("1 RX port(s)", "4 RX port(s)", "workload"),
            ("1 layer(s)", "4 layer(s)", "workload"),
            ("transform_precoding=off", "transform_precoding=on (default)", "workload"),
            ("rx_device_grid=direct-managed", "rx_device_grid=managed", "workload"),
            ("gpu_output=pinned", "gpu_output=vector", "workload"),
            ("54296 bits (6787 bytes)", "54240 bits (6780 bytes)", "workload"),
            ("Codeblocks: 7", "Codeblocks: 6", "workload"),
            ("5 warmup skipped", "0 warmup skipped", "workload"),
            ("CPU: 200/200 pass (0.0% BLER)", "CPU: 199/200 pass (0.5% BLER)", "cpu_decodes"),
            ("CPU: 200/200 pass (0.0% BLER)", "CPU: 0/200 pass (100.0% BLER)", "cpu_decodes"),
            ("GPU: 200/200 pass (0.0% BLER)", "GPU: 199/200 pass (0.5% BLER)", "gpu_decodes"),
            ("Disagreements: 0", "Disagreements: 1", "decoded_bytes"),
            ("Byte mismatches: 0", "Byte mismatches: 3", "decoded_bytes"),
            ("Output guard failures: 0", "Output guard failures: 1", "decoded_bytes"),
            ("Byte mismatches: 0\n", "", "decoded_bytes"),
            ("resident GPU timing: 200/200", "resident GPU timing: 0/200", "cuda_work"),
            ("resident GPU timing: 200/200", "resident GPU timing: 199/200", "cuda_work"),
            ("GPU producer-owned grid: 200/200", "GPU producer-owned grid: 199/200", "cuda_work"),
            ("GPU immutable owned snapshot: 200/200", "GPU immutable owned snapshot: 199/200", "cuda_work"),
            ("ACTIVE (warmup complete, resident timing: enabled)",
             "NOT ACTIVE (warmup complete, resident timing: missing)", "cuda_work"),
            (LDPC, "LDPC decode:            0.0      0.0      0.0     0.0%", "cuda_work"),
            (LDPC + "\n", "", "cuda_work"),
            (GPU_TOTAL + "\n", "", "service_times"),
            (GPU_TOTAL, "GPU total:          256.0      nan    274.0", "service_times"),
            (GPU_TOTAL, "GPU total:          256.0      inf    274.0", "service_times"),
            (GPU_TOTAL, "GPU total:            0.0      0.0      0.0", "service_times"),
            (GPU_TOTAL, "GPU total:          274.0    264.9    256.0", "service_times"),
            (GPU_TOTAL, "GPU total:       1000000000.0      0.0      0.0", "service_times"),
            ("CPU total:          293.0    304.0    315.0", "CPU total:          293.0   -304.0    315.0", "service_times"),
            ("Speedup:                  1.15x", "Speedup:                  1.82x", "service_times"),
            ("Speedup:                  1.15x", "Speedup:                  -nanx", "service_times"),
            (VERDICT, "*** FAIL: 1 GPU output guard corruptions ***", "native_verdict"),
            (VERDICT, "PASS: Both paths agree on BLER, 0 byte mismatches", "native_verdict"),
            (VERDICT, VERDICT + "\n\n*** FAIL: 2 byte-level mismatches (CRCs agreed but TB data differed) ***",
             "native_verdict"),
            (VERDICT + "\n", "", "native_verdict"),
        ]
        for old, new, gate in defects:
            self.assertEqual(PAPER.count(old), 1, old)
            with self.subTest(old=old, new=new):
                failed, lines, record = grade_transcript(PAPER.replace(old, new), "paper")
                self.assertTrue(failed)
                self.assertEqual(failing(PAPER.replace(old, new)), [gate])
                # A failed gate withholds the comparison with the paper and records why.
                self.assertEqual([line for line in lines if line.startswith("NOTE")],
                                 ["NOTE  S1.2 paper comparison with Table II withheld: a gate above failed"])
                self.assertIsNone(record["performance"])
                self.assertEqual(record["correctness_reproduced"], gate == "service_times")
                self.assertEqual(record["service_times_measured"], gate != "service_times")

    def test_failure_lines_quote_what_the_transcript_holds(self):
        cases = [
            ("CPU: 200/200 pass (0.0% BLER)", "CPU: 199/200 pass (0.5% BLER)",
             "FAIL  S1.2 paper CPU decodes: expected 200/200 CRC passes, found CPU: 199/200 pass (0.5% BLER)"),
            ("Byte mismatches: 0", "Byte mismatches: 3",
             "FAIL  S1.2 paper decoded bytes agree: expected 0 of each, found 0 CRC disagreements, 3 byte mismatches, "
             "0 output guard failures"),
            ("1 RX port(s)", "4 RX port(s)", "FAIL  S1.2 paper workload is the Table II 20 MHz one-layer row: 51 PRB, "
                                             "MCS 27 (qam256), 1 layer, 1 RX port: found Config: 51 PRB, 30.0 dB SINR, "
                                             "MCS 27 (table=qam256), 1 layer(s), 4 RX port(s)"),
            (VERDICT, "*** FAIL: 1 GPU output guard corruptions ***",
             "FAIL  S1.2 paper native test verdict: *** FAIL: 1 GPU output guard corruptions ***"),
            (VERDICT + "\n", "", "FAIL  S1.2 paper native test verdict: no single PASS verdict"),
            (GPU_TOTAL + "\n", "", "FAIL  S1.2 paper service times measured: expected positive, finite, ordered "
                                   "min/mean/max and a consistent ratio, found CPU 304.0 us mean (293.0 to 315.0), "
                                   "GPU absent or not finite, speedup 1.15"),
        ]
        for old, new, expected in cases:
            with self.subTest(expected=expected):
                lines = grade_transcript(PAPER.replace(old, new), "paper")[1]
                self.assertTrue([line for line in lines if line.startswith(expected)], "\n".join(lines))

    def test_missing_gpu_residency_fails_the_cuda_gate_and_withholds_the_comparison(self):
        # The pinned test still decodes and times both paths when the GPU path fell back.
        failed, lines, record = grade_transcript(WITHOUT_GPU, "paper")
        self.assertTrue(failed)
        self.assertEqual(failing(WITHOUT_GPU), ["cuda_work", "native_verdict"])
        self.assertIn("FAIL  S1.2 paper CUDA work completed on the device: expected an active resident pipeline and "
                      "200/200 on every counter, found E2E GPU Pipeline: NOT ACTIVE (warmup complete, resident timing: "
                      "missing); resident timing 0/200; producer-owned grid 0/200; immutable snapshot 0/200; "
                      "LDPC decode time absent", lines)
        self.assertIn("FAIL  S1.2 paper native test verdict: *** FAIL: resident GPU pipeline active for 0/200 "
                      "measured iterations ***", lines)
        self.assertEqual((record["gpu_stages_us"], record["performance"]), ({}, None))
        self.assertFalse(record["correctness_reproduced"])

    def test_zero_iterations_truncated_and_repeated_transcripts_fail_closed(self):
        # A run of zero iterations reports no differing byte. Every gate that counts work refuses it.
        self.assertEqual(failing(ZERO), [gate for gate in GATES if gate != "decoded_bytes"])
        self.assertEqual(failing(""), GATES)
        self.assertEqual(failing(PAPER + PAPER), GATES)
        # A test that found no GPU factory prints one line and exits zero. That line is quoted.
        refusal = "ERROR: GPU PUSCH factory not available. Cannot run E2E pipeline test."
        self.assertEqual(failing(refusal + "\n"), GATES)
        self.assertIn("FAIL  S1.2 paper native test verdict: " + refusal, grade_transcript(refusal + "\n", "paper")[1])
        self.assertEqual(failing("ERROR: Invalid lifting size Z=0\n" + PAPER), [])
        # A run cut off before its latency table has decodes and no timing, device stage or verdict.
        self.assertEqual(failing(PAPER.split("=== Latency")[0]), ["cuda_work", "service_times", "native_verdict"])
        self.assertEqual(failing(PAPER.split("=== Results")[0]), GATES)
        # A second result field cannot be told from a second run.
        self.assertEqual(failing(PAPER + "Output guard failures: 2\n"), ["decoded_bytes"])
        self.assertEqual(failing(PAPER + "CPU: 200/200 pass (0.0% BLER)\n"), ["cpu_decodes"])

    def test_smoke_profile_grades_the_four_rx_diagnostic_without_a_paper_comparison(self):
        failed, lines, record = grade_transcript(SMOKE, "smoke")
        self.assertFalse(failed, "\n".join(lines))
        self.assertEqual([line[:4] for line in lines], ["PASS"] * 7)
        self.assertIn("PASS  S1.2 smoke workload is the four-RX diagnostic: 51 PRB, MCS 10 (qam64), 1 layer, 4 RX ports, "
                      "TBS 9736 bits, 20 measured iterations after 5 warmup", lines)
        self.assertIn("PASS  S1.2 smoke GPU decodes: 20/20 CRC passes", lines)
        self.assertNotIn("reference", record)
        self.assertNotIn("performance", record)
        # Each profile refuses the other's transcript: the workload, the counts and the verdict differ.
        other = ["workload", "cpu_decodes", "gpu_decodes", "cuda_work", "native_verdict"]
        self.assertEqual(failing(SMOKE, "paper"), other)
        self.assertEqual(failing(PAPER, "smoke"), other)
        self.assertEqual(failing(SMOKE.replace("4 RX port(s)", "1 RX port(s)"), "smoke"), ["workload"])
        self.assertEqual(failing(SMOKE.replace("TBS: 9736 bits", "TBS: unknown bits"), "smoke"), ["workload"])


class CommandLineTest(unittest.TestCase):
    def run_main(self, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            try:
                status = verify_pusch_log.main([str(argument) for argument in arguments])
            except SystemExit as stop:
                status = stop.code
        return status, output.getvalue(), errors.getvalue()

    def test_grade_is_printed_and_the_record_is_written_also_when_a_gate_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "pusch.log").write_text(PAPER)
            status, output, _ = self.run_main(folder / "pusch.log", "--profile", "paper", "--json", folder / "a.json")
            self.assertEqual(status, 0, output)
            self.assertEqual(output.splitlines(), grade_transcript(PAPER, "paper")[1])
            record = json.loads((folder / "a.json").read_text())
            self.assertEqual((record["schema"], record["profile"]), ("isac-pusch-grade/v1", "paper"))
            self.assertEqual(record["performance"]["gpu_mean_us"], 264.9)
            (folder / "pusch.log").write_text(WITHOUT_GPU)
            status, output, _ = self.run_main(folder / "pusch.log", "--profile", "paper", "--json", folder / "b.json")
            self.assertEqual(status, 1)
            self.assertIn("FAIL  S1.2 paper CUDA work completed on the device", output)
            record = json.loads((folder / "b.json").read_text())
            self.assertEqual((record["gates"]["cuda_work"], record["performance"]), (False, None))
            self.assertEqual(self.run_main(folder / "pusch.log", "--profile", "paper")[0], 1)
            # A transcript with a stray byte is still read and graded.
            (folder / "pusch.log").write_bytes(b"\xff\n" + PAPER.encode())
            self.assertEqual(self.run_main(folder / "pusch.log", "--profile", "paper")[0], 0)
            status, _, errors = self.run_main(folder / "absent.log", "--profile", "paper")
            self.assertEqual(status, 2)
            self.assertIn("Cannot read", errors)
            status, _, errors = self.run_main(folder / "pusch.log", "--profile", "paper", "--json", folder / "no/c.json")
            self.assertEqual(status, 2)
            self.assertIn("Cannot write", errors)
            for arguments in ((folder / "pusch.log",), (folder / "pusch.log", "--profile", "table"),
                              ("--profile", "paper")):
                self.assertEqual(self.run_main(*arguments)[0], 2)


if __name__ == "__main__":
    unittest.main()
