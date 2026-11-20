"""Grade one transcript of the pinned OCUDU pusch_e2e_pipeline_test.

Each requirement of Sprint 1.2 is its own gate: the workload, the CPU decodes, the GPU decodes,
the agreement of the decoded bytes, the completed CUDA work, the measured service times and the
verdict the test printed. The comparison with the published latencies is recorded in NOTE lines
and in the JSON record. It never passes or fails a gate.
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys

NUMBER = r"(\d+(?:\.\d+)?)"
# Row "20 MHz 1L" of Table II in Pennybacker et al., arXiv:2608.04338v1, and the statements about
# that table in Section VI. The paper measured branch wg/nvcuda_accel_01 at 3c89cdee.
TABLE_II = {
    "comparison": "Pennybacker et al., arXiv:2608.04338v1, Table II, 20 MHz 1L",
    "revision": "wg/nvcuda_accel_01 at 3c89cdee",
    "tbs_kbit": 54.3, "cpu_mean_us": 334.0, "gpu_mean_us": 183.9, "speedup": 1.82,
    "cpu_crc_pass": 200, "gpu_crc_pass": 200, "measured": 200, "cpu_gpu_byte_mismatches": 0,
    "largest_sinr_delta_db": 0.02,
}
# Settings of the paper profile that Table II and Section VI-A do not state.
LOCAL_SETTINGS = "30 dB SINR, 5 warmup iterations, 1 RX port, DM-RS symbols 2 and 11"
# The header the pinned test prints for each profile that scripts/native_ocudu.py runs.
PROFILES = {
    "paper": {
        "label": "S1.2 paper", "iterations": 200, "json": "table-ii.json",
        "workload": "the Table II 20 MHz one-layer row: 51 PRB, MCS 27 (qam256), 1 layer, 1 RX port",
        "config": "Config: 51 PRB, 30.0 dB SINR, MCS 27 (table=qam256), 1 layer(s), 1 RX port(s), "
                  "DMRS Type1 [2,11], transform_precoding=off, 200 iterations, 5 warmup, "
                  "rx_device_grid=direct-managed, gpu_output=pinned",
        "tbs": "TBS: 54296 bits (6787 bytes), Base Graph: BG1, Codeblocks: 7, Modulation: 256QAM",
    },
    "smoke": {
        "label": "S1.2 smoke", "iterations": 20, "json": "smoke.json",
        "workload": "the four-RX diagnostic: 51 PRB, MCS 10 (qam64), 1 layer, 4 RX ports",
        "config": "Config: 51 PRB, 30.0 dB SINR, MCS 10 (table=qam64), 1 layer(s), 4 RX port(s), "
                  "DMRS Type1 [2,11], transform_precoding=off, 20 iterations, 5 warmup, "
                  "rx_device_grid=direct-managed, gpu_output=vector",
        "tbs": None,
    },
}
WARMUP = 5
WITHHELD = "comparison with Table II withheld: a gate above failed"
ACTIVE = "E2E GPU Pipeline: ACTIVE (warmup complete, resident timing: enabled)"
# Stages of the resident GPU pipeline that the test times and prints as "name: min mean max share%".
STAGES = ("Ch estimate", "Setup", "Grid staging", "Demod sync", "Deinterleave", "Rate dematch",
          "LDPC decode", "CRC check", "D2H transfer", "Extract bits", "Join/notify", "SINR readback",
          "Decoder cfg", "RM cfg", "Completion wait*", "TB host copy", "Avg iters query")
# Stage the decoder times with CUDA events on its device stream.
DEVICE_STAGE = "LDPC decode"


class Transcript:
    """Lines of one run. A field is valid only when exactly one line carries its prefix."""

    def __init__(self, text):
        self.lines = [line.strip() for line in text.splitlines()]

    def line(self, prefix):
        found = [line for line in self.lines if line.startswith(prefix)]
        return found[0] if len(found) == 1 else None

    def match(self, prefix, pattern):
        line = self.line(prefix)
        return re.fullmatch(pattern, line) if line is not None else None

    def integers(self, prefix, pattern):
        found = self.match(prefix, pattern)
        return tuple(int(value) for value in found.groups()) if found else None

    def numbers(self, prefix, pattern):
        """Finite numbers of one field, or None. The pattern admits digits only, never nan or inf."""
        found = self.match(prefix, pattern)
        values = tuple(float(value) for value in found.groups()) if found else None
        return values if values and all(math.isfinite(value) for value in values) else None


def shown(value, absent="absent"):
    """Text of an observed field for a gate line."""
    if value is None:
        return absent
    if isinstance(value, tuple):
        return "/".join(str(item) for item in value)
    return str(value)


def grade_transcript(text, profile):
    """Grade one transcript. Return (failed, lines, record) with PASS, FAIL and NOTE lines."""
    spec = PROFILES[profile]
    label, count = spec["label"], spec["iterations"]
    log = Transcript(text)
    lines, gates = [], {}

    def check(name, passed, passing, failing):
        gates[name] = bool(passed)
        lines.append(f"PASS  {label} {passing}" if passed else f"FAIL  {label} {failing}")

    # Workload: the header must be the one this profile requests, for the measured count.
    config = log.line("Config:")
    tbs = log.match("TBS:", r"TBS: (\d+) bits \((\d+) bytes\), Base Graph: (BG[12]), "
                            r"Codeblocks: (\d+), Modulation: (\S+)")
    header = log.integers("=== Results (", r"=== Results \((\d+) iterations, (\d+) warmup skipped\) ===")
    same_tbs = tbs is not None and (spec["tbs"] is None or tbs.group() == spec["tbs"])
    check("workload", config == spec["config"] and same_tbs and header == (count, WARMUP),
          f"workload is {spec['workload']}, TBS {tbs.group(1) if tbs else '?'} bits, "
          f"{count} measured iterations after {WARMUP} warmup",
          f"workload is {spec['workload']}: found {shown(config, 'no single Config line')}; "
          f"{shown(tbs.group() if tbs else None, 'no single valid TBS line')}; "
          f"results for {shown(header, 'no single Results header')} iterations/warmup")

    # Decodes. Every measured iteration must pass its CRC on each path.
    cpu = log.integers("CPU:", r"CPU: (\d+)/(\d+) pass \(\S+% BLER\)")
    gpu = log.integers("GPU:", r"GPU: (\d+)/(\d+) pass \(\S+% BLER\)\s+\[resident GPU timing: (\d+)/(\d+)\]")
    check("cpu_decodes", cpu == (count, count), f"CPU decodes: {count}/{count} CRC passes",
          f"CPU decodes: expected {count}/{count} CRC passes, found {shown(log.line('CPU:'))}")
    check("gpu_decodes", gpu is not None and gpu[:2] == (count, count),
          f"GPU decodes: {count}/{count} CRC passes",
          f"GPU decodes: expected {count}/{count} CRC passes, found {shown(log.line('GPU:'))}")

    # Agreement of the two decoded outputs, and the guard bytes behind the GPU output.
    fields = (("crc_disagreements", "Disagreements:", "CRC disagreements"),
              ("cpu_gpu_byte_mismatches", "Byte mismatches:", "byte mismatches"),
              ("output_guard_failures", "Output guard failures:", "output guard failures"))
    differing = {name: log.integers(prefix, prefix + r" (\d+)") for name, prefix, _ in fields}
    check("decoded_bytes", all(value == (0,) for value in differing.values()),
          "decoded bytes agree: 0 CRC disagreements, 0 byte mismatches, 0 output guard failures",
          "decoded bytes agree: expected 0 of each, found " + ", ".join(
              f"{shown(differing[name])} {words}" for name, _, words in fields))

    # Completed CUDA work: the resident path served every measured iteration from the
    # producer-owned device grid, and the decoder timed a device stage with CUDA events.
    active = log.line("E2E GPU Pipeline:")
    grid = log.integers("GPU producer-owned grid:", r"GPU producer-owned grid: (\d+)/(\d+) measured iterations")
    owned = log.integers("GPU immutable owned snapshot:",
                         r"GPU immutable owned snapshot: (\d+)/(\d+) measured iterations")
    stages = {}
    for name in STAGES:
        values = log.numbers(name + ":", re.escape(name) + ":" + (r"\s+" + NUMBER) * 4 + "%")
        if values:
            stages[name] = dict(zip(("min", "mean", "max", "percent"), values))
    device = stages.get(DEVICE_STAGE, {}).get("mean")
    resident = gpu[2:] if gpu else None
    check("cuda_work", active == ACTIVE and resident == (count, count) and grid == (count, count)
          and owned == (count, count) and device is not None and device > 0,
          f"CUDA work completed on the device: resident pipeline active after warmup, resident timing "
          f"{count}/{count}, producer-owned grid {count}/{count}, immutable snapshot {count}/{count}, "
          f"{DEVICE_STAGE} {device} us mean by CUDA events",
          f"CUDA work completed on the device: expected an active resident pipeline and {count}/{count} "
          f"on every counter, found {shown(active, 'no single E2E GPU Pipeline line')}; resident timing "
          f"{shown(resident)}; producer-owned grid {shown(grid)}; immutable snapshot {shown(owned)}; "
          f"{DEVICE_STAGE} time {shown(device)}")

    # Service times: positive, finite and ordered, with a printed ratio that agrees with the means.
    times = {name: log.numbers(name + " total:", name + " total:" + (r"\s+" + NUMBER) * 3) for name in ("CPU", "GPU")}
    speedup = log.numbers("Speedup:", r"Speedup:\s+" + NUMBER + "x")
    ordered = all(value is not None and 0 < value[0] <= value[1] <= value[2] for value in times.values())
    consistent = False
    if ordered and speedup:
        cpu_mean, gpu_mean = times["CPU"][1], times["GPU"][1]
        # Both means are printed to 0.1 us and the ratio to 0.01.
        consistent = ((cpu_mean - .05) / (gpu_mean + .05) - .0051 <= speedup[0]
                      <= (cpu_mean + .05) / max(gpu_mean - .05, .001) + .0051)
    described = ", ".join(
        f"{name} {value[1]} us mean ({value[0]} to {value[2]})" if value else f"{name} absent or not finite"
        for name, value in times.items())
    check("service_times", ordered and consistent,
          f"service times measured over {count} iterations: {described}, "
          f"speedup {speedup[0] if speedup else 0:.2f}x",
          f"service times measured: expected positive, finite, ordered min/mean/max and a consistent "
          f"ratio, found {described}, speedup {shown(speedup)}")

    # The verdict the test printed. It prints at most one, and any FAIL verdict rejects the run.
    # Without a verdict, an ERROR line of the test says why it never measured.
    refused = [line for line in log.lines if "*** FAIL:" in line]
    stopped = [line for line in log.lines if line.startswith("ERROR:")]
    verdict = log.line("PASS:")
    found = (refused or [verdict or (stopped or [None])[0]])[0]
    check("native_verdict", not refused and verdict == f"PASS: Both paths decoded all {count} iterations, "
                                                       "0 byte mismatches",
          f"native test verdict: {verdict}", f"native test verdict: {shown(found, 'no single PASS verdict')}")

    sinr = log.match("SINR (dB):", r"SINR \(dB\):\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S.*)")
    iterations = {name.lower(): log.numbers(name + " avg:", name + " avg: " + NUMBER) for name in ("CPU", "GPU")}
    failed = not all(gates.values())
    correct = all(passed for name, passed in gates.items() if name != "service_times")
    record = {
        "schema": "isac-pusch-grade/v1", "profile": profile, "gates": gates,
        "correctness_reproduced": correct, "service_times_measured": gates["service_times"],
        "workload": {"config": config, "tbs_bits": int(tbs.group(1)) if tbs else None,
                     "tbs_bytes": int(tbs.group(2)) if tbs else None,
                     "base_graph": tbs.group(3) if tbs else None,
                     "codeblocks": int(tbs.group(4)) if tbs else None,
                     "modulation": tbs.group(5) if tbs else None,
                     "measured": header[0] if header else None, "warmup": header[1] if header else None},
        "decodes": {"cpu_crc_pass": cpu[0] if cpu else None, "gpu_crc_pass": gpu[0] if gpu else None,
                    "resident_gpu_timing": resident[0] if resident else None,
                    "producer_owned_grid": grid[0] if grid else None,
                    "immutable_owned_snapshot": owned[0] if owned else None,
                    "pipeline_after_warmup": active, "verdict": verdict,
                    **{name: value[0] if value else None for name, value in differing.items()}},
        "service_time_us": {name.lower(): dict(zip(("min", "mean", "max"), value)) if value else None
                            for name, value in times.items()},
        "reported_speedup": speedup[0] if speedup else None,
        "gpu_stages_us": stages,
        "ldpc_iterations": {name: value[0] if value else None for name, value in iterations.items()},
        "sinr_db": dict(zip(("cpu", "gpu", "delta", "gpu_populated"), sinr.groups())) if sinr else None,
    }
    if profile != "paper":
        return failed, lines, record

    # The paper comparison is a record. It is withheld when a gate failed, because a time
    # measured on a path that fell back or decoded wrongly compares nothing.
    record["reference"] = TABLE_II
    record["local_settings"] = LOCAL_SETTINGS
    record["performance"] = None
    if failed:
        lines.append(f"NOTE  {label} {WITHHELD}")
        return failed, lines, record
    cpu_mean, gpu_mean = times["CPU"][1], times["GPU"][1]
    faster = gpu_mean < cpu_mean
    change = {name: 100 * (mean / TABLE_II[f"{name}_mean_us"] - 1)
              for name, mean in (("cpu", cpu_mean), ("gpu", gpu_mean))}
    record["performance"] = {
        "graded": False, "cpu_mean_us": cpu_mean, "gpu_mean_us": gpu_mean, "reported_speedup": speedup[0],
        "speedup_from_rounded_means": cpu_mean / gpu_mean, "cpu_mean_delta_percent": change["cpu"],
        "gpu_mean_delta_percent": change["gpu"], "gpu_faster_on_this_run": faster,
    }
    lines.append(f"NOTE  {label} correctness reproduced: {count}/{count} CPU and {count}/{count} GPU decodes "
                 "with 0 differing bytes, as Table II reports")
    for name, mean in (("CPU", cpu_mean), ("GPU", gpu_mean)):
        lines.append(f"NOTE  {label} performance, {name} mean: {mean} us here; "
                     f"{TABLE_II[name.lower() + '_mean_us']} us in the paper ({change[name.lower()]:+.1f} percent)")
    lines.append(f"NOTE  {label} performance, speedup: {speedup[0]:.2f}x here; {TABLE_II['speedup']}x in the paper. "
                 f"The GPU path was {'faster' if faster else 'slower'} than the CPU path on this run.")
    if sinr:
        lines.append(f"NOTE  {label} CSI parity, SINR: CPU {sinr.group(1)} dB, GPU {sinr.group(2)} dB, "
                     f"delta {sinr.group(3)} dB, GPU populated: {sinr.group(4)}. The paper reports deltas "
                     f"no larger than {TABLE_II['largest_sinr_delta_db']} dB.")
    lines.append(f"NOTE  {label} settings that Table II does not state: {LOCAL_SETTINGS}")
    return failed, lines, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="combined stdout and stderr of one upstream run")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True,
                        help="paper: the Table II row with 200 measured frames. smoke: the four-RX diagnostic.")
    parser.add_argument("--json", type=Path, help="write the machine-readable grade, also when a gate fails")
    args = parser.parse_args(argv)
    try:
        transcript = args.log.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        parser.exit(2, f"Cannot read {args.log}: {error}\n")
    failed, lines, record = grade_transcript(transcript, args.profile)
    if args.json is not None:
        try:
            args.json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        except OSError as error:
            parser.exit(2, f"Cannot write {args.json}: {error}\n")
    print("\n".join(lines))
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
