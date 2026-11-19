"""Verify a single high-SINR, direct-managed OCUDU PUSCH smoke transcript."""

import argparse
import json
import math
from pathlib import Path
import re
import sys


def verify_transcript(transcript: str, iterations: int) -> list[str]:
    """Return any failed checks against the summary printed by the pinned upstream test."""
    if iterations < 1:
        raise ValueError("expected iterations must be positive")

    lines = [line.strip() for line in transcript.splitlines()]
    issues = []
    if any("*** FAIL:" in line for line in lines):
        issues.append("the test printed a FAIL verdict")
    count = f"{iterations}/{iterations}"
    expected = {
        "=== Results (": rf"=== Results \({iterations} iterations, \d+ warmup skipped\) ===",
        "CPU:": rf"CPU: {count} pass \(0\.0% BLER\)",
        "GPU:": rf"GPU: {count} pass \(0\.0% BLER\)\s+\[resident GPU timing: {count}\]",
        "GPU producer-owned grid:": rf"GPU producer-owned grid: {count} measured iterations",
        "GPU immutable owned snapshot:": rf"GPU immutable owned snapshot: {count} measured iterations",
        "Disagreements:": r"Disagreements: 0",
        "Byte mismatches:": r"Byte mismatches: 0",
        "Output guard failures:": r"Output guard failures: 0",
        "PASS:": rf"PASS: Both paths decoded all {iterations} iterations, 0 byte mismatches",
    }
    for prefix, pattern in expected.items():
        matches = [line for line in lines if line.startswith(prefix)]
        if len(matches) != 1 or re.fullmatch(pattern, matches[0]) is None:
            issues.append(f"missing, repeated or failing summary field {prefix!r}")
    return issues


def compare_table_ii(transcript: str) -> dict:
    """Validate workload identity and compare means with the published 20 MHz row."""
    issues = verify_transcript(transcript, 200)
    if issues:
        raise ValueError(", ".join(issues))
    lines = [line.strip() for line in transcript.splitlines()]

    def field(prefix: str, pattern: str) -> re.Match:
        selected = [line for line in lines if line.startswith(prefix)]
        if len(selected) == 1:
            match = re.fullmatch(pattern, selected[0])
            if match:
                return match
        raise ValueError(f"Expected one valid paper-profile field {prefix!r}")

    field("Config:", r"Config: 51 PRB, 30\.0 dB SINR, MCS 27 \(table=qam256\), "
          r"1 layer\(s\), 1 RX port\(s\), DMRS Type1 \[2,11\], transform_precoding=off, "
          r"200 iterations, 5 warmup, rx_device_grid=direct-managed, gpu_output=pinned")
    field("TBS:", r"TBS: 54296 bits \(6787 bytes\), Base Graph: BG1, Codeblocks: 7, Modulation: 256QAM")
    field("=== Results (", r"=== Results \(200 iterations, 5 warmup skipped\) ===")
    numeric = r"(\d+(?:\.\d+)?)"
    means = {}
    for name in ["CPU", "GPU"]:
        prefix = name + " total:"
        timing = field(prefix, prefix + (r"\s+" + numeric) * 3)
        low, mean, high = map(float, timing.groups())
        if not all(map(math.isfinite, [low, mean, high])) or not 0 < low <= mean <= high:
            raise ValueError(f"Expected positive, ordered completed-work timings for {name}")
        means[name] = mean
    ratio = means["CPU"] / means["GPU"]
    printed = float(field("Speedup:", r"Speedup:\s+" + numeric + "x").group(1))
    # Both means were printed to the nearest 0.1 us and the upstream ratio to 0.01x.
    if not (means["CPU"] - .05) / (means["GPU"] + .05) - .0051 <= printed <= (
            (means["CPU"] + .05) / max(means["GPU"] - .05, .001) + .0051):
        raise ValueError("Printed speedup is inconsistent with CPU/GPU means")
    return {
        "comparison": "Pennybacker et al., arXiv:2608.04338v1, Table II, 20 MHz 1L",
        "paper_commit": "3c89cdee",
        "reference": {"cpu_mean_us": 334.0, "gpu_mean_us": 183.9, "speedup": 1.82,
                      "cpu_crc_pass": 200, "gpu_crc_pass": 200, "measured": 200},
        "observed": {"cpu_mean_us": means["CPU"], "gpu_mean_us": means["GPU"],
                     "reported_speedup": printed, "speedup_from_rounded_means": ratio,
                     "cpu_crc_pass": 200, "gpu_crc_pass": 200, "measured": 200,
                     "tbs_bits": 54296, "cpu_gpu_byte_mismatches": 0},
        "cpu_mean_delta_percent": 100 * (means["CPU"] / 334.0 - 1),
        "gpu_mean_delta_percent": 100 * (means["GPU"] / 183.9 - 1),
        "gpu_faster_on_this_run": means["GPU"] < means["CPU"],
        "correctness_matches_table_row": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="combined stdout and stderr of one upstream run")
    parser.add_argument("--iterations", type=int, required=True, help="measured iterations, excluding warmup")
    parser.add_argument("--paper-json", type=Path, help="check the Table II profile and write its comparison")
    args = parser.parse_args()
    if args.paper_json is not None and args.iterations != 200:
        parser.error("--paper-json requires 200 measured frames")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    try:
        transcript = args.log.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        parser.exit(2, f"Cannot read {args.log}: {error}\n")

    issues = verify_transcript(transcript, args.iterations)
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return 1
    if args.paper_json is not None:
        try:
            comparison = compare_table_ii(transcript)
            args.paper_json.write_text(json.dumps(comparison, indent=2) + "\n")
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
            return 1
        print(f"Table II correctness reproduced. See {args.paper_json} for the measured latency comparison.")
    print(f"Verified {args.iterations} measured PUSCH decodes on both paths and all direct-grid checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
