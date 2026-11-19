"""Verify a single high-SINR, direct-managed OCUDU PUSCH smoke transcript."""

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="combined stdout and stderr of one upstream run")
    parser.add_argument("--iterations", type=int, required=True, help="measured iterations, excluding warmup")
    args = parser.parse_args()
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
    print(f"Verified {args.iterations} measured PUSCH decodes on both paths and all direct-grid checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
