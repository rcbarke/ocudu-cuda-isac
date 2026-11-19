"""Run the synthetic replay smoke check. All stated times come from an input trace."""

import argparse
import json
from pathlib import Path

from .replay import join_timing, materialize


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        episode = json.loads(args.episode.read_text(encoding="utf-8"))
        trace = json.loads(args.trace.read_text(encoding="utf-8"))
        ingress, oracle = materialize(episode)
        summary = join_timing(ingress, trace)
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(2, f"Replay input error: {exc}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "receiver_ingress.jsonl", ingress)
    write_jsonl(args.output_dir / "offline_oracle.jsonl", oracle)
    (args.output_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scheduled_updates": len(ingress), **summary["counts"]}))
    print(f"Fixture outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
