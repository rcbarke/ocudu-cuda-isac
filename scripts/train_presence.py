#!/usr/bin/env python3
"""Train, calibrate and freeze one presence head (Sprint 3.4, selector presence_head).

    python scripts/train_presence.py --training training.jsonl --validation validation.jsonl \\
        --spec spec.json --config configs/presence/dd_linear.json --output-dir build/presence-001

The two sample files hold presence-sample/v1 rows: the vector of one complete window with the
split, the scene, the matched pair and the label that an evaluator joined to it. The script never
reads a record, a feature output or an episode, and it takes no third file: a held-out sample has
no way in. In this order it

  1. reads both files and refuses a split that was not made before its windows: a row in the
     wrong file, a scene, a pair, an episode key or a window in both, two specifications;
  2. writes training_manifest.json and validation_manifest.json, with the hash of each file;
  3. fits the normalizer and the weights on the training rows, and nothing else on them;
  4. takes the logits of the validation rows with the arithmetic of the deployed head and sets
     the threshold on the target-absent ones;
  5. writes the artifact, artifact/head.json and artifact/weights.f32, reads it back and requires
     the same logits within 1e-6;
  6. writes freeze.json, which names the hashes of the artifact and of both manifests.

The output directory must be new. Exit status 2 for a refused input, with one line.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST_SCHEMA = "presence-manifest/v1"
FREEZE_SCHEMA = "presence-freeze/v1"
FIT_SCHEMA = "presence-fit/v1"
ARTIFACT_DIR = "artifact"
RELOAD_TOLERANCE = 1e-6
USED_FOR = {"training": ["normalizer", "weights"], "validation": ["threshold, from its target-absent windows"]}
TEST_ACCESS = ("none: this script read the two sample files that its manifests name and no record, feature output, "
               "label or episode beside them")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha256(values):
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode("ascii")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return path


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest_of(role, path, rows, facts):
    """What one sample file held, by identifiers and counts: a presence-manifest/v1 object."""
    windows, keys, pairs = [row["window"] for row in rows], sorted({row["episode"] for row in rows}), \
        sorted({row["pair"] for row in rows})
    return {"schema": MANIFEST_SCHEMA, "role": role, "used_for": USED_FOR[role],
            "samples": {"file": str(Path(path).resolve()), "sha256": sha256_file(path), "rows": len(rows)},
            "spec_sha256": facts["spec_sha256"], "vector": facts["vector"], "windows": len(rows),
            "target_present": sum(row["label"] for row in rows), "target_absent": sum(1 - row["label"] for row in rows),
            "scenes": facts["scenes"][role], "pairs": len(pairs), "pairs_sha256": ids_sha256(pairs),
            "episode_keys": keys, "episode_keys_sha256": ids_sha256(keys), "window_ids": sorted(windows),
            "window_ids_sha256": ids_sha256(windows), "created_utc": utc_now()}


def train(training_path, validation_path, spec, settings, output_dir):
    """Run the six steps of this script. Returns the freeze record."""
    from ocudu_cuda_isac.sensing import features, presence
    from ocudu_cuda_isac.sensing.presence import PresenceError

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise PresenceError(f"{output_dir} already exists. A training run is never overwritten.")
    features.check_spec(spec)
    presence.check_settings(settings)
    training, validation = presence.read_samples(training_path), presence.read_samples(validation_path)
    facts = presence.check_split(training, validation)
    if facts["spec_sha256"] != features.spec_sha256(spec):
        raise PresenceError(f"the samples were computed with the specification {facts['spec_sha256']}, and "
                            f"{spec['spec_id']} has the hash {features.spec_sha256(spec)}")
    vector = facts["vector"]
    columns = presence.select_columns(vector, settings["columns"])
    output_dir.mkdir(parents=True)
    manifests = {"training": manifest_of("training", training_path, training, facts),
                 "validation": manifest_of("validation", validation_path, validation, facts)}
    for role, manifest in manifests.items():
        write_json(output_dir / f"{role}_manifest.json", manifest)

    normalizer = presence.fit_normalizer(training, vector, columns)
    weights, bias, history = presence.fit_head(training, vector, columns, normalizer, settings["l2"],
                                               settings["max_iterations"], settings["min_per_class"])
    logits = {"training": presence.logits_of(training, vector, columns, normalizer, weights, bias),
              "validation": presence.logits_of(validation, vector, columns, normalizer, weights, bias)}
    threshold = presence.calibrate_threshold(validation, logits["validation"], settings["threshold"]["target_false_alarm"],
                                             settings["threshold"]["min_negatives"])
    head = presence.write_artifact(output_dir / ARTIFACT_DIR, settings["head_id"], spec["spec_id"], facts["spec_sha256"],
                                   vector, columns, normalizer, weights, bias, threshold,
                                   features_entry=spec["steps"]["features"])
    # The artifact as it will be deployed: read back, on the CPU, on every row of both files.
    loaded = presence.Head(presence.open_artifact(output_dir / ARTIFACT_DIR, head["sha256"]), "cpu")
    reload = 0.0
    for role, rows in (("training", training), ("validation", validation)):
        for row, logit in zip(rows, logits[role]):
            reload = max(reload, abs(loaded.score(row["names"], row["values"])["logit"] - logit))
    if reload > RELOAD_TOLERANCE:
        raise PresenceError(f"the artifact read back scores a row {reload:.3e} from the fit, above {RELOAD_TOLERANCE:g}")

    def rates(role, rows):
        positives = [logit for row, logit in zip(rows, logits[role]) if row["label"] == 1]
        negatives = [logit for row, logit in zip(rows, logits[role]) if row["label"] == 0]
        wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
        return {"target_present": len(positives), "detected": sum(value > threshold["logit"] for value in positives),
                "target_absent": len(negatives), "false_alarms": sum(value > threshold["logit"] for value in negatives),
                "area_under_curve": wins / (len(positives) * len(negatives)) if positives and negatives else None}

    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
             for path in sorted((output_dir / ARTIFACT_DIR).iterdir())}
    fit = {"schema": FIT_SCHEMA, "history": history, "normalizer_constant_columns": normalizer["constant"],
           "weights": dict(zip(columns, weights)), "bias": bias, "graded": False,
           "at_the_frozen_threshold": {"training": rates("training", training),
                                       "validation": rates("validation", validation)},
           "note": "the counts of the training and of the validation split are those of the samples the head and its "
                   "threshold were fitted on: they describe the fit and are no result"}
    write_json(output_dir / "fit.json", fit)
    freeze = {"schema": FREEZE_SCHEMA, "head_id": settings["head_id"],
              "artifact": {"directory": ARTIFACT_DIR, "sha256": head["sha256"],
                           "weights_sha256": head["weights"]["sha256"],
                           "normalizer_sha256": head["normalizer"]["sha256"], "files": files},
              "input": {"spec_id": spec["spec_id"], "spec_sha256": facts["spec_sha256"], "columns": columns},
              "threshold": threshold, "settings": settings,
              "settings_sha256": hashlib.sha256(presence.canonical(settings)).hexdigest(),
              "manifests": {role: {"file": f"{role}_manifest.json",
                                   "sha256": sha256_file(output_dir / f"{role}_manifest.json"),
                                   "samples_sha256": manifests[role]["samples"]["sha256"],
                                   "windows": manifests[role]["windows"]} for role in manifests},
              "fit": {"file": "fit.json", "sha256": sha256_file(output_dir / "fit.json"),
                      "initial_loss": history["initial_loss"], "final_loss": history["final_loss"],
                      "evaluations": history["evaluations"]},
              "reload": {"largest_logit_difference": reload, "tolerance": RELOAD_TOLERANCE, "device": "cpu",
                         "rows": len(training) + len(validation)},
              "test_access": TEST_ACCESS, "frozen_utc": utc_now(), "frozen_monotonic_ns": time.monotonic_ns(),
              "software": {"python": platform.python_version(), "torch": presence.torch.__version__,
                           "numpy": presence.np.__version__, "machine": platform.machine(),
                           "script_sha256": sha256_file(__file__),
                           "module_sha256": sha256_file(presence.__file__)}}
    write_json(output_dir / "freeze.json", freeze)
    return freeze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training", required=True, type=Path, help="presence-sample/v1 rows of the training split")
    parser.add_argument("--validation", required=True, type=Path, help="presence-sample/v1 rows of the validation split")
    parser.add_argument("--spec", required=True, type=Path, help="pilot-features-spec/v1 file the samples were computed with")
    parser.add_argument("--config", required=True, type=Path, help="presence-train-config/v1 file")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the artifact and its records")
    args = parser.parse_args(argv)
    try:
        from ocudu_cuda_isac.sensing import features, presence
    except ImportError as error:
        parser.exit(2, f"train_presence: the head needs NumPy and PyTorch in this interpreter: {error}\n")
    try:
        try:
            settings = json.loads(args.config.read_text(encoding="ascii"))
        except (OSError, ValueError) as error:
            raise presence.PresenceError(f"cannot read the training configuration: {error}") from None
        freeze = train(args.training, args.validation, features.load_spec(args.spec), settings, args.output_dir)
    except (presence.PresenceError, features.FeaturesError) as error:
        parser.exit(2, f"train_presence: {error}\n")
    threshold = freeze["threshold"]
    print(f"train_presence: head {freeze['head_id']} {freeze['artifact']['sha256'][:12]} on {len(freeze['input']['columns'])} "
          f"columns from {freeze['manifests']['training']['windows']} training windows; threshold logit "
          f"{threshold['logit']:.4f} from {threshold['negatives']} validation negatives, {threshold['exceeding']} above "
          f"it of {threshold['allowed']} allowed; frozen {freeze['frozen_utc']}")
    print(args.output_dir / "freeze.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
