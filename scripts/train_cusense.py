#!/usr/bin/env python3
"""Fit, calibrate and freeze one cuSense-like model (Sprint 3.5, selector cusense_frontend).

    python scripts/train_cusense.py --training training.jsonl --validation validation.jsonl \\
        --spec spec.json --grid grid.json --config training.json --output-dir build/cusense-001

A sample set is three files with one stem: the rows (cusense-sample/v1, one window per line, with
the split, the scene, the matched pair, the label and the position that an evaluator joined to
it) and the mean and the count of every window as complex64 and uint8. The script reads the two
sample sets, the window specification, the position grid and the training configuration. It
never reads a record, a decision or an episode, and it takes no third sample set: a held-out
window has no way in. In this order it

  1. reads both sets, verifies every tensor against the hash its row names, and refuses a split
     that was not made before its windows: a row in the wrong file, a scene, a pair, an episode
     key or a window in both, two specifications;
  2. writes training_manifest.json and validation_manifest.json, with the hash of every file;
  3. fits the background template on the target-absent training windows, the mean and the
     deviation of the z-score on every training window, and the model on the training inputs;
  4. scores every window with the arithmetic of the deployed model, one window at a time on the
     CPU, and sets the presence threshold on the target-absent validation windows;
  5. writes the artifact of four files, reads it back and requires the same scores;
  6. writes fit.json and freeze.json, which names the hashes of the artifact and of both
     manifests.

The output directory must be new. Exit status 2 for a refused input, with one line.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST_SCHEMA = "cusense-manifest/v1"
FREEZE_SCHEMA = "cusense-freeze/v1"
FIT_SCHEMA = "cusense-fit/v1"
ARTIFACT_DIR = "artifact"
RELOAD_TOLERANCE = 1e-5
RELOAD_TRAINING_ROWS = 512
USED_FOR = {"training": ["template, from its target-absent windows", "normalizer", "weights"],
            "validation": ["threshold, from its target-absent windows", "monitor of the fit, which selects nothing"]}
TEST_ACCESS = ("none: this script read the two sample sets that its manifests name and no record, decision, label or "
               "episode beside them")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha256(values):
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode("ascii")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return path


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def area_under_curve(logits, labels):
    """The probability that a target-present window scores above a target-absent one. Ties count one half."""
    positives = [value for value, label in zip(logits, labels) if label == 1]
    negatives = [value for value, label in zip(logits, labels) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def error_statistics(values):
    """Mean, median and 90th percentile by nearest rank, and root mean square, of position errors in metres."""
    if not values:
        return {"eligible": 0, "mean_m": None, "median_m": None, "p90_m": None, "rmse_m": None, "max_m": None}
    ordered = sorted(values)

    def rank(share):
        return ordered[min(max(math.ceil(share * len(ordered)), 1), len(ordered)) - 1]

    return {"eligible": len(values), "mean_m": sum(values) / len(values), "median_m": rank(0.5), "p90_m": rank(0.9),
            "rmse_m": math.sqrt(sum(value * value for value in values) / len(values)), "max_m": max(values)}


def manifest_of(role, samples, facts):
    """What one sample set held, by identifiers and counts: a cusense-manifest/v1 object."""
    rows = samples.rows
    windows, keys, pairs = [row["window"] for row in rows], sorted({row["episode"] for row in rows}), \
        sorted({row["pair"] for row in rows})
    return {"schema": MANIFEST_SCHEMA, "role": role, "used_for": USED_FOR[role],
            "samples": {"file": str(Path(samples.path).resolve()), "sha256": samples.files[Path(samples.path).name]["sha256"],
                        "rows": len(rows), "files": samples.files},
            "spec_sha256": facts["spec_sha256"], "shape": facts["shape"], "elements": facts["elements"],
            "producer_steps": facts["producer_steps"], "windows": len(rows),
            "target_present": sum(row["label"] for row in rows), "target_absent": sum(1 - row["label"] for row in rows),
            "scenes": facts["scenes"][role], "pairs": len(pairs), "pairs_sha256": ids_sha256(pairs),
            "episode_keys": keys, "episode_keys_sha256": ids_sha256(keys), "window_ids": sorted(windows),
            "window_ids_sha256": ids_sha256(windows), "created_utc": utc_now()}


def train(training_path, validation_path, spec, geometry, settings, output_dir):
    """Run the six steps of this script. Returns the freeze record."""
    from ocudu_cuda_isac.sensing import cusense, presence
    from ocudu_cuda_isac.sensing.cusense import CuSenseError
    from ocudu_cuda_isac.sensing.models import cusense_cnn

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise CuSenseError(f"{output_dir} already exists. A training run is never overwritten.")
    cusense.check_spec(spec)
    cusense.check_settings(settings)
    grid = cusense.grid_record(geometry)
    sets = {"training": cusense.read_samples(training_path), "validation": cusense.read_samples(validation_path)}
    training, validation = sets["training"].rows, sets["validation"].rows
    facts = cusense.check_split(training, validation)
    if facts["spec_sha256"] != cusense.spec_sha256(spec) or facts["shape"] != cusense.shape_of(spec):
        raise CuSenseError(f"the samples were computed with the specification {facts['spec_sha256']} on windows of "
                           f"the shape {facts['shape']}, and {spec['spec_id']} has the hash "
                           f"{cusense.spec_sha256(spec)} and the shape {cusense.shape_of(spec)}")
    output_dir.mkdir(parents=True)
    manifests = {role: manifest_of(role, sets[role], facts) for role in sets}
    for role, manifest in manifests.items():
        write_json(output_dir / f"{role}_manifest.json", manifest)

    # Stage 1 and the statistics of stage 2, on the training split and on no other window.
    absent = [index for index, row in enumerate(training) if row["label"] == 0]
    template = cusense.fit_template([training[index] for index in absent], sets["training"].means[absent],
                                    sets["training"].counts[absent])
    normalizer = cusense.fit_normalizer(training, sets["training"].means, sets["training"].counts, template)
    inputs = {role: cusense.inputs_of(sets[role].means, sets[role].counts, template, normalizer) for role in sets}

    device = cusense.training_device(settings)
    net, history = cusense.fit_model(training, inputs["training"], grid, settings, device,
                                     monitor=(validation, inputs["validation"]))
    data, layout = cusense_cnn.pack(net)
    # The scores of the deployed arithmetic: float32 on the CPU, one window at a time.
    scored = {role: cusense.scores_of(net.config, data, layout, inputs[role], "cpu") for role in sets}
    logits = {role: scored[role][0] for role in sets}
    threshold = presence.calibrate_threshold(validation, logits["validation"], settings["threshold"]["target_false_alarm"],
                                             settings["threshold"]["min_negatives"])
    made = cusense.write_artifact(output_dir / ARTIFACT_DIR, settings["model_id"], spec, facts["elements"],
                                  facts["producer_steps"], template, normalizer, net, grid, threshold,
                                  settings["smoothing"], settings["loss"]["sigma_cells"])
    # The artifact as it will be deployed: read back, on the CPU, from the tensors of the windows.
    loaded = cusense.Model(cusense.open_artifact(output_dir / ARTIFACT_DIR, made["sha256"]), "cpu")
    stride = max(1, math.ceil(len(training) / RELOAD_TRAINING_ROWS))
    chosen = {"training": list(range(0, len(training), stride)), "validation": list(range(len(validation)))}
    reload, moved = 0.0, 0.0
    for role in sets:
        for index in chosen[role]:
            again = loaded.score(sets[role].means[index], sets[role].counts[index])
            reload = max(reload, abs(again["logit"] - logits[role][index]))
            moved = max(moved, float(abs(again["x"][0] - inputs[role][index]).max()))
    if reload > RELOAD_TOLERANCE or moved > 0.0:
        raise CuSenseError(f"the artifact read back scores a window {reload:.3e} from the fit and prepares an input "
                           f"{moved:.3e} from it, above {RELOAD_TOLERANCE:g} and 0")

    def rates(role, rows):
        values = logits[role]
        positives = [value for row, value in zip(rows, values) if row["label"] == 1]
        negatives = [value for row, value in zip(rows, values) if row["label"] == 0]
        errors = [math.dist(cusense.locate(drawn, grid)["position_m"], row["position_m"])
                  for row, drawn in zip(rows, scored[role][1]) if row["label"] == 1]
        return {"target_present": len(positives), "detected": sum(value > threshold["logit"] for value in positives),
                "target_absent": len(negatives), "false_alarms": sum(value > threshold["logit"] for value in negatives),
                "area_under_curve": area_under_curve(values, [row["label"] for row in rows]),
                "position_error": error_statistics(errors)}

    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
             for path in sorted((output_dir / ARTIFACT_DIR).iterdir())}
    fit = {"schema": FIT_SCHEMA, "history": history, "normalizer_constant": normalizer["constant"],
           "normalizer_elements": normalizer["elements"], "template": {name: made["template"][name] for name in (
               "valid_elements", "records_min", "records_max", "fitted_on")},
           "parameters": made["model"]["parameters"], "graded": False,
           "at_the_frozen_threshold": {role: rates(role, sets[role].rows) for role in sets},
           "note": "the counts and the position errors of the training and of the validation split are those of the "
                   "windows the model and its threshold were fitted on: they describe the fit and are no result"}
    write_json(output_dir / "fit.json", fit)
    freeze = {"schema": FREEZE_SCHEMA, "model_id": settings["model_id"],
              "artifact": {"directory": ARTIFACT_DIR, "sha256": made["sha256"],
                           "weights_sha256": made["weights"]["sha256"], "template_sha256": made["template"]["sha256"],
                           "template_count_sha256": made["template"]["count_sha256"],
                           "normalizer_sha256": made["normalizer"]["sha256"], "grid_sha256": made["grid"]["sha256"],
                           "threshold_sha256": made["threshold"]["sha256"], "files": files},
              "input": {"entry": spec["input"]["entry"], "spec_id": spec["spec_id"], "spec_sha256": facts["spec_sha256"],
                        "shape": made["input"]["shape"], "elements": facts["elements"]},
              "threshold": made["threshold"], "settings": settings,
              "settings_sha256": hashlib.sha256(cusense.canonical(settings)).hexdigest(),
              "manifests": {role: {"file": f"{role}_manifest.json",
                                   "sha256": sha256_file(output_dir / f"{role}_manifest.json"),
                                   "samples_sha256": manifests[role]["samples"]["sha256"],
                                   "windows": manifests[role]["windows"]} for role in manifests},
              "fit": {"file": "fit.json", "sha256": sha256_file(output_dir / "fit.json"),
                      "initial_loss": history["initial_loss"], "final_loss": history["final_loss"],
                      "epochs": history["epochs"], "device": history["device"]},
              "reload": {"largest_logit_difference": reload, "largest_input_difference": moved,
                         "tolerance": RELOAD_TOLERANCE, "device": "cpu",
                         "rows": {role: len(chosen[role]) for role in chosen}, "training_stride": stride},
              "test_access": TEST_ACCESS, "frozen_utc": utc_now(), "frozen_monotonic_ns": time.monotonic_ns(),
              "software": {"python": platform.python_version(), "torch": cusense.torch.__version__,
                           "numpy": cusense.np.__version__, "machine": platform.machine(),
                           "script_sha256": sha256_file(__file__), "module_sha256": sha256_file(cusense.__file__),
                           "model_sha256": sha256_file(cusense_cnn.__file__)}}
    write_json(output_dir / "freeze.json", freeze)
    return freeze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training", required=True, type=Path, help="cusense-sample/v1 rows of the training split")
    parser.add_argument("--validation", required=True, type=Path, help="cusense-sample/v1 rows of the validation split")
    parser.add_argument("--spec", required=True, type=Path, help="cusense-spec/v1 file the windows were computed with")
    parser.add_argument("--grid", required=True, type=Path, help="the position grid of the episode set, as its manifest "
                                                               "records it")
    parser.add_argument("--config", required=True, type=Path, help="cusense-train-config/v1 file")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the artifact and its records")
    args = parser.parse_args(argv)
    try:
        from ocudu_cuda_isac.sensing import cusense
    except ImportError as error:
        parser.exit(2, f"train_cusense: the model needs NumPy and PyTorch in this interpreter: {error}\n")
    try:
        try:
            settings = json.loads(args.config.read_text(encoding="ascii"))
            geometry = json.loads(args.grid.read_text(encoding="ascii"))
        except (OSError, ValueError) as error:
            raise cusense.CuSenseError(f"cannot read the training configuration or the grid: {error}") from None
        freeze = train(args.training, args.validation, cusense.load_spec(args.spec), geometry, settings, args.output_dir)
    except (cusense.CuSenseError, cusense.PresenceError) as error:
        parser.exit(2, f"train_cusense: {error}\n")
    threshold = freeze["threshold"]
    print(f"train_cusense: model {freeze['model_id']} {freeze['artifact']['sha256'][:12]} on windows of the shape "
          f"{freeze['input']['shape']} from {freeze['manifests']['training']['windows']} training windows, "
          f"{freeze['fit']['epochs']} epochs on {freeze['fit']['device']}, loss {freeze['fit']['initial_loss']:.4f} to "
          f"{freeze['fit']['final_loss']:.4f}; threshold logit {threshold['logit']:.4f} from {threshold['negatives']} "
          f"validation negatives, {threshold['exceeding']} above it of {threshold['allowed']} allowed; frozen "
          f"{freeze['frozen_utc']}")
    print(args.output_dir / "freeze.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
