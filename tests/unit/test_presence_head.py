#!/usr/bin/env python3
"""Unit gate of Sprint 3.4 (selector presence_head): python tests/unit/test_presence_head.py -v

The reference of the presence head is checked on vectors whose answer can be worked out by hand:
the two weights of the card, a saved head read back, a vector of the wrong shape or in the wrong
order, a checkpoint with one bit changed, a normalizer and a threshold whose values follow from
four numbers, a scene in two splits and samples of the wrong split. A stream of records with one
return then gets one decision per answer, and records of the future and fields outside the vector
must not change it. The training script freezes one artifact from two sample files, and the
evaluator side splits a small episode set, plans its replay and joins its windows.

NumPy and PyTorch are needed. The last class but one attaches the presence sensor to a native
producer of the ring, and needs cmake and a C++17 compiler for that.
"""

import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac.sensing import obs  # noqa: E402

if np is not None:
    from ocudu_cuda_isac.episodes import episode_set as sets  # noqa: E402
    from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
    from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
    from ocudu_cuda_isac.episodes import presence_head as evaluator  # noqa: E402
    from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
    from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
    from ocudu_cuda_isac.episodes.presence_head import PresenceHeadError  # noqa: E402
    from ocudu_cuda_isac.sensing import features  # noqa: E402
    from ocudu_cuda_isac.sensing import presence  # noqa: E402
    from ocudu_cuda_isac.sensing.presence import PresenceError  # noqa: E402
    import presence_head_double as double  # noqa: E402
    import test_pilot_features as feature_tests  # noqa: E402
    import train_presence  # noqa: E402

NEEDS = "the reference of the head is PyTorch on NumPy vectors"
SPEC_SHA = "5" * 64
KEY = "0123456789abcdef"


def threshold_of(logit=0.0, target=0.0, negatives=1, allowed=0, exceeding=0):
    return {"entry": "frozen_threshold", "logit": logit, "probability": 1.0 / (1.0 + math.exp(-logit)),
            "rule": presence.THRESHOLD_RULE, "target_false_alarm": target, "negatives": negatives, "allowed": allowed,
            "exceeding": exceeding}


def make_artifact(directory, vector=("a", "b"), columns=None, weights=(1.0, -1.0), bias=0.0, mean=None, scale=None,
                  spec_sha256=SPEC_SHA, **threshold):
    """An artifact with given weights. By default the head of the card: weights [1, -1], no bias, no normalizer."""
    columns = list(columns or vector)
    presence.write_artifact(directory, "unit-head", "unit-spec", spec_sha256, list(vector), columns,
                            {"mean": list(mean or [0.0] * len(columns)), "scale": list(scale or [1.0] * len(columns))},
                            list(weights), bias, threshold_of(**threshold))
    return presence.open_artifact(directory)


def sample(window, values, label, split="training", scene="walk-1", pair=None, names=("a", "b"), episode=None):
    return {"schema": presence.SAMPLE_SCHEMA, "window": window, "episode": episode or f"{abs(hash(window)) % 16**16:016x}",
            "split": split, "scene": scene, "pair": pair or f"ep-{window}", "label": label, "spec_sha256": SPEC_SHA,
            "names": list(names), "values": [float(value) for value in values]}


def two_classes(count=12, split="training", scene="walk", offset=0.0, names=("a", "b", "c")):
    """Samples of two classes that the first column separates. The third column is noise the head does not read."""
    generator = np.random.Generator(np.random.PCG64(7 if split == "training" else 11))
    rows = []
    for index in range(count):
        label = index % 2
        values = [6.0 * label + generator.normal() + offset, generator.normal(), 100.0 * generator.normal()]
        rows.append(sample(f"{split}-{index}", values, label, split, f"{scene}-{split}-{index % 3}",
                           f"ep-{split}-{index // 2}", names, f"{index + (1000 if split == 'training' else 2000):016x}"))
    return rows


@unittest.skipIf(np is None, NEEDS)
class HeadTests(unittest.TestCase):
    """The artifact, the batch-one interface and what they refuse."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_the_fixed_head_of_the_card_scores_logit_1_and_probability_0_731059(self):
        head = presence.Head(make_artifact(self.root / "card"))
        scored = head.score(["a", "b"], [2.0, 1.0])
        self.assertEqual(scored["logit"], 1.0)
        self.assertAlmostEqual(scored["probability"], 0.731059, places=6)
        self.assertEqual(scored["present"], True)  # the logit lies above the threshold of 0
        # The batch-one interface: float32 of shape [1, columns] in, two floats and a time out.
        x = head.prepare(["a", "b"], [2.0, 1.0])
        self.assertEqual((x.dtype, x.shape), (np.float32, (1, 2)))
        logit, probability, elapsed = head.predict(x)
        self.assertEqual((logit, type(probability), type(elapsed)), (1.0, float, int))
        self.assertEqual(head.predict(torch.tensor([[2.0, 1.0]]))[0], 1.0)
        self.assertEqual(scored["input_sha256"], hashlib.sha256(struct.pack("<2f", 2.0, 1.0)).hexdigest())
        # The threshold is a strict bound, and the normalizer comes first.
        self.assertEqual(head.score(["a", "b"], [1.0, 1.0])["present"], False)
        shifted = presence.Head(make_artifact(self.root / "shifted", mean=[1.0, 1.0], scale=[2.0, 0.5]))
        self.assertEqual(shifted.score(["a", "b"], [5.0, 2.0])["logit"], (5.0 - 1.0) / 2.0 - (2.0 - 1.0) / 0.5)

    def test_a_saved_head_reloads_to_the_same_score_within_1e_6(self):
        generator = np.random.Generator(np.random.PCG64(3))
        weights, bias = [float(value) for value in generator.normal(size=8)], 0.37
        mean, scale = [float(value) for value in generator.normal(size=8)], [float(value) for value in 1 + generator.random(8)]
        vector = [f"f{index}" for index in range(8)]
        artifact = make_artifact(self.root / "saved", vector, None, weights, bias, mean, scale)
        self.assertEqual(sorted(path.name for path in artifact.directory.iterdir()), ["head.json", "weights.f32"])
        self.assertEqual((artifact.directory / "weights.f32").stat().st_size, 4 * 9)
        first, second = presence.Head(artifact), presence.Head(presence.open_artifact(artifact.directory))
        for _ in range(20):
            values = [float(value) for value in 5 * generator.normal(size=8)]
            # The arithmetic by hand, in double precision, from the values the fit handed over.
            wanted = sum(w * (float(np.float32(x)) - m) / s for w, x, m, s in zip(weights, values, mean, scale)) + bias
            a, b = first.score(vector, values), second.score(vector, values)
            self.assertEqual(a["logit"], b["logit"])
            self.assertAlmostEqual(a["logit"], wanted, delta=5e-5)
            self.assertLessEqual(abs(a["logit"] - b["logit"]), 1e-6)
            self.assertAlmostEqual(a["probability"], 1.0 / (1.0 + math.exp(-a["logit"])), places=6)
        # The artifact holds parameters and preprocessing state, and nothing of a training run.
        manifest = json.loads((artifact.directory / "head.json").read_text())
        self.assertEqual(set(manifest), set(presence.ARTIFACT_FIELDS))
        for word in ("label", "split", "scene", "sample", "pair", "episode", "window"):
            self.assertNotIn(f'"{word}', json.dumps(manifest))
        self.assertEqual((manifest["schema"], manifest["weights"]["count"], manifest["input"]["shape"],
                          manifest["input"]["dtype"]), ("presence-head/v1", 9, [1, 8], "<f4"))
        self.assertEqual(manifest["sha256"], presence.identity(manifest))
        self.assertEqual(artifact.model(), {"head_id": "unit-head", "entry": "linear", "artifact_sha256": manifest["sha256"],
                                            "weights_sha256": hashlib.sha256(artifact.weights).hexdigest(),
                                            "normalizer_sha256": presence.normalizer_sha256(mean, scale)})
        with self.assertRaisesRegex(PresenceError, "never overwritten"):
            make_artifact(self.root / "saved")

    def test_wrong_shape_and_wrong_ordering_are_refused(self):
        head = presence.Head(make_artifact(self.root / "card"))
        for wrong, message in ((np.zeros((2, 2), dtype=np.float32), r"shape \[1, 2\] as float32.*\[2, 2\]"),
                               (np.zeros(2, dtype=np.float32), r"has the shape \[2\]"),
                               (np.zeros((1, 3), dtype=np.float32), r"has the shape \[1, 3\]"),
                               (np.zeros((1, 2), dtype=np.float64), "as float64"),
                               (torch.zeros((1, 2), dtype=torch.float64), "as float64"),
                               (torch.zeros((1, 1, 2)), r"has the shape \[1, 1, 2\]"),
                               ([[2.0, 1.0]], "a NumPy array or a tensor"),
                               (np.array([[np.inf, 0.0]], dtype=np.float32), "not finite")):
            with self.subTest(message=message), self.assertRaisesRegex(PresenceError, message):
                head.predict(wrong)
        # A vector arrives with its names. The same names in another order are not the vector of the head.
        with self.assertRaisesRegex(PresenceError, "another order than the head was trained on: position 0 is 'b' and "
                                                   "the head expects 'a'"):
            head.score(["b", "a"], [1.0, 2.0])
        with self.assertRaisesRegex(PresenceError, "1 names against 2, without 'b'"):
            head.score(["a"], [2.0])
        with self.assertRaisesRegex(PresenceError, "3 names against 2, with 'c'"):
            head.score(["a", "b", "c"], [2.0, 1.0, 0.0])
        with self.assertRaisesRegex(PresenceError, "names 2 values and holds 1"):
            head.score(["a", "b"], [2.0])
        for values in ([2.0, float("nan")], [2.0, 1e39], [2.0, "one"]):
            with self.subTest(values=values), self.assertRaisesRegex(PresenceError, "not finite|no number"):
                head.score(["a", "b"], values)

    def test_an_altered_checkpoint_is_refused(self):
        def flip_weight(directory):
            data = bytearray((directory / "weights.f32").read_bytes())
            data[5] ^= 0x10
            (directory / "weights.f32").write_bytes(bytes(data))

        def edit(change):
            def apply(directory):
                manifest = json.loads((directory / "head.json").read_text())
                change(manifest)
                (directory / "head.json").write_text(json.dumps(manifest))
            return apply

        def rehashed(change):
            def apply(manifest):
                change(manifest)
                manifest["sha256"] = presence.identity(manifest)
            return apply

        cases = (
            (flip_weight, "the checkpoint was altered"),
            (lambda directory: (directory / "weights.f32").write_bytes((directory / "weights.f32").read_bytes()[:-4]),
             "the checkpoint was altered"),
            (edit(lambda manifest: manifest["threshold"].update(logit=-3.0)), "altered after it was frozen"),
            (edit(lambda manifest: manifest["input"].update(columns=["b"], shape=[1, 1])), "normalizer.mean must be a list of 1 finite numbers"),
            (edit(lambda manifest: manifest["normalizer"]["mean"].__setitem__(0, 9.0)), "the preprocessing state changed"),
            # A manifest that was edited and hashed again is still found by the hash of its parts.
            (edit(rehashed(lambda manifest: manifest["normalizer"]["scale"].__setitem__(0, 2.0))),
             "the preprocessing state changed"),
            (edit(rehashed(lambda manifest: manifest["weights"].update(sha256="0" * 64))), "the checkpoint was altered"),
            (edit(rehashed(lambda manifest: manifest["threshold"].update(exceeding=3))), "does not follow its rule"),
            (edit(rehashed(lambda manifest: manifest["entries"].update(model="gru"))), "unknown model entry"),
            (edit(lambda manifest: manifest.update(labels=[1, 0])), "requires exactly"),
            (lambda directory: (directory / "training.jsonl").write_text("{}\n"), "holds training.jsonl"),
            (lambda directory: (directory / "head.json").unlink(), "not a complete presence head"))
        for number, (change, message) in enumerate(cases):
            artifact = make_artifact(self.root / f"case-{number}")
            change(artifact.directory)
            with self.subTest(message=message), self.assertRaisesRegex(PresenceError, message):
                presence.open_artifact(artifact.directory)
            with self.subTest(message=message, load=True), self.assertRaises(PresenceError):
                presence.Head(artifact.directory)
        # A freeze record names one artifact. Another one, whole in itself, is not it.
        artifact = make_artifact(self.root / "whole")
        self.assertEqual(presence.open_artifact(artifact.directory, artifact.sha256).sha256, artifact.sha256)
        with self.assertRaisesRegex(PresenceError, "was asked for"):
            presence.open_artifact(artifact.directory, "0" * 64)
        # A weight that float32 cannot hold never reaches a file.
        with self.assertRaisesRegex(PresenceError, "does not fit a float32"):
            make_artifact(self.root / "huge", weights=(1e39, 0.0))
        self.assertFalse((self.root / "huge").exists())

    def test_the_head_binds_to_names_and_reads_only_its_columns(self):
        vector = list(features.LEADING) + ["profile_db@-25Hz", "profile_db@+25Hz"]
        self.assertEqual(presence.select_columns(vector, presence.DEMO_COLUMNS),
                         ["peak_to_median_db", "profile_db@-25Hz", "profile_db@+25Hz"])
        self.assertEqual(presence.select_columns(vector, ["profile_db@+25Hz", "peak_delay_ns"]),
                         ["peak_delay_ns", "profile_db@+25Hz"])  # the order is that of the vector
        with self.assertRaisesRegex(PresenceError, "'peak_to_mean_db' selects no value"):
            presence.select_columns(vector, ["peak_to_mean_db"])
        with self.assertRaisesRegex(PresenceError, "nonempty list"):
            presence.select_columns(vector, [])
        columns = presence.select_columns(vector, presence.DEMO_COLUMNS)
        head = presence.Head(make_artifact(self.root / "named", vector, columns, (1.0, 0.5, -0.5)))
        values = [20.0, -33.0, 163.4, -50.0, -31.2, 47.0, 14.0, 6.0]
        base = head.score(vector, values)
        self.assertEqual(base["logit"], 20.0 + 7.0 - 3.0)
        # The peak delay and the peak Doppler shift locate a return. The head does not read them.
        moved = list(values)
        moved[2], moved[3], moved[1] = 1634.0, -100.0, 0.0
        self.assertEqual({key: head.score(vector, moved)[key] for key in ("logit", "input_sha256")},
                         {key: base[key] for key in ("logit", "input_sha256")})
        moved[6] += 1.0
        self.assertNotEqual(head.score(vector, moved)["input_sha256"], base["input_sha256"])
        self.assertEqual(head.score(vector, moved)["logit"], base["logit"] + 0.5)
        # Columns that are no names of the vector, or not in its order, are no artifact.
        with self.assertRaisesRegex(PresenceError, "in the order of the vector"):
            make_artifact(self.root / "reordered", vector, list(reversed(columns)), (1.0, 0.5, -0.5))

    def test_the_selected_device_agrees_with_the_cpu(self):
        # On a host with a CUDA device this is the parity of the card. Without one both heads are on the CPU.
        device = presence.default_device()
        generator = np.random.Generator(np.random.PCG64(5))
        vector = [f"f{index}" for index in range(8)]
        artifact = make_artifact(self.root / "device", vector, None, [float(v) for v in generator.normal(size=8)], -0.2,
                                 [float(v) for v in 10 * generator.normal(size=8)], [float(v) for v in 1 + 5 * generator.random(8)])
        host, selected = presence.Head(artifact, "cpu"), presence.Head(artifact, device)
        self.assertEqual(selected.device.type, device)
        self.assertGreater(selected.warm_up(), 0.0)
        for _ in range(50):
            values = [float(value) for value in 20 * generator.normal(size=8)]
            a, b = host.score(vector, values), selected.score(vector, values)
            self.assertEqual(a["input_sha256"], b["input_sha256"])
            self.assertLessEqual(abs(a["logit"] - b["logit"]), evaluator.TOLERANCE["logit_absolute"])
            self.assertLessEqual(abs(a["probability"] - b["probability"]), evaluator.TOLERANCE["probability_absolute"])
        facts = presence.device_facts(selected.device)
        self.assertEqual((facts["type"], facts["torch"]), (device, torch.__version__))
        # A device this process cannot reach is refused before a weight is moved.
        for name, message in (("tpu", "unknown device"), ("mps", "neither"), ("cuda:99", "CUDA device")):
            with self.subTest(name=name), self.assertRaisesRegex(PresenceError, message):
                presence.Head(artifact, name)

    def test_registry_entries_state_their_device_and_runtime(self):
        entries = presence.resolve()
        self.assertEqual({stage: entry["entry"] for stage, entry in entries.items()},
                         {"normalizer": "zscore", "model": "linear", "detector": "frozen_threshold"})
        for stage, entry in entries.items():
            self.assertEqual(entry["runtime"], "reference", stage)
            self.assertIn("device", entry)
            self.assertIn("does", entry)
        self.assertEqual((entries["model"]["library"], entries["detector"]["device"]), ("torch", "cpu"))
        self.assertEqual(presence.RUNTIME, features.RUNTIME)


@unittest.skipIf(np is None, NEEDS)
class TrainingTests(unittest.TestCase):
    """The fit, the threshold and the split rules, on numbers that can be checked by hand."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_the_normalizer_is_the_mean_and_the_deviation_of_the_training_split(self):
        rows = [sample("w0", [1.0, 5.0], 0), sample("w1", [3.0, 5.0], 1)]
        made = presence.fit_normalizer(rows, ["a", "b"], ["a", "b"])
        # Column a: values 1 and 3, mean 2 and deviation 1. Column b does not vary and keeps the scale 1.
        self.assertEqual(made, {"entry": "zscore", "mean": [2.0, 5.0], "scale": [1.0, 1.0], "constant": ["b"]})
        np.testing.assert_array_equal(presence.normalized(rows, ["a", "b"], ["a", "b"], made), [[-1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(presence.fit_normalizer(rows, ["a", "b"], ["b"])["mean"], [5.0])
        with self.assertRaisesRegex(PresenceError, "needs a training sample"):
            presence.fit_normalizer([], ["a", "b"], ["a"])

    def test_samples_outside_their_split_cannot_fit(self):
        training, validation = two_classes(), two_classes(8, "validation")
        vector, columns = ["a", "b", "c"], ["a", "b"]
        normalizer = presence.fit_normalizer(training, vector, columns)
        # Validation and test samples used to fit the normalization.
        for other in (validation[:1], [dict(validation[0], split="held_out")]):
            with self.subTest(split=other[0]["split"]), self.assertRaisesRegex(
                    PresenceError, f"of the {other[0]['split']} split was given to the fit of the normalizer, which reads "
                                   "the training split and no other sample"):
                presence.fit_normalizer(training + other, vector, columns)
            with self.subTest(split=other[0]["split"], fit="weights"), self.assertRaisesRegex(
                    PresenceError, "was given to the fit of the weights"):
                presence.fit_head(training + other, vector, columns, normalizer, 0.01)
        with self.assertRaisesRegex(PresenceError, "of the training split was given to the calibration of the threshold"):
            presence.calibrate_threshold(validation + training[:1], [0.0] * 9, 0.05)
        with self.assertRaisesRegex(PresenceError, "needs 7 of each"):
            presence.fit_head(training, vector, columns, normalizer, 0.01, min_per_class=7)

    def test_a_scene_in_two_splits_is_refused(self):
        training, validation = two_classes(), two_classes(8, "validation")
        facts = presence.check_split(training, validation)
        self.assertEqual((facts["spec_sha256"], facts["vector"], facts["pairs"]),
                         (SPEC_SHA, ["a", "b", "c"], {"training": 6, "validation": 4}))
        self.assertEqual(facts["scenes"]["validation"], ["walk-validation-0", "walk-validation-1", "walk-validation-2"])

        def changed(row, **fields):
            return dict(row, **fields)

        cases = (
            (training, validation[1:] + [changed(validation[0], scene=training[0]["scene"])],
             "the scene walk-training-0 is in the training and in the validation split: a split is made by scene before "
             "any window is cut"),
            (training, validation[1:] + [changed(validation[0], pair=training[0]["pair"])], "the matched pair ep-training-0"),
            (training, validation[1:] + [changed(validation[0], episode=training[0]["episode"])], "the episode key"),
            (training, validation[1:] + [changed(validation[0], window=training[0]["window"])], "the window training-0"),
            (training + [changed(validation[0])], validation[1:], "of the validation split is among the training samples"),
            (training + [changed(training[0], window="late", split="held_out")], validation,
             "a held-out sample is read after the freeze and never before"),
            (training, validation + [changed(validation[0], window="other", split="held_out")], "of the held_out split"),
            (training + training[:1], validation, "a window is among the training samples twice"),
            (training, validation[1:] + [changed(validation[0], spec_sha256="6" * 64)], "not computed with one specification"),
            (training, validation[1:] + [changed(validation[0], names=["a", "c", "b"])], "not computed with one specification"),
            (training[1:] + [changed(training[0], label=2)], validation, "a label of 0 or 1"),
            (training[1:] + [changed(training[0], values=[1.0, float("nan"), 0.0])], validation, "finite numbers"),
            ([], [], "there is no sample"))
        for first, second, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(PresenceError, message):
                presence.check_split(first, second)

    def test_the_threshold_is_set_on_validation_negatives_by_its_rule(self):
        rows = [sample(f"v{index}", [0.0, 0.0], label, "validation") for index, label in enumerate([0] * 10 + [1] * 3)]
        negatives = [0.5, -1.0, 2.5, -3.0, 0.0, -0.5, 1.5, -2.0, -1.5, 1.0]
        logits = negatives + [9.0, -9.0, 0.7]  # the three positives are never read
        # No false alarm allowed: the threshold is the largest negative, and none lies above it.
        made = presence.calibrate_threshold(rows, logits, 0.0)
        self.assertEqual((made["logit"], made["allowed"], made["exceeding"], made["negatives"]), (2.5, 0, 0, 10))
        self.assertEqual(made["probability"], 1.0 / (1.0 + math.exp(-2.5)))
        # A target of 0.2 allows two of ten: the third from the top is the threshold.
        made = presence.calibrate_threshold(rows, logits, 0.2)
        self.assertEqual((made["logit"], made["allowed"], made["exceeding"]), (1.0, 2, 2))
        # 0.05 of ten is less than one window: none is allowed.
        self.assertEqual(presence.calibrate_threshold(rows, logits, 0.05)["logit"], 2.5)
        self.assertEqual(presence.calibrate_threshold(rows, logits, 0.99)["logit"], -3.0)
        # Ties: with the two largest equal, one allowed alarm puts the threshold on the tie, and none lies above it.
        tied = presence.calibrate_threshold(rows, [2.5, 2.5] + negatives[2:] + [9.0, -9.0, 0.7], 0.1)
        self.assertEqual((tied["logit"], tied["allowed"], tied["exceeding"]), (2.5, 1, 0))
        changed = presence.calibrate_threshold(rows, negatives + [-50.0, 50.0, 0.0], 0.2)
        self.assertEqual(changed["logit"], 1.0)
        for arguments, message in (((rows, logits, 0.05, 11), "holds 10 target-absent windows, and the calibration needs 11"),
                                   ((rows, logits, 1.0), "from 0 to below 1"), ((rows, logits[:-1], 0.05), "one logit per"),
                                   ((rows[10:], logits[10:], 0.05), "holds 0 target-absent windows")):
            with self.subTest(message=message), self.assertRaisesRegex(PresenceError, message):
                presence.calibrate_threshold(*arguments)
        # The record is the threshold of an artifact as it stands.
        artifact = make_artifact(self.root / "calibrated", logit=1.0, target=0.2, negatives=10, allowed=2, exceeding=2)
        self.assertEqual(artifact.manifest["threshold"], presence.calibrate_threshold(rows, logits, 0.2))

    def test_the_fit_separates_two_classes_and_is_repeatable(self):
        training = two_classes(40)
        vector = ["a", "b", "c"]
        columns = presence.select_columns(vector, ["a", "b"])
        normalizer = presence.fit_normalizer(training, vector, columns)
        weights, bias, history = presence.fit_head(training, vector, columns, normalizer, 0.01)
        self.assertAlmostEqual(history["initial_loss"], math.log(2.0), places=12)
        self.assertLess(history["final_loss"], 0.2)
        self.assertLess(history["gradient_norm"], 1e-5)
        self.assertEqual(history["samples"], {"target_present": 20, "target_absent": 20})
        self.assertGreater(weights[0], 5 * abs(weights[1]))  # the first column carries the class
        again = presence.fit_head(training, vector, columns, normalizer, 0.01)
        self.assertEqual((weights, bias), again[:2])
        logits = presence.logits_of(training, vector, columns, normalizer, weights, bias)
        self.assertGreaterEqual(sum((logit > 0) == bool(row["label"]) for row, logit in zip(training, logits)), 38)
        self.assertEqual(evaluator.area_under_curve([0.1, 0.9, 0.5, 0.5], [0, 1, 0, 1]), 0.875)
        self.assertIsNone(evaluator.area_under_curve([0.1, 0.2], [1, 1]))
        # A stronger penalty gives smaller weights: the minimum is that of the stated loss.
        heavy, _, _ = presence.fit_head(training, vector, columns, normalizer, 1.0)
        self.assertLess(abs(heavy[0]), abs(weights[0]))

    def test_the_training_configuration_is_checked(self):
        settings = presence.demo_settings()
        self.assertEqual((settings["columns"], settings["threshold"]), (["peak_to_median_db", "profile_db@*"],
                                                                      {"target_false_alarm": 0.05, "min_negatives": 20}))
        for change, message in ((dict(schema="other"), "unsupported training configuration"), (dict(l2=0), "l2 must lie"),
                                (dict(columns=[]), "nonempty list"), (dict(max_iterations=0), "max_iterations"),
                                (dict(min_per_class=0), "min_per_class"), (dict(head_id=""), "head_id"),
                                (dict(threshold={"target_false_alarm": 1.0, "min_negatives": 1}), "below 1"),
                                (dict(threshold={"target_false_alarm": 0.1}), "requires exactly")):
            with self.subTest(message=message), self.assertRaisesRegex(PresenceError, message):
                presence.check_settings(dict(settings, **change))

    def test_the_training_script_freezes_one_artifact_from_two_sample_files(self):
        spec = feature_tests.small_spec()
        sha = features.spec_sha256(spec)
        training = [dict(row, spec_sha256=sha) for row in two_classes(40)]
        validation = [dict(row, spec_sha256=sha) for row in two_classes(24, "validation")]
        presence.write_samples(self.root / "training.jsonl", training)
        presence.write_samples(self.root / "validation.jsonl", validation)
        settings = dict(presence.demo_settings("unit-linear"), columns=["a", "b"], min_per_class=5,
                        threshold={"target_false_alarm": 0.1, "min_negatives": 10})
        freeze = train_presence.train(self.root / "training.jsonl", self.root / "validation.jsonl", spec, settings,
                                      self.root / "run")
        run = self.root / "run"
        self.assertEqual(sorted(path.name for path in run.iterdir()),
                         ["artifact", "fit.json", "freeze.json", "training_manifest.json", "validation_manifest.json"])
        artifact = presence.open_artifact(run / "artifact", freeze["artifact"]["sha256"])
        self.assertEqual((artifact.columns, artifact.manifest["input"]["spec_sha256"], artifact.manifest["head_id"]),
                         (["a", "b"], sha, "unit-linear"))
        # The freeze record names the artifact and both manifests by hash, and each manifest its sample file.
        self.assertEqual(freeze, json.loads((run / "freeze.json").read_text()))
        self.assertEqual(freeze["artifact"]["weights_sha256"], hashlib.sha256((run / "artifact/weights.f32").read_bytes()).hexdigest())
        for role, rows in (("training", training), ("validation", validation)):
            manifest = json.loads((run / f"{role}_manifest.json").read_text())
            self.assertEqual(freeze["manifests"][role]["sha256"],
                             hashlib.sha256((run / f"{role}_manifest.json").read_bytes()).hexdigest())
            self.assertEqual(manifest["samples"]["sha256"], hashlib.sha256((self.root / f"{role}.jsonl").read_bytes()).hexdigest())
            self.assertEqual((manifest["role"], manifest["windows"], manifest["target_present"], manifest["target_absent"]),
                             (role, len(rows), len(rows) // 2, len(rows) // 2))
            self.assertEqual(manifest["window_ids"], sorted(row["window"] for row in rows))
            self.assertEqual(manifest["episode_keys"], sorted(row["episode"] for row in rows))
        self.assertEqual(json.loads((run / "training_manifest.json").read_text())["used_for"], ["normalizer", "weights"])
        # The normalizer is that of the training rows, the threshold that of the validation negatives.
        normalizer = presence.fit_normalizer(training, ["a", "b", "c"], ["a", "b"])
        self.assertEqual(artifact.manifest["normalizer"]["mean"], normalizer["mean"])
        head = presence.Head(artifact)
        logits = [head.score(row["names"], row["values"])["logit"] for row in validation]
        self.assertEqual(artifact.manifest["threshold"], presence.calibrate_threshold(validation, logits, 0.1, 10))
        self.assertEqual((freeze["threshold"]["allowed"], freeze["threshold"]["negatives"]), (1, 12))
        self.assertLessEqual(freeze["reload"]["largest_logit_difference"], 1e-6)
        self.assertLess(freeze["fit"]["final_loss"], freeze["fit"]["initial_loss"])
        self.assertIn("no record, feature output, label or episode beside them", freeze["test_access"])
        fit = json.loads((run / "fit.json").read_text())
        self.assertEqual((fit["graded"], sorted(fit["weights"])), (False, ["a", "b"]))
        # The same files give the same artifact, bit for bit.
        again = train_presence.train(self.root / "training.jsonl", self.root / "validation.jsonl", spec, settings,
                                     self.root / "again")
        self.assertEqual(again["artifact"]["sha256"], freeze["artifact"]["sha256"])
        with self.assertRaisesRegex(PresenceError, "never overwritten"):
            train_presence.train(self.root / "training.jsonl", self.root / "validation.jsonl", spec, settings, run)

    def test_the_training_script_refuses_by_its_command_line(self):
        spec = feature_tests.small_spec()
        sha = features.spec_sha256(spec)
        training = [dict(row, spec_sha256=sha) for row in two_classes(40)]
        validation = [dict(row, spec_sha256=sha) for row in two_classes(24, "validation")]
        settings = dict(presence.demo_settings("unit-linear"), columns=["a", "b"], min_per_class=5,
                        threshold={"target_false_alarm": 0.1, "min_negatives": 10})
        files = {"training": training, "validation": validation,
                 "crossed": validation[1:] + [dict(validation[0], scene=training[0]["scene"])],
                 "leaked": training + [dict(training[0], window="late", split="held_out")],
                 "other-spec": [dict(row, spec_sha256="6" * 64) for row in training]}
        for name, rows in files.items():
            presence.write_samples(self.root / f"{name}.jsonl", rows)
        (self.root / "spec.json").write_text(json.dumps(spec))
        (self.root / "settings.json").write_text(json.dumps(settings))
        (self.root / "greedy.json").write_text(json.dumps(dict(settings, columns=["a", "missing"])))

        def run(training="training", validation="validation", config="settings.json", out="out"):
            with contextlib.redirect_stdout(io.StringIO()) as said, contextlib.redirect_stderr(io.StringIO()) as errors:
                try:
                    status = train_presence.main(["--training", str(self.root / f"{training}.jsonl"), "--validation",
                                                  str(self.root / f"{validation}.jsonl"), "--spec",
                                                  str(self.root / "spec.json"), "--config", str(self.root / config),
                                                  "--output-dir", str(self.root / out)])
                except SystemExit as stop:
                    status = stop.code
            return status, said.getvalue(), errors.getvalue()

        for arguments, message in ((dict(validation="crossed"), "a split is made by scene before any window is cut"),
                                   (dict(training="leaked"), "a held-out sample is read after the freeze"),
                                   (dict(training="validation", validation="training"), "is among the training samples"),
                                   (dict(training="other-spec"), "not computed with one specification"),
                                   (dict(training="missing"), "cannot read the samples"),
                                   (dict(config="greedy.json"), "'missing' selects no value"),
                                   (dict(config="absent.json"), "cannot read the training configuration")):
            status, _, errors = run(**arguments)
            with self.subTest(message=message):
                self.assertEqual(status, 2)
                self.assertIn(message, errors)
                self.assertTrue(errors.startswith("train_presence: "))
                self.assertEqual(errors.count("\n"), 1)
                self.assertFalse((self.root / "out").exists())
        status, said, _ = run()
        self.assertEqual(status, 0)
        self.assertIn("train_presence: head unit-linear ", said)
        self.assertIn("from 12 validation negatives, ", said)
        self.assertTrue(said.rstrip().endswith("freeze.json"))
        self.assertEqual(run()[0], 2)


@unittest.skipIf(np is None, NEEDS)
class StreamTests(unittest.TestCase):
    """One decision per answer of a feature stream, and what a sensor leaves."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.records = feature_tests.lattice()
        self.spec = feature_tests.small_spec()
        lines, _, _ = features.run_records(self.records, self.spec)
        self.vector = feature_tests.complete(lines)[0]["features"]["names"]
        self.columns = presence.select_columns(self.vector, presence.DEMO_COLUMNS)
        # One weight on the peak over the median: a window whose peak stands 120 dB up scores 1.2 - 0.5.
        weights = [0.01] + [0.0] * (len(self.columns) - 1)
        self.artifact = make_artifact(self.root / "artifact", self.vector, self.columns, weights, -0.5,
                                      spec_sha256=features.spec_sha256(self.spec))
        self.archive = self.root / "archive"
        self.archive.mkdir()
        rows = []
        for record in self.records:
            (self.archive / obs.record_name(1, record.sequence)).write_bytes(record.to_bytes())
            rows.append({"epoch": 1, "sequence": record.sequence, "opportunity": f"{record.header['opportunity']:016x}",
                         "depth": 0, "bytes": len(record.to_bytes()), "t_publish_ns": record.header["t_publish_ns"],
                         "t_read_ns": 1, "delay_ns": 1, "intact": True, "file": True})
        (self.archive / "consumer.json").write_text(json.dumps({
            "schema": obs.CONSUMER_SCHEMA, "reader": "native", "definition": f"{obs.DEFINITION:016x}", "wait": "spin",
            "epoch": 1, "records": rows}))
        (self.root / "spec.json").write_text(json.dumps(self.spec))

    def made(self, name="out"):
        directory = self.root / name
        directory.mkdir()
        return directory, presence.replay(self.archive, self.spec, self.artifact.directory, directory)

    def test_every_answer_of_the_stream_gets_one_decision(self):
        self.assertEqual(self.columns, ["peak_to_median_db"] + [name for name in self.vector if name.startswith("profile_db@")])
        head = presence.Head(self.artifact)
        ticks = iter(range(10_000, 20_000))
        lines, views, stream, decisions = presence.run_records(self.records, self.spec, head, clock=lambda: next(ticks))
        self.assertEqual([line["status"] for line in decisions], ["insufficient_history"] * 7 + ["ok"])
        self.assertEqual([line["line"] for line in decisions], list(range(8)))
        for decision, line in zip(decisions[:7], lines):
            self.assertEqual((decision["score"], decision["decision"], decision["input"], decision["infer_ns"]),
                             (None, None, None, None))
            self.assertEqual((decision["window"], decision["trigger"]), (line["window"], line["trigger"]))
            self.assertNotIn("contributing", decision)
        scored, line = decisions[7], lines[7]
        self.assertEqual(scored["schema"], "presence-decision/v1")
        self.assertAlmostEqual(scored["score"]["logit"], 0.7, places=6)
        self.assertEqual(scored["decision"], {"present": True, "threshold_logit": 0.0, "detector": "frozen_threshold"})
        self.assertEqual((scored["contributing"], scored["time"], scored["window"]),
                         (line["contributing"], line["time"], line["window"]))
        self.assertEqual(len(scored["contributing"]), 8)
        self.assertEqual(scored["time"]["unit"], "tc")
        self.assertEqual(scored["model"], self.artifact.model())
        self.assertEqual(scored["input"], {"spec_sha256": features.spec_sha256(self.spec), "features_entry": "dd_compact",
                                           "columns": 8, "sha256": head.score(line["features"]["names"],
                                                                              line["features"]["values"])["input_sha256"]})
        self.assertEqual((scored["runtime"], scored["device"]), ("reference", "cpu"))
        # The answer is stamped when it is complete: the features first, then the decision.
        self.assertGreater(scored["t_decision_ns"], scored["t_features_ns"])
        self.assertGreater(scored["infer_ns"], 0)
        # A decision names no label, and nothing but its line made it.
        for word in ("label", "truth", "target_present", "split", "scene"):
            self.assertNotIn(f'"{word}', json.dumps(decisions))
        self.assertEqual([presence.comparable(decision) for decision in presence.decide_lines(head, lines)],
                         [presence.comparable(decision) for decision in decisions])
        # A weak return scores lower with the same head: the score follows the vector.
        weak = feature_tests.lattice(static=3.0)
        noise = np.random.Generator(np.random.PCG64(1))
        noisy = []
        for record in weak:
            values = np.frombuffer(record.payload, dtype="<c8") + (noise.normal(size=64) + 1j * noise.normal(size=64)) * 0.5
            noisy.append(obs.Record(dict(record.header), record.host, record.mask, values.astype("<c8").tobytes()))
        low = presence.run_records(noisy, self.spec, head)[3][-1]
        self.assertLess(low["score"]["logit"], scored["score"]["logit"])
        self.assertEqual(low["decision"]["present"], low["score"]["logit"] > 0.0)

    def test_future_records_and_fields_outside_the_vector_cannot_change_a_decision(self):
        head = presence.Head(self.artifact)
        reference = [presence.comparable(line) for line in presence.run_records(self.records, self.spec, head)[3]]
        last = max(index for index, record in enumerate(self.records) if record.header["absolute_slot"] == 77)
        big = np.full(len(self.records[0].payload) // 8, 1e30 * (1 + 1j), dtype="<c8").tobytes()
        future = self.records[:last + 1] + [obs.Record(dict(record.header), record.host, record.mask, big)
                                            for record in self.records[last + 1:]]
        self.assertEqual([presence.comparable(line) for line in presence.run_records(future, self.spec, head)[3]], reference)
        # Before the completing record there is no score to leak into.
        early = presence.run_records(self.records[:last], self.spec, head)[3]
        self.assertTrue(all(line["score"] is None for line in early))
        self.assertEqual(len(early), 7)
        # Of a line a head reads the vector and the hash of the specification.
        lines, _, _ = features.run_records(self.records, self.spec)
        line = lines[7]
        base = head.decide(line)
        hidden = json.loads(json.dumps(line))
        hidden["view"]["peak"]["doppler_bin"] = -2
        hidden["contributing"] = hidden["contributing"][:1]
        hidden["validity"]["fill"] = 0.1
        hidden["trigger"]["sequence"] = 999
        hidden["host"] = "evaluator only: target present"
        other = head.decide(hidden)
        self.assertEqual((other["score"], other["input"], other["decision"]), (base["score"], base["input"], base["decision"]))
        with self.assertRaisesRegex(PresenceError, "the head was trained on"):
            head.decide(dict(line, spec_sha256="7" * 64))
        with self.assertRaisesRegex(PresenceError, "not that of the head"):
            head.decide(dict(line, features={"entry": "dd_compact", "names": line["features"]["names"][:-1],
                                             "values": line["features"]["values"][:-1]}))
        self.assertEqual(presence.LINE_FIELDS, ("status", "spec_sha256", "features.names", "features.values"))

    def test_the_weights_are_loaded_once(self):
        before = presence.Head.LOADS
        directory, report = self.made()
        self.assertEqual(presence.Head.LOADS, before + 1)  # one head for eight decisions
        self.assertEqual(report["loads"], presence.Head.LOADS)
        head = presence.Head(self.artifact)
        loaded = presence.Head.LOADS
        presence.run_records(self.records * 1, self.spec, head)
        presence.decide_lines(head, features.run_records(self.records, self.spec)[0])
        self.assertEqual(presence.Head.LOADS, loaded)

    def test_an_output_verifies_and_a_second_reader_recovers_the_decision(self):
        directory, report = self.made()
        self.assertEqual(sorted(path.name for path in directory.iterdir()), ["decisions.jsonl", "features", "presence.json"])
        self.assertEqual(report["counts"], {"ok": 1, "insufficient_history": 7, "gap": 0, "invalid": 0, "rejected": 0,
                                            "decisions": 8, "scored": 1, "present": 1, "absent": 0})
        self.assertEqual((report["runtime"]["entry"], report["runtime"]["device"], report["runtime"]["torch"],
                          report["features"]), ("reference", "cpu", torch.__version__, "features"))
        self.assertEqual(report["artifact"]["manifest"], self.artifact.manifest)
        self.assertEqual(report["entries"]["model"]["entry"], "linear")
        output = presence.open_output(directory)
        self.assertEqual((len(output.decisions), len(output.scored), len(output.features.lines)), (8, 1, 8))
        shown = presence.summary(output)
        self.assertEqual((shown["scored_windows"], shown["window"]["contributing"], shown["window"]["present"],
                          shown["artifact_sha256"], shown["device"], shown["source"]),
                         (1, 8, True, self.artifact.sha256, "cpu", "archive"))
        self.assertAlmostEqual(shown["window"]["logit"], 0.7, places=6)
        with self.assertRaisesRegex(PresenceError, "window 1 was asked for"):
            presence.summary(output, 1)
        # The lines of a feature output that a sensor wrote before, scored afterwards: the same decisions.
        (self.root / "scored").mkdir()
        presence.score(directory / "features", self.artifact.directory, self.root / "scored")
        again = presence.open_output(self.root / "scored")
        self.assertEqual([presence.comparable(line) for line in again.decisions],
                         [presence.comparable(line) for line in output.decisions])
        self.assertEqual(again.report["source"]["kind"], "features")
        problems, facts = evaluator.parity(output, again)
        self.assertEqual((problems, facts["joined"], facts["largest_logit_difference"], facts["flips"]), ([], 1, 0.0, []))
        inspected = presence.inspect(self.artifact)
        self.assertEqual((inspected["columns"], inspected["files"], inspected["vector"]),
                         (self.columns, ["head.json", "weights.f32"], 13))
        self.assertAlmostEqual(inspected["weights"]["peak_to_median_db"], 0.01, places=7)
        with self.assertRaisesRegex(PresenceError, "not empty"):
            presence.replay(self.archive, self.spec, self.artifact.directory, directory)
        # A head of another specification is refused before a record is read.
        other = make_artifact(self.root / "other", self.vector, self.columns, [0.0] * 8, 0.0, spec_sha256="8" * 64)
        (self.root / "refused").mkdir()
        with self.assertRaisesRegex(PresenceError, "the head was trained on"):
            presence.replay(self.archive, self.spec, other.directory, self.root / "refused")
        self.assertEqual(list((self.root / "refused").iterdir()), [])

    def test_a_changed_output_is_refused(self):
        def lines_of(directory):
            return [json.loads(text) for text in (directory / "decisions.jsonl").read_text().splitlines()]

        def rewrite(directory, lines):
            (directory / "decisions.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))

        def edit(change):
            def apply(directory):
                lines = lines_of(directory)
                change(lines)
                rewrite(directory, lines)
            return apply

        def report(change):
            def apply(directory):
                made = json.loads((directory / "presence.json").read_text())
                change(made)
                (directory / "presence.json").write_text(json.dumps(made))
            return apply

        cases = ((edit(lambda lines: lines[7]["score"].update(logit=-0.7)), "do not agree"),
                 (edit(lambda lines: lines[7]["decision"].update(present=False)), "do not agree"),
                 (edit(lambda lines: lines[7]["input"].update(sha256="0" * 64)), "do not agree"),
                 (edit(lambda lines: lines[7]["score"].update(probability=0.9)), "do not agree"),
                 (edit(lambda lines: lines[7].update(contributing=lines[7]["contributing"][:7])), "do not agree"),
                 (edit(lambda lines: lines[0].update(score=lines[7]["score"])), "carries a score"),
                 (edit(lambda lines: lines[7]["model"].update(weights_sha256="1" * 64)), "with the model of the report"),
                 (edit(lambda lines: lines.pop()), "7 decisions for 8 lines"),
                 (edit(lambda lines: lines[3].update(status="ok")), "is not the presence-decision/v1 line"),
                 (report(lambda made: made["counts"].update(present=0)), "counts of the report"),
                 (report(lambda made: made["runtime"].update(entry="native")), "the reference runtime"),
                 (report(lambda made: made["artifact"]["manifest"]["threshold"].update(logit=5.0)), "altered after"),
                 (lambda directory: (directory / "labels.json").write_text("{}"), "holds labels.json"),
                 (lambda directory: (directory / "features/features.jsonl").unlink(), "feature output .* does not verify"),
                 (lambda directory: (directory / "presence.json").unlink(), "not a complete presence output"))
        for number, (change, message) in enumerate(cases):
            directory, _ = self.made(f"out-{number}")
            change(directory)
            with self.subTest(message=message), self.assertRaisesRegex(PresenceError, message):
                presence.open_output(directory)

    def test_command_line(self):
        def run(*arguments):
            with contextlib.redirect_stdout(io.StringIO()) as said, contextlib.redirect_stderr(io.StringIO()) as errors:
                try:
                    status = presence.main([str(word) for word in arguments])
                except SystemExit as stop:
                    status = stop.code
            return status, said.getvalue(), errors.getvalue()

        artifact = self.artifact.directory
        status, said, _ = run("settings", "--output", self.root / "settings.json")
        self.assertEqual((status, json.loads((self.root / "settings.json").read_text())), (0, presence.demo_settings()))
        self.assertEqual(run("settings", "--output", self.root / "settings.json")[0], 2)
        status, said, _ = run("device", "--device", "cpu")
        self.assertEqual((status, json.loads(said)["device"]), (0, "cpu"))
        (self.root / "out").mkdir()
        status, said, _ = run("replay", "--archive", self.archive, "--spec", self.root / "spec.json", "--artifact",
                              artifact, "--output-dir", self.root / "out")
        self.assertEqual(status, 0)
        self.assertIn("presence: 8 decisions on cpu, 1 scored: 1 present, 0 absent; 7 of a filling window, 0 gaps, "
                      "0 invalid, 0 rejected; the head unit-head ", said)
        self.assertIn("was loaded ", said)
        status, said, _ = run("show", self.root / "out")
        self.assertEqual((status, json.loads(said)["window"]["present"]), (0, True))
        (self.root / "scored").mkdir()
        self.assertEqual(run("score", "--features", self.root / "out/features", "--artifact", artifact, "--output-dir",
                             self.root / "scored")[0], 0)
        status, said, _ = run("inspect", artifact, "--sha256", self.artifact.sha256)
        self.assertEqual((status, json.loads(said)["artifact_sha256"]), (0, self.artifact.sha256))
        # Refusals leave with status 2 and one line.
        flipped = self.root / "flipped"
        shutil.copytree(artifact, flipped)
        (flipped / "weights.f32").write_bytes(b"\x01" + (flipped / "weights.f32").read_bytes()[1:])
        other = dict(self.spec, view=dict(self.spec["view"], delay_bins=4))
        (self.root / "other.json").write_text(json.dumps(other))
        (self.root / "empty").mkdir()
        replay_arguments = ("replay", "--archive", self.archive, "--spec", self.root / "spec.json", "--artifact")
        for arguments, message in (
                ((*replay_arguments, flipped, "--output-dir", self.root / "empty"), "the checkpoint was altered"),
                ((*replay_arguments, self.root / "missing", "--output-dir", self.root / "empty"), "not a complete presence head"),
                ((*replay_arguments, artifact, "--output-dir", self.root / "out"), "is not empty"),
                ((*replay_arguments, artifact, "--device", "cuda:99", "--output-dir", self.root / "empty"), "CUDA device"),
                (("replay", "--archive", self.archive, "--spec", self.root / "other.json", "--artifact", artifact,
                  "--output-dir", self.root / "empty"), "the head was trained on"),
                (("replay", "--archive", self.root / "empty", "--spec", self.root / "spec.json", "--artifact", artifact,
                  "--output-dir", self.root / "empty"), "holds no readable consumer.json"),
                (("follow", "--socket", self.root / "no.sock", "--spec", self.root / "spec.json", "--artifact", artifact,
                  "--output-dir", self.root / "out"), "is not empty"),
                (("score", "--features", self.root / "empty", "--artifact", artifact, "--output-dir", self.root / "empty"),
                 "does not verify"),
                (("inspect", artifact, "--sha256", "0" * 64), "was asked for"),
                (("show", self.root / "empty"), "not a complete presence output")):
            status, _, errors = run(*arguments)
            with self.subTest(message=message):
                self.assertEqual(status, 2)
                self.assertIn(message, errors)
                self.assertTrue(errors.startswith("presence: "))
                self.assertEqual(errors.count("\n"), 1)
        self.assertEqual(list((self.root / "empty").iterdir()), [])


@unittest.skipIf(np is None, NEEDS)
class EvaluatorTests(unittest.TestCase):
    """The split of a set, the plans of its replay, and the join of windows and decisions to it."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.grant = ROOT / "configs/grants/two_prb_offset.json"
        config = double.set_config(cls.root / "set.json", cls.grant, double.trajectories(4, 2, 2))
        double.analytic_set(config, cls.root / "set")
        cls.set = sets.open_set(cls.root / "set")
        cls.table = evaluator.split_table(cls.set)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_the_split_is_that_of_the_set_by_trajectory(self):
        table = self.table
        self.assertEqual((table["schema"], table["unit"], table["counts"]),
                         ("presence-split/v1", "trajectory", {"training": 4, "validation": 2, "held_out": 2}))
        self.assertEqual(table["assignment_sha256"], self.set.manifest["assignment_sha256"])
        self.assertEqual(table["groups"], {"training": ["unit-training-lanes", "unit-training-line"],
                                           "validation": ["unit-validation-random"], "held_out": ["unit-held-out-line"]})
        lookup = evaluator.by_key(table)
        self.assertEqual(len(lookup), 16)
        for row in table["episodes"]:
            absent, present = lookup[row["keys"]["a"]], lookup[row["keys"]["b"]]
            # Both members of a matched pair share the split, the scene and the pair. Only the label differs.
            self.assertEqual((absent["label"], present["label"]), (0, 1))
            self.assertEqual({key: absent[key] for key in ("split", "scene", "pair")},
                             {"split": row["group"], "scene": row["trajectory"], "pair": row["episode_id"]})
            self.assertEqual({key: present[key] for key in ("split", "scene", "pair")},
                             {key: absent[key] for key in ("split", "scene", "pair")})
            # A key is the leading 64 bits of the hash of a realization identifier: it names no group and no case.
            self.assertRegex(row["keys"]["a"], "^[0-9a-f]{16}$")
            self.assertEqual((row["periods"], row["opportunities"]), (2, 6))
        self.assertEqual(json.loads(json.dumps(table)), evaluator.check_table(json.loads(json.dumps(table))))

        def changed(edit):
            copy = json.loads(json.dumps(table))
            edit(copy)
            return copy

        for edit, message in (
                (lambda copy: copy["groups"]["validation"].append("unit-training-line"), "a trajectory is in two groups"),
                (lambda copy: copy["episodes"][0].update(group="held_out"), "is not in the group of its trajectory"),
                (lambda copy: copy["episodes"][1]["keys"].update(a=copy["episodes"][0]["keys"]["a"]), "names two realizations"),
                (lambda copy: copy["counts"].update(training=5), "counts of the split table"),
                (lambda copy: copy["groups"].update(held_out=[]), "at least one trajectory"),
                (lambda copy: copy.update(schema="other"), "presence-split/v1")):
            with self.subTest(message=message), self.assertRaisesRegex(PresenceHeadError, message):
                evaluator.check_table(changed(edit))

    def test_runs_hold_some_entries_of_one_group_and_one_gain(self):
        runs = evaluator.chunks(self.table, "training", 3)
        self.assertEqual([len(run) for run in runs], [3, 1])
        self.assertEqual({row["group"] for run in runs for row in run}, {"training"})
        self.assertEqual([len(run) for run in evaluator.chunks(self.table, "held_out", 25)], [2])
        # A set whose traces differ in the last bit records two gains: no run mixes them.
        table = json.loads(json.dumps(self.table))
        table["episodes"][1]["grid_gain"] *= 1.0000001
        runs = evaluator.chunks(table, "training", 25)
        self.assertEqual(sorted(len(run) for run in runs), [1, 3])
        self.assertEqual([len({row["grid_gain"] for row in run}) for run in runs], [1, 1])
        with self.assertRaisesRegex(PresenceHeadError, "at least one entry"):
            evaluator.chunks(self.table, "training", 0)

    def test_an_entry_of_a_set_is_replayed_with_the_grids_an_accepted_entry_binds(self):
        import test_grid_apply_driver as stores

        entries = self.set.episodes("training")[:2]
        schedule = entries[0].schedule
        bindings = {}
        for grant_id, grant in schedule.grants.items():
            directory = self.root / "stores" / grant_id / "store"
            if not directory.exists():
                stores.store_of(grant, directory)
            store = grants.open_store(directory)
            bindings[grant_id] = {"x_sha256": store.manifest["files"]["x.npy"]["sha256"], "store": str(directory),
                                  "record_sha256": entries[0].manifest["grants"][grant_id]["record_sha256"]}
        rows = [row for row in self.table["episodes"] if row["group"] == "training"][:2]
        # The exporter of a set binds no transmit grid, so the plan of Sprint 2.3 refuses the entry as it is.
        self.assertTrue(all(described["x"] is None for described in entries[0].manifest["grants"].values()))
        with self.assertRaisesRegex(GridApplyError, "has no bound grid"):
            replay.entry_plan("unit", [row["directory"] for row in rows])
        with self.assertRaisesRegex(GridApplyError, "has no bound grid"):
            replay.entry_plan("unit", [row["directory"] for row in rows], bindings={})
        ring = sr.ring_spec(1, 1, "wait")
        channel, plan = evaluator.write_plans("unit-run", rows, bindings, self.root / "plans", ring, epoch=3)
        self.assertEqual((channel["schema"], len(channel["sources"]), len(channel["opportunities"])),
                         ("grid-apply-plan/v1", 4, 24))
        self.assertEqual((plan["schema"], plan["evidence"], plan["observer"]["epoch"], plan["ring"]["slots"]),
                         ("pilot-ls-plan/v2", {"mode": "timed"}, 3, 4))
        self.assertEqual(sorted(path.name for path in (self.root / "plans").iterdir()),
                         ["channel-plan.json", "plan.json", "replay-plan.json"])
        self.assertEqual({grant["x"]["sha256"] for grant in channel["grants"]},
                         {binding["x_sha256"] for binding in bindings.values()})
        keys = evaluator.keys_of(channel)
        self.assertEqual(keys, [row["keys"][case] for row in rows for case in ("a", "b")])
        # A store with another grid than the binding names is refused, and so is a plan that exists.
        wrong = {name: dict(item, x_sha256="0" * 64) for name, item in bindings.items()}
        with self.assertRaisesRegex(PresenceHeadError, "cannot be replayed: .*the entry binds"):
            evaluator.write_plans("unit-run", rows, wrong, self.root / "wrong", ring)
        with self.assertRaisesRegex(PresenceHeadError, "never overwritten"):
            evaluator.write_plans("unit-run", rows, bindings, self.root / "plans", ring)
        # The bindings of an accepted entry are read from its manifest. An entry of a set records none.
        with self.assertRaisesRegex(PresenceHeadError, "records no transmit grid for the grant"):
            evaluator.bindings_of(entries[0])

    def outputs(self, role, artifact=None, logit_shift=0.0):
        """A feature output, and with an artifact a presence output, on lattice records under the keys of one group."""
        spec = feature_tests.small_spec()
        rows = [row for row in self.table["episodes"] if row["group"] == role]
        records = []
        for row in rows:
            for case, key in sorted(row["keys"].items()):
                for record in feature_tests.lattice(static=3.0 if case == "b" else 30.0):
                    records.append(obs.Record(dict(record.header, episode=int(key, 16), sequence=len(records)),
                                              record.host, record.mask, record.payload))
        directory = Path(tempfile.mkdtemp(dir=self.root))
        lines, views, stream = features.run_records(records, spec)
        if artifact is None:
            features.write_output(directory, spec, lines, views, stream, {"kind": "archive", "directory": "unit",
                                                                         "reader": "native", "records": len(records)})
            return features.open_output(directory), rows
        (directory / "features").mkdir()
        features.write_output(directory / "features", spec, lines, views, stream,
                              {"kind": "archive", "directory": "unit", "reader": "native", "records": len(records)})
        head = presence.Head(artifact)
        decisions = presence.decide_lines(head, lines)
        for decision in decisions:
            if decision["status"] == "ok" and logit_shift:
                decision["score"]["logit"] += logit_shift
                decision["score"]["probability"] = 1.0 / (1.0 + math.exp(-decision["score"]["logit"]))
                decision["decision"]["present"] = decision["score"]["logit"] > head.threshold
        presence.write_output(directory, head, decisions, {"kind": "archive", "directory": "unit", "records": len(records)})
        return presence.open_output(directory), rows

    def test_windows_are_joined_to_the_split_by_their_episode_key(self):
        output, rows = self.outputs("training")
        keys = [row["keys"][case] for row in rows for case in ("a", "b")]
        made, account = evaluator.samples(output, self.table, "training", keys)
        self.assertEqual((len(made), account["realizations"], account["with_a_window"], account["target_present"],
                          account["target_absent"], account["missing"]), (8, 8, 8, 4, 4, []))
        for row in made:
            presence.check_sample(row)
            self.assertEqual((row["split"], row["window"].split("/")[1]), ("training", row["episode"]))
            self.assertIn(row["scene"], self.table["groups"]["training"])
        self.assertEqual(presence.check_split(made, [])["pairs"]["training"], 4)
        # The output of another split is refused, and so is a key that the set does not hold.
        with self.assertRaisesRegex(PresenceHeadError, "belongs to the training split, and this output was read as "
                                                       "validation"):
            evaluator.samples(output, self.table, "validation")
        table = json.loads(json.dumps(self.table))
        table["episodes"] = [row for row in table["episodes"] if row["group"] != "training"]
        with self.assertRaisesRegex(PresenceHeadError, "belongs to no realization of the set"):
            evaluator.samples(output, table, "training")
        # A realization without a complete window stays in the account, with the last answer of its stream.
        other = [row["keys"]["a"] for row in self.table["episodes"] if row["group"] == "held_out"]
        _, account = evaluator.samples(output, self.table, "training", keys + other[:1])
        self.assertEqual(account["missing"], [{"episode": other[0], "pair": evaluator.by_key(self.table)[other[0]]["pair"],
                                               "last_answer": None}])
        self.assertEqual(evaluator.ids_sha256(["b", "a"]), evaluator.ids_sha256(["a", "b"]))

    def test_decisions_are_compared_by_window_and_input_digest_and_counted_by_label(self):
        spec = feature_tests.small_spec()
        vector = features.run_records(feature_tests.lattice(), spec)[0][-1]["features"]["names"]
        columns = presence.select_columns(vector, ["moving_to_static_db"])
        # Moving over static power is -9.5 dB with the strong return and -29.5 dB with the weak one.
        artifact = make_artifact(self.root / "evaluated", vector, columns, [1.0], 0.0, logit=-20.0,
                                 spec_sha256=features.spec_sha256(spec))
        device, rows = self.outputs("held_out", artifact)
        offline, _ = self.outputs("held_out", artifact)
        problems, facts = evaluator.parity(device, offline)
        self.assertEqual((problems, facts["joined"], facts["flips"], facts["schema"]), ([], 4, [], "presence-parity/v1"))
        keys = [row["keys"][case] for row in rows for case in ("a", "b")]
        made = evaluator.evaluate(device, self.table, "held_out", keys)
        self.assertEqual((made["graded"], made["counts"]["target_present"], made["counts"]["target_absent"]),
                         (False, {"episodes": 2, "detected": 2, "missed": 0, "missing": 0},
                          {"episodes": 2, "false_alarm": 0, "quiet": 2, "missing": 0}))
        self.assertEqual(made["area_under_curve"], 1.0)
        # A realization without a decision stays in its denominator as missing.
        extra = [row["keys"]["b"] for row in self.table["episodes"] if row["group"] == "held_out"]
        short = evaluator.evaluate(device, self.table, "held_out", keys + [keys[1]])
        self.assertEqual(short["counts"]["target_present"]["episodes"], 3)
        self.assertEqual(len(extra), 2)
        with self.assertRaisesRegex(PresenceHeadError, "no window of the validation split"):
            evaluator.evaluate(device, self.table, "validation")
        # Scores that drift beyond the tolerance, and a decision that flips, are named.
        drift, _ = self.outputs("held_out", artifact, logit_shift=1e-3)
        problems, facts = evaluator.parity(drift, offline)
        self.assertEqual(len(problems), 4)
        self.assertIn("the logit differs by 1.000e-03", problems[0])
        flipped, _ = self.outputs("held_out", artifact, logit_shift=-15.0)
        problems, facts = evaluator.parity(flipped, offline)
        self.assertEqual(len(facts["flips"]), 2)  # the two target-present windows fall below the threshold
        self.assertAlmostEqual(facts["flips"][0]["margin"], -9.542 - 15.0 + 20.0, places=2)
        other = make_artifact(self.root / "other", vector, columns, [2.0], 0.0, logit=-20.0,
                              spec_sha256=features.spec_sha256(spec))
        problems, _ = evaluator.parity(self.outputs("held_out", other)[0], offline)
        self.assertIn("the two outputs were scored with different artifacts", problems)


@unittest.skipIf(np is None, NEEDS)
class LiveSensorTests(unittest.TestCase):
    """The presence sensor in a second process, on a native producer of the ring."""

    @classmethod
    def setUpClass(cls):
        cls.missing = [tool for tool in ("cmake",) if not shutil.which(tool)]
        if not any(shutil.which(compiler) for compiler in ("c++", "g++", "clang++")):
            cls.missing.append("a C++17 compiler")
        if cls.missing:
            return
        cls.temporary = tempfile.TemporaryDirectory()
        cls.build = Path(cls.temporary.name) / "build"
        cls.log, cls.status = "", 0
        for step in (["cmake", "-S", str(ROOT / "native/obs_ring"), "-B", str(cls.build)],
                     ["cmake", "--build", str(cls.build), "--target", "obs_feed", "-j2"]):
            result = subprocess.run(step, capture_output=True, text=True, timeout=1800, check=False)
            cls.log += result.stdout + result.stderr
            cls.status = cls.status or result.returncode

    @classmethod
    def tearDownClass(cls):
        if not cls.missing:
            cls.temporary.cleanup()

    def setUp(self):
        if self.missing:
            self.skipTest("the native producer needs " + ", ".join(self.missing))
        if self.status:
            self.fail("the native ring does not build:\n" + self.log[-3000:])
        scratch = tempfile.TemporaryDirectory(dir=self.temporary.name)
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def test_the_sensor_loads_once_warms_up_and_decides_on_a_native_ring(self):
        # obs_feed publishes one record per 20 slots, slot 8 of a period of 20, with the shape of the
        # demonstration and sample times of its own, which are not those of a slot: every record is
        # rejected by the stream, and every rejection is a decision without a score.
        spec = features.demo_spec()
        spec["radio"]["period_slots"] = 20
        spec["sequence"]["slots_in_period"] = [8]
        (self.scratch / "spec.json").write_text(json.dumps(spec))
        vector = list(features.LEADING) + [f"profile_db@{value:+g}Hz" for value in (-50.0, -37.5, -25.0, -12.5, 12.5, 25.0, 37.5)]
        columns = presence.select_columns(vector, presence.DEMO_COLUMNS)
        artifact = make_artifact(self.scratch / "artifact", vector, columns, [0.1] * len(columns), -1.0,
                                 spec_sha256=features.spec_sha256(spec))
        out = self.scratch / "sensor"
        out.mkdir()
        socket, ready = self.scratch / "ring.sock", self.scratch / "ready.json"
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        sensor = subprocess.Popen([sys.executable, "-m", "ocudu_cuda_isac.sensing.presence", "follow", "--socket",
                                   str(socket), "--spec", str(self.scratch / "spec.json"), "--artifact",
                                   str(artifact.directory), "--output-dir", str(out), "--wait", "wait", "--ready-file",
                                   str(ready)], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # The producer starts when the sensor says that it is warm, as the launcher of the gate does.
        for _ in range(1200):
            if ready.is_file() or sensor.poll() is not None:
                break
            subprocess.run(["sleep", "0.05"], check=False)
        self.assertTrue(ready.is_file(), "the sensor did not become ready")
        feed = subprocess.run([str(self.build / "obs_feed"), "--socket", str(socket), "--records", "24",
                               "--period-us", "5000"], capture_output=True, text=True, timeout=120, check=False)
        said, errors = sensor.communicate(timeout=120)
        self.assertEqual((feed.returncode, sensor.returncode), (0, 0), feed.stderr + errors)
        fed = json.loads(feed.stdout)
        self.assertEqual((fed["published"] + fed["dropped"], fed["released"], fed["pending"]), (24, fed["published"], 0))
        self.assertEqual(fed["dropped"], 0, "the sensor did not keep the pace of one record per 5 ms")
        self.assertIn("presence: 24 decisions on ", said)
        self.assertIn("was loaded 1 time(s)", said)
        output = presence.open_output(out)
        report = output.report
        self.assertEqual((report["counts"]["decisions"], report["counts"]["rejected"], report["counts"]["scored"],
                          report["loads"], report["source"]["kind"], report["source"]["ended"], report["source"]["records"]),
                         (24, 24, 0, 1, "ring", "closed", 24))
        self.assertTrue(all(line["status"] == "rejected" and line["score"] is None for line in output.decisions))
        self.assertTrue(all(type(line["t_decision_ns"]) is int and line["t_decision_ns"] >= line["t_features_ns"]
                            for line in output.decisions))
        # It was warm before it attached: the file names the device and the artifact, and both warm-ups ran.
        warm = json.loads(ready.read_text())
        self.assertEqual((warm["device"], warm["artifact_sha256"]), (str(presence.device_of(presence.default_device())),
                                                                     artifact.sha256))
        self.assertGreater(warm["warm_up_ms"]["features"], 0.0)
        self.assertGreater(warm["warm_up_ms"]["head"], 0.0)
        self.assertEqual(report["source"]["warm_up_ms"], warm["warm_up_ms"])
        first = min(record.header["t_publish_ns"] for record in obs.open_archive(out / "features/records").records)
        self.assertLess(warm["t_ready_ns"], first)
        self.assertEqual(report["runtime"]["type"], presence.default_device())
        # With sample times of a slot the records it kept fill windows, and the same head scores them offline.
        fixed = []
        for record in obs.open_archive(out / "features/records").records:
            slot = record.header["absolute_slot"]
            header = dict(record.header, t_first_sample=features.symbol_time_tc(30, slot, 2),
                          t_last_sample=features.symbol_time_tc(30, slot, 11) + features.estimator.useful_tc(30))
            fixed.append(obs.Record(header, record.host, record.mask, record.payload))
        decisions = presence.run_records(fixed, spec, presence.Head(artifact))[3]
        self.assertEqual([line["status"] for line in decisions], (["insufficient_history"] * 7 + ["ok"]) * 3)
        self.assertEqual(len([line for line in decisions if line["score"] is not None]), 3)


class TreeTests(unittest.TestCase):
    FILES = ("src/ocudu_cuda_isac/sensing/presence.py", "src/ocudu_cuda_isac/episodes/presence_head.py",
             "src/ocudu_cuda_isac/episodes/grid_apply.py", "scripts/train_presence.py", "scripts/accept_s3_4.sh",
             "tests/unit/test_presence_head.py", "tests/unit/presence_head_double.py",
             "tests/unit/test_presence_head_driver.py", "tests/unit/test_core_imports.py",
             "tests/integration/presence_head.py", "configs/integration/presence_head.json", "tests/test_accept_s3_4.py",
             "docs/interfaces/PRESENCE_HEAD.md", "Makefile")

    def test_files_are_ascii_and_do_not_name_the_project(self):
        # The working name lives on line 1 of the top-level README and nowhere else.
        name = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").split(":")[0].strip()
        for relative in self.FILES:
            data = (ROOT / relative).read_bytes()
            self.assertTrue(data.isascii(), relative)
            if name and name.lower() != "ocudu-cuda-isac":
                self.assertNotIn(name.lower(), data.decode("ascii").lower(), relative)

    def test_the_gate_and_the_configuration_name_the_demonstration(self):
        config = json.loads((ROOT / "configs/integration/presence_head.json").read_text())
        self.assertEqual((config["schema"], config["selector"], config["backends"]),
                         ("integration-config/presence_head/v1", "presence_head", ["cpu", "cuda"]))
        # The specification and the ring are those of Sprint 3.3, whose live run is the control.
        earlier = json.loads((ROOT / "configs/integration/pilot_features.json").read_text())
        self.assertEqual(config["spec"], earlier["spec"])
        self.assertEqual(config["ring"], earlier["ring"])
        self.assertEqual(config["ring"]["slots"], obs.slots_for(config["ring"]["window_records"],
                                                                config["ring"]["burst_records"]))
        self.assertEqual(config["control"], {"driver": "pilot_features", "run": "live"})
        # The demonstration set: 300 training, 40 validation and 10 held-out pairs, by trajectory.
        demo = json.loads((ROOT / "configs/episodes/demo_set.json").read_text())
        counts = dict.fromkeys(("training", "validation", "held_out"), 0)
        for trajectory in demo["trajectories"]:
            counts[trajectory["group"]] += trajectory["episodes"]
            self.assertGreaterEqual(trajectory["periods"], config["spec"]["window"]["periods"])
        self.assertEqual(config["set"]["expect"], {"set_id": demo["set_id"], "episodes": counts})
        self.assertEqual(counts, {"training": 300, "validation": 40, "held_out": 10})
        # The head: the peak over the median and the Doppler profile, by name. A false-alarm target of 0.05
        # allows two of the forty target-absent validation windows.
        training = config["training"]
        self.assertEqual((training["schema"], training["columns"]), ("presence-train-config/v1",
                                                                    ["peak_to_median_db", "profile_db@*"]))
        self.assertEqual(math.floor(training["threshold"]["target_false_alarm"] * counts["validation"]), 2)
        self.assertLessEqual(training["threshold"]["min_negatives"], counts["validation"])
        self.assertLessEqual(training["min_per_class"], counts["training"])
        self.assertEqual((config["sensor"]["device"], config["receiver"]), ("cuda", {"keep_grids": False}))
        self.assertEqual(config["tolerance"], {"logit_absolute": 1e-4, "probability_absolute": 1e-5, "reload_absolute": 1e-6})
        self.assertIn("accept-s3.4:", (ROOT / "Makefile").read_text())
        runner = (ROOT / "scripts/accept_s3_4.sh").read_text()
        for needed in ("tests/unit/test_presence_head.py", "tests/unit/test_core_imports.py",
                       "tests/integration/presence_head.py --config configs/integration/presence_head.json",
                       "-m ocudu_cuda_isac.sensing.presence inspect", "-m ocudu_cuda_isac.sensing.presence show",
                       "-m ocudu_cuda_isac.sensing.presence device", "sha256sum"):
            self.assertIn(needed, runner)
        if np is not None:
            self.assertEqual(training, presence.demo_settings())
            self.assertEqual(config["spec"], features.demo_spec())
            self.assertEqual(config["tolerance"], evaluator.TOLERANCE)
            names = features.feature_names({"doppler": {"hz": [25.0 * index for index in range(-4, 4)]}})
            self.assertEqual(len(presence.select_columns(names, training["columns"])), 8)

    def test_this_sprint_adds_no_native_source_and_amends_one_plan_builder(self):
        for directory in ("native/obs_ring", "native/ocudu_adapter", "native/sense_core"):
            for path in (ROOT / directory).rglob("*"):
                self.assertNotIn("presence", path.name, path)
        # The plan of Sprint 2.3 takes one more argument. Without it an entry that binds no grid is refused as before.
        source = (ROOT / "src/ocudu_cuda_isac/episodes/grid_apply.py").read_text()
        self.assertIn('def entry_plan(run_id, entries, stores=(), cases=("a", "b"), bindings=None):', source)
        self.assertIn("            if binding is None and bindings is not None:\n"
                      "                binding = bindings.get(grant_id)\n", source)
        # The head names its runtime and its device in every registry entry, and the core imports no evaluator.
        head = (ROOT / "src/ocudu_cuda_isac/sensing/presence.py").read_text()
        self.assertIn('RUNTIME = features.RUNTIME', head)
        self.assertNotIn("episodes", "".join(line for line in head.splitlines() if line.startswith(("import", "from"))))
        script = (ROOT / "scripts/train_presence.py").read_text()
        self.assertNotIn("ocudu_cuda_isac.episodes", script)


if __name__ == "__main__":
    unittest.main()
