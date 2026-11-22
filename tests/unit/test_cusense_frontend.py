#!/usr/bin/env python3
"""Unit gate of Sprint 3.5 (selector cusense_frontend): python tests/unit/test_cusense_frontend.py -v

The reference of the cuSense-like branch is checked on values whose answer can be worked out by
hand: the template of two valid samples and one invalid zero, the window of two slots, the
magnitude left after a template of 0.5, the z-score with a mean of 0.2 and a deviation of 0.1, a
slot at resource block 2, the element of the receiver's layout that the card names, a map that
sums to one for both input widths, the KL divergence of a distribution with itself and of a
uniform map against one cell, and a target-absent batch that sends no gradient into the map.
Statistics that name a validation or a held-out window are refused, and records of the future
and fields of the truth must not change a window. The training script then freezes one artifact
of four files from two sample sets, and the evaluator side plans the replay of a small episode
set, joins its windows to positions and counts what a sensor decided.

NumPy and PyTorch are needed. The last class but one attaches the sensor to a native producer of
the ring, and needs cmake and a C++17 compiler for that.
"""

import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
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
    from ocudu_cuda_isac.episodes import cusense_frontend as evaluator  # noqa: E402
    from ocudu_cuda_isac.episodes import episode_set as sets  # noqa: E402
    from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
    from ocudu_cuda_isac.episodes import native_estimate as ne  # noqa: E402
    from ocudu_cuda_isac.episodes.cusense_frontend import CuSenseFrontendError  # noqa: E402
    from ocudu_cuda_isac.sensing import cusense  # noqa: E402
    from ocudu_cuda_isac.sensing import features  # noqa: E402
    from ocudu_cuda_isac.sensing import native_estimate as o3_reference  # noqa: E402
    from ocudu_cuda_isac.sensing import pilot_ls as o2_reference  # noqa: E402
    from ocudu_cuda_isac.sensing import presence  # noqa: E402
    from ocudu_cuda_isac.sensing.cusense import CuSenseError  # noqa: E402
    from ocudu_cuda_isac.sensing.models import cusense_cnn  # noqa: E402
    from ocudu_cuda_isac.sensing.models.cusense_cnn import ModelError  # noqa: E402
    import presence_head_double as set_double  # noqa: E402
    import train_cusense  # noqa: E402

NEEDS = "the reference of the cuSense-like branch is NumPy and PyTorch"
KEY = 0x0123456789ABCDEF
SYMBOLS = (2, 11)
AREA = {"x_m": [0.61, 7.39], "y_m": [1.47, 11.53]}


# ---------------------------------------------------------------- fixtures
def times_of(slot, symbols=SYMBOLS):
    return (features.symbol_time_tc(30, slot, symbols[0]),
            features.symbol_time_tc(30, slot, symbols[-1]) + features.estimator.useful_tc(30))


def header_of(slot, sequence, shape, strides, stage, layout, dtype, steps, step, count, payload, mask, valid, episode=KEY,
              epoch=1, first_subcarrier=0, antenna_map=None, symbols=SYMBOLS, evidence=2, opportunity=None):
    marks = 0
    for symbol in symbols:
        marks |= 1 << symbol
    rx = shape[2] if stage == 3 else shape[0]
    first, last = times_of(slot, symbols)
    return obs.new_header(
        epoch=epoch, sequence=sequence, episode=episode, opportunity=1000 + sequence if opportunity is None else opportunity,
        grant=7, absolute_slot=slot, cell=1, ue=0x4601, stage=stage, source=1, layout=layout, dtype=dtype,
        producer_steps=steps, scale=1.4125375747680664, rank=len(shape), shape=list(shape), strides=list(strides),
        first_subcarrier=first_subcarrier, subcarrier_step=step, subcarrier_count=count, dmrs_symbols=marks, rx_count=rx,
        layer_count=1, antenna_map=list(antenna_map or range(rx)), mask_bytes=len(mask), payload_bytes=len(payload),
        host_bytes=0, time_unit=1, digest_kind=0, evidence=evidence, port=0, t_first_sample=first, t_last_sample=last,
        t_ready_ns=1_000_000 * sequence, t_publish_ns=1_000_000 * sequence + 500, valid=valid)


def o3_record(values, slot, sequence, valid=None, steps=7, **options):
    """A record of the stage O3 from values [A, K, S]: the payload is [dmrs_symbol][prb][rx_port][12] as FP16."""
    values = np.asarray(values, dtype=np.complex128)
    planes, positions, symbols = values.shape
    blocks = positions // 12
    ordered = values.reshape(planes, blocks, 12, symbols).transpose(3, 1, 0, 2)
    payload = np.stack([ordered.real, ordered.imag], axis=-1).astype("<f2").tobytes()
    bits = np.ones(values.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    bits = bits.reshape(planes, blocks, 12, symbols).transpose(3, 1, 0, 2).reshape(-1)
    mask = np.packbits(bits, bitorder="little").tobytes()
    shape = [symbols, blocks, planes, 12]
    header = header_of(slot, sequence, shape, [blocks * planes * 12, planes * 12, 12, 1], 3, 1, 2, steps, 1, positions,
                       payload, mask, int(bits.sum()), **options)
    return obs.Record(header, b"", mask, payload)


def o2_record(values, slot, sequence, valid=None, **options):
    """A record of the stage O2 from values [A, pilots, S]: the payload is [rx_plane][reference_symbol][pilot] as complex64."""
    values = np.asarray(values, dtype=np.complex128)
    planes, pilots, symbols = values.shape
    payload = values.transpose(0, 2, 1).astype("<c8").tobytes()
    bits = np.ones(values.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    bits = bits.transpose(0, 2, 1).reshape(-1)
    mask = np.packbits(bits, bitorder="little").tobytes()
    header = header_of(slot, sequence, [planes, symbols, pilots], [symbols * pilots, pilots, 1], 2, 2, 1, 1, 2, pilots,
                       payload, mask, int(bits.sum()), **options)
    return obs.Record(header, b"", mask, payload)


def small_spec(entry="o3_fp16", subcarriers=48, rx=2, slots=(7, 8, 9), **options):
    return cusense.demo_spec(entry, subcarriers=subcarriers, rx=rx, slots=slots, **options)


def grid_of(rows=4, cols=3):
    return cusense.grid_record({"rows": rows, "cols": cols,
                                "cell_m": [(AREA["y_m"][1] - AREA["y_m"][0]) / rows, (AREA["x_m"][1] - AREA["x_m"][0]) / cols],
                                "origin_m": [AREA["x_m"][0], AREA["y_m"][0]], "area": AREA, "rule": "unit grid"})


def channel(generator, rx=2, positions=48, symbols=2, cell=None, noise=0.02):
    """One slot of a static channel near 1, with a ripple whose period and phase name a cell when a target is there."""
    values = np.ones((rx, positions, symbols), dtype=np.complex128)
    if cell is not None:
        row, col = cell
        along = np.arange(positions)[None, :, None]
        element = np.arange(rx)[:, None, None]
        values = values + 0.2 * np.exp(2j * np.pi * ((row + 1) * along / positions + 0.25 * (col + 1) * element))
    return values + noise * (generator.standard_normal(values.shape) + 1j * generator.standard_normal(values.shape))


def periods_of(count, generator, cell=None, episode=KEY, first_sequence=0, slots=(7, 8, 9), rx=2, positions=48, **options):
    """The records of count periods of one realization, in the order of the ring."""
    records = []
    for period in range(count):
        for slot in slots:
            records.append(o3_record(channel(generator, rx, positions, cell=cell), 10 * period + slot,
                                     first_sequence + len(records), episode=episode, **options))
    return records


def rows_of(lines, split="training", label=0, scene="walk", pair="ep", cell=None, first=0):
    """Sample rows for the complete windows of a sealed stream."""
    grid = grid_of()
    rows = []
    for line in lines:
        if line["status"] != "ok":
            continue
        rows.append({"schema": cusense.SAMPLE_SCHEMA, "window": line["window"]["id"], "episode": line["window"]["episode"],
                     "split": split, "scene": scene, "pair": pair, "label": label,
                     "cell": list(cell) if label else None,
                     "position_m": cusense.cell_centre(grid, cell) if label else None,
                     "spec_sha256": line["spec_sha256"], "elements": line["rx"]["elements"],
                     "producer_steps": line["input"]["producer_steps"], "shape": line["mean"]["shape"],
                     "index": first + len(rows), "mean_sha256": line["mean"]["sha256"],
                     "count_sha256": line["count"]["sha256"]})
    return rows


def sample_set(split, pairs=6, periods=2, seed=3, spec=None):
    """A sample set of matched pairs: (rows, means, counts). Pair n walks through the cells of a 4 by 3 grid."""
    spec = spec or small_spec()
    generator = np.random.Generator(np.random.PCG64(seed))
    rows, means, counts = [], [], []
    offset = {"training": 0x1000, "validation": 0x2000, "held_out": 0x3000}[split]
    for pair in range(pairs):
        cell = (pair % 4, pair % 3)
        for label in (0, 1):
            records = periods_of(periods, generator, cell if label else None, episode=offset + 2 * pair + label)
            lines, stream = cusense.run_records(records, spec)
            mean, count = cusense.stacked(stream)
            rows += rows_of(lines, split, label, f"walk-{split}-{pair % 2}", f"ep-{split}-{pair}", cell, len(rows))
            means.append(mean)
            counts.append(count)
    return rows, np.concatenate(means), np.concatenate(counts)


def small_settings(epochs=60, **options):
    settings = cusense.demo_settings("unit-model")
    settings["model"].update(stem_channels=4, block_channels=[4, 8, 8], hidden=[8, 8], dropout=[0.0, 0.0])
    settings["optimizer"].update(epochs=epochs, batch_size=8, learning_rate=0.02)
    settings["loss"].update(sigma_cells=0.5)
    settings.update(min_per_class=1, device="cpu")
    settings["threshold"].update(min_negatives=1)
    settings.update(options)
    return settings


def threshold_of(logit=0.0, target=0.0, negatives=1, allowed=0, exceeding=0):
    return {"entry": "frozen_threshold", "logit": logit, "probability": 1.0 / (1.0 + math.exp(-logit)),
            "rule": presence.THRESHOLD_RULE, "target_false_alarm": target, "negatives": negatives, "allowed": allowed,
            "exceeding": exceeding}


def make_artifact(directory, spec=None, template=None, mean=0.0, scale=1.0, seed=5, smoothing=None, grid=None,
                  elements=None, steps=("pilot_ls", "frequency_interpolation", "smoothing"), **threshold):
    """An artifact with a template of ones, a normalizer that changes nothing and a small model with random weights."""
    spec = spec or small_spec()
    grid = grid or grid_of()
    shape = cusense.shape_of(spec)
    fitted = {"split": "training", "windows": 4, "windows_sha256": "a" * 64}
    template = template if template is not None else {"values": np.ones(shape, dtype="<c8"),
                                                      "count": np.full(shape, 12, dtype="<u4")}
    settings = small_settings()
    torch.manual_seed(seed)
    net = cusense_cnn.CuSenseCNN(cusense.model_config(settings, shape[0], grid)).eval()
    with torch.no_grad():
        for layer in (net.location, net.presence):
            layer.weight.normal_(0.0, 0.5)
    cusense.write_artifact(directory, "unit-model", spec, elements or list(range(shape[0])), list(steps),
                           dict(template, fitted_on=template.get("fitted_on", fitted)),
                           {"mean": mean, "scale": scale, "fitted_on": fitted}, net, grid, threshold_of(**threshold),
                           smoothing or {"entry": "none"}, 0.5)
    return cusense.open_artifact(directory)


@unittest.skipIf(np is None, NEEDS)
class FrontEndTests(unittest.TestCase):
    """The three stages on values of the card."""

    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def test_the_template_of_the_card_is_2_from_two_valid_samples(self):
        # Calibration values 1+1j, 3-1j and one invalid zero at one element.
        mean, count = cusense.coherent_mean([1 + 1j, 3 - 1j, 0j], [True, True, False])
        self.assertEqual((complex(mean), int(count)), (2 + 0j, 2))
        # The same through the fit: three calibration windows of one record each, the third without a valid sample.
        shape = (1, 16, 1)
        means = np.stack([np.full(shape, value, dtype="<c8") for value in (1 + 1j, 3 - 1j, 0j)])
        counts = np.stack([np.full(shape, value, dtype="u1") for value in (1, 1, 0)])
        rows = [{"window": f"w{index}", "split": "training", "label": 0} for index in range(3)]
        template = cusense.fit_template(rows, means, counts)
        self.assertTrue(np.all(template["values"] == 2 + 0j))
        self.assertTrue(np.all(template["count"] == 2))
        self.assertEqual((template["values"].dtype, template["count"].dtype, template["fitted_on"]["split"],
                          template["fitted_on"]["windows"]), (np.dtype("<c8"), np.dtype("<u4"), "training", 3))
        # A zero is invalid by its magnitude as well: the rule that leaves zero padding out of every average.
        spec = small_spec(subcarriers=24, rx=1, slots=(7, 8, 9))
        records = [o3_record(np.full((1, 24, 2), value), slot, index)
                   for index, (slot, value) in enumerate(((7, 1 + 1j), (8, 3 - 1j), (9, 0j)))]
        lines, stream = cusense.run_records(records, spec)
        self.assertEqual([line["status"] for line in lines], ["insufficient_history", "insufficient_history", "ok"])
        mean, count = stream.last
        self.assertTrue(np.all(mean == 2 + 0j) and np.all(count == 2))
        self.assertEqual(lines[-1]["validity"]["records_per_element"], {"min": 2, "max": 2})
        # A window that holds twice the records counts twice in a template.
        merged, total = cusense.merge_means([[2 + 0j], [5 + 0j]], [[2], [1]])
        self.assertEqual((complex(merged[0]), int(total[0])), (3 + 0j, 3))

    def test_the_window_of_the_card_averages_to_0_5_plus_0_5j(self):
        mean, count = cusense.coherent_mean([1 + 0j, 0 + 1j], [True, True])
        self.assertEqual((complex(mean), int(count)), (0.5 + 0.5j, 2))
        self.assertAlmostEqual(abs(complex(mean)), 0.7071, places=4)
        spec = small_spec(subcarriers=24, rx=1, slots=(7, 8))
        records = [o3_record(np.full((1, 24, 2), 1 + 0j), 7, 0), o3_record(np.full((1, 24, 2), 0 + 1j), 8, 1)]
        lines, stream = cusense.run_records(records, spec)
        self.assertEqual([line["status"] for line in lines], ["insufficient_history", "ok"])
        mean, count = stream.last
        self.assertEqual((mean.shape, mean.dtype, count.dtype), ((1, 24, 2), np.dtype("<c8"), np.dtype("u1")))
        self.assertTrue(np.all(mean == np.complex64(0.5 + 0.5j)) and np.all(count == 2))
        line = lines[-1]
        self.assertEqual((line["window"]["id"], line["window"]["period"],
                          [item["absolute_slot"] for item in line["contributing"]]), (f"1/{KEY:016x}/0", 0, [7, 8]))
        self.assertEqual(line["mean"]["sha256"], hashlib.sha256(mean.tobytes()).hexdigest())
        # The window lasts from the first DM-RS symbol of its first slot to the last one of its last slot.
        self.assertEqual((line["time"]["t_first_sample"], line["time"]["t_last_sample"]), (times_of(7)[0], times_of(8)[1]))
        self.assertAlmostEqual(line["time"]["acquisition_us"], 500.0 + 321.09 + 33.33, delta=0.5)

    def test_the_template_magnitude_and_the_z_score_of_the_card(self):
        window = np.full((1, 16, 2), 0.5 + 0.5j)
        template = np.full((1, 16, 2), 0.5 + 0j)
        count = np.ones((1, 16, 2), dtype="u1")
        d, valid = cusense.background_subtract(window, count, template, count)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(float(d[0, 0]), 0.2071, places=4)
        self.assertEqual(float(d[0, 0]), math.sqrt(0.5) - 0.5)
        x = cusense.zscore(d, valid, 0.2, 0.1)
        self.assertEqual((x.dtype, x.shape), (np.dtype(np.float32), (1, 16)))
        self.assertAlmostEqual(float(x[0, 0]), 0.071, places=3)
        self.assertEqual(float(x[0, 0]), float(np.float32((math.sqrt(0.5) - 0.5 - 0.2) / (0.1 + 1e-8))))
        # The subtraction is of magnitudes: a template of another phase leaves the same value.
        turned, _ = cusense.background_subtract(window, count, template * 1j, count)
        self.assertEqual(float(turned[0, 0]), float(d[0, 0]))
        # The DM-RS symbols are averaged, over those that hold a sample in the window and in the template.
        two = np.stack([np.full((1, 16), 0.7 + 0j), np.full((1, 16), 0.9 + 0j)], axis=2)
        d, valid = cusense.background_subtract(two, count, template, count)
        self.assertAlmostEqual(float(d[0, 3]), 0.3)
        holes = count.copy()
        holes[0, 3, 1] = 0
        d, valid = cusense.background_subtract(two, holes, template, count)
        self.assertAlmostEqual(float(d[0, 3]), 0.2)
        holes[0, 3, 0] = 0
        d, valid = cusense.background_subtract(two, holes, template, count)
        self.assertEqual((float(d[0, 3]), bool(valid[0, 3]), bool(valid[0, 4])), (0.0, False, True))
        # A position without a valid sample is 0 after the z-score, whatever the statistics are.
        self.assertEqual(float(cusense.zscore(d, valid, 5.0, 2.0)[0, 3]), 0.0)
        with self.assertRaisesRegex(CuSenseError, "are not one \\[A, K, S\\] support"):
            cusense.background_subtract(window, count, template[:, :8], count[:, :8])

    def test_a_slot_at_resource_block_2_lands_on_subcarrier_24(self):
        spec = small_spec(subcarriers=48, rx=1, slots=(7, 8))
        self.assertEqual(cusense.place(24, spec["support"], 24), 24)
        # Two slots with allocations of two blocks each, the second from resource block 2.
        records = [o3_record(np.full((1, 24, 2), 1 + 0j), 7, 0, first_subcarrier=0),
                   o3_record(np.full((1, 24, 2), 3 + 0j), 8, 1, first_subcarrier=24)]
        lines, stream = cusense.run_records(records, spec)
        mean, count = stream.last
        self.assertEqual([item["first_position"] for item in lines[-1]["contributing"]], [0, 24])
        self.assertTrue(np.all(mean[0, :24] == 1) and np.all(mean[0, 24:] == 3) and np.all(count == 1))
        self.assertEqual(complex(mean[0, 24, 0]), 3 + 0j)
        # Both on the same blocks: each element is the mean of two slots.
        both = [o3_record(np.full((1, 24, 2), 1 + 0j), 7, 0, first_subcarrier=24),
                o3_record(np.full((1, 24, 2), 3 + 0j), 8, 1, first_subcarrier=24)]
        mean, count = cusense.run_records(both, dict(spec, window=dict(spec["window"], min_valid_fraction=0.5)))[1].last
        self.assertTrue(np.all(mean[0, 24:] == 2) and np.all(count[0, 24:] == 2) and np.all(count[0, :24] == 0))
        # On the comb a pilot lies on every second subcarrier: resource block 2 starts at position 12.
        comb = small_spec("o2_comb", subcarriers=24, rx=1)
        self.assertEqual(cusense.place(24, comb["support"], 12), 12)
        for first, count in ((25, 12), (24, 13), (-2, 4)):
            with self.subTest(first=first), self.assertRaisesRegex(CuSenseError, "does not lie on the support"):
                cusense.place(first, comb["support"], count)
        # An allocation beyond the support is no input: its record is rejected before its payload is read.
        beyond = o3_record(np.ones((1, 24, 2)), 7, 0, first_subcarrier=36)
        self.assertIn("does not lie on the support", cusense.refusal(beyond, spec))

    def test_an_element_of_the_receiver_layout_arrives_where_the_card_names_it(self):
        # Element [1][3][2][5] of an ocudu_o3_fp16 payload is DM-RS symbol 1, receive port 2 and subcarrier 41.
        shape = [2, 4, 4, 12]
        position = o3_reference.element(shape, 1, 3, 2, 5)
        self.assertEqual(o3_reference.coordinate(shape, position)["subcarrier"], 41)
        generator = np.random.Generator(np.random.PCG64(1))
        values = generator.standard_normal((4, 48, 2)) + 1j * generator.standard_normal((4, 48, 2))
        record = o3_record(values, 7, 0)
        spec = small_spec(subcarriers=48, rx=4)
        made, valid = cusense.planes(record, spec)
        listed = o3_reference.payload_values(record.payload)
        self.assertEqual(complex(made[2, 41, 1]), listed[position])
        # Every element, against the reader of Sprint 5.2: widened from FP16 once, with no scale.
        for index, value in enumerate(listed):
            where = o3_reference.coordinate(shape, index)
            self.assertEqual(complex(made[where["rx_port"], where["subcarrier"], where["dmrs_symbol"]]), value)
        self.assertTrue(valid.all())
        self.assertEqual(complex(made[2, 41, 1]), complex(np.float16(values[2, 41, 1].real), np.float16(values[2, 41, 1].imag)))
        # The receive planes are sorted by their array element, whatever the order of the payload.
        turned = o3_record(values, 7, 0, antenna_map=[3, 2, 1, 0])
        self.assertTrue(np.array_equal(cusense.planes(turned, spec)[0], made[::-1]))
        # The same values on the pilot comb, in the layout of the stage O2.
        comb = small_spec("o2_comb", subcarriers=24, rx=4)
        pilots = o2_record(values[:, 0::2, :], 7, 0)
        made_o2, _ = cusense.planes(pilots, comb)
        self.assertEqual(made_o2.shape, (4, 24, 2))
        self.assertTrue(np.array_equal(made_o2, values[:, 0::2, :].astype("<c8").astype(np.complex128)))
        listed = o2_reference.payload_values(pilots.payload)
        self.assertEqual(complex(made_o2[2, 5, 1]), listed[(2 * 2 + 1) * 24 + 5])

    def test_both_input_entries_run_the_same_pipeline(self):
        generator = np.random.Generator(np.random.PCG64(2))
        o3, o2 = small_spec("o3_fp16", subcarriers=48, rx=2), small_spec("o2_comb", subcarriers=24, rx=2)
        first, second = [], []
        for index, slot in enumerate((7, 8, 9)):
            values = channel(generator)
            exact = np.float16(values.real).astype(np.float64) + 1j * np.float16(values.imag).astype(np.float64)
            first.append(o3_record(exact, slot, index))
            second.append(o2_record(exact[:, 0::2, :], slot, index))
        full, half = cusense.run_records(first, o3), cusense.run_records(second, o2)
        self.assertEqual([line["status"] for line in full[0]], [line["status"] for line in half[0]])
        self.assertEqual((full[1].last[0].shape, half[1].last[0].shape), ((2, 48, 2), (2, 24, 2)))
        self.assertTrue(np.array_equal(full[1].last[0][:, 0::2, :], half[1].last[0]))
        self.assertEqual([item["opportunity"] for item in full[0][-1]["contributing"]],
                         [item["opportunity"] for item in half[0][-1]["contributing"]])
        self.assertEqual((full[0][-1]["input"]["entry"], half[0][-1]["input"]["entry"]), ("o3_fp16", "o2_comb"))
        # A record of the other stage is no input of a specification.
        self.assertIn("the specification reads O3", cusense.refusal(second[0], o3))
        self.assertIn("the specification reads O2", cusense.refusal(first[0], o2))

    def test_an_invalid_element_takes_part_in_no_average(self):
        spec = small_spec(subcarriers=24, rx=1, slots=(7, 8))
        values = np.full((1, 24, 2), 1 + 0j)
        clean = cusense.run_records([o3_record(values, 7, 0), o3_record(values * 3, 8, 1)], spec)[1].last
        # One element of the first slot is a hole that holds a large sentinel.
        spoiled, valid = values.copy(), np.ones(values.shape, dtype=bool)
        spoiled[0, 5, 1], valid[0, 5, 1] = 6e4, False
        lines, stream = cusense.run_records([o3_record(spoiled, 7, 0, valid=valid), o3_record(values * 3, 8, 1)], spec)
        mean, count = stream.last
        self.assertEqual((complex(mean[0, 5, 1]), int(count[0, 5, 1]), complex(mean[0, 5, 0]), int(count[0, 5, 0])),
                         (3 + 0j, 1, 2 + 0j, 2))
        other = np.ones(mean.shape, dtype=bool)
        other[0, 5, 1] = False
        self.assertTrue(np.array_equal(mean[other], clean[0][other]))
        self.assertEqual(lines[-1]["validity"]["records_per_element"], {"min": 1, "max": 2})
        # A value that is not finite is a hole even with its validity bit set.
        broken = values.copy()
        broken[0, 7, 0] = complex(float("inf"), 0.0)
        mean, count = cusense.run_records([o3_record(broken, 7, 0), o3_record(values * 3, 8, 1)], spec)[1].last
        self.assertEqual((complex(mean[0, 7, 0]), int(count[0, 7, 0])), (3 + 0j, 1))
        # A window whose valid support is too small is answered as invalid and carries no tensor.
        lines, stream = cusense.run_records([o3_record(values, 7, 0, valid=np.zeros(values.shape, dtype=bool)),
                                             o3_record(values * 0, 8, 1)], spec)
        self.assertEqual((lines[-1]["status"], "mean" in lines[-1], len(stream.tensors)), ("invalid", False, 0))
        self.assertIn("0 of 48 elements of the window hold a valid sample", lines[-1]["reason"])

    def test_statistics_that_name_a_validation_or_a_held_out_window_are_refused(self):
        rows, means, counts = sample_set("training", pairs=2)
        absent = [index for index, row in enumerate(rows) if row["label"] == 0]
        template = cusense.fit_template([rows[index] for index in absent], means[absent], counts[absent])
        for split, purpose in (("validation", "the fit of the template"), ("held_out", "the fit of the template")):
            other = dict(rows[absent[0]], split=split)
            with self.subTest(split=split), self.assertRaisesRegex(
                    CuSenseError, f"of the {split} split was given to {purpose}, which reads the training split"):
                cusense.fit_template([other], means[absent[:1]], counts[absent[:1]])
        present = [index for index, row in enumerate(rows) if row["label"] == 1][:1]
        with self.assertRaisesRegex(CuSenseError, "target-present window .* was given to the fit of the template"):
            cusense.fit_template([rows[index] for index in present], means[present], counts[present])
        for split in ("validation", "held_out"):
            with self.subTest(split=split), self.assertRaisesRegex(
                    CuSenseError, f"of the {split} split was given to the fit of the normalizer"):
                cusense.fit_normalizer(rows[:-1] + [dict(rows[-1], split=split)], means, counts, template)
        # Statistics carry what they were fitted on, and a record that names another split is refused.
        normalizer = cusense.fit_normalizer(rows, means, counts, template)
        self.assertEqual((template["fitted_on"]["split"], template["fitted_on"]["windows"], normalizer["fitted_on"]["windows"]),
                         ("training", len(absent), len(rows)))
        self.assertEqual(normalizer["fitted_on"]["windows_sha256"], cusense.ids_sha256(row["window"] for row in rows))
        for split in ("validation", "held_out"):
            with self.subTest(named=split), self.assertRaisesRegex(
                    CuSenseError, f"names the {split} split: the template and the statistics of the z-score are fitted "
                                  "on windows of the training split"):
                cusense.check_fitted_on(dict(normalizer["fitted_on"], split=split), "the normalizer")
            with self.subTest(artifact=split), self.assertRaisesRegex(CuSenseError, f"the template names the {split} split"):
                make_artifact(self.scratch / f"named-{split}",
                              template=dict(template, fitted_on=dict(template["fitted_on"], split=split)))
        self.assertFalse(any(self.scratch.iterdir()))

    def test_the_normalizer_is_one_mean_and_one_deviation_of_the_training_split(self):
        rows, means, counts = sample_set("training", pairs=3)
        absent = [index for index, row in enumerate(rows) if row["label"] == 0]
        template = cusense.fit_template([rows[index] for index in absent], means[absent], counts[absent])
        # The template is the mean of the records behind the target-absent windows, element by element.
        expected = means[absent].astype(np.complex128).mean(axis=0)
        self.assertTrue(np.allclose(template["values"], expected, rtol=0, atol=1e-6))
        self.assertTrue(np.all(template["count"] == 3 * len(absent)))
        normalizer = cusense.fit_normalizer(rows, means, counts, template)
        d = np.abs(means.astype(np.complex128)) - np.abs(template["values"].astype(np.complex128))[None]
        d = d.mean(axis=3)
        self.assertAlmostEqual(normalizer["mean"], float(d.mean()), places=12)
        self.assertAlmostEqual(normalizer["scale"], float(d.std()), places=12)
        self.assertEqual((normalizer["entry"], normalizer["epsilon"], normalizer["constant"], normalizer["elements"]),
                         ("zscore_global", 1e-8, False, d.size))
        x = cusense.inputs_of(means, counts, template, normalizer)
        self.assertEqual((x.shape, x.dtype), ((len(rows), 2, 48), np.dtype(np.float32)))
        self.assertAlmostEqual(float(x.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(x.std()), 1.0, places=5)
        # With the windows of another split added, the statistics would be other ones.
        more, more_means, more_counts = sample_set("training", pairs=3, seed=9)
        other = cusense.fit_normalizer(rows + [dict(row, window=row["window"] + "x") for row in more],
                                       np.concatenate([means, more_means]), np.concatenate([counts, more_counts]), template)
        self.assertNotEqual(other["mean"], normalizer["mean"])
        # A set that does not vary keeps the scale 1 and says so.
        flat = cusense.fit_normalizer(rows[:1], template["values"][None], np.ones((1, 2, 48, 2), dtype="u1"), template)
        self.assertEqual((flat["scale"], flat["constant"], flat["mean"]), (1.0, True, 0.0))

    def test_registry_entries_state_their_device_and_runtime(self):
        for stage, entries in cusense.REGISTRY.items():
            for name, entry in entries.items():
                with self.subTest(stage=stage, entry=name):
                    self.assertEqual(entry["runtime"], "reference")
                    self.assertIn(entry["library"], ("numpy", "torch", "python"))
                    self.assertTrue(entry["device"] and entry["does"])
        resolved = cusense.resolve(cusense.demo_spec(), "map_mean_kalman")
        self.assertEqual({stage: entry["entry"] for stage, entry in resolved.items()},
                         {"input": "o3_fp16", "ingest": "ocudu_o3_fp16", "window": "coherent_mean",
                          "clutter": "background_magnitude", "symbols": "dmrs_mean", "normalizer": "zscore_global",
                          "model": "cusense_cnn", "detector": "frozen_threshold", "smoothing": "map_mean_kalman"})
        self.assertEqual(resolved["model"]["library"], "torch")
        self.assertEqual({resolved[stage]["library"] for stage in ("ingest", "window", "clutter", "symbols", "normalizer")},
                         {"numpy"})

    def test_the_specification_of_the_demonstration_and_its_refusals(self):
        o3, o2 = cusense.demo_spec(), cusense.demo_spec("o2_comb")
        self.assertEqual((cusense.shape_of(o3), cusense.shape_of(o2)), ([4, 612, 2], [4, 306, 2]))
        self.assertEqual((o3["window"]["slots_in_period"], o3["radio"], o3["validity"]),
                         ([7, 8, 9], {"scs_khz": 30, "period_slots": 10}, {"tau": 1e-10}))
        self.assertEqual((o3["input"]["layout"], o3["input"]["dtype"], o2["input"]["layout"], o2["input"]["dtype"]),
                         ("ocudu_o3_fp16", "<cfp16", "ocudu_o2_comb_c64", "<c8"))
        self.assertAlmostEqual(cusense.period_s(o3), 0.005)
        self.assertNotEqual(cusense.spec_sha256(o3), cusense.spec_sha256(o2))

        def changed(edit, base=o3):
            copy = json.loads(json.dumps(base))
            edit(copy)
            return copy

        for edit, message in (
                (lambda spec: spec.update(schema="other"), "unsupported specification schema"),
                (lambda spec: spec["input"].update(entry="o1_grid"), "unknown input entry"),
                (lambda spec: spec["input"].update(layout="ocudu_o2_comb_c64"), "the input entry o3_fp16 reads the stage O3"),
                (lambda spec: spec["support"].update(subcarrier_step=2), "support.subcarrier_step is 1"),
                (lambda spec: spec["window"].update(slots_in_period=[9, 7]), "a nonempty rising list"),
                (lambda spec: spec["window"].update(slots_in_period=[10]), "an entry of window.slots_in_period"),
                (lambda spec: spec["window"].update(min_valid_fraction=0), "window.min_valid_fraction"),
                (lambda spec: spec["support"].update(subcarriers=8), "support.subcarriers"),
                (lambda spec: spec["steps"].update(clutter="mti_2"), "unknown clutter step"),
                (lambda spec: spec["steps"].update(ingest="ocudu_o2_comb_c64"), "the ingest step of the input entry"),
                (lambda spec: spec["validity"].update(tau=-1), "validity.tau"),
                (lambda spec: spec["radio"].update(period_slots=3), "a whole number of subframes"),
                (lambda spec: spec.pop("validity"), "requires exactly")):
            with self.subTest(message=message), self.assertRaisesRegex(CuSenseError, message):
                cusense.check_spec(changed(edit))


@unittest.skipIf(np is None, NEEDS)
class ModelTests(unittest.TestCase):
    """The model entry cusense_cnn and its losses."""

    def small(self, rows=4, cols=3, antennas=2):
        return cusense_cnn.CuSenseCNN(cusense_cnn.check_config(
            {"entry": "cusense_cnn", "antennas": antennas, "rows": rows, "cols": cols, "stem_channels": 4, "stem_kernel": 7,
             "block_channels": [4, 8, 8], "block_kernel": 3, "hidden": [8, 8], "dropout": [0.0, 0.0],
             "presence_head": True}))

    def test_the_map_sums_to_one_for_both_input_widths(self):
        # The published widths on the grid and the array of the demonstration.
        config = cusense_cnn.published_config(4, 50, 34)
        self.assertEqual((config["stem_channels"], config["block_channels"], config["hidden"], config["dropout"]),
                         (64, [128, 256, 512], [512, 256], [0.3, 0.15]))
        torch.manual_seed(1)
        net = cusense_cnn.CuSenseCNN(config).eval()
        self.assertEqual(cusense_cnn.parameters_of(net), 2_384_933)
        with torch.no_grad():
            torch.nn.init.normal_(net.location.weight, 0.0, 0.1)
            for width in (612, 306):
                drawn, presence_logits = net.grid(torch.randn(2, 4, width))
                with self.subTest(width=width):
                    self.assertEqual((tuple(drawn.shape), tuple(presence_logits.shape)), ((2, 50, 34), (2,)))
                    self.assertTrue(torch.allclose(drawn.sum(dim=(1, 2)), torch.ones(2), atol=1e-5))
                    self.assertTrue(bool((drawn >= 0).all()))
                    self.assertGreater(float(drawn.max() - drawn.min()), 0.0)
            # Any number of subcarriers from 16 on gives one vector: an allocation of fewer blocks is the same model.
            self.assertEqual(tuple(net.grid(torch.randn(1, 4, 24))[0].shape), (1, 50, 34))
        with self.assertRaisesRegex(ModelError, "needs 16 subcarriers or more"):
            net(torch.zeros(1, 4, 12))
        with self.assertRaisesRegex(ModelError, "takes \\[batch, 4, subcarriers\\]"):
            net(torch.zeros(1, 2, 612))

    def test_an_untrained_model_answers_with_a_uniform_map_and_the_logit_0(self):
        net = self.small().eval()
        drawn, logits = net.grid(torch.randn(3, 2, 48))
        self.assertTrue(torch.equal(drawn, torch.full((3, 4, 3), 1.0 / 12)))
        self.assertTrue(torch.equal(logits, torch.zeros(3)))
        self.assertEqual(sorted(cusense_cnn.LOSSES), ["location_grid", "presence"])
        for loss in cusense_cnn.LOSSES.values():
            self.assertTrue(loss["unit"].startswith("nats") and loss["eligible"])

    def test_the_kl_divergence_of_the_card(self):
        # Of a distribution with itself: 0.
        target = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        self.assertAlmostEqual(float(cusense_cnn.kl_divergence(target, torch.log(target))), 0.0, places=6)
        # A uniform prediction over four cells against a one-hot target: ln 4.
        one_hot = cusense_cnn.smoothed_target(torch.tensor([[1, 0]]), 2, 2, 0)
        self.assertTrue(torch.equal(one_hot, torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])))
        value = float(cusense_cnn.kl_divergence(one_hot, torch.zeros(1, 4)))
        self.assertAlmostEqual(value, 1.386294, places=6)
        self.assertAlmostEqual(value, math.log(4.0), places=6)
        # A cell with a target of zero adds nothing, whatever the prediction holds there.
        self.assertAlmostEqual(float(cusense_cnn.kl_divergence(one_hot, torch.tensor([[-50.0, 3.0, 9.0, 3.0]]))),
                               float(-torch.log_softmax(torch.tensor([-50.0, 3.0, 9.0, 3.0]), 0)[2]), places=5)
        with self.assertRaisesRegex(ModelError, "the target has the shape"):
            cusense_cnn.kl_divergence(one_hot, torch.zeros(1, 5))

    def test_the_target_is_the_cell_smoothed_and_normalized_on_the_grid(self):
        target = cusense_cnn.smoothed_target(torch.tensor([[25, 17], [0, 0], [49, 33]]), 50, 34, 8.0)
        self.assertEqual(tuple(target.shape), (3, 50, 34))
        self.assertTrue(torch.allclose(target.sum(dim=(1, 2)), torch.ones(3), atol=1e-5))
        for index, cell in enumerate(((25, 17), (0, 0), (49, 33))):
            where = int(target[index].argmax())
            self.assertEqual((where // 34, where % 34), cell)
        # Along one axis the ratio of two cells follows the Gaussian: exp(-d^2 / (2 sigma^2)).
        self.assertAlmostEqual(float(target[0, 25, 21] / target[0, 25, 17]), math.exp(-16 / 128), places=5)
        self.assertAlmostEqual(float(target[0, 33, 17] / target[0, 25, 17]), math.exp(-64 / 128), places=5)
        # A cell in a corner keeps a whole distribution, so its peak is higher than that of a cell in the middle.
        self.assertGreater(float(target[1].max()), float(target[0].max()))
        for cells, sigma, message in ((torch.tensor([[50, 0]]), 1.0, "outside the grid"),
                                      (torch.tensor([[0, 0]]), -1.0, "sigma_cells"),
                                      (torch.tensor([0, 0]), 1.0, "shape \\[batch, 2\\]")):
            with self.subTest(message=message), self.assertRaisesRegex(ModelError, message):
                cusense_cnn.smoothed_target(cells, 50, 34, sigma)

    def test_target_absent_samples_send_no_gradient_into_the_map(self):
        torch.manual_seed(3)
        grid_logits = torch.randn(4, 12, requires_grad=True)
        presence_logits = torch.randn(4, requires_grad=True)
        targets = cusense_cnn.smoothed_target(torch.tensor([[1, 1], [2, 0], [0, 0], [0, 0]]), 4, 3, 0.5)
        # A batch of target-absent samples: the location part is zero and its gradient is zero everywhere.
        total, presence_part, location_part = cusense_cnn.joint_loss(grid_logits, presence_logits, targets,
                                                                    torch.zeros(4, dtype=torch.long))
        total.backward()
        self.assertEqual((float(location_part.detach()), float(total.detach())), (0.0, float(presence_part.detach())))
        self.assertTrue(torch.equal(grid_logits.grad, torch.zeros(4, 12)))
        self.assertGreater(float(presence_logits.grad.abs().min()), 0.0)
        # A mixed batch: only the rows of the target-present samples carry a gradient of the map.
        grid_logits.grad = presence_logits.grad = None
        labels = torch.tensor([1, 1, 0, 0])
        total, presence_part, location_part = cusense_cnn.joint_loss(grid_logits, presence_logits, targets, labels, 2.0)
        total.backward()
        self.assertTrue(torch.equal(grid_logits.grad[2:], torch.zeros(2, 12)))
        self.assertGreater(float(grid_logits.grad[:2].abs().max()), 0.0)
        # The location part is the mean over the eligible samples, and the weight multiplies it.
        each = cusense_cnn.kl_divergence(targets, grid_logits.detach())
        self.assertAlmostEqual(float(location_part.detach()), float(each[:2].mean()), places=6)
        self.assertAlmostEqual(float(total.detach()),
                               float(presence_part.detach()) + 2.0 * float(location_part.detach()), places=6)

    def test_the_weights_are_one_flat_file_with_a_layout(self):
        torch.manual_seed(4)
        net = self.small().eval()
        with torch.no_grad():
            net.location.weight.normal_()
            net.blocks[0][1].running_mean.normal_()
        data, layout = cusense_cnn.pack(net)
        self.assertEqual(len(data), 4 * sum(item["count"] for item in layout))
        self.assertEqual(sum(item["count"] for item in layout), layout[-1]["offset"] + layout[-1]["count"])
        names = [item["name"] for item in layout]
        self.assertIn("blocks.0.1.running_mean", names)   # the statistics of a normalization layer are read
        self.assertFalse(any(name.endswith("num_batches_tracked") for name in names))   # its counter is not
        again = cusense_cnn.unpack(self.small(), data, layout).eval()
        x = torch.randn(2, 2, 48)
        for first, second in zip(net.grid(x), again.grid(x)):
            self.assertTrue(torch.equal(first, second))
        with self.assertRaisesRegex(ModelError, "another architecture wrote it"):
            cusense_cnn.unpack(self.small(rows=5), data, layout)
        with self.assertRaisesRegex(ModelError, "holds .* bytes, and its layout names"):
            cusense_cnn.unpack(self.small(), data[:-4], layout)
        broken = bytearray(data)
        broken[0:4] = np.float32("nan").tobytes()
        with self.assertRaisesRegex(ModelError, "not finite"):
            cusense_cnn.unpack(self.small(), bytes(broken), layout)

    def test_the_configuration_of_a_model_is_checked(self):
        good = cusense_cnn.published_config(4, 50, 34)
        for edit, message in ((lambda config: config.update(entry="gru"), "unknown model entry"),
                              (lambda config: config.update(stem_kernel=6), "must be odd"),
                              (lambda config: config.update(block_channels=[64, 128]), "three plain blocks"),
                              (lambda config: config.update(hidden=[512]), "two widths of the trunk"),
                              (lambda config: config.update(dropout=[0.3, 1.0]), "two rates from 0 to below 1"),
                              (lambda config: config.update(presence_head=False), "presence_head must be true"),
                              (lambda config: config.update(antennas=0), "antennas must be an integer"),
                              (lambda config: config.pop("rows"), "requires exactly")):
            copy = json.loads(json.dumps(good))
            edit(copy)
            with self.subTest(message=message), self.assertRaisesRegex(ModelError, message):
                cusense_cnn.check_config(copy)


@unittest.skipIf(np is None, NEEDS)
class StreamTests(unittest.TestCase):
    """The causal window stream: what it answers at each record, and what cannot change an answer."""

    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)
        self.spec = small_spec()
        self.generator = np.random.Generator(np.random.PCG64(11))

    def test_every_period_is_answered_at_its_last_slot(self):
        records = periods_of(3, self.generator)
        lines, stream = cusense.run_records(records, self.spec)
        self.assertEqual([line["status"] for line in lines], ["insufficient_history", "insufficient_history", "ok"] * 3)
        self.assertEqual([line["history"]["have"] for line in lines], [1, 2, 3] * 3)
        self.assertEqual(stream.counts, {"records": 9, "passed": 0, "resets": 0, "ok": 3, "insufficient_history": 6,
                                         "gap": 0, "invalid": 0, "rejected": 0})
        for period, line in enumerate(line for line in lines if line["status"] == "ok"):
            self.assertEqual((line["window"]["period"], line["trigger"]["absolute_slot"], line["mean"]["index"]),
                             (period, 10 * period + 9, period))
            self.assertEqual(line["trigger"]["sequence"], line["contributing"][-1]["sequence"])
            self.assertEqual((line["validity"]["fill"], line["rx"]["elements"], line["input"]["producer_steps"]),
                             (1.0, [0, 1], ["pilot_ls", "frequency_interpolation", "smoothing"]))
        means, counts = cusense.stacked(stream)
        self.assertEqual((means.shape, counts.shape, int(counts.min()), int(counts.max())), ((3, 2, 48, 2), (3, 2, 48, 2), 3, 3))
        # A record of a slot outside the window is passed over and causes no answer.
        other = o3_record(channel(self.generator), 4, 99)
        self.assertEqual((stream.push(other), stream.counts["passed"]), ([], 1))

    def test_a_window_without_one_of_its_slots_is_a_gap(self):
        records = periods_of(3, self.generator)
        del records[4]   # slot 8 of the second period never arrives
        lines, stream = cusense.run_records(records, self.spec)
        self.assertEqual([line["status"] for line in lines],
                         ["insufficient_history", "insufficient_history", "ok", "insufficient_history", "gap",
                          "insufficient_history", "insufficient_history", "ok"])
        gap = lines[4]
        self.assertEqual((gap["missing"], gap["history"], "mean" in gap),
                         ([{"period": 1, "slot_in_period": 8}], {"have": 2, "need": 3}, False))
        # The last slot is missing: the window is closed by the first record of a later period, with no trigger.
        records = periods_of(2, self.generator)
        del records[2]
        lines, _ = cusense.run_records(records, self.spec)
        self.assertEqual([line["status"] for line in lines],
                         ["insufficient_history", "insufficient_history", "gap", "insufficient_history",
                          "insufficient_history", "ok"])
        self.assertEqual((lines[2]["trigger"], lines[2]["window"]["period"]), (None, 0))
        # A window that is still filling at the end is neither a gap nor a window: it is reported as open.
        lines, stream = cusense.run_records(periods_of(1, self.generator)[:2], self.spec)
        self.assertEqual(([line["status"] for line in lines], stream.open_samples()), (["insufficient_history"] * 2, (2, 3)))

    def test_a_history_never_spans_two_episodes_or_two_epochs(self):
        first = periods_of(1, self.generator, episode=0xA)[:2]
        second = periods_of(1, self.generator, episode=0xB, first_sequence=2)
        lines, stream = cusense.run_records(first + second, self.spec)
        self.assertEqual([line["status"] for line in lines], ["insufficient_history"] * 4 + ["ok"])
        self.assertEqual((stream.counts["resets"], lines[-1]["window"]["episode"],
                          [item["sequence"] for item in lines[-1]["contributing"]]), (1, f"{0xB:016x}", [2, 3, 4]))
        # The two slots of the other episode never enter a window. A new epoch resets as well.
        third = periods_of(1, self.generator, episode=0xB, first_sequence=5, epoch=2)
        self.assertEqual([line["status"] for line in cusense.run_records(second[:2] + third, self.spec)[0]],
                         ["insufficient_history"] * 4 + ["ok"])

    def test_a_record_that_is_no_input_is_rejected_before_its_payload_is_read(self):
        values = channel(self.generator)
        good = o3_record(values, 7, 0)
        self.assertIsNone(cusense.refusal(good, self.spec))
        first, last = times_of(7)
        cases = ((o2_record(values[:, 0::2, :], 7, 0), "the specification reads O3"),
                 (o3_record(channel(self.generator, rx=4), 7, 0), "holds 4 receive planes and 2 DM-RS symbols"),
                 (o3_record(values, 7, 0, symbols=(2, 7, 11)), "the shape is not"),
                 (o3_record(values, 7, 0, steps=1), "after frequency interpolation and before time interpolation"),
                 (o3_record(values, 7, 0, steps=15), "after frequency interpolation and before time interpolation"),
                 (obs.Record(dict(good.header, t_first_sample=first + 1), b"", good.mask, good.payload), "sample times"),
                 (obs.Record(dict(good.header, t_last_sample=last - 1), b"", good.mask, good.payload), "sample times"),
                 (obs.Record(dict(good.header, layer_count=2), b"", good.mask, good.payload), "one layer"),
                 (obs.Record(dict(good.header, antenna_map=[0, 0]), b"", good.mask, good.payload), "one array element"),
                 (obs.Record(dict(good.header, subcarrier_step=2), b"", good.mask, good.payload), "subcarrier step of 2"),
                 (obs.Record(dict(good.header, strides=[96, 24, 12, 2]), b"", good.mask, good.payload), "the shape is not"))
        for record, message in cases:
            with self.subTest(message=message):
                self.assertIn(message, cusense.refusal(record, self.spec))
                lines, stream = cusense.run_records([record], self.spec)
                self.assertEqual((lines[0]["status"], lines[0]["window"], stream.counts["rejected"]), ("rejected", None, 1))
        # An allocation of one resource block is not smoothed by the receiver, and is still an input.
        self.assertIsNone(cusense.refusal(o3_record(values, 7, 0, steps=3), self.spec))
        # Slots of one window that were estimated by two chains of steps are no window.
        mixed = [o3_record(values, 7, 0), o3_record(values, 8, 1, steps=3), o3_record(values, 9, 2)]
        self.assertEqual(cusense.run_records(mixed, self.spec)[0][-1]["status"], "invalid")

    def test_future_records_and_fields_of_the_truth_leave_a_window_unchanged(self):
        records = periods_of(3, self.generator, cell=(1, 2))
        lines, stream = cusense.run_records(records, self.spec)
        wanted = [cusense.comparable_window(line) for line in lines[:6]]
        tensors = [(mean.tobytes(), count.tobytes()) for _, mean, count in stream.tensors[:2]]
        # The records after the second window are replaced by a large value: the first two windows are the same.
        huge = np.full((2, 48, 2), 6e4 + 6e4j)
        future = records[:6] + [o3_record(huge, record.header["absolute_slot"], record.sequence) for record in records[6:]]
        lines_again, stream_again = cusense.run_records(future, self.spec)
        self.assertEqual([cusense.comparable_window(line) for line in lines_again[:6]], wanted)
        self.assertEqual([(mean.tobytes(), count.tobytes()) for _, mean, count in stream_again.tensors[:2]], tensors)
        # And a window is complete when its last record is pushed: nothing later is waited for.
        early = cusense.WindowStream(self.spec)
        answers = [line for record in records[:3] for line in early.push(record)]
        self.assertEqual((answers[-1]["status"], early.last[0].tobytes()), ("ok", tensors[0][0]))
        # A stream reads the fields of INPUT_FIELDS, the validity bits and the payload. Every other field of a
        # header, the grant, the cell, the UE, the digest and the host block among them, changes nothing.
        unread = sorted(name for name, _, _ in obs.FIELDS if name not in cusense.INPUT_FIELDS)
        self.assertEqual(unread, ["cell", "digest", "digest_kind", "grant", "header_bytes", "host_bytes", "reserved",
                                  "schema", "ue", "valid"])
        touched = []
        for record in records:
            header = dict(record.header, cell=77, ue=0x1234, grant=0xDEAD, digest=b"\x01" * 32, digest_kind=1, valid=1,
                          reserved=9, host_bytes=8)
            touched.append(obs.Record(header, b"\x05" * 8, record.mask, record.payload))
        lines_touched, stream_touched = cusense.run_records(touched, self.spec)
        self.assertEqual([cusense.comparable_window(line) for line in lines_touched],
                         [cusense.comparable_window(line) for line in lines])
        # The truth of a window, its label and the position of its target, lives in a sample row and nowhere in a
        # record. Changing it changes neither the template, nor the statistics, nor an input of the model.
        rows, means, counts = sample_set("training", pairs=2)
        absent = [index for index, row in enumerate(rows) if row["label"] == 0]
        template = cusense.fit_template([rows[index] for index in absent], means[absent], counts[absent])
        normalizer = cusense.fit_normalizer(rows, means, counts, template)
        x = cusense.inputs_of(means, counts, template, normalizer)
        moved = [dict(row, cell=[3, 2], position_m=[7.0, 11.0], scene="elsewhere", pair="other") if row["label"] else row
                 for row in rows]
        self.assertEqual(cusense.fit_normalizer(moved, means, counts, template)["mean"], normalizer["mean"])
        self.assertTrue(np.array_equal(cusense.inputs_of(means, counts, template, normalizer), x))
        self.assertNotIn("label", json.dumps(lines))
        self.assertNotIn("position_m", json.dumps(lines))

    def test_no_digest_is_taken_while_a_stream_reads(self):
        stream = cusense.WindowStream(self.spec)
        lines = [line for record in periods_of(2, self.generator) for line in stream.push(record)]
        complete = [line for line in lines if line["status"] == "ok"]
        # A window is answered without a hash: the digests are taken by seal(), after the last record.
        self.assertEqual([(line["mean"]["sha256"], line["count"]["sha256"]) for line in complete], [(None, None)] * 2)
        cusense.seal(lines, stream)
        for line, (_, mean, count) in zip(complete, stream.tensors):
            self.assertEqual((line["mean"]["sha256"], line["count"]["sha256"]),
                             (hashlib.sha256(mean.tobytes()).hexdigest(), hashlib.sha256(count.tobytes()).hexdigest()))
        source = (ROOT / "src/ocudu_cuda_isac/sensing/cusense.py").read_text()
        for name in ("    def push(self, record):", "    def _close(self, period, record):", "    def decide(self, line",
                     "    def predict(self, x):", "    def score(self, mean, count):"):
            body = source.split(name)[1].split("\n    def ")[0].split("\ndef ")[0]
            self.assertNotIn("sha256(", body, name)

    def test_a_window_output_verifies_and_a_changed_one_is_refused(self):
        records = periods_of(2, self.generator)
        lines, stream = cusense.run_records(records, self.spec)
        out = self.scratch / "windows"
        out.mkdir()
        report = cusense.write_windows(out, self.spec, lines, stream, {"kind": "unit"})
        self.assertEqual((report["schema"], report["windows"], report["shape"], report["runtime"]["entry"],
                          report["open_at_end"]), ("cusense-windows-report/v1", 2, [2, 48, 2], "reference",
                                                   {"have": 0, "need": 3}))
        self.assertEqual(report["input_fields"], list(cusense.INPUT_FIELDS))
        made = cusense.open_windows(out)
        self.assertEqual((len(made.lines), len(made.ok), sorted(made.by_window)),
                         (6, 2, [f"1/{KEY:016x}/0", f"1/{KEY:016x}/1"]))
        with self.assertRaisesRegex(CuSenseError, "holds a result already"):
            cusense.write_windows(out, self.spec, lines, stream, {"kind": "unit"})

        def changed(label, edit):
            copy = self.scratch / label
            shutil.copytree(out, copy)
            written = [json.loads(text) for text in (copy / "windows.jsonl").read_text().splitlines()]
            report = json.loads((copy / "windows.json").read_text())
            edit(written, report, copy)
            (copy / "windows.jsonl").write_text("".join(json.dumps(line) + "\n" for line in written))
            (copy / "windows.json").write_text(json.dumps(report))
            return copy

        for label, edit, message in (
                ("status", lambda written, report, copy: written[2].update(status="gap"), "names a window tensor"),
                ("slots", lambda written, report, copy: written[2]["contributing"].reverse(),
                 "does not name the slots of its period in order"),
                ("count", lambda written, report, copy: report["counts"].update(ok=3), "counts of the report"),
                ("spec", lambda written, report, copy: report["spec"]["window"].update(min_valid_fraction=0.5),
                 "does not name its specification"),
                ("extra", lambda written, report, copy: (copy / "labels.json").write_text("{}"),
                 "holds labels.json, which its report does not list"),
                ("line", lambda written, report, copy: written.pop(0), "is not a cusense-window/v1 line")):
            with self.subTest(label=label), self.assertRaisesRegex(CuSenseError, message):
                cusense.open_windows(changed(label, edit))


@unittest.skipIf(np is None, NEEDS)
class ArtifactTests(unittest.TestCase):
    """The artifact of four files, the model loaded from it and the decision it gives."""

    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)
        self.spec = small_spec()
        self.generator = np.random.Generator(np.random.PCG64(21))

    def test_one_artifact_holds_every_part_with_its_hash(self):
        artifact = make_artifact(self.scratch / "artifact", logit=-0.25)
        manifest = artifact.manifest
        self.assertEqual(sorted(path.name for path in artifact.directory.iterdir()),
                         ["model.json", "template.c64", "template.count.u4", "weights.f32"])
        self.assertEqual((manifest["schema"], manifest["entries"]),
                         ("cusense-artifact/v1", {"input": "o3_fp16", "ingest": "ocudu_o3_fp16", "window": "coherent_mean",
                                                  "clutter": "background_magnitude", "symbols": "dmrs_mean",
                                                  "normalizer": "zscore_global", "model": "cusense_cnn",
                                                  "detector": "frozen_threshold", "smoothing": "none"}))
        # The template, the normalizer, the weights, the threshold and the grid geometry, each with a hash.
        directory = artifact.directory
        self.assertEqual(manifest["template"]["sha256"], hashlib.sha256((directory / "template.c64").read_bytes()).hexdigest())
        self.assertEqual(manifest["template"]["count_sha256"],
                         hashlib.sha256((directory / "template.count.u4").read_bytes()).hexdigest())
        self.assertEqual(manifest["weights"]["sha256"], hashlib.sha256((directory / "weights.f32").read_bytes()).hexdigest())
        self.assertEqual(manifest["normalizer"]["sha256"], cusense.normalizer_sha256(manifest["normalizer"]))
        self.assertEqual(manifest["threshold"]["sha256"], cusense.part_sha256(manifest["threshold"]))
        self.assertEqual(manifest["grid"]["sha256"], cusense.part_sha256(manifest["grid"]))
        self.assertEqual(manifest["sha256"], cusense.identity(manifest))
        self.assertEqual(artifact.model(), {
            "model_id": "unit-model", "entry": "cusense_cnn", "artifact_sha256": manifest["sha256"],
            "weights_sha256": manifest["weights"]["sha256"], "template_sha256": manifest["template"]["sha256"],
            "normalizer_sha256": manifest["normalizer"]["sha256"], "grid_sha256": manifest["grid"]["sha256"],
            "threshold_sha256": manifest["threshold"]["sha256"]})
        # The grid with its units: rows along y, columns along x, metres.
        grid = manifest["grid"]
        self.assertEqual((grid["rows"], grid["cols"], grid["origin_m"], grid["units"]), (4, 3, [0.61, 1.47], cusense.GRID_UNITS))
        self.assertAlmostEqual(grid["cell_m"][0], 10.06 / 4)
        self.assertAlmostEqual(grid["cell_m"][1], 6.78 / 3)
        for made, wanted in zip(cusense.cell_centre(grid, (0, 0)), (0.61 + 6.78 / 6, 1.47 + 10.06 / 8)):
            self.assertAlmostEqual(made, wanted, places=12)
        self.assertEqual((manifest["input"]["shape"], manifest["input"]["dtype"], manifest["input"]["elements"],
                          manifest["input"]["spec"], manifest["template"]["shape"]),
                         ([1, 2, 48], "<f4", [0, 1], self.spec, [2, 48, 2]))
        self.assertEqual((manifest["model"]["heads"], manifest["model"]["target"]["sigma_cells"],
                          manifest["threshold"]["logit"]), (["location_grid", "presence"], 0.5, -0.25))
        # It holds no sample and no label of a training run, and the one split it names is the training split.
        text = (directory / "model.json").read_text()
        for word in ('"label"', '"scene"', '"pair"', '"episode"', "held_out", '"split": "validation"'):
            self.assertNotIn(word, text)
        shown = cusense.inspect(artifact)
        self.assertEqual((shown["artifact_sha256"], shown["input_shape"], shown["files"], shown["tensors"]),
                         (artifact.sha256, [1, 2, 48], sorted(cusense.ARTIFACT_FILES), len(manifest["weights"]["tensors"])))
        with self.assertRaisesRegex(CuSenseError, "never overwritten"):
            make_artifact(self.scratch / "artifact")

    def test_an_altered_artifact_is_refused(self):
        artifact = make_artifact(self.scratch / "artifact")

        def copy_of(label, edit):
            target = self.scratch / label
            shutil.copytree(artifact.directory, target)
            edit(target)
            return target

        def flip(name):
            def edit(target):
                data = bytearray((target / name).read_bytes())
                data[0] ^= 1
                (target / name).write_bytes(bytes(data))
            return edit

        def rewrite(change):
            def edit(target):
                manifest = json.loads((target / "model.json").read_text())
                change(manifest)
                (target / "model.json").write_text(json.dumps(manifest))
            return edit

        def rehash(change):
            # The part is changed and the hash of the whole is written again: only the hash of the part still tells.
            def edit(manifest):
                change(manifest)
                manifest["sha256"] = cusense.identity(manifest)
            return rewrite(edit)

        cases = (("weights", flip("weights.f32"), "the checkpoint was altered"),
                 ("template", flip("template.c64"), "the background was altered"),
                 ("counts", flip("template.count.u4"), "the background was altered"),
                 ("threshold", rewrite(lambda manifest: manifest["threshold"].update(logit=-5.0)),
                  "it was moved after it was frozen"),
                 ("name", rewrite(lambda manifest: manifest.update(model_id="another-model")), "altered after it was frozen"),
                 ("threshold-part", rehash(lambda manifest: manifest["threshold"].update(logit=-5.0, probability=0.0067)),
                  "it was moved after it was frozen"),
                 ("normalizer-part", rehash(lambda manifest: manifest["normalizer"].update(mean=0.5)),
                  "the preprocessing state changed"),
                 ("grid-part", rehash(lambda manifest: manifest["grid"].update(origin_m=[0.0, 0.0])),
                  "does not follow from its area"),
                 ("split", rehash(lambda manifest: manifest["normalizer"]["fitted_on"].update(split="validation")),
                  "the normalizer names the validation split"),
                 ("extra", lambda target: (target / "training.jsonl").write_text("{}\n"),
                  "holds training.jsonl: a deployed model is"),
                 ("missing", lambda target: (target / "template.count.u4").unlink(), "is not a complete model"))
        for label, edit, message in cases:
            with self.subTest(label=label), self.assertRaisesRegex(CuSenseError, message):
                cusense.open_artifact(copy_of(label, edit))
        with self.assertRaisesRegex(CuSenseError, "was asked for"):
            cusense.open_artifact(artifact.directory, "0" * 64)
        self.assertEqual(cusense.open_artifact(artifact.directory, artifact.sha256).sha256, artifact.sha256)

    def test_a_saved_model_reloads_to_the_same_map_and_score(self):
        artifact = make_artifact(self.scratch / "artifact")
        first, second = cusense.Model(artifact), cusense.Model(cusense.open_artifact(artifact.directory))
        records = periods_of(1, self.generator, cell=(2, 1))
        mean, count = cusense.run_records(records, self.spec)[1].last
        one, two = first.score(mean, count), second.score(mean, count)
        self.assertEqual((one["logit"], one["probability"]), (two["logit"], two["probability"]))
        self.assertTrue(np.array_equal(one["map"], two["map"]) and np.array_equal(one["x"], two["x"]))
        self.assertEqual((one["x"].shape, one["x"].dtype, one["map"].shape, one["map"].dtype),
                         ((1, 2, 48), np.dtype("<f4"), (4, 3), np.dtype("<f4")))
        self.assertAlmostEqual(float(one["map"].astype(np.float64).sum()), 1.0, places=5)
        self.assertAlmostEqual(one["probability"], 1.0 / (1.0 + math.exp(-one["logit"])), places=6)
        self.assertEqual(one["present"], one["logit"] > artifact.threshold)
        # The input is what the formula gives: here the template is 1 and the normalizer changes nothing.
        expected = ((np.abs(mean.astype(np.complex128)) - 1.0).mean(axis=2) / (1.0 + 1e-8)).astype(np.float32)
        self.assertTrue(np.array_equal(one["x"][0], expected))
        self.assertEqual(cusense.input_sha256(one["x"]), hashlib.sha256(one["x"].tobytes()).hexdigest())

    def test_wrong_shape_dtype_and_values_are_refused(self):
        model = cusense.Model(make_artifact(self.scratch / "artifact"))
        for label, x, message in (
                ("a batch of two windows", np.zeros((2, 2, 48), dtype=np.float32), "one window of shape"),
                ("a window without its batch axis", np.zeros((2, 48), dtype=np.float32), "one window of shape"),
                ("another array", np.zeros((1, 4, 48), dtype=np.float32), "one window of shape"),
                ("float64", np.zeros((1, 2, 48), dtype=np.float64), "as float32"),
                ("not finite", np.full((1, 2, 48), np.nan, dtype=np.float32), "not finite"),
                ("too few subcarriers", np.zeros((1, 2, 8), dtype=np.float32), "16 subcarriers or more"),
                ("a list", [[0.0] * 48] * 2, "a NumPy array or a tensor")):
            with self.subTest(label=label), self.assertRaisesRegex(CuSenseError, message):
                model.predict(x)
        with self.assertRaisesRegex(CuSenseError, "and the model was trained on \\[2, 48, 2\\]"):
            model.prepare(np.zeros((2, 24, 2), dtype="<c8"), np.zeros((2, 24, 2), dtype="u1"))
        for name in ("tpu", "cuda:99"):
            with self.subTest(device=name), self.assertRaises(CuSenseError):
                cusense.Model(model.artifact, name)

    def test_a_window_of_another_specification_or_array_is_refused(self):
        model = cusense.Model(make_artifact(self.scratch / "artifact"))
        lines, stream = cusense.run_records(periods_of(1, self.generator), self.spec)
        line = lines[-1]
        decision, x, drawn = model.decide(line, stream.last)
        self.assertEqual((decision["status"], x.shape, drawn.shape), ("ok", (1, 2, 48), (4, 3)))
        other = small_spec(slots=(7, 8))
        with self.assertRaisesRegex(CuSenseError, "was computed with the specification .*, and the model was trained on"):
            model.decide(dict(line, spec_sha256=cusense.spec_sha256(other)), stream.last)
        with self.assertRaisesRegex(CuSenseError, "on the array elements \\[4, 5\\]"):
            model.decide(dict(line, rx=dict(line["rx"], elements=[4, 5])), stream.last)
        with self.assertRaisesRegex(CuSenseError, "by the steps \\['pilot_ls', 'frequency_interpolation'\\]"):
            model.decide(dict(line, input=dict(line["input"], producer_steps=["pilot_ls", "frequency_interpolation"])),
                         stream.last)
        with self.assertRaisesRegex(CuSenseError, "is decided from its mean and its count"):
            model.decide(line)

    def test_the_selected_device_agrees_with_the_cpu(self):
        artifact = make_artifact(self.scratch / "artifact")
        mean, count = cusense.run_records(periods_of(1, self.generator, cell=(1, 1)), self.spec)[1].last
        cpu = cusense.Model(artifact, "cpu").score(mean, count)
        chosen = cusense.Model(artifact, cusense.default_device())
        other = chosen.score(mean, count)
        self.assertTrue(np.array_equal(cpu["x"], other["x"]))   # the front end is NumPy on the CPU on both
        self.assertLessEqual(abs(cpu["logit"] - other["logit"]), evaluator.TOLERANCE["logit_absolute"])
        self.assertLessEqual(float(np.abs(cpu["map"] - other["map"]).max()), evaluator.TOLERANCE["map_absolute"])
        facts = cusense.device_facts(chosen.device)
        self.assertEqual((facts["type"], facts["torch"]), (cusense.default_device(), torch.__version__))
        self.assertGreater(chosen.warm_up(), 0.0)


@unittest.skipIf(np is None, NEEDS)
class TrainingTests(unittest.TestCase):
    """What the fit, the threshold and the training script may read, and what they freeze."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.spec, cls.grid = small_spec(), grid_of()
        cls.sets = {split: sample_set(split, pairs=6, seed=seed) for split, seed in (("training", 3), ("validation", 5))}
        for split, (rows, means, counts) in cls.sets.items():
            cusense.write_samples(cls.root / f"{split}.jsonl", rows, means, counts)
        (cls.root / "spec.json").write_text(json.dumps(cls.spec))
        (cls.root / "grid.json").write_text(json.dumps({name: cls.grid[name] for name in ("rows", "cols", "cell_m",
                                                                                         "origin_m", "area", "rule")}))
        cls.settings = small_settings()
        (cls.root / "training.json").write_text(json.dumps(cls.settings))
        cls.freeze = train_cusense.train(cls.root / "training.jsonl", cls.root / "validation.jsonl", cls.spec,
                                         json.loads((cls.root / "grid.json").read_text()), cls.settings, cls.root / "run")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def inputs(self, split="training"):
        rows, means, counts = self.sets["training"]
        absent = [index for index, row in enumerate(rows) if row["label"] == 0]
        template = cusense.fit_template([rows[index] for index in absent], means[absent], counts[absent])
        normalizer = cusense.fit_normalizer(rows, means, counts, template)
        chosen = self.sets[split]
        return chosen[0], cusense.inputs_of(chosen[1], chosen[2], template, normalizer)

    def test_a_sample_set_is_three_files_and_every_tensor_has_the_hash_its_row_names(self):
        made = cusense.read_samples(self.root / "training.jsonl")
        rows, means, counts = self.sets["training"]
        self.assertEqual((made.rows, sorted(made.files)), (rows, ["training.c64", "training.jsonl", "training.u1"]))
        self.assertTrue(np.array_equal(made.means, means) and np.array_equal(made.counts, counts))
        self.assertEqual((made.means.dtype, made.counts.dtype, made.means.shape),
                         (np.dtype("<c8"), np.dtype("u1"), (24, 2, 48, 2)))
        self.assertEqual((sum(row["label"] for row in rows), rows[1]["cell"], rows[0]["cell"]), (12, None, None))
        present = next(row for row in rows if row["label"])
        self.assertEqual(present["position_m"], cusense.cell_centre(self.grid, present["cell"]))
        with self.assertRaisesRegex(CuSenseError, "never overwritten"):
            cusense.write_samples(self.root / "training.jsonl", rows, means, counts)
        # One changed value of a tensor, and a row that names another place, are refused.
        for name in ("training.jsonl", "training.c64", "training.u1"):
            shutil.copy(self.root / name, self.scratch / name)
        data = bytearray((self.scratch / "training.c64").read_bytes())
        data[100] ^= 1
        (self.scratch / "training.c64").write_bytes(bytes(data))
        with self.assertRaisesRegex(CuSenseError, "do not have the hashes its row names"):
            cusense.read_samples(self.scratch / "training.jsonl")
        with self.assertRaisesRegex(CuSenseError, "does not name its place and the hashes"):
            cusense.write_samples(self.scratch / "other.jsonl", [dict(rows[0], index=1)] + rows[1:], means, counts)
        for edit, message in ((lambda row: row.update(label=2), "a label of 0 or 1"),
                              (lambda row: row.update(cell=[1, 1]), "names a position: it has none"),
                              (lambda row: row.update(split="test"), "with a split"),
                              (lambda row: row.pop("scene"), "requires exactly")):
            copy = dict(rows[0])
            edit(copy)
            with self.subTest(message=message), self.assertRaisesRegex(CuSenseError, message):
                cusense.check_sample(copy)
        with self.assertRaisesRegex(CuSenseError, "needs the cell and the position of its target"):
            cusense.check_sample(dict(present, cell=None))

    def test_a_split_that_was_not_made_before_its_windows_is_refused(self):
        training, validation = self.sets["training"][0], self.sets["validation"][0]
        facts = cusense.check_split(training, validation)
        self.assertEqual((facts["shape"], facts["elements"], facts["pairs"]),
                         ([2, 48, 2], [0, 1], {"training": 6, "validation": 6}))
        held = dict(training[0], split="held_out")
        crossed = dict(validation[0], scene=training[0]["scene"])
        for label, first, second, message in (
                ("held-out", training + [held], validation, "a held-out sample is read after the freeze and never before"),
                ("scene", training, validation[1:] + [crossed], "the scene .* is in the training and in the validation split"),
                ("pair", training, validation[1:] + [dict(validation[0], pair=training[0]["pair"])], "the matched pair"),
                ("key", training, validation[1:] + [dict(validation[0], episode=training[0]["episode"])], "the episode key"),
                ("swapped", validation, training, "of the validation split is among the training samples"),
                ("twice", training + [training[0]], validation, "a window is among the training samples twice"),
                ("spec", training, validation[1:] + [dict(validation[0], spec_sha256="0" * 64)],
                 "were not computed with one specification"),
                ("array", training, validation[1:] + [dict(validation[0], elements=[2, 3])],
                 "were not computed with one specification, on one array")):
            with self.subTest(label=label), self.assertRaisesRegex(CuSenseError, message):
                cusense.check_split(first, second)

    def test_samples_outside_the_training_split_cannot_fit_the_weights(self):
        rows, x = self.inputs()
        with self.assertRaisesRegex(CuSenseError, "of the validation split was given to the fit of the weights"):
            cusense.fit_model(rows[:-1] + [dict(rows[-1], split="validation")], x, self.grid, self.settings)
        with self.assertRaisesRegex(CuSenseError, "of the training split was given to the monitor of the fit"):
            cusense.fit_model(rows, x, self.grid, self.settings, monitor=(rows, x))
        with self.assertRaisesRegex(CuSenseError, "holds 12 target-present and 12 target-absent windows, and the fit needs 13"):
            cusense.fit_model(rows, x, self.grid, dict(self.settings, min_per_class=13))
        with self.assertRaisesRegex(CuSenseError, "one input of shape \\[A, K\\] as float32 per training window"):
            cusense.fit_model(rows, x[:-1], self.grid, self.settings)
        beyond = [dict(row, cell=[9, 9]) if row["label"] else row for row in rows]
        with self.assertRaisesRegex(CuSenseError, "lies outside the grid of 4 by 3"):
            cusense.fit_model(beyond, x, self.grid, self.settings)

    def test_the_fit_starts_at_a_known_loss_falls_and_is_repeatable_on_the_cpu(self):
        rows, x = self.inputs()
        validation, x_validation = self.inputs("validation")
        net, history = cusense.fit_model(rows, x, self.grid, self.settings, monitor=(validation, x_validation))
        # Zero output layers: a uniform map over 12 cells and the presence logit 0.
        expected = history["initial_expected"]
        self.assertAlmostEqual(expected["presence"], math.log(2.0), places=12)
        target = cusense_cnn.smoothed_target(torch.tensor([row["cell"] for row in rows if row["label"]]), 4, 3, 0.5).double()
        logs = torch.where(target > 0, torch.log(target.clamp_min(1e-300)), torch.zeros_like(target))
        entropy = -(target * logs).sum(dim=(1, 2))
        self.assertAlmostEqual(expected["location"], float((math.log(12.0) - entropy).mean()), places=9)
        self.assertAlmostEqual(history["initial_loss"], expected["loss"], places=5)
        self.assertLess(history["final_loss"], 0.75 * history["initial_loss"])
        self.assertLess(history["final"]["presence"], 0.5 * math.log(2.0))
        self.assertEqual((history["epochs"], len(history["per_epoch"]), history["device"], history["samples"]),
                         (60, 60, "cpu", {"target_present": 12, "target_absent": 12}))
        self.assertEqual(history["steps"], 60 * 3)
        self.assertTrue(all("validation" in entry and math.isfinite(entry["validation"]["loss"])
                            for entry in history["per_epoch"]))
        self.assertFalse(net.training)
        # The validation windows are measured after every epoch and select nothing: without them the same weights.
        data, layout = cusense_cnn.pack(net)
        alone, _ = cusense.fit_model(rows, x, self.grid, self.settings)
        self.assertEqual(cusense_cnn.pack(alone)[0], data)
        # Another seed gives other weights, so the seed is what a fit starts from.
        other, _ = cusense.fit_model(rows, x, self.grid, dict(self.settings, seed=7))
        self.assertNotEqual(cusense_cnn.pack(other)[0], data)
        # The deployed arithmetic, one window at a time, separates the two classes of this set and finds the cell.
        logits, maps = cusense.scores_of(net.config, data, layout, x_validation)
        labels = [row["label"] for row in validation]
        self.assertGreaterEqual(evaluator.area_under_curve(logits, labels), 0.9)
        found = [cusense.locate(drawn, self.grid)["cell"] == row["cell"] for row, drawn in zip(validation, maps) if row["label"]]
        self.assertGreaterEqual(sum(found), 4)   # one of twelve by chance
        self.assertTrue(np.allclose(maps.astype(np.float64).sum(axis=(1, 2)), 1.0, atol=1e-5))

    def test_the_threshold_is_set_on_validation_negatives_by_the_rule_of_sprint_3_4(self):
        rows = self.sets["validation"][0]
        logits = [float(index) - 6.0 if row["label"] == 0 else 50.0 for index, row in enumerate(rows)]
        negatives = sorted((value for row, value in zip(rows, logits) if row["label"] == 0), reverse=True)
        made = presence.calibrate_threshold(rows, logits, 0.25, 1)
        self.assertEqual((made["negatives"], made["allowed"], made["logit"], made["exceeding"]), (12, 3, negatives[3], 3))
        record = cusense.threshold_record(made)
        self.assertEqual(({name: record[name] for name in presence.THRESHOLD_FIELDS}, record["sha256"]),
                         (made, cusense.part_sha256(record)))
        with self.assertRaisesRegex(presence.PresenceError,
                                    "of the training split was given to the calibration of the threshold"):
            presence.calibrate_threshold(self.sets["training"][0], logits, 0.05)

    def test_the_training_configuration_is_checked(self):
        demo = cusense.demo_settings()
        self.assertEqual((demo["schema"], demo["model"], demo["optimizer"]["entry"], demo["device"]),
                         ("cusense-train-config/v1", {name: cusense_cnn.PUBLISHED[name] for name in cusense.MODEL_SETTINGS},
                          "adam", "auto"))
        self.assertEqual(cusense.model_config(demo, 4, grid_of(50, 34)), cusense_cnn.published_config(4, 50, 34))
        for edit, message in ((lambda settings: settings.update(schema="other"), "unsupported training configuration"),
                              (lambda settings: settings["optimizer"].update(entry="sgd"), 'optimizer.entry is "adam"'),
                              (lambda settings: settings["optimizer"].update(learning_rate=0), "optimizer.learning_rate"),
                              (lambda settings: settings["optimizer"].update(epochs=0), "optimizer.epochs"),
                              (lambda settings: settings["loss"].update(sigma_cells=-1), "loss.sigma_cells"),
                              (lambda settings: settings["loss"].update(grid_weight=-1), "loss.grid_weight"),
                              (lambda settings: settings.update(device="tpu"), 'device is "auto", "cpu" or "cuda"'),
                              (lambda settings: settings["threshold"].update(target_false_alarm=1.0), "from 0 to below 1"),
                              (lambda settings: settings["smoothing"].update(entry="lstm"), "unknown smoothing entry"),
                              (lambda settings: settings["smoothing"].update(maps=0), "smoothing.maps"),
                              (lambda settings: settings.update(smoothing={"entry": "none", "maps": 3}), "takes no setting"),
                              (lambda settings: settings["model"].pop("hidden"), "model requires exactly")):
            copy = json.loads(json.dumps(demo))
            edit(copy)
            with self.subTest(message=message), self.assertRaisesRegex(CuSenseError, message):
                cusense.check_settings(copy)
        with self.assertRaisesRegex(CuSenseError, "must be odd"):
            cusense.model_config(dict(demo, model=dict(demo["model"], stem_kernel=4)), 4, grid_of())

    def test_the_training_script_freezes_one_artifact_from_two_sample_sets(self):
        freeze, run = self.freeze, self.root / "run"
        self.assertEqual(sorted(path.name for path in run.iterdir()),
                         ["artifact", "fit.json", "freeze.json", "training_manifest.json", "validation_manifest.json"])
        artifact = cusense.open_artifact(run / "artifact", freeze["artifact"]["sha256"])
        manifest = artifact.manifest
        self.assertEqual((freeze["schema"], freeze["model_id"], freeze["test_access"]),
                         ("cusense-freeze/v1", "unit-model", train_cusense.TEST_ACCESS))
        self.assertEqual(freeze["artifact"]["files"], {path.name: {"bytes": path.stat().st_size,
                                                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                                       for path in (run / "artifact").iterdir()})
        for part in ("weights", "template", "normalizer", "grid", "threshold"):
            self.assertEqual(freeze["artifact"][f"{part}_sha256"], manifest[part]["sha256"])
        self.assertEqual(freeze["threshold"], manifest["threshold"])
        # The manifests are written before the fit and name every file of their sample set by its hash.
        for role in ("training", "validation"):
            listed = json.loads((run / f"{role}_manifest.json").read_text())
            self.assertEqual(freeze["manifests"][role]["sha256"],
                             hashlib.sha256((run / f"{role}_manifest.json").read_bytes()).hexdigest())
            self.assertEqual({name: entry["sha256"] for name, entry in listed["samples"]["files"].items()},
                             {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                              for name in (f"{role}.jsonl", f"{role}.c64", f"{role}.u1")})
            self.assertEqual((listed["role"], listed["windows"], listed["target_present"], listed["target_absent"],
                              listed["used_for"]), (role, 24, 12, 12, train_cusense.USED_FOR[role]))
            self.assertEqual(listed["window_ids"], sorted(row["window"] for row in self.sets[role][0]))
        # The template and the statistics are those of the training split, recomputed here from its rows alone.
        rows, means, counts = self.sets["training"]
        absent = [index for index, row in enumerate(rows) if row["label"] == 0]
        template = cusense.fit_template([rows[index] for index in absent], means[absent], counts[absent])
        self.assertTrue(np.array_equal(artifact.template["values"], template["values"]))
        self.assertTrue(np.array_equal(artifact.template["count"], template["count"]))
        normalizer = cusense.fit_normalizer(rows, means, counts, template)
        self.assertEqual((manifest["normalizer"]["mean"], manifest["normalizer"]["scale"]),
                         (normalizer["mean"], normalizer["scale"]))
        self.assertEqual((manifest["template"]["fitted_on"], manifest["normalizer"]["fitted_on"]),
                         (template["fitted_on"], normalizer["fitted_on"]))
        # The threshold is the rule of Sprint 3.4 on the target-absent validation windows, in the deployed arithmetic.
        model = cusense.Model(artifact)
        validation, v_means, v_counts = self.sets["validation"]
        logits = [model.score(mean, count)["logit"] for mean, count in zip(v_means, v_counts)]
        again = presence.calibrate_threshold(validation, logits, self.settings["threshold"]["target_false_alarm"], 1)
        self.assertEqual(cusense.threshold_record(again), manifest["threshold"])
        self.assertEqual(freeze["reload"], {"largest_logit_difference": 0.0, "largest_input_difference": 0.0,
                                            "tolerance": 1e-5, "device": "cpu", "rows": {"training": 24, "validation": 24},
                                            "training_stride": 1})
        fit = json.loads((run / "fit.json").read_text())
        self.assertEqual((fit["schema"], fit["graded"], fit["history"]["epochs"]), ("cusense-fit/v1", False, 60))
        self.assertLess(fit["history"]["final_loss"], fit["history"]["initial_loss"])
        rates = fit["at_the_frozen_threshold"]["validation"]
        self.assertEqual((rates["target_present"], rates["target_absent"], rates["position_error"]["eligible"]), (12, 12, 12))
        self.assertEqual(rates["area_under_curve"], evaluator.area_under_curve(logits, [row["label"] for row in validation]))
        # The statistics of the script and of the evaluator are one definition.
        errors = [0.1 * value for value in range(1, 12)]
        self.assertEqual(train_cusense.error_statistics(errors), evaluator.error_statistics(errors))
        # The script names no evaluator module and reads no episode.
        script = (ROOT / "scripts/train_cusense.py").read_text()
        self.assertNotIn("ocudu_cuda_isac.episodes", script)

    def test_the_training_script_refuses_by_its_command_line(self):
        rows, means, counts = self.sets["training"]
        validation = self.sets["validation"]
        cusense.write_samples(self.scratch / "held.jsonl", rows[:-1] + [dict(rows[-1], split="held_out")], means, counts)
        crossed = [dict(validation[0][0], scene=rows[0]["scene"])] + validation[0][1:]
        cusense.write_samples(self.scratch / "crossed.jsonl", crossed, validation[1], validation[2])
        other = small_spec(slots=(7, 8))
        (self.scratch / "other-spec.json").write_text(json.dumps(other))
        (self.scratch / "bad-config.json").write_text(json.dumps(dict(self.settings, min_per_class=100)))
        (self.scratch / "few.json").write_text(json.dumps(dict(self.settings, threshold={"target_false_alarm": 0.05,
                                                                                         "min_negatives": 100})))
        base = {"--training": self.root / "training.jsonl", "--validation": self.root / "validation.jsonl",
                "--spec": self.root / "spec.json", "--grid": self.root / "grid.json", "--config": self.root / "training.json"}
        cases = (({"--training": self.scratch / "held.jsonl"}, "a held-out sample is read after the freeze"),
                 ({"--validation": self.scratch / "crossed.jsonl"}, "is in the training and in the validation split"),
                 ({"--training": self.root / "validation.jsonl", "--validation": self.root / "training.jsonl"},
                  "of the validation split is among the training samples"),
                 ({"--spec": self.scratch / "other-spec.json"}, "the samples were computed with the specification"),
                 ({"--config": self.scratch / "bad-config.json"}, "and the fit needs 100 of each"),
                 ({"--config": self.scratch / "few.json"}, "and the calibration needs 100"),
                 ({"--grid": self.root / "spec.json"}, "the position grid is incomplete"),
                 ({"--training": self.scratch / "nowhere.jsonl"}, "cannot read the samples"))
        for number, (changes, message) in enumerate(cases):
            arguments = dict(base, **changes)
            target = self.scratch / f"out-{number}"
            words = [str(word) for pair in arguments.items() for word in pair] + ["--output-dir", str(target)]
            with self.subTest(message=message), contextlib.redirect_stderr(io.StringIO()) as errors, \
                    self.assertRaises(SystemExit) as stop:
                train_cusense.main(words)
            self.assertEqual(stop.exception.code, 2)
            self.assertTrue(errors.getvalue().startswith("train_cusense: "), errors.getvalue())
            self.assertEqual(errors.getvalue().count("\n"), 1)
            self.assertIn(message, errors.getvalue())
            self.assertFalse(target.exists() and (target / "artifact").exists())
        with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
            train_cusense.main([str(word) for pair in base.items() for word in pair] + ["--output-dir", str(self.root / "run")])
        self.assertIn("A training run is never overwritten", errors.getvalue())


@unittest.skipIf(np is None, NEEDS)
class DecisionTests(unittest.TestCase):
    """One decision per answer of the stream, the optional stage, and what a sensor writes."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.spec = small_spec()
        smoothing = dict(cusense.demo_settings()["smoothing"], maps=2)
        cls.artifact = make_artifact(cls.root / "artifact", smoothing=smoothing, logit=0.0)
        cls.plain = make_artifact(cls.root / "plain")
        generator = np.random.Generator(np.random.PCG64(31))
        cls.records = periods_of(3, generator, cell=(1, 2), episode=0xA) \
            + periods_of(2, generator, episode=0xB, first_sequence=9)
        del cls.records[10]   # one slot of the first period of the second episode was never published
        cls.records = [obs.Record(dict(record.header, sequence=index), record.host, record.mask, record.payload)
                       for index, record in enumerate(cls.records)]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def written(self, artifact=None, records=None, name="out"):
        model = cusense.Model(artifact or self.artifact)
        lines, stream, decisions, inputs, maps = cusense.run_model(records or self.records, model)
        out = self.scratch / name
        (out / "windows").mkdir(parents=True)
        cusense.write_windows(out / "windows", model.spec, lines, stream, {"kind": "unit", "records": len(self.records)})
        cusense.write_output(out, model, decisions, inputs, maps, {"kind": "unit", "records": len(self.records)})
        return out, decisions, inputs, maps

    def test_every_answer_of_the_stream_gets_one_decision(self):
        model = cusense.Model(self.artifact)
        loads = cusense.Model.LOADS
        lines, stream, decisions, inputs, maps = cusense.run_model(self.records, model)
        self.assertEqual([decision["status"] for decision in decisions],
                         ["insufficient_history", "insufficient_history", "ok"] * 3
                         + ["insufficient_history", "gap", "insufficient_history", "insufficient_history", "ok"])
        self.assertEqual((len(decisions), len(lines), len(inputs), len(maps), cusense.Model.LOADS), (14, 14, 4, 4, loads))
        for decision, line in zip(decisions, lines):
            self.assertEqual((decision["schema"], decision["line"], decision["window"], decision["trigger"], decision["runtime"]),
                             ("cusense-decision/v1", line["line"], line["window"], line["trigger"], "reference"))
            self.assertEqual(decision["model"], self.artifact.model())
            if decision["status"] != "ok":
                # A filling window and a gap give a decision under their status and no score.
                self.assertEqual([decision[name] for name in ("score", "decision", "location", "smoothed", "input")], [None] * 5)
                continue
            score, location = decision["score"], decision["location"]
            self.assertEqual(decision["decision"], {"present": score["logit"] > 0.0, "threshold_logit": 0.0,
                                                    "detector": "frozen_threshold"})
            self.assertEqual((decision["contributing"], decision["time"]), (line["contributing"], line["time"]))
            drawn = maps[decision["location"]["map"]["index"]]
            row, col = location["cell"]
            self.assertEqual((float(drawn.max()), location["probability"]), (float(drawn[row, col]), float(drawn[row, col])))
            self.assertEqual(location["position_m"], cusense.cell_centre(self.artifact.grid, (row, col)))
            self.assertEqual(decision["input"]["sha256"], cusense.input_sha256(inputs[decision["input"]["index"]][None]))
            self.assertEqual(location["map"]["sha256"], hashlib.sha256(drawn.tobytes()).hexdigest())
            self.assertIsInstance(decision["infer_ns"], int)
        # Two runs on the same records agree on every field but the host times, the durations and the device.
        again = cusense.run_model(self.records, cusense.Model(self.artifact))[2]
        self.assertEqual([cusense.comparable(decision) for decision in again],
                         [cusense.comparable(decision) for decision in decisions])

    def test_the_optional_stage_has_its_own_output_and_changes_no_raw_result(self):
        smoothed = cusense.run_model(self.records, cusense.Model(self.artifact))
        plain = cusense.run_model(self.records, cusense.Model(self.plain))
        both = [(a, b) for a, b in zip(smoothed[2], plain[2]) if a["status"] == "ok"]
        self.assertEqual(len(both), 4)
        for first, second in both:
            # The same weights were drawn for both artifacts: the raw map, the score and the decision are the same.
            self.assertEqual((first["score"], first["location"]["cell"], first["location"]["map"]["sha256"]),
                             (second["score"], second["location"]["cell"], second["location"]["map"]["sha256"]))
            self.assertIsNone(second["smoothed"])
        stage = [first["smoothed"] for first, _ in both]
        # Three windows of one history, with a mean over the newest two maps, and a new history for the next episode.
        self.assertEqual([item["maps"] for item in stage], [1, 2, 2, 1])
        self.assertEqual({item["entry"] for item in stage}, {"map_mean_kalman"})
        maps = smoothed[4]
        expected = cusense.locate(np.mean([maps[1].astype(np.float64), maps[2].astype(np.float64)], axis=0), self.artifact.grid)
        self.assertEqual((stage[2]["cell"], stage[2]["position_m"]), (expected["cell"], expected["position_m"]))
        self.assertEqual((stage[0]["kalman_position_m"], stage[0]["kalman_velocity_m_s"]), (stage[0]["position_m"], [0.0, 0.0]))
        self.assertEqual(stage[3]["kalman_position_m"], stage[3]["position_m"])

    def test_the_filter_holds_a_position_and_follows_a_walk(self):
        grid = grid_of(50, 34)
        settings = cusense.demo_settings()["smoothing"]
        still = cusense.Smoother(settings, grid, 0.005)

        def one_hot(cell):
            drawn = np.zeros((50, 34))
            drawn[cell] = 1.0
            return drawn

        for period in range(20):
            made = still.update(("1", "a"), period, one_hot((10, 10)))
        centre = cusense.cell_centre(grid, (10, 10))
        self.assertEqual((made["maps"], made["cell"]), (10, [10, 10]))
        self.assertAlmostEqual(made["kalman_position_m"][0], centre[0], places=9)
        self.assertAlmostEqual(made["kalman_position_m"][1], centre[1], places=9)
        # A target that walks one cell along y per 170 ms, 1.2 m/s: the filter follows it.
        walk = cusense.Smoother(dict(settings, maps=1), grid, 0.005)
        for period in range(0, 34 * 20, 34):
            made = walk.update(("1", "a"), period, one_hot((period // 34, 5)))
        truth = cusense.cell_centre(grid, (19, 5))
        self.assertLess(math.dist(made["kalman_position_m"], truth), 0.25)
        self.assertGreater(made["kalman_velocity_m_s"][1], 0.5)
        # A period that does not follow, another episode and another epoch each start a new history.
        for key, period in ((("1", "a"), 3), (("1", "b"), 700), (("2", "b"), 701)):
            self.assertEqual(walk.update(key, period, one_hot((0, 0)))["maps"], 1)
        self.assertIsNone(cusense.Smoother({"entry": "none"}, grid, 0.005).update(("1", "a"), 0, one_hot((0, 0))))

    def test_an_output_verifies_and_a_second_reader_recovers_the_decision(self):
        out, decisions, inputs, maps = self.written()
        self.assertEqual(sorted(path.name for path in out.iterdir()),
                         ["cusense.json", "decisions.jsonl", "inputs.f32", "maps.f32", "windows"])
        output = cusense.open_output(out)
        report = output.report
        self.assertEqual((report["schema"], report["counts"]["decisions"], report["counts"]["scored"], report["counts"]["gap"],
                          report["runtime"]["entry"], report["window_fields"]),
                         ("cusense-report/v1", 14, 4, 1, "reference", list(cusense.WINDOW_FIELDS)))
        self.assertEqual(report["files"], {"inputs.f32": {"shape": [4, 2, 48], "dtype": "<f4"},
                                           "maps.f32": {"shape": [4, 4, 3], "dtype": "<f4"}})
        self.assertEqual((output.inputs.shape, output.maps.shape), ((4, 2, 48), (4, 4, 3)))
        scored = output.scored[1]
        self.assertTrue(np.array_equal(output.input_of(scored), inputs[1]) and np.array_equal(output.map_of(scored), maps[1]))
        shown = cusense.summary(output, 1)
        self.assertEqual((shown["artifact_sha256"], shown["input"], shown["input_shape"], shown["grid"], shown["scored_windows"]),
                         (self.artifact.sha256, "o3_fp16", [1, 2, 48], [4, 3], 4))
        window = shown["window"]
        self.assertEqual((window["id"], window["contributing"], window["logit"], window["cell"], window["input_sha256"]),
                         (f"1/{0xA:016x}/1", 3, scored["score"]["logit"], scored["location"]["cell"], scored["input"]["sha256"]))
        self.assertAlmostEqual(window["map_sum"], 1.0, places=5)
        with self.assertRaisesRegex(CuSenseError, "holds 4 scored windows, and window 4 was asked for"):
            cusense.summary(output, 4)
        # What a sensor writes names nothing of the truth: the one split it names is the one its statistics were
        # fitted on, in the manifest of its artifact.
        for path in out.rglob("*.json*"):
            for word in ('"label"', '"scene"', '"pair"', "truth", "held_out", '"split": "validation"'):
                self.assertNotIn(word, path.read_text(), path.name)
        self.assertNotIn('"split"', (out / "decisions.jsonl").read_text() + (out / "windows/windows.jsonl").read_text())

    def test_a_changed_output_is_refused(self):
        out, _, _, _ = self.written()

        def changed(label, edit):
            copy = self.scratch / label
            shutil.copytree(out, copy)
            decisions = [json.loads(text) for text in (copy / "decisions.jsonl").read_text().splitlines()]
            report = json.loads((copy / "cusense.json").read_text())
            edit(decisions, report, copy)
            (copy / "decisions.jsonl").write_text("".join(json.dumps(line) + "\n" for line in decisions))
            (copy / "cusense.json").write_text(json.dumps(report))
            return copy

        def flip(name, at=0):
            def edit(decisions, report, copy):
                data = bytearray((copy / name).read_bytes())
                data[at] ^= 1
                (copy / name).write_bytes(bytes(data))
            return edit

        cases = (("logit", lambda decisions, report, copy: decisions[2]["score"].update(logit=decisions[2]["score"]["logit"] + 1),
                  "decision 2: the digest of its input, its map, its location, its score or its decision"),
                 ("verdict", lambda decisions, report, copy: decisions[2]["decision"].update(
                     present=not decisions[2]["decision"]["present"]), "decision 2"),
                 ("cell", lambda decisions, report, copy: decisions[2]["location"].update(cell=[0, 0], position_m=[0.0, 0.0]),
                  "decision 2"),
                 ("input", flip("inputs.f32"), "decision 2"),
                 ("map", flip("maps.f32", 4 * 12 + 1), "decision 5"),
                 ("score-on-gap", lambda decisions, report, copy: decisions[10].update(score={"logit": 1.0, "probability": 0.7}),
                  "has the status gap and carries a score"),
                 ("model", lambda decisions, report, copy: decisions[0]["model"].update(weights_sha256="0" * 64),
                  "decision 0 is not the cusense-decision/v1 line"),
                 ("dropped", lambda decisions, report, copy: decisions.pop(), "13 decisions for 14 lines"),
                 ("counts", lambda decisions, report, copy: report["counts"].update(present=9), "counts of the report"),
                 ("artifact", lambda decisions, report, copy: report["artifact"]["manifest"]["threshold"].update(logit=-9.0),
                  "it was moved after it was frozen"),
                 ("extra", lambda decisions, report, copy: (copy / "truth.json").write_text("{}"),
                  "holds truth.json, which its report does not list"),
                 ("short", lambda decisions, report, copy: (copy / "maps.f32").write_bytes(b""),
                  "do not hold one tensor per scored decision"))
        for label, edit, message in cases:
            with self.subTest(label=label), self.assertRaisesRegex(CuSenseError, message):
                cusense.open_output(changed(label, edit))

    def test_the_weights_are_loaded_once(self):
        loads = cusense.Model.LOADS
        model = cusense.Model(self.artifact)
        for _ in range(3):
            cusense.run_model(self.records[:3], model)
        self.assertEqual(cusense.Model.LOADS, loads + 1)

    def archive(self, records=None):
        """The records as the archive of a consumer."""
        directory = self.scratch / "archive"
        directory.mkdir()
        rows = []
        for record in records or self.records:
            header = record.header
            data = record.to_bytes()
            (directory / obs.record_name(header["epoch"], header["sequence"])).write_bytes(data)
            rows.append({"epoch": header["epoch"], "sequence": header["sequence"],
                         "opportunity": f"{header['opportunity']:016x}", "depth": 0, "bytes": len(data),
                         "t_publish_ns": header["t_publish_ns"], "t_read_ns": header["t_publish_ns"] + 400,
                         "delay_ns": 400, "intact": True, "file": True})
        (directory / "consumer.json").write_text(json.dumps({
            "schema": obs.CONSUMER_SCHEMA, "reader": "native", "definition": f"{obs.DEFINITION:016x}", "wait": "wait",
            "epoch": 1, "pid": 1, "cpu": -1, "hold_us": 0, "window": 0, "slot_count": 8, "slot_bytes": 20288,
            "attached_ns": 1, "ended": "closed", "counts": {"read": len(rows), "stale": 0, "released": len(rows),
                                                            "skipped": 0, "sleeps": 0, "kept": len(rows), "not_kept": 0,
                                                            "system_calls": 0}, "records": rows}))
        return directory

    def test_command_line(self):
        def run(*words):
            said, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(said), contextlib.redirect_stderr(errors):
                try:
                    status = cusense.main([str(word) for word in words])
                except SystemExit as stop:
                    status = stop.code
            return status, said.getvalue(), errors.getvalue()

        # The two specifications and the training configuration of the demonstration.
        for entry, name in (("o3_fp16", "cusense-o3-period"), ("o2_comb", "cusense-o2-period")):
            status, said, _ = run("spec", "--output", self.scratch / f"{entry}.json", "--entry", entry)
            self.assertEqual((status, json.loads((self.scratch / f"{entry}.json").read_text())),
                             (0, cusense.demo_spec(entry)))
            self.assertEqual(json.loads((self.scratch / f"{entry}.json").read_text())["spec_id"], name)
        self.assertEqual(run("spec", "--output", self.scratch / "o3_fp16.json")[0], 2)
        status, _, _ = run("settings", "--output", self.scratch / "training.json")
        self.assertEqual((status, json.loads((self.scratch / "training.json").read_text())), (0, cusense.demo_settings()))
        status, said, _ = run("device", "--device", "cpu")
        self.assertEqual((status, json.loads(said)["type"]), (0, "cpu"))
        # The windows of an archive, without a model, and the decisions of an archive, with one.
        archive = self.archive()
        (self.scratch / "spec.json").write_text(json.dumps(self.spec))
        (self.scratch / "windows").mkdir()
        status, said, _ = run("windows", "--archive", archive, "--spec", self.scratch / "spec.json", "--output-dir",
                              self.scratch / "windows")
        self.assertEqual(status, 0)
        self.assertIn("cusense: 14 records, 4 complete windows, 9 lines of a filling window, 1 gaps", said)
        self.assertEqual(len(cusense.open_windows(self.scratch / "windows").ok), 4)
        (self.scratch / "replay").mkdir()
        status, said, _ = run("replay", "--archive", archive, "--artifact", self.artifact.directory, "--output-dir",
                              self.scratch / "replay")
        self.assertEqual(status, 0)
        self.assertIn("cusense: 14 decisions on cpu, 4 scored: ", said)
        self.assertIn("1 gaps, 0 invalid, 0 rejected; the model unit-model ", said)
        output = cusense.open_output(self.scratch / "replay")
        self.assertEqual((output.report["source"]["kind"], output.report["source"]["records"], output.report["loads"] >= 1),
                         ("archive", 14, True))
        status, said, _ = run("show", self.scratch / "replay", "--window", 3)
        self.assertEqual((status, json.loads(said)["window"]["id"]), (0, f"1/{0xB:016x}/1"))
        status, said, _ = run("inspect", self.artifact.directory, "--sha256", self.artifact.sha256)
        self.assertEqual((status, json.loads(said)["artifact_sha256"]), (0, self.artifact.sha256))
        # Refusals: status 2, one line, nothing written.
        (self.scratch / "empty").mkdir()
        for words, message in (
                (("inspect", self.artifact.directory, "--sha256", "0" * 64), "was asked for"),
                (("replay", "--archive", archive, "--artifact", self.scratch / "nowhere", "--output-dir", self.scratch / "empty"),
                 "is not a complete model"),
                (("replay", "--archive", self.scratch / "nowhere", "--artifact", self.artifact.directory, "--output-dir",
                  self.scratch / "empty"), "holds no readable consumer.json"),
                (("replay", "--archive", archive, "--artifact", self.artifact.directory, "--output-dir", self.scratch / "replay"),
                 "does not exist or is not empty"),
                (("replay", "--archive", archive, "--artifact", self.artifact.directory, "--device", "cuda:99", "--output-dir",
                  self.scratch / "empty"), "cuda"),
                (("show", self.scratch / "empty"), "is not a complete sensor output")):
            status, said, errors = run(*words)
            with self.subTest(message=message):
                self.assertEqual((status, errors.count("\n"), errors.startswith("cusense: ")), (2, 1, True), errors)
                self.assertIn(message, errors)
        self.assertEqual(list((self.scratch / "empty").iterdir()), [])


@unittest.skipIf(np is None, NEEDS)
class EvaluatorTests(unittest.TestCase):
    """The plans of a set replay, the positions, the samples, the oracle, the parity and the counts."""

    @classmethod
    def setUpClass(cls):
        import test_grid_apply_driver as stores

        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.grant = ROOT / "configs/grants/two_prb_offset.json"
        config = set_double.set_config(cls.root / "set.json", cls.grant, set_double.trajectories(4, 2, 2))
        set_double.analytic_set(config, cls.root / "set")
        cls.set = sets.open_set(cls.root / "set")
        cls.table = evaluator.split_table(cls.set)
        entry = cls.set.episodes("training")[0]
        cls.bindings = {}
        for grant_id, grant in entry.schedule.grants.items():
            directory = cls.root / "stores" / grant_id / "store"
            stores.store_of(grant, directory)
            store = grants.open_store(directory)
            cls.bindings[grant_id] = {"x_sha256": store.manifest["files"]["x.npy"]["sha256"], "store": str(directory),
                                      "record_sha256": entry.manifest["grants"][grant_id]["record_sha256"]}
        cls.rings = {name: {"stream": name, "window_records": 3, "burst_records": 3, "slots": 8, "wait": "wait",
                            "attach_timeout_ms": 30000, "drain_ms": 5000} for name in ("o2", "o3")}
        cls.estimate = {"copy_stream": "producer", "landing": "slot", "timeout_us": 2000}
        # The two-block grant: 24 subcarriers from subcarrier 180, one receive port.
        cls.spec = cusense.demo_spec(subcarriers=24, rx=1, first_subcarrier=180)
        cls.grid = cusense.grid_record(cls.set.manifest["position_grid"])
        cls.plans, cls.channels, cls.records = {}, {}, {}
        generator = np.random.Generator(np.random.PCG64(41))
        for role in ("training", "validation", "held_out"):
            rows = [row for row in cls.table["episodes"] if row["group"] == role]
            cls.channels[role], cls.plans[role] = evaluator.write_plans(
                f"unit-{role}", rows, cls.bindings, cls.root / "plans" / role, cls.rings, cls.estimate, epoch=2)
            cls.records[role] = cls.records_of(cls.channels[role], generator)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def records_of(cls, channel_plan, generator):
        """O3 records for every opportunity of a channel plan, under its keys: a target adds a ripple of its cell."""
        truth = evaluator.positions(channel_plan)
        records = []
        for opportunity in channel_plan["opportunities"]:
            episode = o2_reference.opaque_key(opportunity["realization_id"])
            key = o2_reference.opaque_key(opportunity["opportunity_id"])
            place = truth[episode]["opportunities"][key]
            slot = 20 * (opportunity["index"] // 6) + (7, 8, 9, 17, 18, 19)[opportunity["index"] % 6]
            cell = None if place["cell"] is None else (place["cell"][0] % 4, place["cell"][1] % 3)
            records.append(o3_record(channel(generator, rx=1, positions=24, cell=cell), slot, len(records),
                                     episode=int(episode, 16), opportunity=int(key, 16), first_subcarrier=180, epoch=2))
        return records

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def test_a_set_is_replayed_with_both_stages_on_their_rings(self):
        channel, plan = self.channels["training"], self.plans["training"]
        self.assertEqual(sorted(path.name for path in (self.root / "plans/training").iterdir()),
                         ["channel-plan.json", "plan.json", "replay-plan.json"])
        self.assertEqual((channel["schema"], len(channel["sources"]), len(channel["opportunities"])),
                         ("grid-apply-plan/v1", 8, 48))
        self.assertEqual((plan["schema"], plan["evidence"], plan["observer"]["tap"], plan["observer"]["epoch"]),
                         ("pilot-ls-plan/v3", {"mode": "timed"}, "pilot_ls", 2))
        estimate = plan["estimate"]
        self.assertEqual((estimate["tap"], estimate["stage"], estimate["layout"], estimate["backend"], estimate["copy_stream"],
                          estimate["landing"]), ("native_estimate", "O3", "ocudu_o3_fp16", "cuda", "producer", "slot"))
        # One ring per stream, each first in, first out with eight slots: a window of three, a burst of three and one.
        self.assertEqual(([plan["ring"]["stream"], plan["ring"]["slots"], plan["ring"]["attach"]],
                          [estimate["ring"]["stream"], estimate["ring"]["slots"], estimate["ring"]["attach"]]),
                         (["o2", 8, "required"], ["o3", 8, "required"]))
        self.assertEqual(ne.load_plan(self.root / "plans/training/plan.json"), plan)
        self.assertEqual(str(ne.socket_path("native", plan)), "native/estimate/o3.sock")
        self.assertEqual(evaluator.keys_of(channel),
                         [row["keys"][case] for row in self.table["episodes"] if row["group"] == "training"
                          for case in ("a", "b")])
        rows = [row for row in self.table["episodes"] if row["group"] == "training"]
        with self.assertRaisesRegex(CuSenseFrontendError, "cannot be replayed: .*never overwritten"):
            evaluator.write_plans("unit-training", rows, self.bindings, self.root / "plans" / "training", self.rings,
                                  self.estimate)
        wrong = {name: dict(item, x_sha256="0" * 64) for name, item in self.bindings.items()}
        with self.assertRaisesRegex(CuSenseFrontendError, "cannot be replayed: .*the entry binds"):
            evaluator.write_plans("unit", rows, wrong, self.scratch / "wrong", self.rings, self.estimate)
        with self.assertRaisesRegex(CuSenseFrontendError, "the ring of a stream is unusable"):
            evaluator.write_plans("unit", rows, self.bindings, self.scratch / "ring",
                                  dict(self.rings, o3=dict(self.rings["o3"], window_records=-1)), self.estimate)
        with self.assertRaisesRegex(CuSenseFrontendError, "cannot be replayed: .*different names"):
            evaluator.write_plans("unit", rows, self.bindings, self.scratch / "names",
                                  dict(self.rings, o3=dict(self.rings["o3"], stream="o2")), self.estimate)

    def test_positions_are_those_the_set_recorded_at_each_opportunity(self):
        channel = self.channels["held_out"]
        truth = evaluator.positions(channel)
        lookup = evaluator.by_key(self.table)
        self.assertEqual(sorted(truth), sorted(evaluator.keys_of(channel)))
        for key, known in truth.items():
            self.assertEqual((known["present"], len(known["opportunities"])), (bool(lookup[key]["label"]), 6))
            entry = next(item for item in self.set.episodes("held_out") if str(Path(item.directory).resolve()) == known["entry"])
            for place in known["opportunities"].values():
                if not known["present"]:
                    self.assertEqual((place["cell"], place["position_m"]), (None, None))
                    continue
                recorded = entry.truth(place["index"])
                self.assertEqual((place["cell"], place["position_m"]), (recorded["cell"], recorded["position_m"][:2]))
                self.assertEqual(place["cell"], sets.cell(self.set.manifest["config"], *place["position_m"]))
        with self.assertRaisesRegex(CuSenseFrontendError, "does not replay episode entries"):
            evaluator.positions({"sources": [{"realization_id": "rz-x", "h": {"path": "/tmp/h.npy"}}], "opportunities": []})

    def windows_of(self, role):
        lines, stream = cusense.run_records(self.records[role], self.spec)
        means, counts = cusense.stacked(stream)
        return lines, means, counts

    def test_windows_are_joined_to_the_split_and_to_the_position_of_their_last_opportunity(self):
        lines, means, counts = self.windows_of("training")
        channel = self.channels["training"]
        truth, keys = evaluator.positions(channel), evaluator.keys_of(channel)
        rows, means, counts, account = evaluator.samples(lines, means, counts, self.table, "training", truth, keys)
        # Two periods of three slots per realization: two windows each.
        self.assertEqual(account, {"role": "training", "realizations": 8, "with_a_window": 8, "windows": 16,
                                   "per_realization": [2], "target_present": 8, "target_absent": 8, "missing": []})
        lookup = evaluator.by_key(self.table)
        complete = [line for line in lines if line["status"] == "ok"]
        for row, line in zip(rows, complete):
            cusense.check_sample(row)
            known = lookup[row["episode"]]
            self.assertEqual((row["split"], row["scene"], row["pair"], row["label"]),
                             ("training", known["scene"], known["pair"], known["label"]))
            place = truth[row["episode"]]["opportunities"][line["contributing"][-1]["opportunity"]]
            self.assertEqual((row["cell"], row["position_m"]), (place["cell"], place["position_m"]))
            self.assertEqual(place["index"] % 3, 2)   # the last opportunity of the window: slot 9 or 19 of its period
            self.assertEqual((row["mean_sha256"], row["shape"]), (line["mean"]["sha256"], [1, 24, 2]))
        self.assertEqual(cusense.check_split(rows, [])["pairs"]["training"], 4)
        cusense.write_samples(self.scratch / "training.jsonl", rows, means, counts)
        self.assertEqual(len(cusense.read_samples(self.scratch / "training.jsonl").rows), 16)
        # Several runs of one split are one sample set, with the rows numbered again.
        merged = evaluator.merge_samples([(rows[:6], means[:6], counts[:6]), (rows[6:], means[6:], counts[6:])])
        self.assertEqual(([row["index"] for row in merged[0]], merged[1].shape), (list(range(16)), (16, 1, 24, 2)))
        # A window of another split is refused, and so is a key the set does not hold.
        with self.assertRaisesRegex(CuSenseFrontendError,
                                    "belongs to the training split, and this stream was read as validation"):
            evaluator.samples(lines, means, counts, self.table, "validation", truth)
        table = json.loads(json.dumps(self.table))
        table["episodes"] = [row for row in table["episodes"] if row["group"] != "training"]
        with self.assertRaisesRegex(CuSenseFrontendError, "belongs to no realization of the set"):
            evaluator.samples(lines, means, counts, table, "training", truth)
        with self.assertRaisesRegex(CuSenseFrontendError, "belongs to no realization of the run"):
            evaluator.samples(lines, means, counts, self.table, "training", {})
        # A realization without a window stays in the account, with the last answer of its stream.
        other = [row["keys"]["a"] for row in self.table["episodes"] if row["group"] == "held_out"][:1]
        account = evaluator.samples(lines, means, counts, self.table, "training", truth, keys + other)[3]
        self.assertEqual((account["per_realization"], account["missing"]),
                         ([0, 2], [{"episode": other[0], "pair": lookup[other[0]]["pair"], "last_answer": None}]))

    def test_the_oracle_gives_the_window_and_the_input_of_the_stream(self):
        records = self.records["training"]
        lines, means, counts = self.windows_of("training")
        complete = [line for line in lines if line["status"] == "ok"]
        by_sequence = {record.sequence: record for record in records}
        for number in (0, 5, 15):
            line = complete[number]
            mean, count = evaluator.oracle_window([by_sequence[item["sequence"]] for item in line["contributing"]], self.spec)
            self.assertTrue(np.array_equal(count, counts[number]))
            self.assertTrue(np.allclose(mean, means[number], rtol=0, atol=1e-7))
        # The stage O2 through its own reader, on the comb.
        comb = cusense.demo_spec("o2_comb", subcarriers=24, rx=1, first_subcarrier=180)
        generator = np.random.Generator(np.random.PCG64(43))
        pilots = [o2_record(channel(generator, rx=1, positions=24), slot, index, first_subcarrier=180)
                  for index, slot in enumerate((7, 8, 9))]
        mean, count = evaluator.oracle_window(pilots, comb)
        made = cusense.run_records(pilots, comb)[1].last
        self.assertTrue(np.allclose(mean, made[0], rtol=0, atol=1e-7) and np.array_equal(count, made[1]))
        # The input of the model at every second position, from the formula of the card with Python floats.
        template = {"values": means[1], "count": counts[1].astype("<u4")}
        normalizer = {"mean": 0.01, "scale": 0.05, "epsilon": 1e-8}
        x = cusense.zscore(*cusense.background_subtract(means[0], counts[0], template["values"], template["count"]), 0.01, 0.05)
        points = evaluator.oracle_input(means[0], counts[0], template["values"], template["count"], normalizer, step=2)
        self.assertEqual(len(points), 12)
        for element, position, value in points:
            self.assertAlmostEqual(float(x[element, position]), value, places=5)

    def outputs(self, role="held_out", device_shift=0.0, drop=None, smoothing=True):
        """A sensor output on the records of one role, with a small artifact, and the keys the run replayed."""
        name = "artifact-smoothed" if smoothing else "artifact-plain"
        if not (self.root / name).exists():
            make_artifact(self.root / name, spec=self.spec, grid=self.grid, logit=0.0,
                          smoothing=dict(cusense.demo_settings()["smoothing"], maps=2) if smoothing else None)
        model = cusense.Model(cusense.open_artifact(self.root / name))
        records = [record for index, record in enumerate(self.records[role]) if index != drop]
        lines, stream, decisions, inputs, maps = cusense.run_model(records, model)
        if device_shift:
            for decision in decisions:
                if decision["status"] == "ok":
                    decision["score"]["logit"] += device_shift
                    decision["score"]["probability"] = 1.0 / (1.0 + math.exp(-decision["score"]["logit"]))
                    decision["decision"]["present"] = decision["score"]["logit"] > model.threshold
        directory = Path(tempfile.mkdtemp(dir=self.scratch))
        (directory / "windows").mkdir()
        cusense.write_windows(directory / "windows", self.spec, lines, stream, {"kind": "unit"})
        cusense.write_output(directory, model, decisions, inputs, maps, {"kind": "unit"})
        return cusense.open_output(directory), evaluator.keys_of(self.channels[role])

    def test_decisions_are_compared_by_window_and_input_digest(self):
        device, _ = self.outputs()
        offline, _ = self.outputs()
        problems, facts = evaluator.parity(device, offline)
        self.assertEqual((problems, facts["joined"], facts["flips"], facts["locations_moved"], facts["schema"]),
                         ([], 8, [], [], "cusense-parity/v1"))
        self.assertEqual((facts["largest_logit_difference"], facts["largest_map_difference"], facts["tolerance"]),
                         (0.0, 0.0, evaluator.TOLERANCE))
        # Scores that drift beyond the tolerance are named, window by window.
        drift, _ = self.outputs(device_shift=5e-3)
        problems, facts = evaluator.parity(drift, offline)
        self.assertEqual(len([problem for problem in problems if "the logit differs by 5.000e-03" in problem]), 8)
        # A drift inside the tolerance passes. A decision that flips there is listed and is no problem.
        near, _ = self.outputs(device_shift=5e-4)
        problems, facts = evaluator.parity(near, offline)
        self.assertEqual(problems, [])
        self.assertTrue(all(abs(flip["margin"]) <= 2e-3 for flip in facts["flips"]))
        # The tolerance of a logit grows with its magnitude: a saturated logit is a large float32 value.
        self.assertEqual((evaluator.TOLERANCE["logit_absolute"], evaluator.TOLERANCE["logit_relative"]), (1e-3, 1e-4))
        # A window that one side did not score, and another artifact, are problems.
        short, _ = self.outputs(drop=2)
        problems, _ = evaluator.parity(short, offline)
        self.assertIn("1 windows are scored on one side only", problems[0])
        other, _ = self.outputs(smoothing=False)
        self.assertIn("the two outputs were scored with different artifacts", evaluator.parity(other, offline)[0])

    def test_presence_is_counted_with_its_denominators_and_position_error_in_metres(self):
        output, keys = self.outputs()
        truth = evaluator.positions(self.channels["held_out"])
        made = evaluator.evaluate(output, self.table, truth, "held_out", keys, 2)
        self.assertEqual((made["schema"], made["graded"], made["input"], made["artifact_sha256"]),
                         ("cusense-evaluation/v1", False, "o3_fp16", output.manifest["sha256"]))
        by_window, by_episode = made["presence"]["windows"], made["presence"]["episodes"]
        # Two held-out pairs: four realizations, two windows each. Every one is in a denominator.
        self.assertEqual((by_window["target_present"]["total"], by_window["target_absent"]["total"],
                          by_episode["target_present"]["total"], by_episode["target_absent"]["total"]), (4, 4, 2, 2))
        for side in (by_window, by_episode):
            present, absent = side["target_present"], side["target_absent"]
            self.assertEqual(present["detected"] + present["missed"] + present["missing"], present["total"])
            self.assertEqual(absent["false_alarm"] + absent["quiet"] + absent["missing"], absent["total"])
        self.assertEqual(by_episode["rule"], evaluator.SELECTION_RULE["last"])
        scored = [row for row in made["rows"] if row["window"] is not None]
        self.assertEqual((len(scored), sum(row["selected"] for row in scored), made["presence"]["scored_windows"]), (8, 4, 8))
        for row in scored:
            self.assertEqual(row["selected"], row["period"] == 1)   # the last window of the episode
        # The position error: on the scored windows of target-present realizations, in metres, three ways.
        position = made["position"]
        self.assertEqual((position["unit"], position["raw"]["eligible"], position["smoothed"]["eligible"],
                          position["kalman"]["eligible"], position["cell_m"]), ("m", 4, 4, 4, self.grid["cell_m"]))
        present = [row for row in scored if row["label"] == 1]
        errors = [math.dist(row["position_m"], row["truth_m"]) for row in present]
        self.assertEqual([row["error_m"] for row in present], errors)
        self.assertEqual(position["raw"], evaluator.error_statistics(errors))
        self.assertTrue(all(row["error_m"] is None and row["truth_m"] is None for row in scored if row["label"] == 0))
        # A realization that lost a window keeps its denominator, and one without any decision is missing.
        short, _ = self.outputs(drop=2)
        counted = evaluator.evaluate(short, self.table, truth, "held_out", keys, 2)["presence"]
        label = "target_present" if evaluator.by_key(self.table)[keys[0]]["label"] else "target_absent"
        self.assertEqual((counted["windows"][label]["total"], counted["windows"][label]["missing"]), (4, 1))
        other = [row["keys"]["b"] for row in self.table["episodes"] if row["group"] == "training"][:1]
        with self.assertRaisesRegex(CuSenseFrontendError, "is no window of the validation split"):
            evaluator.evaluate(output, self.table, truth, "validation", keys, 2)
        more = evaluator.evaluate(output, self.table, truth, "held_out", keys + other, 2)
        self.assertEqual((more["presence"]["episodes"]["target_present"]["total"],
                          more["presence"]["episodes"]["target_present"]["missing"],
                          more["presence"]["windows"]["target_present"]["missing"]), (3, 1, 2))
        with self.assertRaisesRegex(CuSenseFrontendError, "unknown selection"):
            evaluator.evaluate(output, self.table, truth, "held_out", keys, 2, selection="best")

    def test_error_statistics_use_the_nearest_rank(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual((evaluator.nearest_rank(values, 0.5), evaluator.nearest_rank(values, 0.95),
                          evaluator.nearest_rank(values, 0.9), evaluator.nearest_rank([7.0], 0.9)), (3.0, 5.0, 5.0, 7.0))
        self.assertEqual(evaluator.nearest_rank(list(range(1, 11)), 0.9), 9)
        made = evaluator.error_statistics([0.3, 0.4, 1.2])
        self.assertEqual((made["eligible"], made["median_m"], made["p90_m"], made["max_m"]), (3, 0.4, 1.2, 1.2))
        self.assertAlmostEqual(made["mean_m"], 1.9 / 3)
        self.assertAlmostEqual(made["rmse_m"], math.sqrt((0.09 + 0.16 + 1.44) / 3))
        self.assertEqual(evaluator.error_statistics([]), {"eligible": 0, "mean_m": None, "median_m": None, "p90_m": None,
                                                          "rmse_m": None, "max_m": None})

    def test_the_published_figures_are_recorded_beside_the_measured_ones_and_never_graded(self):
        output, keys = self.outputs()
        truth = evaluator.positions(self.channels["held_out"])
        made = evaluator.evaluate(output, self.table, truth, "held_out", keys, 2)
        latency = {"runtime": "reference", "device": "cpu", "inference_us": {"count": 8, "p50": 400.0}}
        record = evaluator.reference_record(made, latency, {"bandwidth_mhz": 20})
        self.assertEqual((record["schema"], record["graded"], record["published"]["position_error_cm"]["mean"],
                          record["published"]["live_inference_us"]["value"]), ("cusense-reference/v1", False, 73.3, 241))
        measured = record["measured"]["position_error"]
        self.assertAlmostEqual(measured["raw"]["mean_cm"], 100.0 * made["position"]["raw"]["mean_m"])
        self.assertEqual((measured["raw"]["eligible"], measured["kalman"]["eligible"], record["measured"]["latency"]),
                         (4, 4, latency))
        # Every difference of the card is listed: the scene, the bandwidth and the GPU.
        listed = " ".join(record["differences"])
        for words in ("simulated scene", "20 MHz", "100 MHz", "GB10", "GH200", "reference runtime"):
            self.assertIn(words, listed)
        self.assertIn("S3.6", record["latency_comparison"])
        self.assertTrue(json.dumps(record).isascii())
        compared = evaluator.compare_arms({"o3": made}, None)
        self.assertEqual((compared["o3"]["target_present"], compared["o3"]["target_absent"]), (2, 2))


@unittest.skipIf(np is None, NEEDS)
class LiveSensorTests(unittest.TestCase):
    """The cuSense-like sensor in a second process, on a native producer of the ring."""

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
        # obs_feed publishes one record of the stage O2 per 20 slots, slot 8 of a period of 20, with the shape of
        # the demonstration and sample times of its own, which are not those of a slot: every record is rejected
        # by the stream, and every rejection is a decision without a score.
        spec = cusense.demo_spec("o2_comb", slots=(8,))
        spec["radio"]["period_slots"] = 20
        artifact = make_artifact(self.scratch / "artifact", spec=spec, steps=("pilot_ls",),
                                 smoothing=cusense.demo_settings()["smoothing"])
        out = self.scratch / "sensor"
        out.mkdir()
        socket, ready = self.scratch / "ring.sock", self.scratch / "ready.json"
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        sensor = subprocess.Popen([sys.executable, "-m", "ocudu_cuda_isac.sensing.cusense", "follow", "--socket",
                                   str(socket), "--artifact", str(artifact.directory), "--output-dir", str(out), "--wait",
                                   "wait", "--ready-file", str(ready)], env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
        # The producer starts when the sensor says that it is warm, as the launcher of the gate does.
        for _ in range(2400):
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
        self.assertIn("cusense: 24 decisions on ", said)
        self.assertIn("was loaded 1 time(s)", said)
        output = cusense.open_output(out)
        report = output.report
        self.assertEqual((report["counts"]["decisions"], report["counts"]["rejected"], report["counts"]["scored"],
                          report["loads"], report["source"]["kind"], report["source"]["ended"], report["source"]["records"]),
                         (24, 24, 0, 1, "ring", "closed", 24))
        self.assertTrue(all(line["status"] == "rejected" and line["score"] is None for line in output.decisions))
        self.assertTrue(all(type(line["t_decision_ns"]) is int and line["t_decision_ns"] >= line["t_window_ns"]
                            for line in output.decisions))
        # It was warm before it attached: the file names the device and the artifact, and both warm-ups ran.
        warm = json.loads(ready.read_text())
        self.assertEqual((warm["device"], warm["artifact_sha256"]), (str(cusense.device_of(cusense.default_device())),
                                                                     artifact.sha256))
        self.assertGreater(warm["warm_up_ms"]["frontend"], 0.0)
        self.assertGreater(warm["warm_up_ms"]["model"], 0.0)
        self.assertEqual(report["source"]["warm_up_ms"], warm["warm_up_ms"])
        kept = obs.open_archive(out / "windows/records").records
        self.assertLess(warm["t_ready_ns"], min(record.header["t_publish_ns"] for record in kept))
        self.assertEqual(report["runtime"]["type"], cusense.default_device())
        # With the sample times of a slot, and a payload of a channel in place of the counting bytes of the tool,
        # the headers it wrote are windows of one slot, and the same model scores them.
        generator = np.random.Generator(np.random.PCG64(51))
        fixed = []
        for record in kept:
            first, last = times_of(record.header["absolute_slot"])
            payload = channel(generator, rx=4, positions=306).transpose(0, 2, 1).astype("<c8").tobytes()
            self.assertEqual(len(payload), len(record.payload))
            fixed.append(obs.Record(dict(record.header, t_first_sample=first, t_last_sample=last), record.host, record.mask,
                                    payload))
        decisions = cusense.run_model(fixed, cusense.Model(artifact))[2]
        self.assertEqual([line["status"] for line in decisions], ["ok"] * 24)
        self.assertEqual([line["smoothed"]["maps"] for line in decisions[:12]], list(range(1, 11)) + [10, 10])
        self.assertTrue(all(line["input"]["shape"] == [1, 4, 306] for line in decisions))


class TreeTests(unittest.TestCase):
    FILES = ("src/ocudu_cuda_isac/sensing/cusense.py", "src/ocudu_cuda_isac/sensing/models/__init__.py",
             "src/ocudu_cuda_isac/sensing/models/cusense_cnn.py", "src/ocudu_cuda_isac/episodes/cusense_frontend.py",
             "scripts/train_cusense.py", "scripts/accept_s3_5.sh", "tests/unit/test_cusense_frontend.py",
             "tests/unit/cusense_frontend_double.py", "tests/unit/test_cusense_frontend_driver.py",
             "tests/unit/test_core_imports.py", "tests/integration/cusense_frontend.py",
             "configs/integration/cusense_frontend.json", "tests/test_accept_s3_5.py",
             "docs/interfaces/CUSENSE_FRONTEND.md", "Makefile")

    def test_files_are_ascii_and_do_not_name_the_project(self):
        # The working name lives on line 1 of the top-level README and nowhere else.
        name = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").split(":")[0].strip()
        for relative in self.FILES:
            data = (ROOT / relative).read_bytes()
            self.assertTrue(data.isascii(), relative)
            if name and name.lower() != "ocudu-cuda-isac":
                self.assertNotIn(name.lower(), data.decode("ascii").lower(), relative)

    def test_the_gate_and_the_configuration_name_the_demonstration(self):
        config = json.loads((ROOT / "configs/integration/cusense_frontend.json").read_text())
        self.assertEqual((config["schema"], config["selector"], config["backends"], config["adapter_build"]),
                         ("integration-config/cusense_frontend/v1", "cusense_frontend", ["cpu", "cuda"],
                          "build/ocudu-adapter-o3"))
        self.assertEqual(config["control"], {"driver": "presence_head"})
        # The rings are those of Sprint 5.2: one per stream, eight slots for a window of three and a burst of three.
        earlier = json.loads((ROOT / "configs/integration/native_estimate.json").read_text())
        for name in ("o2", "o3"):
            ring = config["rings"][name]
            self.assertEqual({key: ring[key] for key in earlier["rings"][name]}, earlier["rings"][name])
            self.assertEqual(ring["slots"], obs.slots_for(ring["window_records"], ring["burst_records"]))
        self.assertEqual(config["estimate"], {key: earlier["estimate"][key] for key in ("copy_stream", "landing", "timeout_us")})
        # The demonstration set of Sprint 3.4, with eight windows of one period per realization.
        demo = json.loads((ROOT / "configs/episodes/demo_set.json").read_text())
        counts = dict.fromkeys(("training", "validation", "held_out"), 0)
        for trajectory in demo["trajectories"]:
            counts[trajectory["group"]] += trajectory["episodes"]
            self.assertEqual(trajectory["periods"], config["set"]["expect"]["windows_per_realization"])
        self.assertEqual(config["set"]["expect"], {"set_id": demo["set_id"], "episodes": counts, "windows_per_realization": 8,
                                                   "grid": [demo["position_grid"]["rows"], demo["position_grid"]["cols"]]})
        previous = json.loads((ROOT / "configs/integration/presence_head.json").read_text())
        self.assertEqual((config["set"]["entries_per_run"], config["receiver"]),
                         (previous["set"]["entries_per_run"], previous["receiver"]))
        self.assertEqual((config["sensor"]["device"], config["selection"], sorted(config["arms"])),
                         ("cuda", {"window": "last"}, ["o2", "o3"]))
        self.assertEqual(config["expect"]["published"], {"mean_position_error_cm": 73, "live_inference_us": 241})
        self.assertIn("accept-s3.5:", (ROOT / "Makefile").read_text())
        runner = (ROOT / "scripts/accept_s3_5.sh").read_text()
        for needed in ("tests/unit/test_cusense_frontend.py", "tests/unit/test_core_imports.py",
                       "tests/integration/cusense_frontend.py --config configs/integration/cusense_frontend.json",
                       "-m ocudu_cuda_isac.sensing.cusense inspect", "-m ocudu_cuda_isac.sensing.cusense show",
                       "-m ocudu_cuda_isac.sensing.cusense device", "sha256sum", "cusense_eval.json",
                       "cusense_reference.json"):
            self.assertIn(needed, runner)
        if np is not None:
            self.assertEqual(config["arms"]["o3"]["spec"], cusense.demo_spec("o3_fp16"))
            self.assertEqual(config["arms"]["o2"]["spec"], cusense.demo_spec("o2_comb"))
            self.assertEqual(config["training"], cusense.demo_settings())
            self.assertEqual(config["tolerance"], evaluator.TOLERANCE)
            # A false-alarm target of 0.05 on the 320 target-absent validation windows allows sixteen.
            negatives = counts["validation"] * config["set"]["expect"]["windows_per_realization"]
            self.assertEqual(math.floor(config["training"]["threshold"]["target_false_alarm"] * negatives), 16)
            self.assertEqual(round(evaluator.PUBLISHED["position_error_cm"]["mean"]),
                             config["expect"]["published"]["mean_position_error_cm"])

    def test_this_sprint_adds_no_native_source(self):
        for directory in ("native/obs_ring", "native/ocudu_adapter", "native/sense_core", "patches"):
            for path in (ROOT / directory).rglob("*"):
                self.assertNotIn("cusense", path.name.lower(), path)
        # The core imports no evaluator, and every registry entry carries the reference runtime.
        for relative in ("src/ocudu_cuda_isac/sensing/cusense.py", "src/ocudu_cuda_isac/sensing/models/cusense_cnn.py"):
            source = (ROOT / relative).read_text()
            self.assertNotIn("episodes", "".join(line for line in source.splitlines() if line.startswith(("import", "from"))))
        self.assertIn("RUNTIME = features.RUNTIME", (ROOT / "src/ocudu_cuda_isac/sensing/cusense.py").read_text())
        # The model takes no batched product with broadcast operands: convolutions and plain linear layers only.
        model = (ROOT / "src/ocudu_cuda_isac/sensing/models/cusense_cnn.py").read_text()
        for word in ("torch.matmul", "torch.bmm", "einsum", " @ "):
            self.assertNotIn(word, model)


if __name__ == "__main__":
    unittest.main()
