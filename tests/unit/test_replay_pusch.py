#!/usr/bin/env python3
"""Sprint 2.4 unit gate (selector replay_pusch): python tests/unit/test_replay_pusch.py -v

Two parts. The Python part needs the standard library only: the outcome rule, the transport block
per opportunity, the replay plan and its bindings, and the reader and evaluator of the native
output, on a double of the native tool. The native part builds native/ocudu_adapter/receiver on
its own, with a C++17 compiler and CMake and no OCUDU source or build, runs its CTest cases, in
which doubles of a PUSCH processor return a valid CRC with a wrong byte, swap their completions,
raise, deliver twice and never call back, and compares the native outcome rule with the Python
one. No Sionna, no NumPy, no OCUDU build and no GPU.
"""

import contextlib
import copy
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes import schedule as schedules  # noqa: E402
from ocudu_cuda_isac.episodes.replay_pusch import ReplayPuschError  # noqa: E402
import replay_pusch_double as double  # noqa: E402
import test_grid_apply as apply_tests  # noqa: E402

RECEIVER = ROOT / "native/ocudu_adapter/receiver"
CASES = ("compare", "decoded", "valid_crc_wrong_byte", "crc_fail", "swapped_completions", "decoder_error",
         "duplicate_delivery", "silent_timeout", "owned_input", "changed_input", "guard", "accounting")
SEED = "unit-block"


class OutcomeTests(unittest.TestCase):
    """The rule that turns what a decoder did into an outcome, and the byte comparison."""

    def test_the_rule_has_one_row_per_outcome(self):
        self.assertEqual(rp.classify(False, False, True, True), "timeout")
        self.assertEqual(rp.classify(True, True, True, True), "decoder_error")
        self.assertEqual(rp.classify(True, False, True, True, guard_intact=False), "decoder_error")
        self.assertEqual(rp.classify(True, False, True, True, input_intact=False), "decoder_error")
        self.assertEqual(rp.classify(True, False, False, True), "crc_fail")
        self.assertEqual(rp.classify(True, False, True, False), "payload_mismatch")
        self.assertEqual(rp.classify(True, False, True, True), "decoded")
        # A CRC failure is never a decode, whatever the bytes are: nothing forces an acknowledgement.
        self.assertEqual(rp.classify(True, False, False, False), "crc_fail")
        for flags in itertools.product((False, True), repeat=6):
            self.assertIn(rp.classify(*flags), rp.OUTCOMES)
        self.assertEqual(set(rp.COMPLETED) | {"timeout", "not_dispatched"}, set(rp.OUTCOMES))

    def test_a_valid_crc_with_one_wrong_byte_fails_the_comparison(self):
        original = grants.generated_block("unit", 1217)
        self.assertEqual(rp.compare(original, original), {"equal": True, "first_differing_byte": -1,
                                                          "bytes_compared": 1217})
        for position in (0, 600, 1216):
            wrong = original[:position] + bytes([original[position] ^ 0x10]) + original[position + 1:]
            compared = rp.compare(wrong, original)
            self.assertEqual((compared["equal"], compared["first_differing_byte"]), (False, position))
            self.assertEqual(rp.classify(True, False, True, compared["equal"]), "payload_mismatch")
        # A shorter or an empty block is never equal, even when every compared byte is.
        self.assertEqual(rp.compare(original[:-1], original), {"equal": False, "first_differing_byte": -1,
                                                               "bytes_compared": 1216})
        self.assertFalse(rp.compare(b"", b"")["equal"])


def demo_schedule(periods=2):
    base = grants.load(ROOT / "configs/grants/demo_51prb_4rx.json")
    return schedules.build({"schema": schedules.SCHEMA, "schedule_id": "unit", "tdd": {"pattern": "DDDDDDSUUU",
                            "ul_slots": [7, 8, 9]}, "start_sfn": 0, "periods": periods, "payload": "fixed"}, base)


class PayloadTests(unittest.TestCase):
    """A transport block per opportunity, built where the decoded bytes are compared."""

    def test_a_block_per_opportunity_is_generated_from_the_seed_and_the_index(self):
        blocks = [rp.opportunity_block(SEED, index, 1217) for index in range(24)]
        self.assertEqual(len(set(blocks)), 24)
        self.assertTrue(all(len(block) == 1217 for block in blocks))
        self.assertEqual(blocks[3], grants.generated_block("unit-block/000003", 1217))
        self.assertEqual(blocks[3], rp.opportunity_block(SEED, 3, 1217))
        self.assertNotEqual(blocks[3], rp.opportunity_block("other", 3, 1217))
        self.assertEqual(rp.opportunity_block(SEED, 3, 48), blocks[3][:48])
        for seed, index in (("", 0), (" padded", 0), ("caf" + chr(233), 0), (SEED, -1), (SEED, 1.0), (None, 0)):
            with self.assertRaises(ReplayPuschError):
                rp.block_seed(seed, index)

    def test_a_payload_grant_is_the_slot_grant_with_its_own_block(self):
        schedule = demo_schedule()
        built = rp.payload_grants(schedule, SEED)
        self.assertEqual(len(built), len(schedule.opportunities), 6)
        self.assertEqual([grant.grant_id for grant in built[:4]],
                         ["demo-51prb-4rx.p000000", "demo-51prb-4rx.s08.p000001", "demo-51prb-4rx.s09.p000002",
                          "demo-51prb-4rx.s17.p000003"])
        for item, grant in zip(schedule.opportunities, built):
            self.assertEqual(grant.tb, rp.opportunity_block(SEED, item.index, 1217))
            self.assertNotEqual(grant.tb, schedule.base.tb)
            self.assertTrue(rp.same_allocation(grant, item.grant))
            # The allocation, the pilots and every coordinate are those of the slot grant.
            self.assertEqual((grant.slot, grant.pilot_index, grant.pilot_reference, grant.data_index, grant.tbs_bits),
                             (item.grant.slot, item.grant.pilot_index, item.grant.pilot_reference, item.grant.data_index,
                              item.grant.tbs_bits))
            self.assertNotEqual(schedules.record_sha256(grant.record()), schedules.record_sha256(item.grant.record()))
        self.assertFalse(rp.same_allocation(built[0], built[1]))  # another slot is another allocation
        with self.assertRaisesRegex(ReplayPuschError, "validated grant schedule"):
            rp.payload_grants("schedule", SEED)

    def test_the_payload_set_binds_every_grid_and_refuses_another_block(self):
        schedule = demo_schedule(periods=1)
        with tempfile.TemporaryDirectory() as temporary:
            root, stores = Path(temporary), {}
            for grant in rp.payload_grants(schedule, SEED):
                grants.export(grant, root / grant.grant_id, {"x": {"values": double.transmit_grid(grant),
                                                                  "provenance": {"double": True}}})
                stores[grant.grant_id] = grants.open_store(root / grant.grant_id)
            record = rp.payload_set(schedule, SEED, stores)
            self.assertEqual((record["schema"], record["rule"], record["generator"], record["seed"]),
                             (rp.PAYLOAD_SCHEMA, "per_opportunity", grants.TB_GENERATOR, SEED))
            self.assertEqual([block["index"] for block in record["blocks"]], [0, 1, 2])
            self.assertEqual(len({block["tb_sha256"] for block in record["blocks"]}), 3)
            first = record["blocks"][0]
            self.assertEqual(first["tb_sha256"], hashlib.sha256(rp.opportunity_block(SEED, 0, 1217)).hexdigest())
            self.assertEqual(first["x_sha256"], hashlib.sha256((root / first["grant_id"] / "x.npy").read_bytes()).hexdigest())
            self.assertEqual((first["slot_grant_id"], first["slot"], first["tb_seed"]), ("demo-51prb-4rx", 7, "unit-block/000000"))
            # The grid of another block has the pilots of its slot and other data symbols.
            slot_grant = schedule.opportunities[0].grant
            grants.export(slot_grant, root / "slot", {"x": {"values": double.transmit_grid(slot_grant), "provenance": {}}})
            same, moved = rp.pilots_equal(stores[first["grant_id"]], grants.open_store(root / "slot"))
            self.assertEqual(same, len(slot_grant.pilot_index))
            self.assertGreater(moved, 6000)
            with self.assertRaisesRegex(ReplayPuschError, "another record than the payload rule gives"):
                rp.payload_set(schedule, "another-seed", stores)
            missing = dict(stores)
            missing.pop(first["grant_id"])
            with self.assertRaisesRegex(ReplayPuschError, "no transmit grid was realized"):
                rp.payload_set(schedule, SEED, missing)


class StimulusTests(unittest.TestCase):
    """The plan on a small entry, the double of the native tool, the reader and the evaluator."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.entry = apply_tests.make_stimulus(cls.root)
        cls.channel = replay.entry_plan("unit-run", [cls.entry])
        replay.write_plan(cls.channel, cls.root / "channel-plan.json")
        cls.plan = rp.build_plan("unit-run", cls.root / "channel-plan.json")
        rp.write_plan(cls.plan, cls.root / "plan.json")
        cls.output = cls.root / "out"
        cls.output.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.status = double.main(["--plan", str(cls.root / "plan.json"), "--output-dir", str(cls.output)])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def scratch(self):
        directory = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def test_a_fixed_plan_binds_the_record_and_the_block_of_every_grant(self):
        plan = self.plan
        self.assertEqual(tuple(sorted(plan)), tuple(sorted(rp.PLAN_FIELDS)))
        self.assertEqual((plan["schema"], plan["access"]), ("replay-pusch-plan/v1", "offline_stimulus_and_evaluator"))
        self.assertEqual(plan["payload"], {"rule": "fixed", "generator": None, "seed": None, "blocks": 1})
        self.assertEqual(plan["receiver"], {"backends": ["cpu", "cuda"], "ldpc_iterations": 6, "early_stop": True,
                                            "timeout_ms": 30000, "tap": "none"})
        self.assertEqual(plan["channel_plan"]["sha256"], hashlib.sha256((self.root / "channel-plan.json").read_bytes()).hexdigest())
        self.assertEqual([grant["grant_id"] for grant in plan["grants"]], [grant["grant_id"] for grant in self.channel["grants"]])
        for grant, bound in zip(plan["grants"], self.channel["grants"]):
            store = grants.open_store(Path(bound["x"]["path"]).parent)
            # One hash binds the grid, the PDU and the block: the record with its transport block.
            self.assertEqual(grant["record_sha256"], bound["record_sha256"])
            self.assertEqual(grant["record_sha256"], schedules.record_sha256(store.grant.record()))
            self.assertEqual(Path(grant["record"]["path"]).name, "grant.json")
            self.assertEqual(Path(grant["tb"]["path"]).read_bytes(), store.grant.tb)
            self.assertEqual((grant["tb"]["sha256"], grant["tb_bytes"]), (hashlib.sha256(store.grant.tb).hexdigest(), 48))
        self.assertEqual(rp.check_files(plan)["opportunities"], self.channel["opportunities"])
        self.assertEqual(rp.load_plan(self.root / "plan.json"), plan)
        with self.assertRaisesRegex(ReplayPuschError, "never overwritten"):
            rp.write_plan(plan, self.root / "plan.json")

    def test_every_plan_rule_of_the_ticket(self):
        scratch = self.scratch()
        second = self.plan["grants"][1]

        def channel_change(change):
            def apply(plan):
                edited = copy.deepcopy(self.channel)
                change(edited)
                path = scratch / f"channel-{len(list(scratch.iterdir()))}.json"
                path.write_text(json.dumps(edited))
                plan["channel_plan"] = replay.file_ref(path)
            return apply

        def shorter_block(plan):
            block = Path(second["tb"]["path"]).read_bytes()[:-1]
            (scratch / "tb.bin").write_bytes(block)
            plan["grants"][1].update(tb=replay.file_ref(scratch / "tb.bin"), tb_bytes=len(block))

        def edited_record(plan):
            record = json.loads(Path(second["record"]["path"]).read_text())
            record["rnti"] += 1
            record["tb"] = {"hex": Path(second["tb"]["path"]).read_bytes().hex()}
            (scratch / "grant.json").write_text(json.dumps(record))
            plan["grants"][1]["record"] = replay.file_ref(scratch / "grant.json")

        rules = (
            (shorter_block, "mismatched payload: the transport block of grant .* holds 376 bits"),
            (lambda plan: plan["grants"][1]["tb"].update(sha256="0" * 64), "does not have the hash its record binds"),
            (lambda plan: plan["grants"][1]["tb"].update(sha256=None), "is bound by no hash"),
            (edited_record, "changed allocation: the record of grant .* with its transport block has the hash"),
            (lambda plan: plan["grants"].pop(1), "names no record and no transport block for the grant two-prb-offset.s08"),
            (lambda plan: plan["grants"].append(dict(plan["grants"][0], grant_id="another")),
             "exactly the grants of the channel plan"),
            (lambda plan: plan["grants"].append(dict(plan["grants"][0])), "is listed twice"),
            (channel_change(lambda channel: channel["opportunities"][1].update(slot=7)), "DM-RS depends on the slot index"),
            (channel_change(lambda channel: channel["opportunities"][1].update(
                opportunity_id=channel["opportunities"][0]["opportunity_id"])), "is listed twice"),
            (channel_change(lambda channel: channel.pop("gain")), "the channel plan is refused: .*names no grid gain"),
            (lambda plan: plan["channel_plan"].update(sha256="0" * 64), "channel plan does not have the hash"),
            (lambda plan: plan["receiver"].update(backends=["cpu", "fpga"]), "unknown backend 'fpga'"),
            (lambda plan: plan["receiver"].update(backends=["cuda", "cuda"]), "listed twice"),
            (lambda plan: plan["receiver"].update(backends=[]), "names no backend"),
            (lambda plan: plan["receiver"].update(tap="o2"), "decodes without an observer"),
            (lambda plan: plan["receiver"].update(timeout_ms=0), "timeout_ms must be"),
            (lambda plan: plan["receiver"].update(ldpc_iterations=0), "ldpc_iterations must be"),
            (lambda plan: plan["payload"].update(rule="random"), 'must be "fixed" or "per_opportunity"'),
            (lambda plan: plan["payload"].update(seed="unit"), "names its generator and its seed"),
            (lambda plan: plan["payload"].update(rule="per_opportunity", generator=grants.TB_GENERATOR, seed=SEED),
             "does not carry the block that the payload rule generates for index 0"),
            (lambda plan: plan["payload"].update(blocks=2), "declares 2 transport blocks and schedules 1"),
            (lambda plan: plan.update(access="offline_stimulus"), 'access must be "offline_stimulus_and_evaluator"'),
            (lambda plan: plan.update(labels={}), "requires exactly"),
        )
        for change, text in rules:
            plan = copy.deepcopy(self.plan)
            change(plan)
            with self.subTest(text=text), self.assertRaisesRegex(ReplayPuschError, text):
                rp.check_files(plan)

    def test_a_plan_per_opportunity_carries_the_generated_blocks(self):
        scratch = self.scratch()
        episode = episodes.open_episode(self.entry)
        stores = {}
        for grant in rp.payload_grants(episode.schedule, SEED):
            grants.export(grant, scratch / grant.grant_id / "store",
                          {"x": {"values": double.transmit_grid(grant), "provenance": {"double": True}}})
            stores[grant.grant_id] = grants.open_store(scratch / grant.grant_id / "store")
        payload = rp.payload_set(episode.schedule, SEED, stores)
        channel = rp.per_opportunity_channel_plan("unit-payload", [self.entry], payload)
        self.assertEqual(len(channel["grants"]), 3)
        self.assertEqual(len(channel["opportunities"]), 6)
        # Both members of the pair carry the same block at the same opportunity.
        by_index = {}
        for item in channel["opportunities"]:
            by_index.setdefault(item["index"], set()).add(item["grant_id"])
        self.assertEqual(by_index, {0: {"two-prb-offset.s07.p000000"}, 1: {"two-prb-offset.s08.p000001"},
                                    2: {"two-prb-offset.s09.p000002"}})
        self.assertEqual((channel["gain"], channel["sources"]), (self.channel["gain"], self.channel["sources"]))
        replay.write_plan(channel, scratch / "channel-plan.json")
        plan = rp.build_plan("unit-payload", scratch / "channel-plan.json", payload)
        self.assertEqual(plan["payload"], {"rule": "per_opportunity", "generator": grants.TB_GENERATOR, "seed": SEED,
                                           "blocks": 3})
        self.assertEqual(len({grant["tb"]["sha256"] for grant in plan["grants"]}), 3)
        rp.check_files(plan)
        # A plan whose seed is not the one its blocks were generated from is refused.
        other = copy.deepcopy(plan)
        other["payload"]["seed"] = "another-seed"
        with self.assertRaisesRegex(ReplayPuschError, "payload rule generates for index"):
            rp.check_files(other)
        # A payload set for another schedule position, or with another grid, is refused by the builder.
        damaged = copy.deepcopy(payload)
        damaged["blocks"][1]["x_sha256"] = "0" * 64
        with self.assertRaisesRegex(ReplayPuschError, "differs from the hashes the payload set binds"):
            rp.per_opportunity_channel_plan("unit-payload", [self.entry], damaged)
        damaged = copy.deepcopy(payload)
        damaged["blocks"][1]["slot_grant_id"] = damaged["blocks"][0]["slot_grant_id"]
        with self.assertRaisesRegex(ReplayPuschError, "was built for the grant"):
            rp.per_opportunity_channel_plan("unit-payload", [self.entry], damaged)
        with self.assertRaisesRegex(ReplayPuschError, "holds no block for opportunity 2"):
            rp.per_opportunity_channel_plan("unit-payload", [self.entry], dict(payload, blocks=payload["blocks"][:2]))
        # The double decodes every opportunity to its own block.
        rp.write_plan(plan, scratch / "plan.json")
        (scratch / "out").mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(double.main(["--plan", str(scratch / "plan.json"), "--output-dir", str(scratch / "out")]), 0)
        output = rp.open_output(scratch / "out")
        for record in output.records:
            index = int(record["opportunity_id"].rpartition(".")[2])
            self.assertEqual(output.decoded(record), rp.opportunity_block(SEED, index, 48))
        self.assertEqual(len({record["decoded_sha256"] for record in output.records}), 3)

    def test_the_double_decodes_the_pair_and_the_reader_verifies_it(self):
        self.assertEqual(self.status, 0)
        output = rp.open_output(self.output)
        self.assertEqual((output.run_id, len(output.records), len(output.descriptors), output.backends()),
                         ("unit-run", 12, 6, ["cpu", "cuda"]))
        self.assertEqual(len(output.by_key), 12)
        self.assertEqual([record["key"] for record in output.records[:2]],
                         [f"{self.channel['opportunities'][0]['opportunity_id']}/cpu",
                          f"{self.channel['opportunities'][0]['opportunity_id']}/cuda"])
        for record in output.records:
            self.assertEqual(tuple(sorted(record)), tuple(sorted(rp.RADIO_FIELDS)))
            self.assertEqual(rp.record_violations(record), [])
            self.assertEqual((record["outcome"], record["crc_ok"], record["payload_match"]), ("decoded", True, True))
            self.assertEqual(record["tb"], {"id": record["grant_id"], "bytes": 48, "bits": 384})
            self.assertEqual(record["allocation"]["rx_ports"], [0, 1])
            self.assertEqual((record["allocation"]["crb_start"], record["allocation"]["prb_count"]), (15, 2))
            self.assertEqual(record["path"]["executed"], "resident" if record["backend"] == "cuda" else "host")
            self.assertEqual((record["tap"], record["retry"]), ("none", 0))
        # Both backends were handed the same grid: one digest per opportunity, named by both records.
        for first, second in zip(output.records[::2], output.records[1::2]):
            self.assertEqual(first["ingress"]["readback_sha256"], second["ingress"]["readback_sha256"])
            self.assertEqual(first["opportunity_id"], second["opportunity_id"])
        digests = [descriptor["grid"]["sha256"] for descriptor in output.descriptors]
        self.assertEqual(digests, [record["ingress"]["readback_sha256"] for record in output.records[::2]])
        # The grid the receiver decoded is the one the channel application of Sprint 2.3 stores.
        stored = self.root / "stored"
        stored.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            apply_tests.double.main(["--plan", str(self.root / "channel-plan.json"), "--output-dir", str(stored)])
        self.assertEqual(digests, [item["grid"]["sha256"] for item in replay.open_output(stored).descriptors])
        # The receiver side of its input names no channel, no noise, no transport block and no label.
        secrets = [item["sha256"] for item in json.loads((self.entry / "manifest.json").read_text())["files"].values()]
        secrets += [grant["tb"]["sha256"] for grant in self.plan["grants"]] + ["tb.bin", "truth", "target_present"]
        self.assertEqual(replay.ingress_violations(self.output, secrets), [])

    def test_the_evaluator_joins_every_decode_to_its_original_block(self):
        output = rp.open_output(self.output)
        rows, totals = rp.evaluate(output, self.plan)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["consistent"] and row["payload_match"] and row["evaluated"] == "decoded" for row in rows))
        for backend in ("cpu", "cuda"):
            self.assertEqual({key: totals[backend][key] for key in ("planned", "decoded", "payload_matches", "crc_fail",
                                                                    "inconsistent", "matched_bytes")},
                             {"planned": 6, "decoded": 6, "payload_matches": 6, "crc_fail": 0, "inconsistent": 0,
                              "matched_bytes": 288})
        # A tool that claims a match for a block with one wrong byte is caught by the evaluator's own comparison.
        for fault, outcome, consistent in (("forged-match", "decoded", False), ("wrong-byte", "payload_mismatch", True)):
            out = self.scratch()
            with unittest.mock.patch.dict("os.environ", {"REPLAY_PUSCH_FAULT": fault}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(double.main(["--plan", str(self.root / "plan.json"), "--output-dir", str(out)]), 0)
            rows, totals = rp.evaluate(rp.open_output(out), self.plan)
            cuda = [row for row in rows if row["backend"] == "cuda"]
            self.assertTrue(all(row["outcome"] == outcome and row["evaluated"] == "payload_mismatch"
                                and row["first_differing_byte"] == 47 and row["consistent"] is consistent for row in cuda))
            self.assertEqual((totals["cuda"]["payload_matches"], totals["cpu"]["payload_matches"]), (0, 6))
            self.assertEqual(totals["cuda"]["inconsistent"], 0 if consistent else 6)
        # The original blocks come from the files the plan binds: a block that changed afterwards is refused.
        changed = copy.deepcopy(self.plan)
        changed["grants"][0]["tb"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ReplayPuschError, "changed after the plan was built"):
            rp.evaluate(output, changed)

    def test_the_reader_refuses_a_damaged_output(self):
        def damaged(change):
            copy_dir = self.scratch() / "out"
            shutil.copytree(self.output, copy_dir)
            change(copy_dir)
            return copy_dir

        def edit_record(position, **fields):
            def change(directory):
                path = directory / "radio/radio.jsonl"
                lines = path.read_text().splitlines()
                record = json.loads(lines[position])
                record.update(fields)
                lines[position] = json.dumps(record)
                path.write_text("\n".join(lines) + "\n")
            return change

        def flip_byte(directory):
            path = directory / "radio/0001.cuda.bin"
            data = bytearray(path.read_bytes())
            data[5] ^= 1
            path.write_bytes(bytes(data))

        def drop_line(directory):
            path = directory / "radio/radio.jsonl"
            path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")

        cases = (
            (flip_byte, "decoded bytes of radio record 3 differ from their hash"),
            (drop_line, "holds 11 radio records for 12 planned decodes"),
            (edit_record(3, key=json.loads((self.output / "radio/radio.jsonl").read_text().splitlines()[2])["key"]),
             "two radio records have the same key"),
            (edit_record(4, outcome="crc_fail"), "radio record 4: outcome crc_fail with crc_ok True"),
            (edit_record(4, payload_match=False), "radio record 4: outcome decoded with crc_ok True and payload_match False"),
            (edit_record(4, truth="present"), "does not hold exactly the fields of a radio/v1 record"),
            (edit_record(4, tap="o2"), "a tap or a retry on the untapped first transmission"),
            (edit_record(4, ingress={"grid_index": 0, "generation": 1, "readback_sha256": "0" * 64}),
             "does not name the grid of its opportunity"),
            (edit_record(4, sequence=7), "out of sequence"),
            (lambda directory: (directory / "radio/extra.bin").write_bytes(b"x"), "radio/ holds extra.bin"),
            (lambda directory: (directory / "ingress/0002.c64").write_bytes(b"\0" * 8), "differs from its readback digest"),
            (lambda directory: (directory / "native.json").unlink(), "not a complete native output"),
        )
        for change, text in cases:
            with self.subTest(text=text), self.assertRaisesRegex(ReplayPuschError, text):
                rp.open_output(damaged(change))
        # Faults of the tool that the reader sees on its own.
        for fault, text in (("drop-record", "radio records for 12 planned decodes"),
                            ("other-grid", "does not name the grid of its opportunity"),
                            ("leak", "not an o1-grid/v1 record")):
            out = self.scratch()
            with unittest.mock.patch.dict("os.environ", {"REPLAY_PUSCH_FAULT": fault}), \
                    contextlib.redirect_stdout(io.StringIO()):
                double.main(["--plan", str(self.root / "plan.json"), "--output-dir", str(out)])
            with self.subTest(fault=fault), self.assertRaisesRegex(ReplayPuschError, text):
                rp.open_output(out)

    def test_a_decode_without_a_completion_keeps_its_place_in_the_denominator(self):
        out = self.scratch()
        with unittest.mock.patch.dict("os.environ", {"REPLAY_PUSCH_FAULT": "timeout"}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(double.main(["--plan", str(self.root / "plan.json"), "--output-dir", str(out)]), 1)
        output = rp.open_output(out)
        rows, totals = rp.evaluate(output, self.plan)
        self.assertEqual(len(output.records), 12)
        self.assertEqual({key: totals["cuda"][key] for key in ("planned", "decoded", "timeout", "not_dispatched")},
                         {"planned": 6, "decoded": 1, "timeout": 1, "not_dispatched": 4})
        self.assertEqual(totals["cpu"]["decoded"], 6)
        skipped = [record for record in output.records if record["outcome"] == "not_dispatched"]
        self.assertTrue(all("did not complete" in record["error"] and record["decoded_file"] is None for record in skipped))
        self.assertTrue(all(row["consistent"] for row in rows))
        self.assertEqual(output.report["totals"]["grids_not_released"], 1)

    def test_the_double_refuses_what_the_native_tool_refuses(self):
        for change in (lambda plan: plan["receiver"].update(backends=["fpga"]),
                       lambda plan: plan["grants"][1]["tb"].update(sha256="0" * 64),
                       lambda plan: plan["channel_plan"].update(sha256="0" * 64)):
            plan = copy.deepcopy(self.plan)
            change(plan)
            scratch = self.scratch()
            (scratch / "plan.json").write_text(json.dumps(plan))
            (scratch / "out").mkdir()
            with contextlib.redirect_stderr(io.StringIO()) as said:
                status = double.main(["--plan", str(scratch / "plan.json"), "--output-dir", str(scratch / "out")])
            self.assertEqual(status, 2)
            self.assertIn("replay_pusch: input error:", said.getvalue())
            self.assertEqual(list((scratch / "out").iterdir()), [])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(double.main(["--plan", str(self.root / "plan.json"), "--output-dir", str(self.output)]), 2)
            self.assertEqual(double.main(["--plan", str(self.root / "plan.json"), "--output-dir", str(self.scratch()),
                                          "--only", "no-such-opportunity"]), 2)

    def test_one_opportunity_is_repeated_exactly(self):
        chosen = self.channel["opportunities"][4]["opportunity_id"]
        out = self.scratch()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(double.main(["--plan", str(self.root / "plan.json"), "--output-dir", str(out),
                                          "--only", chosen]), 0)
        again, first = rp.open_output(out), rp.open_output(self.output)
        self.assertEqual((len(again.records), again.report["selected"]), (2, [chosen]))
        for record in again.records:
            earlier = first.by_key[record["key"]]
            self.assertEqual(record["opportunity_sequence"], 4)
            for key in ("outcome", "decoded_sha256", "grant_record_sha256", "sfn", "slot", "decoded_file"):
                self.assertEqual(record[key], earlier[key])
            self.assertEqual(record["ingress"]["readback_sha256"], earlier["ingress"]["readback_sha256"])

    def test_command_line_builds_a_plan_and_reads_an_output_back(self):
        scratch = self.scratch()
        environment = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
        command = [sys.executable, "-m", "ocudu_cuda_isac.episodes.replay_pusch"]
        made = subprocess.run(command + ["plan", "--episode", str(self.entry), "--run-id", "unit-cli", "--backend", "cpu",
                                         "--output-dir", str(scratch)], capture_output=True, text=True, env=environment)
        self.assertEqual(made.returncode, 0, made.stderr)
        self.assertEqual((json.loads(made.stdout)["opportunities"], json.loads(made.stdout)["backends"]), (6, ["cpu"]))
        self.assertEqual(json.loads((scratch / "plan.json").read_text())["grants"], self.plan["grants"])
        again = subprocess.run(command + ["plan", "--episode", str(self.entry), "--run-id", "unit-cli", "--output-dir",
                                          str(scratch)], capture_output=True, text=True, env=environment)
        self.assertEqual(again.returncode, 2)
        self.assertIn("never overwritten", again.stderr)
        shown = subprocess.run(command + ["show", str(self.output), "--sequence", "3", "--backend", "cuda"],
                               capture_output=True, text=True, env=environment)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        record = json.loads(shown.stdout)
        self.assertEqual((record["verified_records"], record["outcome"], record["executed"], record["tb_bytes"]),
                         (12, "decoded", "resident", 48))
        self.assertEqual(record["decoded_sha256"], hashlib.sha256((self.output / "radio/0003.cuda.bin").read_bytes()).hexdigest())
        self.assertEqual(record["ingress_sha256"], hashlib.sha256((self.output / "ingress/0003.c64").read_bytes()).hexdigest())
        missing = subprocess.run(command + ["show", str(scratch)], capture_output=True, text=True, env=environment)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("not a complete native output", missing.stderr)


class NativeCoreTests(unittest.TestCase):
    """The host-free dispatch core, built on its own: its CTest cases and its agreement with the Python rule."""

    @classmethod
    def setUpClass(cls):
        cls.missing = [tool for tool in ("cmake", "ctest") if not shutil.which(tool)]
        if not any(shutil.which(compiler) for compiler in ("c++", "g++", "clang++")):
            cls.missing.append("a C++17 compiler")
        if cls.missing:
            return
        cls.temporary = tempfile.TemporaryDirectory()
        cls.build = Path(cls.temporary.name) / "build"
        steps = (["cmake", "-S", str(RECEIVER), "-B", str(cls.build)], ["cmake", "--build", str(cls.build)])
        cls.log, cls.status = "", 0
        for step in steps:
            result = subprocess.run(step, capture_output=True, text=True, timeout=900, check=False)
            cls.log += result.stdout + result.stderr
            cls.status = cls.status or result.returncode
        cls.test = cls.build / "dispatch_core_test"

    @classmethod
    def tearDownClass(cls):
        if not cls.missing:
            cls.temporary.cleanup()

    def setUp(self):
        if self.missing:
            self.skipTest("the native core needs " + ", ".join(self.missing))
        if self.status:
            self.fail("the native core does not build:\n" + self.log[-3000:])

    def test_core_builds_without_a_warning_and_links_only_its_modules(self):
        self.assertNotRegex(self.log, r"warning:")
        link = (self.build / "CMakeFiles/dispatch_core_test.dir/link.txt").read_text()
        self.assertEqual(re.findall(r"\S+\.(?:a|so)\b", link), ["libdispatch_core.a", "libreplay_core.a"])
        self.assertNotIn("ocudu", link.lower().replace(str(ROOT).lower(), ""))
        # The core and its unit target include the channel application core and the standard library.
        for name in ("dispatch_core.h", "dispatch_core.cpp", "tests/dispatch_core_test.cpp"):
            for include in re.findall(r'#include [<"]([^>"]+)[>"]', (RECEIVER / name).read_text()):
                self.assertTrue(include in ("dispatch_core.h", "replay_core.h") or "/" not in include, include)
        # The OCUDU-bound files are only built where the pinned build is imported.
        self.assertIn("if(TARGET ocudu_pin_fixture)", (RECEIVER / "CMakeLists.txt").read_text())
        self.assertNotIn("replay_pusch", "".join(path.name for path in self.build.iterdir()))

    def test_every_native_unit_case_passes(self):
        listed = subprocess.run([str(self.test), "--list"], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(tuple(listed), CASES)
        result = subprocess.run(["ctest", "--output-on-failure"], cwd=self.build, capture_output=True, text=True,
                                timeout=600, check=False)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn(f"100% tests passed, 0 tests failed out of {len(CASES)}", result.stdout)
        for name in CASES:
            self.assertRegex(result.stdout, rf"dispatch_core_{name} \.+\s+Passed")
        unknown = subprocess.run([str(self.test), "no_such_case"], capture_output=True, text=True, check=False)
        self.assertEqual(unknown.returncode, 2)

    def test_native_rule_and_python_rule_give_the_same_outcome(self):
        for flags in itertools.product((False, True), repeat=6):
            words = ["1" if flag else "0" for flag in flags]
            result = subprocess.run([str(self.test), "--classify", *words], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), rp.classify(*flags), words)
        refused = subprocess.run([str(self.test), "--classify", "1", "0", "2", "1", "1", "1"], capture_output=True,
                                 text=True, check=False)
        self.assertEqual(refused.returncode, 2)


class TreeTests(unittest.TestCase):
    FILES = ("native/ocudu_adapter/receiver/CMakeLists.txt", "native/ocudu_adapter/receiver/dispatch_core.h",
             "native/ocudu_adapter/receiver/dispatch_core.cpp", "native/ocudu_adapter/receiver/grant_pdu.h",
             "native/ocudu_adapter/receiver/receiver_plan.h", "native/ocudu_adapter/receiver/replay_pusch.cpp",
             "native/ocudu_adapter/receiver/tests/dispatch_core_test.cpp", "src/ocudu_cuda_isac/episodes/replay_pusch.py",
             "tests/unit/test_replay_pusch.py", "tests/unit/replay_pusch_double.py",
             "tests/unit/test_replay_pusch_driver.py", "tests/integration/replay_pusch.py",
             "configs/integration/replay_pusch.json", "scripts/accept_s2_4.sh", "scripts/accept_s2.sh",
             "scripts/verify_sprint2.py", "tests/test_accept_s2_4.py", "tests/test_accept_s2.py",
             "tests/test_verify_sprint2.py", "docs/interfaces/REPLAY_PUSCH.md")

    def test_files_are_ascii_and_do_not_name_the_project(self):
        # The working name lives on line 1 of the top-level README and nowhere else.
        name = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").split(":")[0].strip()
        for relative in self.FILES:
            data = (ROOT / relative).read_bytes()
            self.assertTrue(data.isascii(), relative)
            if name and name.lower() != "ocudu-cuda-isac":
                self.assertNotIn(name.lower(), data.decode("ascii").lower(), relative)

    def test_the_tool_hands_one_owned_grid_to_both_pinned_processors(self):
        tool = (RECEIVER / "replay_pusch.cpp").read_text()
        # The grid comes from the channel application of Sprint 2.3 and is read through the pinned reader.
        self.assertIn("replay.deliver(sequence, pool, key)", tool)
        self.assertIn("grids.at(job.grid.grid)->ocudu_grid().get_reader()", tool)
        self.assertIn("ocudu::MAX_NSYMB_PER_SLOT, ocudu::MAX_NOF_SUBCARRIERS", tool)  # sized like the pinned test's rx_grid
        # Both processors are created as the pinned test creates them, with the settings Sprint 1 measured.
        self.assertIn("ocudu::create_sw_pusch_processor_factory(executor,", tool)
        self.assertIn('name == "cuda" ? "gpu" : "auto"', tool)
        self.assertIn("ocudu::port_channel_estimator_td_interpolation_strategy::average", tool)
        self.assertIn("ocudu::channel_equalizer_algorithm_type::mmse", tool)
        # Everything is checked before a processor exists or a file is written.
        self.assertLess(tool.index("const session     replay(spec);"), tool.index("make_directory(ingress_path)"))
        self.assertLess(tool.index("record_sha256(entry.record, block)"), tool.index("make_directory(ingress_path)"))
        self.assertLess(tool.index("make_pdu(grant.record, grant.coding, item.sfn, item.slot)"),
                        tool.index("std::make_unique<processor_backend>"))
        # The CRC flag is the decoder's own, and the resident flag is the pinned decoder's.
        self.assertIn("sink.complete(token, sch_.data.tb_crc_ok, std::string());", tool)
        self.assertIn('timing.valid ? "resident" : "host"', tool)
        for word in ("truth", "target_present", "position_m"):
            self.assertNotIn(word, tool)
        # A worker is still inside its task after the decoder's callback: on every way out of the run
        # and of the self-test the workers are joined before a processor is destroyed.
        self.assertIn("~joined_workers() { pool.pool->stop(); }", tool)
        self.assertEqual(tool.count("const joined_workers "), 3)
        self.assertLess(tool.index("std::make_unique<processor_backend>"),
                        tool.rindex("const joined_workers join(pool_of_workers);"))
        # The payload grids are made without the loopback of the fixture adapter, which does not join its workers.
        driver = (ROOT / "tests/integration/replay_pusch.py").read_text()
        self.assertIn('"--dmrs-reference", "crb0", "--no-loopback"]', driver)
        self.assertNotIn("fixture.run_adapter(", driver)
        # The PDU is the grant's with the PUSCH DM-RS: no reference point of the pinned test's transmit side.
        pdu = (RECEIVER / "grant_pdu.h").read_text()
        self.assertNotIn("PRB0", pdu)
        self.assertIn("ocudu::slot_point(ocudu::to_numerology_value(spacing_of(grant)), sfn, slot)", pdu)
        self.assertIn('canonical["tb"]          = {{"hex", hex}};', pdu)

    def test_the_adapter_build_and_the_configuration_name_the_tool(self):
        self.assertIn("add_subdirectory(receiver)", (ROOT / "native/ocudu_adapter/CMakeLists.txt").read_text())
        cmake = (RECEIVER / "CMakeLists.txt").read_text()
        self.assertEqual(tuple(re.search(r"set\(DISPATCH_CORE_CASES ([^)]*)\)", cmake)[1].split()), CASES)
        self.assertIn("add_test(NAME replay_pusch_host_decode COMMAND replay_pusch --self-test)", cmake)
        self.assertIn('"replay_pusch": Path("receiver/replay_pusch")', (ROOT / "scripts/native_adapter.py").read_text())
        config = json.loads((ROOT / "configs/integration/replay_pusch.json").read_text())
        self.assertEqual((config["schema"], config["selector"], config["backends"]),
                         ("integration-config/replay_pusch/v1", "replay_pusch", ["cpu", "cuda"]))
        self.assertEqual(config["receiver"], {"ldpc_iterations": 6, "early_stop": True, "timeout_ms": 30000})
        self.assertEqual(tuple(config["tolerance"]), ("epre_db", "sinr_db", "storage_sqnr_db_min"))
        # The original bytes of the three shipped grants, as Sprint 2.1 accepted them.
        self.assertEqual(config["unity"]["expect"]["bytes"], {"demo-51prb-4rx": 1217, "table2-51prb-1rx": 6787,
                                                              "two-prb-offset": 48})
        self.assertEqual(config["pair"]["expect"]["tb_sha256"],
                         hashlib.sha256(grants.load(ROOT / "configs/grants/demo_51prb_4rx.json").tb).hexdigest())


if __name__ == "__main__":
    import unittest.mock  # noqa: F401
    unittest.main()
