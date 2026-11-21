"""Receiver replay of an episode: the payload, the plan and the radio records (Sprint 2.4, selector replay_pusch).

The native harness (native/ocudu_adapter/receiver) fills the receiver grid of every opportunity
with the channel application of Sprint 2.3, hands that grid and the PDU of the opportunity's grant
to the host and to the CUDA PUSCH processor of the pinned source, and compares the decoded bytes
with the original transport block. This module is the offline side of that step. It

  * builds a transport block per opportunity. Sprint 2.2 gave every opportunity the block of the
    base grant (payload "fixed"). Under "per_opportunity" the block of opportunity i is generated
    from a seed and i, and its grant is the grant of the opportunity's slot with that block. Both
    cases of a pair carry the same block at the same opportunity, so the waveform stays fixed;
  * writes the replay-pusch-plan/v1 file the native tool reads: the channel plan of Sprint 2.3 by
    the hash that binds it, and per grant the record its PDU is built from and the original block;
  * states the rule that turns what a decoder did into an outcome, as the native core applies it;
  * reads the native output back: the radio/v1 records, the decoded bytes and the grid
    descriptors, and joins every decode to its original block, as an evaluator does after a run.

Standard library only. A plan is an offline stimulus and evaluator record: it names transport
blocks. The receiver side of a run reads none of it.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import grant as grants
from . import grid_apply as replay
from . import schedule as schedules
from .episode import EpisodeError
from .grant import GrantError
from .grid_apply import GridApplyError
from ..replay import ContractError, load_json

PLAN_SCHEMA = "replay-pusch-plan/v1"
NATIVE_SCHEMA = "replay-pusch-native/v1"
RADIO_SCHEMA = "radio/v1"
PAYLOAD_SCHEMA = "pusch-payload-set/v1"
ACCESS = "offline_stimulus_and_evaluator"
BACKENDS = ("cpu", "cuda")
PAYLOAD_RULES = ("fixed", "per_opportunity")
# Every decode of a plan ends in exactly one of these.
OUTCOMES = ("decoded", "payload_mismatch", "crc_fail", "decoder_error", "timeout", "not_dispatched")
# Outcomes with a true decoder callback: the denominators of the radio KPIs count all of OUTCOMES.
COMPLETED = ("decoded", "payload_mismatch", "crc_fail", "decoder_error")
# The settings of the pinned test, which the Sprint 1 gates measured.
RECEIVER = {"ldpc_iterations": 6, "early_stop": True, "timeout_ms": 30000, "tap": "none"}
# The resident CUDA path of the pinned processor takes these numbers of receive ports.
RESIDENT_PORTS = (1, 2, 4, 8)
PLAN_FIELDS = ("schema", "run_id", "access", "channel_plan", "payload", "receiver", "grants")
PAYLOAD_FIELDS = ("rule", "generator", "seed", "blocks")
RECEIVER_FIELDS = ("backends", "ldpc_iterations", "early_stop", "timeout_ms", "tap")
GRANT_FIELDS = ("grant_id", "record_sha256", "record", "tb", "tb_bytes")
# Everything a radio record holds. A field outside this list fails the reader.
RADIO_FIELDS = ("schema", "run_id", "sequence", "key", "opportunity_id", "opportunity_sequence", "backend", "sfn",
                "slot", "grant_id", "grant_record_sha256", "allocation", "ingress", "tb", "outcome", "crc_ok",
                "payload_match", "decoded_sha256", "decoded_file", "error", "path", "decoder", "timing", "retry", "tap")
ALLOCATION_FIELDS = ("crb_start", "prb_count", "symbol_start", "symbol_count", "dmrs_symbols", "rx_ports", "layers",
                     "active_re", "active_sha256")


class ReplayPuschError(ValueError):
    """A payload, a plan or a native output violates the documented interface."""


# ---------------------------------------------------------------- outcome
def classify(completed, error, crc_ok, bytes_equal, guard_intact=True, input_intact=True):
    """The outcome of one decode, by the rule of the native core (dispatch_core.h).

    No completion is a timeout. An error, a damaged guard behind the decoded bytes or an input
    grid that changed is a decoder error. Then the decoder's own CRC flag decides, and a CRC pass
    counts as decoded only when every byte equals the original transport block.
    """
    if not completed:
        return "timeout"
    if error or not guard_intact or not input_intact:
        return "decoder_error"
    if not crc_ok:
        return "crc_fail"
    return "decoded" if bytes_equal else "payload_mismatch"


def compare(decoded, original):
    """Decoded bytes against the original ones. Blocks of different lengths are never equal."""
    compared = min(len(decoded), len(original))
    first = next((index for index in range(compared) if decoded[index] != original[index]), -1)
    return {"equal": len(decoded) == len(original) and len(original) != 0 and first < 0,
            "first_differing_byte": first, "bytes_compared": compared}


# ---------------------------------------------------------------- payload
def block_seed(seed, index):
    """Seed of the transport block of opportunity index. The generator is grant.TB_GENERATOR."""
    if not isinstance(seed, str) or not seed or not seed.isascii() or seed.strip() != seed:
        raise ReplayPuschError("a payload seed is a nonempty ASCII string")
    if type(index) is not int or not 0 <= index < schedules.MAX_OPPORTUNITIES:
        raise ReplayPuschError("an opportunity index is a non-negative integer")
    return f"{seed}/{index:06d}"


def opportunity_block(seed, index, size):
    """The transport block of one opportunity: size bytes from the seed and the opportunity index."""
    try:
        return grants.generated_block(block_seed(seed, index), size)
    except GrantError as error:
        raise ReplayPuschError(str(error)) from None


def payload_grant(slot_grant, index, seed):
    """The grant of one opportunity under the per-opportunity rule.

    It is the grant of the opportunity's slot with the block of the opportunity and the identifier
    "<grant_id>.pNNNNNN". The allocation, the pilots and every coordinate are those of the slot grant.
    """
    if not isinstance(slot_grant, grants.Grant):
        raise ReplayPuschError("a payload grant is built on a validated grant")
    record = slot_grant.record()
    record["grant_id"] = f"{slot_grant.grant_id}.p{index:06d}"
    record["tb"] = {"generator": grants.TB_GENERATOR, "seed": block_seed(seed, index), "bytes": len(slot_grant.tb)}
    try:
        return grants.build(record)
    except GrantError as error:
        raise ReplayPuschError(f"the payload grant of opportunity {index} is not legal: {error}") from None


def payload_grants(schedule, seed):
    """One grant per opportunity of a schedule, in opportunity order, each with its own block."""
    if not isinstance(schedule, schedules.Schedule):
        raise ReplayPuschError("a payload is built on a validated grant schedule")
    built = [payload_grant(item.grant, item.index, seed) for item in schedule.opportunities]
    if len({grant.tb for grant in built}) != len(built):
        raise ReplayPuschError("two opportunities received the same transport block")
    return built


def same_allocation(grant, slot_grant):
    """True when two grant records differ in nothing but the identifier and the transport block."""
    first, second = grant.record(), slot_grant.record()
    for record in (first, second):
        record.pop("grant_id")
        record.pop("tb")
    return first == second


def payload_set(schedule, seed, stores):
    """The pusch-payload-set/v1 record of a schedule: what binds every per-opportunity transmit grid.

    stores maps the identifier of each payload grant to its verified store entry, which holds the
    transmit grid the fixture adapter realized. Every entry is checked against the rule: the
    record is the slot grant's apart from the identifier and the block, the block is the generated
    one, and the grid carries the pilots of the slot grant.
    """
    blocks = []
    for item, grant in zip(schedule.opportunities, payload_grants(schedule, seed)):
        store = stores.get(grant.grant_id)
        if store is None or store.x is None:
            raise ReplayPuschError(f"no transmit grid was realized for the payload grant {grant.grant_id}")
        if store.grant.record() != grant.record() or not same_allocation(store.grant, item.grant):
            raise ReplayPuschError(f"the store entry of {grant.grant_id} holds another record than the payload rule gives")
        block = opportunity_block(seed, item.index, len(item.grant.tb))
        if store.grant.tb != block:
            raise ReplayPuschError(f"the store entry of {grant.grant_id} holds another block than the generated one")
        files = store.manifest["files"]
        blocks.append({"index": item.index, "slot": item.slot, "slot_grant_id": item.grant.grant_id,
                       "grant_id": grant.grant_id, "record_sha256": schedules.record_sha256(grant.record()),
                       "tb_seed": block_seed(seed, item.index), "tb_bytes": len(block),
                       "tb_sha256": hashlib.sha256(block).hexdigest(), "x_sha256": files["x.npy"]["sha256"],
                       "store": str(Path(store.directory).resolve())})
    return {"schema": PAYLOAD_SCHEMA, "rule": "per_opportunity", "generator": grants.TB_GENERATOR, "seed": seed,
            "schedule_id": schedule.schedule_id, "base_grant_id": schedule.base.grant_id,
            "base_record_sha256": schedules.record_sha256(schedule.base.record()), "blocks": blocks,
            "binding": "each transmit grid is the fixture adapter's grid of its slot grant with the generated block; "
                       "both cases of a pair carry the same block at the same opportunity"}


def pilots_equal(store, slot_store):
    """Pilots of two transmit grids that are bit for bit the same, and data symbols that differ."""
    width = store.grant.grid_subcarriers
    same = sum(store.x[symbol * width + k] == slot_store.x[symbol * width + k] for symbol, k in store.grant.pilot_index)
    moved = sum(store.x[symbol * width + k] != slot_store.x[symbol * width + k] for symbol, k in store.grant.data_index)
    return same, moved


# ---------------------------------------------------------------- channel plans
def per_opportunity_channel_plan(run_id, entries, payload, stores=(), cases=("a", "b")):
    """The channel plan of episode entries with a transport block per opportunity.

    It starts from the plan of Sprint 2.3, which verifies every entry and the transmit grid the
    entry binds to each slot grant. Then every opportunity takes the grant of its index from the
    payload set, whose grid has the hash the set records. The gain, the channel and the noise are
    those of the entry. payload is a pusch-payload-set/v1 record.
    """
    fixed = replay.entry_plan(run_id, entries, stores, cases)
    if not isinstance(payload, dict) or payload.get("schema") != PAYLOAD_SCHEMA \
            or not isinstance(payload.get("blocks"), list) or not payload["blocks"]:
        raise ReplayPuschError(f"a per-opportunity plan needs a {PAYLOAD_SCHEMA} record")
    by_index = {block["index"]: block for block in payload["blocks"]}
    slot_records = {grant["grant_id"]: grant["record_sha256"] for grant in fixed["grants"]}
    bound = {}
    for opportunity in fixed["opportunities"]:
        block = by_index.get(opportunity["index"])
        if block is None:
            raise ReplayPuschError(f"the payload set holds no block for opportunity {opportunity['index']}")
        if block["slot_grant_id"] != opportunity["grant_id"] or block["slot_grant_id"] not in slot_records:
            raise ReplayPuschError(f"the block of opportunity {opportunity['index']} was built for the grant "
                                   f"{block['slot_grant_id']}, and the entry schedules {opportunity['grant_id']}")
        if block["grant_id"] not in bound:
            try:
                store = grants.open_store(block["store"])
            except GrantError as error:
                raise ReplayPuschError(f"the store entry of {block['grant_id']} is unusable: {error}") from None
            item = replay.grant_from_store(store)
            if item["x"]["sha256"] != block["x_sha256"] or item["record_sha256"] != block["record_sha256"]:
                raise ReplayPuschError(f"the store entry of {block['grant_id']} differs from the hashes the payload "
                                       "set binds")
            bound[block["grant_id"]] = item
        opportunity["grant_id"] = block["grant_id"]
    fixed["grants"] = [bound[grant_id] for grant_id in sorted(bound)]
    return replay.check_plan(fixed)


# ---------------------------------------------------------------- plan
def _keys(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ReplayPuschError(f"{name} requires exactly {', '.join(fields)}")
    return value


def _bound(value, name):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"} or not isinstance(value["path"], str) \
            or not value["path"]:
        raise ReplayPuschError(f"{name} requires path and sha256")
    digest = value["sha256"]
    if digest is None:
        raise ReplayPuschError(f"{name} is bound by no hash")
    if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        raise ReplayPuschError(f"{name}.sha256 must be 64 hexadecimal digits")
    return value


def check_receiver(receiver):
    """The receiver settings of a plan: the backends in dispatch order and the decoder limits."""
    _keys(receiver, RECEIVER_FIELDS, "receiver")
    backends = receiver["backends"]
    if not isinstance(backends, list) or not backends:
        raise ReplayPuschError("receiver.backends names no backend")
    for name in backends:
        if name not in BACKENDS:
            raise ReplayPuschError(f"unknown backend {name!r}: a backend is \"cpu\" or \"cuda\"")
    if len(set(backends)) != len(backends):
        raise ReplayPuschError("a backend is listed twice")
    if type(receiver["ldpc_iterations"]) is not int or not 1 <= receiver["ldpc_iterations"] <= 64:
        raise ReplayPuschError("receiver.ldpc_iterations must be an integer from 1 to 64")
    if type(receiver["timeout_ms"]) is not int or not 1 <= receiver["timeout_ms"] <= 600000:
        raise ReplayPuschError("receiver.timeout_ms must be an integer from 1 to 600000")
    if receiver["early_stop"] not in (True, False):
        raise ReplayPuschError("receiver.early_stop must be true or false")
    if receiver["tap"] != "none":
        raise ReplayPuschError('receiver.tap must be "none": this harness decodes without an observer')
    return receiver


def check_plan(plan):
    """Apply every rule of a plan that needs no file: the fields, the payload rule and the receiver."""
    _keys(plan, PLAN_FIELDS, "a plan")
    if plan["schema"] != PLAN_SCHEMA:
        raise ReplayPuschError(f"unsupported plan schema {plan['schema']!r}")
    if not isinstance(plan["run_id"], str) or not plan["run_id"] or not plan["run_id"].isascii():
        raise ReplayPuschError("run_id must be a nonempty ASCII string")
    if plan["access"] != ACCESS:
        raise ReplayPuschError(f'access must be "{ACCESS}"')
    _bound(plan["channel_plan"], "the channel plan")
    payload = _keys(plan["payload"], PAYLOAD_FIELDS, "payload")
    if payload["rule"] not in PAYLOAD_RULES:
        raise ReplayPuschError('payload.rule must be "fixed" or "per_opportunity"')
    generated = payload["rule"] == "per_opportunity"
    if generated != (payload["generator"] is not None) or generated != (payload["seed"] is not None):
        raise ReplayPuschError("a payload per opportunity names its generator and its seed, and a fixed payload "
                               "names neither")
    if generated and payload["generator"] != grants.TB_GENERATOR:
        raise ReplayPuschError(f"unknown transport block generator {payload['generator']!r}")
    if generated:
        block_seed(payload["seed"], 0)
    if type(payload["blocks"]) is not int or payload["blocks"] < 1:
        raise ReplayPuschError("payload.blocks must be a positive integer")
    check_receiver(plan["receiver"])
    if not isinstance(plan["grants"], list) or not plan["grants"]:
        raise ReplayPuschError("a plan needs at least one grant")
    seen = set()
    for grant in plan["grants"]:
        _keys(grant, GRANT_FIELDS, "a grant")
        name = grant["grant_id"]
        if not isinstance(name, str) or not name or name in seen:
            raise ReplayPuschError(f"the grant {name!r} is listed twice or has no identifier")
        seen.add(name)
        if not isinstance(grant["record_sha256"], str) or not grant["record_sha256"]:
            raise ReplayPuschError(f"the grant {name} names no record hash")
        _bound(grant["record"], f"the record of grant {name}")
        _bound(grant["tb"], f"the transport block of grant {name}")
        if type(grant["tb_bytes"]) is not int or grant["tb_bytes"] < 1:
            raise ReplayPuschError(f"tb_bytes of grant {name} must be a positive integer")
    return plan


def _file_sha256(path, name):
    try:
        return replay.sha256_file(path)
    except GridApplyError as error:
        raise ReplayPuschError(f"{name}: {error}") from None


def check_files(plan):
    """Verify everything a plan binds and return the channel plan it names.

    The channel plan must have its hash and pass the file check of Sprint 2.3. Every grant of the
    channel plan needs one entry here, whose record with its transport block has the hash the
    channel plan carries for the grid. So one hash binds the grid, the PDU and the block. The
    block has the size the grant carries, every opportunity lies in the slot its grant was built
    for, and under the per-opportunity rule every block is the generated one of its opportunity.
    """
    check_plan(plan)
    reference = plan["channel_plan"]
    if _file_sha256(reference["path"], "the channel plan") != reference["sha256"]:
        raise ReplayPuschError(f"the channel plan does not have the hash its record binds ({reference['path']})")
    try:
        channel = replay.check_files(replay.load_plan(reference["path"]))
    except GridApplyError as error:
        raise ReplayPuschError(f"the channel plan is refused: {error}") from None
    listed = {grant["grant_id"]: grant for grant in plan["grants"]}
    for bound in channel["grants"]:
        if bound["grant_id"] not in listed:
            raise ReplayPuschError(f"the plan names no record and no transport block for the grant {bound['grant_id']}")
    if len(listed) != len(channel["grants"]):
        raise ReplayPuschError("the plan must name a record and a transport block for exactly the grants of the "
                               "channel plan")
    records, blocks = {}, {}
    for bound in channel["grants"]:
        name, item = bound["grant_id"], listed[bound["grant_id"]]
        if _file_sha256(item["record"]["path"], f"the record of grant {name}") != item["record"]["sha256"]:
            raise ReplayPuschError(f"the record of grant {name} does not have the hash its record binds "
                                   f"({item['record']['path']})")
        try:
            block = Path(item["tb"]["path"]).read_bytes()
            grant = grants.load(item["record"]["path"])
        except (OSError, GrantError) as error:
            raise ReplayPuschError(f"the grant {name} is unusable: {error}") from None
        if hashlib.sha256(block).hexdigest() != item["tb"]["sha256"]:
            raise ReplayPuschError(f"the transport block of grant {name} does not have the hash its record binds "
                                   f"({item['tb']['path']})")
        if len(block) != item["tb_bytes"] or 8 * len(block) != grant.tbs_bits or block != grant.tb:
            raise ReplayPuschError(f"mismatched payload: the transport block of grant {name} holds {8 * len(block)} "
                                   f"bits, the plan names {8 * item['tb_bytes']} and the grant carries {grant.tbs_bits}")
        canonical = schedules.record_sha256(grant.record())
        if grant.grant_id != name or canonical != bound["record_sha256"] or item["record_sha256"] != canonical:
            raise ReplayPuschError(f"changed allocation: the record of grant {name} with its transport block has the "
                                   f"hash {canonical}, and the channel plan binds the grid of this grant to "
                                   f"{bound['record_sha256']}")
        carrier = channel["carrier"]
        if len(grant.rx_ports) != carrier["rx_ports"] or grant.antenna_ports != carrier["tx_ports"] \
                or grant.grid_subcarriers != carrier["subcarriers"]:
            raise ReplayPuschError(f"changed allocation: the grant {name} has {len(grant.rx_ports)} receive ports on "
                                   f"{grant.grid_subcarriers} subcarriers, and the carrier grid of the run has "
                                   f"{carrier['rx_ports']} on {carrier['subcarriers']}")
        records[name], blocks[name] = grant, block
    payload, used = plan["payload"], set()
    for opportunity in channel["opportunities"]:
        grant = records[opportunity["grant_id"]]
        if opportunity["slot"] != grant.slot:
            raise ReplayPuschError(f"the opportunity {opportunity['opportunity_id']} lies in slot {opportunity['slot']}, "
                                   f"and its grant {grant.grant_id} was built for slot {grant.slot}: the DM-RS depends "
                                   "on the slot index")
        block = blocks[opportunity["grant_id"]]
        if payload["rule"] == "per_opportunity" \
                and block != opportunity_block(payload["seed"], opportunity["index"], len(block)):
            raise ReplayPuschError(f"the opportunity {opportunity['opportunity_id']} does not carry the block that the "
                                   f"payload rule generates for index {opportunity['index']}")
        used.add(hashlib.sha256(block).hexdigest())
    if len(used) != payload["blocks"]:
        raise ReplayPuschError(f"the plan declares {payload['blocks']} transport blocks and schedules {len(used)}")
    return channel


def build_plan(run_id, channel_plan_path, payload=None, backends=BACKENDS, receiver=None):
    """A replay plan around a channel plan file that was written before.

    Every grant of the channel plan must come from a grant store: the directory of its transmit
    grid holds grant.json and tb.bin, and their hashes are taken from the verified store entry.
    payload is None for the fixed rule, or a pusch-payload-set/v1 record.
    """
    path = Path(channel_plan_path).resolve()
    try:
        channel = replay.load_plan(path)
    except GridApplyError as error:
        raise ReplayPuschError(str(error)) from None
    listed, blocks = [], set()
    for bound in channel["grants"]:
        directory = Path(bound["x"]["path"]).parent
        try:
            store = grants.open_store(directory)
        except GrantError as error:
            raise ReplayPuschError(f"the grant {bound['grant_id']} has no usable store entry at {directory}: "
                                   f"{error}") from None
        files = store.manifest["files"]
        if files.get("x.npy", {}).get("sha256") != bound["x"]["sha256"]:
            raise ReplayPuschError(f"the store entry of grant {bound['grant_id']} holds another transmit grid than "
                                   "the channel plan binds")
        listed.append({"grant_id": bound["grant_id"], "record_sha256": bound["record_sha256"],
                       "record": {"path": str(directory.resolve() / "grant.json"), "sha256": files["grant.json"]["sha256"]},
                       "tb": {"path": str(directory.resolve() / "tb.bin"), "sha256": files["tb.bin"]["sha256"]},
                       "tb_bytes": files["tb.bin"]["bytes"]})
    used = {grant["grant_id"] for grant in listed} & {item["grant_id"] for item in channel["opportunities"]}
    blocks = {grant["tb"]["sha256"] for grant in listed if grant["grant_id"] in used}
    settings = dict(RECEIVER, backends=list(backends))
    settings.update(receiver or {})
    if payload is None:
        rule = {"rule": "fixed", "generator": None, "seed": None, "blocks": len(blocks)}
    else:
        if not isinstance(payload, dict) or payload.get("schema") != PAYLOAD_SCHEMA:
            raise ReplayPuschError(f"a payload is None for the fixed rule or a {PAYLOAD_SCHEMA} record")
        rule = {"rule": "per_opportunity", "generator": payload["generator"], "seed": payload["seed"],
                "blocks": len(blocks)}
    plan = {"schema": PLAN_SCHEMA, "run_id": run_id, "access": ACCESS,
            "channel_plan": {"path": str(path), "sha256": replay.sha256_file(path)}, "payload": rule,
            "receiver": settings, "grants": listed}
    check_files(plan)
    return plan


def write_plan(plan, path):
    """Check a plan and write it. The file must not exist."""
    path = Path(path)
    if path.exists():
        raise ReplayPuschError(f"{path} already exists. A plan is never overwritten.")
    check_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return path


def load_plan(path):
    try:
        return check_plan(load_json(Path(path)))
    except (ContractError, OSError) as error:
        raise ReplayPuschError(f"cannot read the plan: {error}") from None


# ---------------------------------------------------------------- native output
class Output:
    """A verified native output: the report, the radio records and the grid descriptors."""

    def __init__(self, directory, report, records, descriptors):
        self.directory, self.report, self.records, self.descriptors = directory, report, records, descriptors
        self.run_id = report["run_id"]
        self.by_key = {record["key"]: record for record in records}

    def decoded(self, record):
        """The decoded bytes of a record whose CRC passed, or None."""
        if record["decoded_file"] is None:
            return None
        return (self.directory / "radio" / record["decoded_file"]).read_bytes()

    def backends(self):
        return [backend["name"] for backend in self.report["backends"]]

    def of(self, backend):
        return [record for record in self.records if record["backend"] == backend]


def record_violations(record):
    """What is wrong with one radio record on its own. An empty list means it is consistent."""
    found = []
    if not isinstance(record, dict) or set(record) != set(RADIO_FIELDS):
        return ["the record does not hold exactly the fields of a radio/v1 record"]
    if record["schema"] != RADIO_SCHEMA:
        found.append(f"schema {record['schema']!r}")
    outcome, crc_ok, match = record["outcome"], record["crc_ok"], record["payload_match"]
    if outcome not in OUTCOMES:
        return found + [f"unknown outcome {outcome!r}"]
    if record["key"] != f"{record['opportunity_id']}/{record['backend']}":
        found.append("the key is not the opportunity and the backend")
    if crc_ok not in (True, False) or match not in (True, False) or (match and not crc_ok):
        found.append("payload_match without a CRC pass")
    wanted = {"decoded": (True, True), "payload_mismatch": (True, False), "crc_fail": (False, False),
              "timeout": (False, False), "not_dispatched": (False, False)}.get(outcome)
    if wanted is not None and (crc_ok, match) != wanted:
        found.append(f"outcome {outcome} with crc_ok {crc_ok} and payload_match {match}")
    if (record["decoded_file"] is not None) != crc_ok or (record["decoded_sha256"] is not None) != crc_ok:
        found.append("decoded bytes are kept exactly for a CRC pass")
    if outcome in COMPLETED and (not isinstance(record["path"], dict) or record["path"].get("executed") is None) \
            and outcome != "decoder_error":
        found.append("a completed decode names no executed path")
    if outcome == "not_dispatched" and record["error"] is None:
        found.append("a decode that was not dispatched names no reason")
    if not isinstance(record["allocation"], dict) or set(record["allocation"]) != set(ALLOCATION_FIELDS):
        found.append("the allocation does not hold exactly its fields")
    if not isinstance(record["tb"], dict) or set(record["tb"]) != {"id", "bytes", "bits"} \
            or record["tb"]["bits"] != 8 * record["tb"]["bytes"]:
        found.append("the transport block entry names no identifier and size")
    if record["tap"] != "none" or record["retry"] != 0:
        found.append("a tap or a retry on the untapped first transmission")
    return found


def open_output(directory):
    """Verify a native output and return it.

    Every planned decode has exactly one radio record under a unique key, in sequence. A record is
    consistent on its own, names the readback digest of the grid its opportunity was filled with,
    and its decoded bytes have the hash it names. Every stored grid has its readback digest. The
    report lists the same decodes with the same outcomes. radio/ and ingress/ hold nothing else.
    """
    directory = Path(directory)
    try:
        report = load_json(directory / "native.json")
        radio = (directory / "radio" / "radio.jsonl").read_text(encoding="ascii").splitlines()
        records = [json.loads(line) for line in radio if line.strip()]
        ingress = (directory / "ingress" / "ingress.jsonl").read_text(encoding="ascii").splitlines()
        descriptors = [json.loads(line) for line in ingress if line.strip()]
    except (ContractError, OSError, ValueError) as error:
        raise ReplayPuschError(f"{directory} is not a complete native output: {error}") from None
    if not isinstance(report, dict) or report.get("schema") != NATIVE_SCHEMA \
            or not isinstance(report.get("opportunities"), list) or not isinstance(report.get("backends"), list):
        raise ReplayPuschError(f"{directory / 'native.json'} is not a {NATIVE_SCHEMA} report")
    listed = [(item, decode) for item in report["opportunities"] for decode in item["decodes"]]
    planned = len(report["opportunities"]) * len(report["backends"])
    if not records or len(records) != planned or len(listed) != planned or report["totals"]["planned"] != planned:
        raise ReplayPuschError(f"the output holds {len(records)} radio records for {planned} planned decodes")
    if len({record.get("key") for record in records if isinstance(record, dict)}) != len(records):
        raise ReplayPuschError("two radio records have the same key")
    by_sequence = {}
    for descriptor in descriptors:
        grid = descriptor.get("grid") if isinstance(descriptor, dict) else None
        if not isinstance(grid, dict) or set(descriptor) != set(replay.INGRESS_FIELDS) \
                or set(grid) != set(replay.INGRESS_GRID_FIELDS) or descriptor["schema"] != replay.INGRESS_SCHEMA \
                or descriptor["state"] != "ready":
            raise ReplayPuschError("a grid descriptor is not an o1-grid/v1 record of a ready grid")
        if grid["file"] is not None:
            name = grid["file"]
            if not isinstance(name, str) or Path(name).name != name:
                raise ReplayPuschError("a grid descriptor names a file outside ingress/")
            if _file_sha256(directory / "ingress" / name, "a stored grid") != grid["sha256"]:
                raise ReplayPuschError(f"the stored grid of opportunity {descriptor['sequence']} differs from its "
                                       "readback digest")
        by_sequence[descriptor["sequence"]] = descriptor
    kept = {"radio.jsonl"}
    for position, (record, (item, decode)) in enumerate(zip(records, listed)):
        problems = record_violations(record)
        if problems:
            raise ReplayPuschError(f"radio record {position}: {problems[0]}")
        if record["sequence"] != position or record["run_id"] != report["run_id"]:
            raise ReplayPuschError(f"radio record {position} is out of sequence or of another run")
        if record["key"] != decode.get("key") or record["outcome"] != decode.get("outcome") \
                or record["payload_match"] != decode.get("payload_match") \
                or record["opportunity_id"] != item.get("opportunity_id") \
                or record["opportunity_sequence"] != item.get("sequence"):
            raise ReplayPuschError(f"radio record {position} differs from the report")
        descriptor = by_sequence.get(record["opportunity_sequence"])
        if record["ingress"] is not None:
            if descriptor is None or record["ingress"]["readback_sha256"] != descriptor["grid"]["sha256"] \
                    or record["grant_record_sha256"] != descriptor["grant_record_sha256"] \
                    or record["ingress"]["readback_sha256"] != item.get("readback_sha256"):
                raise ReplayPuschError(f"radio record {position} does not name the grid of its opportunity")
        elif record["outcome"] != "not_dispatched":
            raise ReplayPuschError(f"radio record {position} names no grid")
        if record["decoded_file"] is not None:
            name = record["decoded_file"]
            if not isinstance(name, str) or Path(name).name != name:
                raise ReplayPuschError(f"radio record {position} names a file outside radio/")
            path = directory / "radio" / name
            if _file_sha256(path, "decoded bytes") != record["decoded_sha256"] \
                    or path.stat().st_size != record["tb"]["bytes"]:
                raise ReplayPuschError(f"the decoded bytes of radio record {position} differ from their hash or size")
            kept.add(name)
    extra = sorted(path.name for path in (directory / "radio").iterdir() if path.name not in kept or not path.is_file())
    if extra:
        raise ReplayPuschError(f"radio/ holds {extra[0]}, which no record names")
    return Output(directory, report, records, descriptors)


def evaluate(output, plan):
    """Join every radio record to its original transport block, as an evaluator does after a run.

    The original blocks are read from the files the plan binds, never from the native output. Each
    row names the outcome that the rule gives for the decoder's CRC flag and the evaluator's own
    byte comparison, and whether the native record agrees with it.
    """
    originals = {}
    for grant in plan["grants"]:
        block = Path(grant["tb"]["path"]).read_bytes()
        if hashlib.sha256(block).hexdigest() != grant["tb"]["sha256"]:
            raise ReplayPuschError(f"the transport block of grant {grant['grant_id']} changed after the plan was built")
        originals[grant["grant_id"]] = block
    rows, totals = [], {}
    for record in output.records:
        original = originals.get(record["grant_id"])
        if original is None:
            raise ReplayPuschError(f"the record {record['key']} names the grant {record['grant_id']}, which the plan "
                                   "does not hold")
        decoded = output.decoded(record)
        compared = compare(decoded, original) if decoded is not None else None
        equal = bool(compared and compared["equal"])
        completed = record["outcome"] in COMPLETED
        if record["outcome"] == "not_dispatched":
            wanted = "not_dispatched"
        elif record["outcome"] == "decoder_error":
            wanted = "decoder_error"  # the reason is the receiver's: an error, a guard or a changed grid
        else:
            wanted = classify(completed, False, record["crc_ok"], equal)
        row = {"key": record["key"], "backend": record["backend"], "opportunity_id": record["opportunity_id"],
               "grant_id": record["grant_id"], "outcome": record["outcome"], "evaluated": wanted,
               "crc_ok": record["crc_ok"], "payload_match": equal,
               "first_differing_byte": compared["first_differing_byte"] if compared else None,
               "bytes_compared": compared["bytes_compared"] if compared else 0, "original_bytes": len(original),
               "consistent": wanted == record["outcome"] and equal == record["payload_match"]
               and record["tb"]["bytes"] == len(original)}
        rows.append(row)
        total = totals.setdefault(record["backend"], dict({name: 0 for name in OUTCOMES}, planned=0, payload_matches=0,
                                                         inconsistent=0, matched_bytes=0))
        total["planned"] += 1
        total[record["outcome"]] += 1
        total["payload_matches"] += equal
        total["matched_bytes"] += len(original) if equal else 0
        total["inconsistent"] += not row["consistent"]
    return rows, totals


def summary(output, sequence=0, backend=None):
    """What a second reader recovers from a native output: one decode, its grid digest and its bytes."""
    names = output.backends()
    backend = names[0] if backend is None else backend
    chosen = [record for record in output.of(backend) if record["opportunity_sequence"] == sequence]
    if backend not in names or not chosen:
        raise ReplayPuschError(f"the output holds no decode of opportunity {sequence} by the backend {backend!r}")
    record = chosen[0]
    decoded = output.decoded(record)
    counts = {name: sum(item["outcome"] == name for item in output.records) for name in OUTCOMES}
    return {"run_id": output.run_id, "records": len(output.records), "verified_records": len(output.records),
            "backends": names, "outcomes": {name: count for name, count in counts.items() if count},
            "payload": output.report["payload"], "key": record["key"], "opportunity_id": record["opportunity_id"],
            "sfn": record["sfn"], "slot": record["slot"], "grant_id": record["grant_id"], "backend": record["backend"],
            "outcome": record["outcome"], "crc_ok": record["crc_ok"], "payload_match": record["payload_match"],
            "executed": record["path"]["executed"] if isinstance(record["path"], dict) else None,
            "tb_bytes": record["tb"]["bytes"], "decoded_file": record["decoded_file"],
            "decoded_sha256": record["decoded_sha256"],
            "decoded_first_bytes": decoded[:8].hex() if decoded is not None else None,
            "ingress_sha256": record["ingress"]["readback_sha256"] if record["ingress"] else None,
            "service_us": record["timing"]["service_us"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a receiver replay plan from an episode entry, or read a "
                                                 "native output.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("plan", help="verify an episode entry and write its channel plan and its replay plan "
                                            "with the fixed payload")
    make.add_argument("--episode", required=True, action="append", type=Path, help="episode entry directory")
    make.add_argument("--grant-store", action="append", type=Path, default=[],
                      help="directory that holds <grant_id>/store, searched after the path the entry recorded")
    make.add_argument("--run-id", required=True)
    make.add_argument("--backend", action="append", choices=BACKENDS, help="backend to decode with (default: both)")
    make.add_argument("--output-dir", required=True, type=Path, help="directory for channel-plan.json and plan.json")
    show = commands.add_parser("show", help="verify a native output and print one decode with its grid digest")
    show.add_argument("directory", type=Path)
    show.add_argument("--sequence", type=int, default=0, help="opportunity in the order of the plan (default 0)")
    show.add_argument("--backend", choices=BACKENDS, help="backend of the decode (default: the first of the run)")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            channel = replay.entry_plan(args.run_id, args.episode, args.grant_store)
            replay.write_plan(channel, args.output_dir / "channel-plan.json")
            plan = build_plan(args.run_id, args.output_dir / "channel-plan.json", None, args.backend or BACKENDS)
            write_plan(plan, args.output_dir / "plan.json")
            print(json.dumps({"plan": str(args.output_dir / "plan.json"), "run_id": plan["run_id"],
                              "payload": plan["payload"], "backends": plan["receiver"]["backends"],
                              "grants": len(plan["grants"]), "opportunities": len(channel["opportunities"])}, indent=2))
            return 0
        print(json.dumps(summary(open_output(args.directory), args.sequence, args.backend), indent=2))
        return 0
    except (ReplayPuschError, GridApplyError, EpisodeError, GrantError) as error:
        parser.exit(2, f"replay_pusch: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
