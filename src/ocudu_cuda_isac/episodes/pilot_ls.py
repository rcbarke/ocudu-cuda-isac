"""Tapped receiver replay: the plan, the reader and the joins of the stage O2 (Sprint 3.1, selector pilot_ls).

The native tool (native/ocudu_adapter/observer/pilot_ls) decodes a receiver replay of Sprint 2.4
and, before each grid is decoded, lets the observer compute the pilot least squares of the grant's
sensing source on it. This module is the offline side of that step. It

  * writes the pilot-ls-plan/v1 file the tool reads: the replay plan by the hash that binds it,
    the tap, and per grant the sensing-source record of the grant's store entry with its two
    arrays and the antenna map of the receive planes;
  * reads the native output back: the radio records and the grid descriptors with the reader of
    Sprint 2.4, and the observation records, payloads and validity bits of observer/;
  * recomputes every estimate from the stored grid and the source record with the reference of
    ocudu_cuda_isac.sensing.pilot_ls, which must give the bytes the tool wrote;
  * joins the radio records of a tapped run to those of a control run, field by field.

Standard library only. A plan is an offline stimulus and evaluator record, because the replay
plan it names holds transport blocks. The observer reads the grid and the source record only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import grant as grants
from . import grid_apply as replay
from . import replay_pusch as rp
from .episode import EpisodeError
from .grant import GrantError
from .grid_apply import GridApplyError
from .replay_pusch import ReplayPuschError
from ..replay import ContractError, load_json
from ..sensing import pilot_ls as reference
from ..sensing.pilot_ls import PilotLsError

PLAN_SCHEMA = "pilot-ls-plan/v1"
NATIVE_SCHEMA = "pilot-ls-native/v1"
ACCESS = "offline_stimulus_and_evaluator"
TAPS = ("none", "pilot_ls")
SOURCE = "pusch_dmrs"
OBSERVER_DIR = "observer"
RECORDS_FILE = "o2.jsonl"
PLAN_FIELDS = ("schema", "run_id", "access", "replay_plan", "observer", "sources")
OBSERVER_FIELDS = ("tap", "stage", "source", "layout", "epoch")
SOURCE_FIELDS = ("grant_id", "record_sha256", "source", "pilot_index", "pilot_ref", "antenna_map")
# Fields of a radio record that a tap must leave as they are. The run, the tap itself, the times
# and the values the decoder measured are not compared: they are not deterministic or name the run.
RADIO_INVARIANT = ("key", "opportunity_id", "opportunity_sequence", "backend", "sfn", "slot", "grant_id",
                   "grant_record_sha256", "allocation", "tb", "outcome", "crc_ok", "payload_match", "decoded_sha256",
                   "decoded_file", "error", "retry")


class PilotLsPlanError(ValueError):
    """A plan or a native output of the tapped replay violates the documented interface."""


# ---------------------------------------------------------------- plan
def _keys(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise PilotLsPlanError(f"{name} requires exactly {', '.join(fields)}")
    return value


def _bound(value, name):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"} or not isinstance(value["path"], str) \
            or not value["path"]:
        raise PilotLsPlanError(f"{name} requires path and sha256")
    digest = value["sha256"]
    if digest is None:
        raise PilotLsPlanError(f"{name} is bound by no hash")
    if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        raise PilotLsPlanError(f"{name}.sha256 must be 64 hexadecimal digits")
    return value


def check_plan(plan):
    """Apply every rule of a plan that needs no file: the fields, the tap and the sources."""
    _keys(plan, PLAN_FIELDS, "a plan")
    if plan["schema"] != PLAN_SCHEMA:
        raise PilotLsPlanError(f"unsupported plan schema {plan['schema']!r}")
    if not isinstance(plan["run_id"], str) or not plan["run_id"] or not plan["run_id"].isascii():
        raise PilotLsPlanError("run_id must be a nonempty ASCII string")
    if plan["access"] != ACCESS:
        raise PilotLsPlanError(f'access must be "{ACCESS}"')
    _bound(plan["replay_plan"], "the replay plan")
    observer = _keys(plan["observer"], OBSERVER_FIELDS, "observer")
    if observer["tap"] not in TAPS:
        raise PilotLsPlanError(f"unknown tap {observer['tap']!r}: observer.tap is \"none\" or \"pilot_ls\"")
    if type(observer["epoch"]) is not int or observer["epoch"] < 1:
        raise PilotLsPlanError("observer.epoch is at least 1")
    if not isinstance(plan["sources"], list):
        raise PilotLsPlanError("sources must be a list")
    if observer["tap"] == "none":
        if observer["stage"] is not None or observer["layout"] is not None or observer["source"] is not None \
                or plan["sources"]:
            raise PilotLsPlanError("a run without a tap names no stage, no layout and no sensing source")
        return plan
    if observer["stage"] != reference.STAGE or observer["layout"] != reference.LAYOUT \
            or not isinstance(observer["source"], str) or not observer["source"]:
        raise PilotLsPlanError(f"the tap pilot_ls exports the stage {reference.STAGE} in the layout "
                               f"{reference.LAYOUT} and names its sensing source")
    seen = set()
    for source in plan["sources"]:
        _keys(source, SOURCE_FIELDS, "a source")
        name = source["grant_id"]
        if not isinstance(name, str) or not name:
            raise PilotLsPlanError("a source names no grant")
        if name in seen:
            raise PilotLsPlanError(f"the sensing source of grant {name} is listed twice")
        seen.add(name)
        if not isinstance(source["record_sha256"], str) or not source["record_sha256"]:
            raise PilotLsPlanError(f"the source of grant {name} names no record hash")
        _bound(source["source"], f"the source record of grant {name}")
        _bound(source["pilot_index"], f"the pilot coordinates of grant {name}")
        _bound(source["pilot_ref"], f"the reference symbols of grant {name}")
        antenna_map = source["antenna_map"]
        if not isinstance(antenna_map, list) or not antenna_map \
                or any(type(element) is not int or not 0 <= element <= 4095 for element in antenna_map):
            raise PilotLsPlanError(f"the antenna map of grant {name} holds array elements from 0 to 4095")
    return plan


def _file_sha256(path, name):
    try:
        return replay.sha256_file(path)
    except GridApplyError as error:
        raise PilotLsPlanError(f"{name}: {error}") from None


def load_source(bound):
    """The sensing source of one grant of a plan, from the files the plan binds, each against its hash."""
    name = bound["grant_id"]
    if _file_sha256(bound["source"]["path"], f"the source record of grant {name}") != bound["source"]["sha256"]:
        raise PilotLsPlanError(f"the source record of grant {name} does not have the hash its record binds "
                               f"({bound['source']['path']})")
    for key, what in (("pilot_index", "pilot coordinates"), ("pilot_ref", "reference symbols")):
        if _file_sha256(bound[key]["path"], f"the {what} of grant {name}") != bound[key]["sha256"]:
            raise PilotLsPlanError(f"the {what} of grant {name} do not have the hash their record binds "
                                   f"({bound[key]['path']})")
    try:
        record = load_json(Path(bound["source"]["path"]))
        index_type, index_shape, index = grants.read_npy(bound["pilot_index"]["path"])
        value_type, value_shape, values = grants.read_npy(bound["pilot_ref"]["path"])
    except (ContractError, GrantError, OSError) as error:
        raise PilotLsPlanError(f"the sensing source of grant {name} is unreadable: {error}") from None
    try:
        named_index, named_values = record["mask"]["index"], record["reference"]["values"]
        hashes = (named_index["sha256"], named_values["sha256"])
    except (KeyError, TypeError):
        raise PilotLsPlanError(f"the source record of grant {name} must name its pilot coordinates and its "
                               "reference symbols by file, as a store entry does") from None
    if hashes != (bound["pilot_index"]["sha256"], bound["pilot_ref"]["sha256"]):
        raise PilotLsPlanError(f"the source record of grant {name} names other arrays than the plan binds")
    ports = record.get("ports")
    if index_type != "<i4" or len(index_shape) != 2 or index_shape[1] != 2:
        raise PilotLsPlanError(f"the pilot coordinates of grant {name} must be int32 [pilot, 2]")
    if value_type != "<c8" or not isinstance(ports, list) or value_shape != (len(ports), index_shape[0]):
        raise PilotLsPlanError(f"the reference symbols of grant {name} must be complex64 [port, pilot]")
    inline = json.loads(json.dumps(record))
    inline["mask"]["index"] = [[index[i], index[i + 1]] for i in range(0, len(index), 2)]
    count = index_shape[0]
    inline["reference"]["values"] = [[[value.real, value.imag] for value in values[row * count:(row + 1) * count]]
                                     for row in range(len(ports))]
    try:
        return reference.Source(inline, bound["antenna_map"])
    except PilotLsError as error:
        raise PilotLsPlanError(f"the sensing source of grant {name} is refused: {error}") from None


def check_files(plan):
    """Verify everything a plan binds. Returns (replay plan, channel plan, sources by grant record hash).

    The replay plan must have its hash, be the control plan of Sprint 2.4 and pass its file check.
    With the tap, every grant of the channel plan needs one source, bound to the hash that binds
    the grid of the grant, whose record and arrays have the hashes the plan names and agree with
    the record of the grant: the carrier grid, the slot, the reference symbols, the port and
    pilots inside the allocation. The antenna map names one array element per receive plane.
    """
    check_plan(plan)
    bound = plan["replay_plan"]
    if _file_sha256(bound["path"], "the replay plan") != bound["sha256"]:
        raise PilotLsPlanError(f"the replay plan does not have the hash its record binds ({bound['path']})")
    try:
        replay_plan = rp.load_plan(bound["path"])
        channel = rp.check_files(replay_plan)
    except ReplayPuschError as error:
        raise PilotLsPlanError(f"the replay plan is refused: {error}") from None
    compiled = {}
    if plan["observer"]["tap"] == "none":
        return replay_plan, channel, compiled
    listed = {source["grant_id"]: source for source in plan["sources"]}
    records = {grant["grant_id"]: grant for grant in replay_plan["grants"]}
    for grant in channel["grants"]:
        name = grant["grant_id"]
        if name not in listed:
            raise PilotLsPlanError(f"the plan binds no sensing source for the grant {name}")
        item = listed[name]
        if item["record_sha256"] != grant["record_sha256"]:
            raise PilotLsPlanError(f"changed allocation: the sensing source of grant {name} is bound to the record "
                                   f"{item['record_sha256']}, and the channel plan binds the grid of this grant to "
                                   f"{grant['record_sha256']}")
        source = load_source(item)
        try:
            record = grants.load(records[name]["record"]["path"])
        except GrantError as error:
            raise PilotLsPlanError(f"the grant {name} is unusable: {error}") from None
        carrier = channel["carrier"]
        if source.name != plan["observer"]["source"]:
            raise PilotLsPlanError(f"the sensing source of grant {name} is the source {source.name}, and the plan taps "
                                   f"{plan['observer']['source']}")
        if (source.grid_symbols, source.grid_subcarriers, source.scs_khz) \
                != (carrier["symbols"], carrier["subcarriers"], record.scs_khz):
            raise PilotLsPlanError(f"the sensing source of grant {name} was built for another carrier grid than the "
                                   "run decodes")
        if source.slot != record.slot or source.symbols != list(record.dmrs["symbols"]) \
                or [source.port] != list(record.dmrs["ports"]):
            raise PilotLsPlanError(f"changed allocation: the sensing source of grant {name} names another slot, other "
                                   "reference symbols or other ports than the record of the grant")
        if source.rx_planes != carrier["rx_ports"]:
            raise PilotLsPlanError(f"the antenna map of grant {name} names {source.rx_planes} receive planes, and the "
                                   f"carrier grid of the run has {carrier['rx_ports']}")
        first, last = record.subcarrier_start, record.subcarrier_start + record.subcarrier_count
        outside = [k for k in source.subcarriers if not first <= k < last]
        if outside:
            raise PilotLsPlanError(f"changed allocation: the sensing source of grant {name} has a pilot on subcarrier "
                                   f"{outside[0]}, outside the allocation of the grant")
        compiled[grant["record_sha256"]] = source
    if len(listed) != len(channel["grants"]):
        raise PilotLsPlanError("the plan lists a sensing source for a grant that the channel plan does not bind")
    return replay_plan, channel, compiled


def build_plan(run_id, replay_plan_path, tap="pilot_ls", antenna_map=None, epoch=1, source=SOURCE):
    """A plan around a replay plan file that was written before.

    Every grant of the replay plan comes from a grant store: the directory of its record holds
    source.json, pilot_index.npy and pilot_ref.npy, and their hashes are taken from the verified
    store entry. antenna_map names the array element behind each receive plane and defaults to
    the receive ports of the grant, in order. tap "none" writes the control plan of this tool.
    """
    path = Path(replay_plan_path).resolve()
    try:
        replay_plan = rp.load_plan(path)
    except ReplayPuschError as error:
        raise PilotLsPlanError(str(error)) from None
    sources = []
    for grant in replay_plan["grants"] if tap != "none" else ():
        directory = Path(grant["record"]["path"]).parent
        try:
            store = grants.open_store(directory)
        except GrantError as error:
            raise PilotLsPlanError(f"the grant {grant['grant_id']} has no usable store entry at {directory}: "
                                   f"{error}") from None
        files = store.manifest["files"]
        if files["grant.json"]["sha256"] != grant["record"]["sha256"]:
            raise PilotLsPlanError(f"the store entry of grant {grant['grant_id']} holds another record than the replay "
                                   "plan binds")
        sources.append({
            "grant_id": grant["grant_id"], "record_sha256": grant["record_sha256"],
            "source": {"path": str(directory.resolve() / "source.json"), "sha256": files["source.json"]["sha256"]},
            "pilot_index": {"path": str(directory.resolve() / "pilot_index.npy"),
                            "sha256": files["pilot_index.npy"]["sha256"]},
            "pilot_ref": {"path": str(directory.resolve() / "pilot_ref.npy"), "sha256": files["pilot_ref.npy"]["sha256"]},
            "antenna_map": list(antenna_map) if antenna_map is not None else list(store.grant.rx_ports)})
    tapped = tap != "none"
    plan = {"schema": PLAN_SCHEMA, "run_id": run_id, "access": ACCESS,
            "replay_plan": {"path": str(path), "sha256": _file_sha256(path, "the replay plan")},
            "observer": {"tap": tap, "stage": reference.STAGE if tapped else None, "source": source if tapped else None,
                         "layout": reference.LAYOUT if tapped else None, "epoch": epoch},
            "sources": sources}
    check_files(plan)
    return plan


def write_plan(plan, path):
    """Check a plan and write it. The file must not exist."""
    path = Path(path)
    if path.exists():
        raise PilotLsPlanError(f"{path} already exists. A plan is never overwritten.")
    check_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return path


def load_plan(path):
    try:
        return check_plan(load_json(Path(path)))
    except (ContractError, OSError) as error:
        raise PilotLsPlanError(f"cannot read the plan: {error}") from None


# ---------------------------------------------------------------- native output
class Output:
    """A verified native output: the radio side as Sprint 2.4 reads it, and the observations."""

    def __init__(self, directory, radio, observations):
        self.directory, self.radio, self.observations = directory, radio, observations
        self.report, self.records, self.descriptors = radio.report, radio.records, radio.descriptors
        self.run_id, self.tap = radio.report["run_id"], radio.report["observer"]["tap"]
        self.by_sequence = {record["host"]["opportunity_sequence"]: record for record in observations}

    def payload(self, record):
        return (self.directory / OBSERVER_DIR / record["payload"]["file"]).read_bytes()

    def mask(self, record):
        return (self.directory / OBSERVER_DIR / record["validity"]["file"]).read_bytes()

    def values(self, record):
        return reference.payload_values(self.payload(record))

    def grid(self, record):
        """Read access to the stored grid an observation was computed from, or None when it was not kept."""
        descriptor = self.descriptors_by_sequence()[record["host"]["opportunity_sequence"]]
        if descriptor["grid"]["file"] is None:
            return None
        planes, symbols, subcarriers = descriptor["grid"]["shape"]
        return reference.stored_grid((self.directory / "ingress" / descriptor["grid"]["file"]).read_bytes(), planes,
                                     symbols, subcarriers)

    def descriptors_by_sequence(self):
        return {descriptor["sequence"]: descriptor for descriptor in self.descriptors}


def open_output(directory, schema=NATIVE_SCHEMA):
    """Verify a native output of the tapped replay and return it.

    schema is the report schema of this ticket unless a later plan version names its own (an
    audited run of Sprint 3.2 writes the same files under the schema of its plan version).

    The radio records, the decoded bytes and the grid descriptors are verified as Sprint 2.4
    verifies them, with the tap of the run in every record. With the tap "none" the output holds
    no observer directory. With the tap "pilot_ls" every opportunity that was filled has exactly
    one observation record, in sequence, consistent on its own, whose payload and validity bits
    have the hashes and sizes it names, whose set bits are the count it names, and which names the
    readback digest of the grid of its opportunity. observer/ holds nothing else.
    """
    directory = Path(directory)
    try:
        report = load_json(directory / "native.json")
        tap = report["observer"]["tap"]
    except (ContractError, OSError, KeyError, TypeError) as error:
        raise PilotLsPlanError(f"{directory} is not a complete native output: {error}") from None
    if tap not in TAPS:
        raise PilotLsPlanError(f"the report names the unknown tap {tap!r}")
    try:
        radio = rp.open_output(directory, schema=schema, taps=(tap,))
    except ReplayPuschError as error:
        raise PilotLsPlanError(str(error)) from None
    if report["receiver"]["tap"] != tap or any(record["tap"] != tap for record in radio.records):
        raise PilotLsPlanError("the report and the radio records name different taps")
    observer_dir = directory / OBSERVER_DIR
    filled = [item for item in report["opportunities"] if item.get("grid_index") is not None]
    if tap == "none":
        if observer_dir.exists() or report["observer"]["records"] != 0 or report["totals"].get("observations") != 0 \
                or any(item.get("observer") is not None for item in report["opportunities"]):
            raise PilotLsPlanError("a run without a tap holds an observation")
        return Output(directory, radio, [])
    try:
        lines = (observer_dir / RECORDS_FILE).read_text(encoding="ascii").splitlines()
        observations = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError) as error:
        raise PilotLsPlanError(f"{observer_dir} holds no readable observation records: {error}") from None
    if len(observations) != len(filled) or report["observer"]["records"] != len(observations) \
            or report["totals"].get("observations") != len(observations):
        raise PilotLsPlanError(f"the output holds {len(observations)} observation records for {len(filled)} filled "
                               "grids")
    descriptors = {descriptor["sequence"]: descriptor for descriptor in radio.descriptors}
    kept = {RECORDS_FILE}
    for position, (record, item) in enumerate(zip(observations, filled)):
        problems = reference.record_violations(record)
        if problems:
            raise PilotLsPlanError(f"observation record {position}: {problems[0]}")
        host = record["host"]
        if record["keys"]["sequence"] != position or record["run_id"] != report["run_id"] \
                or record["keys"]["epoch"] != report["observer"]["epoch"]:
            raise PilotLsPlanError(f"observation record {position} is out of sequence, of another run or of another "
                                   "epoch")
        descriptor = descriptors.get(host["opportunity_sequence"])
        if host["opportunity_sequence"] != item["sequence"] or host["opportunity_id"] != item["opportunity_id"] \
                or descriptor is None:
            raise PilotLsPlanError(f"observation record {position} does not belong to opportunity {item['sequence']} "
                                   "of the report")
        if host["ingress"]["readback_sha256"] != descriptor["grid"]["sha256"] \
                or host["grant_record_sha256"] != descriptor["grant_record_sha256"] \
                or host["ingress"]["generation"] != descriptor["generation"] \
                or host["ingress"]["grid_index"] != descriptor["grid_index"] \
                or (record["radio"]["sfn"], record["radio"]["slot"]) != (descriptor["sfn"], descriptor["slot"]):
            raise PilotLsPlanError(f"observation record {position} does not name the grid of its opportunity")
        if record["allocation"]["rx_count"] != descriptor["grid"]["shape"][0]:
            raise PilotLsPlanError(f"observation record {position} has another number of receive planes than its grid")
        for name in (record["payload"]["file"], record["validity"]["file"]):
            if Path(name).name != name or name in kept:
                raise PilotLsPlanError(f"observation record {position} names a file outside observer/ or one that "
                                       "another record names")
            kept.add(name)
        try:
            payload = (observer_dir / record["payload"]["file"]).read_bytes()
            mask = (observer_dir / record["validity"]["file"]).read_bytes()
        except OSError as error:
            raise PilotLsPlanError(f"observation record {position}: {error}") from None
        if len(payload) != record["payload"]["bytes"] or hashlib.sha256(payload).hexdigest() != record["payload"]["digest"]:
            raise PilotLsPlanError(f"the payload of observation record {position} differs from its digest or size")
        if len(mask) != record["validity"]["mask_bytes"] or hashlib.sha256(mask).hexdigest() != record["validity"]["sha256"]:
            raise PilotLsPlanError(f"the validity bits of observation record {position} differ from their hash or size")
        if sum(bin(byte).count("1") for byte in mask) != record["validity"]["valid"]:
            raise PilotLsPlanError(f"observation record {position} counts other valid elements than its bits set")
        listed = item.get("observer")
        if not isinstance(listed, dict) or listed.get("payload_sha256") != record["payload"]["digest"] \
                or listed.get("ingress_sha256") != host["ingress"]["readback_sha256"]:
            raise PilotLsPlanError(f"observation record {position} differs from the report")
    extra = sorted(path.name for path in observer_dir.iterdir() if path.name not in kept or not path.is_file())
    if extra:
        raise PilotLsPlanError(f"observer/ holds {extra[0]}, which no record names")
    return Output(directory, radio, observations)


def observer_violations(directory, forbidden=()):
    """What the observer side holds that it must not. An empty list means the access rule holds.

    observer/ may hold the records and the files they name. forbidden lists strings that must not
    appear in the records: hashes and paths of H, W and transport blocks, and labels.
    """
    path = Path(directory) / OBSERVER_DIR / RECORDS_FILE
    try:
        text = path.read_text(encoding="ascii")
    except OSError as error:
        return [f"{RECORDS_FILE} is unreadable: {error}"]
    found = []
    for line in text.splitlines():
        if line.strip():
            record = json.loads(line)
            found += [f"record field {key!r}" for key in sorted(set(record) - set(reference.RECORD_FIELDS))]
            for group, fields in reference.GROUPS.items():
                if isinstance(record.get(group), dict):
                    found += [f"record field {group}.{key}" for key in sorted(set(record[group]) - set(fields))]
    found += [f"the records hold {item!r}" for item in forbidden if item and item in text]
    return found


def oracle(output, sources):
    """Recompute every observation from the stored grid and the source record, with the reference.

    sources maps the hash of a grant record to its Source, as check_files returns them. Each row
    says whether the payload and the validity bits of the tool equal the reference bit for bit.
    """
    rows = []
    for record in output.observations:
        source = sources.get(record["host"]["grant_record_sha256"])
        grid = output.grid(record)
        if source is None or grid is None:
            raise PilotLsPlanError(f"the observation of {record['host']['opportunity_id']} has no source in the plan or "
                                   "no stored grid")
        made = reference.estimate(source, grid)
        allocation = record["allocation"]
        rows.append({
            "sequence": record["keys"]["sequence"], "opportunity_id": record["host"]["opportunity_id"],
            "payload_equal": made.payload() == output.payload(record), "mask_equal": made.mask == output.mask(record),
            "valid": made.valid, "holes": made.holes,
            "coordinates_equal": allocation["coordinates_sha256"] == source.coordinates_sha256
            and allocation["dmrs_symbols"] == source.symbols and allocation["first_subcarrier"] == source.subcarriers[0]
            and allocation["subcarrier_step"] == source.subcarrier_step
            and record["shape"]["shape"] == source.shape and allocation["antenna_map"] == source.antenna_map,
            "identity_equal": (record["what"]["source"], record["what"]["port"], record["what"]["scale"])
            == (source.name, source.port, source.amplitude)})
    return rows


def expected_times(output, sources):
    """Every observation against the slot count and the sample times the reference derives. Returns mismatches."""
    clock, found = reference.SlotClock(), []
    entries = {item["sequence"]: item for item in output.report["opportunities"]}
    for record in output.observations:
        source = sources[record["host"]["grant_record_sha256"]]
        episode = entries[record["host"]["opportunity_sequence"]]["realization_id"]
        absolute = clock.absolute_slot(episode, source.scs_khz, record["radio"]["sfn"], record["radio"]["slot"])
        first, last = reference.sample_times(source, absolute)
        if (record["radio"]["absolute_slot"], record["time"]["t_first_sample"], record["time"]["t_last_sample"]) \
                != (absolute, first, last) or record["keys"]["episode"] != reference.opaque_key(episode):
            found.append(record["host"]["opportunity_id"])
    return found


def radio_differences(tapped, control):
    """Fields of the radio records of a tapped run that differ from those of a control run.

    Both arguments are lists of radio/v1 records. Every key of the control must exist in the
    tapped run and the reverse, and every field of RADIO_INVARIANT, the readback digest, the index
    and the generation of the grid and the requested and executed path must be equal.
    """
    found = []
    by_key = {record["key"]: record for record in tapped}
    for key in sorted(set(by_key) ^ {record["key"] for record in control}):
        found.append(f"{key}: the key exists in one run only")
    for record in control:
        other = by_key.get(record["key"])
        if other is None:
            continue
        for field in RADIO_INVARIANT:
            if other[field] != record[field]:
                found.append(f"{record['key']}: {field} is {other[field]!r} with the tap and {record[field]!r} without")
        ingress, mine = record["ingress"] or {}, other["ingress"] or {}
        for field in ("readback_sha256", "grid_index", "generation", "intact"):
            if mine.get(field) != ingress.get(field):
                found.append(f"{record['key']}: ingress.{field} differs")
        path, taken = record["path"] or {}, other["path"] or {}
        for field in ("requested", "executed", "fallback"):
            if taken.get(field) != path.get(field):
                found.append(f"{record['key']}: path.{field} differs")
    return found


def summary(output, sequence=0, plane=0, row=0, pilot=0):
    """What a second reader recovers from a native output: one observation and one of its estimates."""
    if output.tap == "none":
        raise PilotLsPlanError("the output is a control run: it holds no observation")
    if type(sequence) is not int or sequence not in output.by_sequence:
        raise PilotLsPlanError(f"the output holds no observation of opportunity {sequence}")
    record = output.by_sequence[sequence]
    shape, allocation = record["shape"]["shape"], record["allocation"]
    for position, size in zip((plane, row, pilot), shape):
        if type(position) is not int or not 0 <= position < size:
            raise PilotLsPlanError(f"index {[plane, row, pilot]} is outside a payload of shape {shape}")
    element = (plane * shape[1] + row) * shape[2] + pilot
    value = output.values(record)[element]
    step = allocation["subcarrier_step"]
    return {"run_id": output.run_id, "tap": output.tap, "observations": len(output.observations),
            "verified_observations": len(output.observations), "radio_records": len(output.records),
            "sequence": sequence, "opportunity_id": record["host"]["opportunity_id"],
            "grant_id": record["host"]["grant_id"], "sfn": record["radio"]["sfn"], "slot": record["radio"]["slot"],
            "absolute_slot": record["radio"]["absolute_slot"], "stage": record["what"]["stage"],
            "producer": record["what"]["producer"], "source": record["what"]["source"],
            "layout": record["what"]["layout"], "scale": record["what"]["scale"], "shape": shape,
            "valid": record["validity"]["valid"], "holes": record["validity"]["holes"],
            "payload_file": record["payload"]["file"], "payload_sha256": record["payload"]["digest"],
            "ingress_sha256": record["host"]["ingress"]["readback_sha256"],
            "t_first_sample": record["time"]["t_first_sample"], "t_last_sample": record["time"]["t_last_sample"],
            "element": {"rx_plane": plane, "array_element": allocation["antenna_map"][plane],
                        "symbol": allocation["dmrs_symbols"][row],
                        "subcarrier": allocation["first_subcarrier"] + step * pilot if step else None,
                        "valid": bool(output.mask(record)[element // 8] >> (element % 8) & 1),
                        "value": [value.real, value.imag], "magnitude": abs(value)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the plan of a tapped receiver replay, or read a native output.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("plan", help="verify a replay plan of Sprint 2.4 and write the plan of a run on it")
    make.add_argument("--replay-plan", required=True, type=Path, help="plan.json of a receiver replay")
    make.add_argument("--run-id", required=True)
    make.add_argument("--tap", choices=TAPS, default="pilot_ls", help="pilot_ls (default) or none for the control")
    make.add_argument("--epoch", type=int, default=1)
    make.add_argument("--output", required=True, type=Path, help="the plan file to write")
    show = commands.add_parser("show", help="verify a native output and print one observation with one estimate")
    show.add_argument("directory", type=Path)
    show.add_argument("--sequence", type=int, default=0, help="opportunity in the order of the plan (default 0)")
    show.add_argument("--rx", type=int, default=0, help="receive plane (default 0)")
    show.add_argument("--row", type=int, default=0, help="reference symbol of the payload (default 0)")
    show.add_argument("--pilot", type=int, default=0, help="pilot of the payload (default 0)")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            plan = build_plan(args.run_id, args.replay_plan, args.tap, epoch=args.epoch)
            write_plan(plan, args.output)
            print(json.dumps({"plan": str(args.output), "run_id": plan["run_id"], "observer": plan["observer"],
                              "sources": len(plan["sources"]), "replay_plan": plan["replay_plan"]["path"]}, indent=2))
            return 0
        print(json.dumps(summary(open_output(args.directory), args.sequence, args.rx, args.row, args.pilot), indent=2))
        return 0
    except (PilotLsPlanError, PilotLsError, ReplayPuschError, GridApplyError, EpisodeError, GrantError) as error:
        parser.exit(2, f"pilot_ls: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
