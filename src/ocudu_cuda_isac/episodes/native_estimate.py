"""The receiver's own estimate on the staged harness: the plan, the reader and the parity (Sprint 5.2, selector native_estimate).

The native tool pilot_ls reads a third plan version from this sprint on. pilot-ls-plan/v3 is the
plan of Sprint 3.2 with one more object:

  estimate  null, or the capture of the receiver's own channel estimate, the stage O3: the tap
            native_estimate, the backend that produces the estimate, the stream the copy is
            enqueued on, where a timed copy lands, how long a capture asks for the completion of
            its copy, and the ring of the O3 stream or null.

Only the pilot_ls of the adapter build against the patched receiver runs a plan that names an
estimate. Every other build refuses it. A plan of this module that names an estimate also taps
the stage O2, which stays the parity reference and the fallback input, as a stream of its own.

This module is the offline side. It

  * writes and checks a plan of version 3 around a receiver replay plan of Sprint 2.4;
  * reads a native output: the radio side, the observer side and ring/ with the readers of the
    earlier sprints, and estimate/ with the account of the capture, the records and the payloads;
  * balances the account: every decode of the producing backend ends as published, captured,
    dropped, absent, refused or failed, and the counts of the O3 ring are those of its lines;
  * joins the records that a consumer of the O3 stream kept to the lines of the producer;
  * compares every captured estimate with the matched estimator on the stage O2 of the same
    opportunity, and writes the counts of that comparison.

Standard library only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import pilot_ls as pl
from . import snapshot_ring as sr
from .snapshot_ring import SnapshotRingError
from ..replay import ContractError, load_json
from ..sensing import native_estimate as reference
from ..sensing import obs
from ..sensing import pilot_ls as o2_reference
from ..sensing.native_estimate import NativeEstimateError
from ..sensing.obs import ObsError

PLAN_SCHEMA = "pilot-ls-plan/v3"
NATIVE_SCHEMA = "pilot-ls-native/v3"
CAPTURE_SCHEMA = "o3-capture/v1"
PARITY_SCHEMA = "o3-parity/v1"
TAP = "native_estimate"
BACKEND = "cuda"
ESTIMATE_DIR = "estimate"
CAPTURES_FILE = "captures.jsonl"
RECORDS_FILE = "o3.jsonl"
RING_FILE = "ring.json"
COPY_STREAMS = ("producer", "side")
LANDINGS = ("slot", "staging")
OUTCOMES = ("published", "captured", "dropped", "absent", "refused", "failed")
REASONS = {"published": (None,), "captured": (None,), "dropped": ("no_credit",), "absent": ("no_estimate",),
           "refused": ("null_estimate", "layers", "not_contiguous", "not_interpolated", "other_pdu", "other_allocation",
                       "size"),
           "failed": ("submit", "copy", "timeout", "overtaken", "busy")}
PLAN_FIELDS = sr.PLAN_FIELDS + ("estimate",)
ESTIMATE_FIELDS = ("tap", "stage", "source", "layout", "backend", "copy_stream", "landing", "timeout_us", "ring")
CAPTURE_FIELDS = ("schema", "run_id", "mode", "stage", "backend", "opportunity_id", "opportunity_sequence", "outcome",
                  "reason", "admitted", "seen", "landed", "producer", "copy", "ring", "keys", "absolute_slot", "grid",
                  "payload_bytes", "valid", "holes", "nonzero", "t_expect_ns", "t_tap_ns", "t_submit_ns", "t_resolve_ns",
                  "t_complete_ns", "t_publish_ns", "t_done_ns", "resolve_us")
TOLERANCE_FIELDS = ("abs_ulps", "relative", "nmse", "smoothing_fraction")


class NativeEstimatePlanError(ValueError):
    """A plan, a native output or an archive of the O3 stream violates the documented interface."""


# ---------------------------------------------------------------- plan
def estimate_spec(copy_stream="producer", landing="slot", timeout_us=2000, ring=None, source=pl.SOURCE):
    """The estimate object of a plan. ring is a ring object of snapshot_ring.ring_spec, with its own stream, or None."""
    return {"tap": TAP, "stage": reference.STAGE, "source": source, "layout": reference.LAYOUT, "backend": BACKEND,
            "copy_stream": copy_stream, "landing": landing, "timeout_us": timeout_us, "ring": ring}


def _second_version(plan):
    """The part of a plan that is a plan of Sprint 3.2."""
    inner = {key: plan[key] for key in sr.PLAN_FIELDS}
    inner["schema"] = sr.PLAN_SCHEMA
    return inner


def check_plan(plan):
    """Apply every rule of a plan of version 3 that needs no file."""
    if not isinstance(plan, dict) or set(plan) != set(PLAN_FIELDS):
        raise NativeEstimatePlanError(f"a plan requires exactly {', '.join(PLAN_FIELDS)}")
    if plan["schema"] != PLAN_SCHEMA:
        raise NativeEstimatePlanError(f"unsupported plan schema {plan['schema']!r}")
    try:
        sr.check_plan(_second_version(plan))
    except SnapshotRingError as error:
        raise NativeEstimatePlanError(str(error)) from None
    estimate = plan["estimate"]
    if estimate is None:
        return plan
    if not isinstance(estimate, dict) or set(estimate) != set(ESTIMATE_FIELDS):
        raise NativeEstimatePlanError(f"estimate requires exactly {', '.join(ESTIMATE_FIELDS)}, or is null")
    if (estimate["tap"], estimate["stage"], estimate["layout"]) != (TAP, reference.STAGE, reference.LAYOUT):
        raise NativeEstimatePlanError(f"the estimate of a plan is the tap {TAP}, which exports the stage "
                                      f"{reference.STAGE} in the layout {reference.LAYOUT}")
    if estimate["backend"] != BACKEND:
        raise NativeEstimatePlanError('estimate.backend is "cuda": only the resident CUDA path of the receiver leaves '
                                      "its estimate on the device")
    if estimate["copy_stream"] not in COPY_STREAMS:
        raise NativeEstimatePlanError('estimate.copy_stream is "producer" or "side"')
    if estimate["landing"] not in LANDINGS:
        raise NativeEstimatePlanError('estimate.landing is "slot" or "staging"')
    if type(estimate["timeout_us"]) is not int or not 0 <= estimate["timeout_us"] <= 60_000_000:
        raise NativeEstimatePlanError("estimate.timeout_us must be an integer from 0 to 60000000")
    observer = plan["observer"]
    if observer["tap"] != "pilot_ls" or estimate["source"] != observer["source"]:
        raise NativeEstimatePlanError("a plan that names an estimate taps the stage O2 of the same sensing source as "
                                      "well: O2 is the parity reference and the fallback input")
    ring = estimate["ring"]
    if ring is None:
        if plan["evidence"]["mode"] == "timed":
            raise NativeEstimatePlanError("a timed estimate exists in a slot of its ring only: the plan names an "
                                          "estimate and no ring for it")
        return plan
    try:
        # The ring of the O3 stream obeys the rules of the ring of the O2 stream.
        probe = dict(_second_version(plan), ring=ring, evidence={"mode": "audit"})
        sr.check_plan(probe)
    except SnapshotRingError as error:
        raise NativeEstimatePlanError("estimate." + str(error)) from None
    if plan["ring"] is not None and plan["ring"]["stream"] == ring["stream"]:
        raise NativeEstimatePlanError("the O2 stream and the O3 stream of a plan have different names")
    return plan


def check_files(plan):
    """Verify everything a plan binds, as Sprint 3.2 does. Returns (replay plan, channel plan, sources)."""
    check_plan(plan)
    try:
        replay_plan, channel, sources = sr.check_files(_second_version(plan))
    except SnapshotRingError as error:
        raise NativeEstimatePlanError(str(error)) from None
    if plan["estimate"] is not None and BACKEND not in replay_plan["receiver"]["backends"]:
        raise NativeEstimatePlanError(f"the plan names the estimate of the backend {BACKEND}, and the replay plan does "
                                      "not decode on it")
    return replay_plan, channel, sources


def build_plan(run_id, replay_plan_path, mode="audit", ring=None, estimate=None, tap="pilot_ls", antenna_map=None,
               epoch=1, source=pl.SOURCE):
    """A plan of version 3 around a replay plan file that was written before."""
    try:
        plan = sr.build_plan(run_id, replay_plan_path, mode, ring, tap, antenna_map, epoch, source)
    except SnapshotRingError as error:
        raise NativeEstimatePlanError(str(error)) from None
    plan.update(schema=PLAN_SCHEMA, estimate=estimate)
    check_files(plan)
    return plan


def write_plan(plan, path):
    """Check a plan and write it. The file must not exist."""
    path = Path(path)
    if path.exists():
        raise NativeEstimatePlanError(f"{path} already exists. A plan is never overwritten.")
    check_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return path


def load_plan(path):
    try:
        return check_plan(load_json(Path(path)))
    except (ContractError, OSError) as error:
        raise NativeEstimatePlanError(f"cannot read the plan: {error}") from None


def socket_path(native_directory, plan):
    """Where a consumer of the O3 stream of a run attaches."""
    return Path(native_directory) / ESTIMATE_DIR / f"{plan['estimate']['ring']['stream']}.sock"


# ---------------------------------------------------------------- native output
class Output:
    """A verified native output of a plan of version 3."""

    def __init__(self, directory, base, captures, estimates):
        self.directory, self.base, self.captures, self.estimates = Path(directory), base, captures, estimates
        self.report, self.radio, self.observer = base.report, base.radio, base.observer
        # The radio side under the names of the earlier readers, so that their checks read this output too.
        self.records, self.descriptors = base.records, base.descriptors
        self.mode, self.run_id = base.mode, base.run_id
        self.estimate = self.report["estimate"]
        self.ring = self.estimate["ring"] if self.estimate else None
        self.published = [line for line in captures if line["outcome"] == "published"]

    def payload(self, record):
        return (self.directory / ESTIMATE_DIR / record["payload"]["file"]).read_bytes()

    def values(self, record):
        return reference.payload_values(self.payload(record))

    def by_opportunity(self):
        return {record["host"]["opportunity_id"]: record for record in self.estimates}


def capture_violations(line):
    """What is wrong with one line of the account of the capture on its own."""
    if not isinstance(line, dict) or set(line) != set(CAPTURE_FIELDS):
        return ["the line does not hold exactly the fields of an o3-capture/v1 line"]
    found = []
    outcome = line["outcome"]
    if line["schema"] != CAPTURE_SCHEMA or line["mode"] not in sr.MODES or line["stage"] != reference.STAGE \
            or line["backend"] != BACKEND:
        found.append("another schema, an unknown evidence mode or another stage or backend")
    if outcome not in OUTCOMES:
        return found + [f"unknown outcome {outcome!r}"]
    if line["reason"] not in REASONS[outcome]:
        found.append(f"the outcome {outcome} does not have the reason {line['reason']!r}")
    keys = line["keys"]
    if not isinstance(keys, dict) or set(keys) != {"episode", "opportunity", "grant"} \
            or keys["opportunity"] != o2_reference.opaque_key(line["opportunity_id"]):
        found.append("the opportunity key is not the key of the opportunity")
    producer, copy, ring = line["producer"], line["copy"], line["ring"]
    if line["seen"] != isinstance(producer, dict) or (producer is not None
                                                      and set(producer) != set(reference.PRODUCER_FIELDS)):
        found.append("a line names the trace of the producer exactly when the receiver reported an estimate")
    if (outcome == "published") != isinstance(ring, dict) or (outcome == "published" and not line["admitted"]):
        found.append("a published estimate names its place on the ring, and no other does")
    landed = outcome in ("published", "captured") or (outcome == "dropped" and line["mode"] == "audit" and line["seen"])
    if line["landed"] != landed or (line["landed"] and (copy is None or line["valid"] is None
                                                       or line["t_complete_ns"] is None)):
        found.append("a complete payload exists for a published or captured estimate, and for a dropped one of an audit")
    if not line["landed"] and any(line[name] is not None for name in ("valid", "holes", "nonzero")):
        found.append("an estimate without a payload names no counts")
    if copy is not None and (set(copy) != {"stream", "landing", "completion", "polls", "submit_us"}
                             or copy["stream"] not in COPY_STREAMS or copy["landing"] not in LANDINGS
                             or copy["completion"] != "cuda_event"):
        found.append("the copy names its stream, its landing and its completion event")
    if outcome == "published":
        times = [line[name] for name in ("t_expect_ns", "t_tap_ns", "t_submit_ns", "t_resolve_ns", "t_complete_ns",
                                         "t_publish_ns", "t_done_ns")]
        if any(time is None for time in times) or times != sorted(times):
            found.append("the times of a published estimate are not in order: expect, tap, submit, resolve, complete, "
                         "publish, done")
        elif line["valid"] + line["holes"] != line["payload_bytes"] // reference.ELEMENT_BYTES:
            found.append("valid and holes do not add up to the elements of the payload")
    if outcome == "absent" and (line["seen"] or copy is not None):
        found.append("an absent estimate was never reported and never copied")
    return found


def open_output(directory, schema=NATIVE_SCHEMA):
    """Verify a native output of a plan of version 3 and return it.

    schema is the report schema of this ticket unless a later plan version names its own (a run
    of Sprint 4.1 writes the same files under the schema of its plan version).

    Everything a plan of version 2 writes is verified as Sprint 3.2 verifies it. With an estimate,
    estimate/ holds one account line per decode of the producing backend that was started, in the
    order of the opportunities, consistent with the entry of its opportunity in the report. An
    audited run holds one obs-descriptor/v1 record with its payload and its validity bits per
    line that says landed, and the payload has the digest, the size and the counts its record
    names. A timed run holds no record file: its estimates exist in the slots of the ring only.
    The counts of the O3 ring are those of the lines. estimate/ holds nothing else.
    """
    directory = Path(directory)
    try:
        base = sr.open_output(directory, schema=schema)
        report = base.report
        estimate, route = report["estimate"], report["o3_route"]
    except SnapshotRingError as error:
        raise NativeEstimatePlanError(str(error)) from None
    except (KeyError, TypeError) as error:
        raise NativeEstimatePlanError(f"{directory} is not a complete native output of a plan of version 3: "
                                      f"{error!r}") from None
    estimate_dir = directory / ESTIMATE_DIR
    entries = report["opportunities"]
    if estimate is None:
        if estimate_dir.exists() or any(item.get("estimate") is not None for item in entries):
            raise NativeEstimatePlanError("the report names no estimate, and the output holds one")
        return Output(directory, base, [], [])
    if route is not True or estimate.get("tap") != TAP or estimate.get("stage") != reference.STAGE \
            or estimate.get("layout") != reference.LAYOUT or estimate.get("backend") != BACKEND:
        raise NativeEstimatePlanError("the report names an estimate that is not the stage O3 of a build with the route "
                                      "to the patched receiver")
    try:
        lines = [json.loads(line) for line in (estimate_dir / CAPTURES_FILE).read_text(encoding="ascii").splitlines()
                 if line.strip()]
    except (OSError, ValueError) as error:
        raise NativeEstimatePlanError(f"{estimate_dir} holds no readable account of the capture: {error}") from None
    expected = [item for item in entries if item.get("estimate") is not None]
    if len(lines) != len(expected) or estimate["eligible"] != len(lines):
        raise NativeEstimatePlanError(f"the account holds {len(lines)} lines for {len(expected)} decodes with an "
                                      "expected estimate")
    last, counts, landed = {}, dict.fromkeys(OUTCOMES, 0), 0
    for position, (line, item) in enumerate(zip(lines, expected)):
        problems = capture_violations(line)
        if problems:
            raise NativeEstimatePlanError(f"capture line {position}: {problems[0]}")
        listed = item["estimate"]
        if line["run_id"] != report["run_id"] or line["mode"] != base.mode \
                or line["opportunity_id"] != item["opportunity_id"] or line["opportunity_sequence"] != item["sequence"] \
                or listed["outcome"] != line["outcome"] or listed["reason"] != line["reason"]:
            raise NativeEstimatePlanError(f"capture line {position} does not belong to opportunity {item['sequence']} "
                                          "of the report")
        if line["grid"] != {"grid_index": item["grid_index"], "generation": item["generation"]}:
            raise NativeEstimatePlanError(f"capture line {position} names another grid than its opportunity was filled "
                                          "with")
        if line["outcome"] == "published":
            place = line["ring"]
            if place["sequence"] != last.get(place["epoch"], -1) + 1 or listed["ring"] != place["sequence"]:
                raise NativeEstimatePlanError(f"capture line {position} has the sequence {place['sequence']} of epoch "
                                              f"{place['epoch']}, where {last.get(place['epoch'], -1) + 1} follows")
            last[place["epoch"]] = place["sequence"]
        counts[line["outcome"]] += 1
        landed += line["landed"]
    if estimate["outcomes"] != counts or estimate["landed"] != landed:
        raise NativeEstimatePlanError("the outcomes of the report are not those of the account lines")
    kept = {CAPTURES_FILE}
    ring = estimate["ring"]
    if ring is not None:
        try:
            written = load_json(estimate_dir / RING_FILE)
        except (OSError, ContractError) as error:
            raise NativeEstimatePlanError(f"{estimate_dir} holds no readable report of the O3 ring: {error}") from None
        totals = ring.get("counts", {})
        if written != ring or ring.get("schema") != sr.RING_SCHEMA or set(totals) != set(sr.COUNT_FIELDS) \
                or ring["definition"] != f"{obs.DEFINITION:016x}" or ring["eligible"] != len(lines):
            raise NativeEstimatePlanError("estimate/ring.json is not the ring of the report, or was built from another "
                                          "record definition than this reader")
        with_ring = [line for line in lines if line["reason"] != "busy"]
        admitted = sum(line["admitted"] for line in lines)
        if totals["published"] != counts["published"] or totals["admitted"] != admitted \
                or totals["dropped"] != len(with_ring) - admitted or totals["cancelled"] != admitted - counts["published"]:
            raise NativeEstimatePlanError("the counts of the O3 ring are not those of the account lines")
        kept.add(RING_FILE)
    elif counts["published"] or any(line["admitted"] for line in lines):
        raise NativeEstimatePlanError("the report names no O3 ring, and a line was admitted to one")
    records = []
    if base.mode == "audit":
        try:
            text = (estimate_dir / RECORDS_FILE).read_text(encoding="ascii")
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        except (OSError, ValueError) as error:
            raise NativeEstimatePlanError(f"{estimate_dir} holds no readable records of the stage O3: {error}") from None
        kept.add(RECORDS_FILE)
        holders = [(line, item) for line, item in zip(lines, expected) if line["landed"]]
        if len(records) != len(holders) or estimate["archived"] != len(records):
            raise NativeEstimatePlanError(f"the output holds {len(records)} records of the stage O3 for {len(holders)} "
                                          "captured estimates")
        descriptors = {descriptor["sequence"]: descriptor for descriptor in base.radio.descriptors}
        for position, (record, (line, item)) in enumerate(zip(records, holders)):
            problems = reference.record_violations(record)
            if problems:
                raise NativeEstimatePlanError(f"O3 record {position}: {problems[0]}")
            host, descriptor = record["host"], descriptors.get(item["sequence"])
            if record["keys"]["sequence"] != position or record["run_id"] != report["run_id"] \
                    or host["opportunity_id"] != item["opportunity_id"] or host["opportunity_sequence"] != item["sequence"] \
                    or record["keys"]["opportunity"] != line["keys"]["opportunity"] or descriptor is None:
                raise NativeEstimatePlanError(f"O3 record {position} does not belong to its opportunity of the report")
            if host["ingress"]["readback_sha256"] != descriptor["grid"]["sha256"] \
                    or host["grant_record_sha256"] != descriptor["grant_record_sha256"] \
                    or host["producer"] != line["producer"] or host["ring_sequence"] != item["estimate"]["ring"] \
                    or (record["radio"]["sfn"], record["radio"]["slot"]) != (descriptor["sfn"], descriptor["slot"]) \
                    or (host["producer"]["sfn"], host["producer"]["slot"]) != (descriptor["sfn"], descriptor["slot"]):
                raise NativeEstimatePlanError(f"O3 record {position} does not name the grid, the grant and the PDU of "
                                              "its opportunity")
            for name in (record["payload"]["file"], record["validity"]["file"]):
                if Path(name).name != name or name in kept:
                    raise NativeEstimatePlanError(f"O3 record {position} names a file outside estimate/ or one that "
                                                  "another record names")
                kept.add(name)
            try:
                payload = (estimate_dir / record["payload"]["file"]).read_bytes()
                mask = (estimate_dir / record["validity"]["file"]).read_bytes()
            except OSError as error:
                raise NativeEstimatePlanError(f"O3 record {position}: {error}") from None
            if len(payload) != record["payload"]["bytes"] or hashlib.sha256(payload).hexdigest() != record["payload"]["digest"] \
                    or item["estimate"]["payload_sha256"] != record["payload"]["digest"]:
                raise NativeEstimatePlanError(f"the payload of O3 record {position} differs from its digest or size")
            bits, valid, holes, nonzero = reference.scan(payload)
            if mask != bits or hashlib.sha256(mask).hexdigest() != record["validity"]["sha256"] \
                    or (valid, holes, nonzero) != (record["validity"]["valid"], record["validity"]["holes"], host["nonzero"]) \
                    or (valid, holes, nonzero) != (line["valid"], line["holes"], line["nonzero"]):
                raise NativeEstimatePlanError(f"the validity bits or the counts of O3 record {position} are not those of "
                                              "its payload")
    extra = sorted(path.name for path in estimate_dir.iterdir() if path.name not in kept or not path.is_file())
    if extra:
        raise NativeEstimatePlanError(f"estimate/ holds {extra[0]}, which no record names")
    return Output(directory, base, lines, records)


def account(output):
    """Where the estimate of every decode of the producing backend ended. balanced: nothing is unaccounted for."""
    states = dict.fromkeys(OUTCOMES, 0)
    for line in output.captures:
        states[line["outcome"]] += 1
    states.update(eligible=len(output.captures), unexpected=output.estimate["unexpected"],
                  overruns=output.estimate["overruns"])
    balanced = states["eligible"] == sum(states[name] for name in OUTCOMES)
    if output.ring is not None:
        counts = output.ring["counts"]
        states.update(released=counts["released"], reclaimed=counts["reclaimed"], pending=counts["pending"],
                      pending_at_cutoff=output.ring["pending_at_cutoff"], faults=counts["faults"])
        balanced = balanced and states["published"] == states["released"] + states["reclaimed"] + states["pending"] \
            and states["pending"] == states["pending_at_cutoff"]
    states["balanced"] = balanced
    return states


def join_archive(output, archive):
    """The records a consumer of the O3 stream kept, against the lines of the producer. Returns (pairs, problems)."""
    by_place = {(line["ring"]["epoch"], line["ring"]["sequence"]): line for line in output.published}
    pairs, problems = [], []
    for record in archive.records:
        header = record.header
        name = obs.record_name(header["epoch"], header["sequence"])
        line = by_place.get((header["epoch"], header["sequence"]))
        if line is None:
            problems.append(f"{name}: the producer published no record at this place")
            continue
        described = record.described()
        what, producer = described["what"], line["producer"]
        try:
            host = obs.staged_host(record.host)
        except ObsError as error:
            problems.append(f"{name}: {error}")
            continue
        if (what["stage"], what["layout"], what["dtype"]) != (reference.STAGE, reference.LAYOUT, reference.DTYPE) \
                or what["producer_steps"] != producer["steps"] or what["evidence"] != output.mode:
            problems.append(f"{name}: the record is not an estimate of the stage O3 with the steps of its producer, "
                            "published in the mode of the run")
        if f"{header['opportunity']:016x}" != line["keys"]["opportunity"] or f"{header['grant']:016x}" != line["keys"]["grant"] \
                or f"{header['episode']:016x}" != line["keys"]["episode"] or header["absolute_slot"] != line["absolute_slot"]:
            problems.append(f"{name}: the join keys or the slot are not those of its account line")
        if (header["t_publish_ns"], header["t_ready_ns"], header["valid"], header["payload_bytes"]) \
                != (line["t_publish_ns"], line["t_complete_ns"], line["valid"], line["payload_bytes"]):
            problems.append(f"{name}: the times, the valid count or the payload size are not those of its account line")
        if (host["grid_index"], host["grid_generation"], host["opportunity_sequence"]) \
                != (line["grid"]["grid_index"], line["grid"]["generation"], line["opportunity_sequence"]):
            problems.append(f"{name}: the host block names another grid ticket or another opportunity")
        if list(record.shape) != [len(producer["dmrs_symbols"]), producer["prb_count"], producer["rx_ports"], 12] \
                or header["first_subcarrier"] != 12 * producer["start_prb"] or header["scale"] != producer["scale"]:
            problems.append(f"{name}: the shape, the first subcarrier or the scale are not those its producer reported")
        if output.mode == "timed" and (header["digest_kind"] != 0 or any(header["digest"])):
            problems.append(f"{name}: a timed record carries a digest")
        if output.mode == "audit" and described["payload"]["digest"] != hashlib.sha256(record.payload).hexdigest():
            problems.append(f"{name}: the digest of an audited record is not the SHA-256 of its payload")
        if reference.scan(record.payload)[:2] != (record.mask, header["valid"]):
            problems.append(f"{name}: the validity bits are not those of the payload")
        pairs.append((line, record))
    return pairs, problems


# ---------------------------------------------------------------- parity
def check_tolerance(tolerance, name="a tolerance"):
    if not isinstance(tolerance, dict) or set(tolerance) != set(TOLERANCE_FIELDS):
        raise NativeEstimatePlanError(f"{name} requires exactly {', '.join(TOLERANCE_FIELDS)}")
    for key, value in tolerance.items():
        if value is not None and (type(value) not in (int, float) or not value >= 0):
            raise NativeEstimatePlanError(f"{name}.{key} is null or a number that is not negative")
    if tolerance["nmse"] is None:
        raise NativeEstimatePlanError(f"{name}.nmse is a number: every comparison has an NMSE bound")
    return tolerance


def compare(o3_values, o3_shape, o2_values, o2_shape, tolerance, smoothed=True):
    """One estimate of the receiver against the matched estimator on the stage O2 of the same slot.

    The verdict is taken on the common physical pilot support. The NMSE bound of a comparison is
    the declared nmse, or the declared fraction of the NMSE between the matched estimator without
    and with its smoothing step when that is larger: an estimate must lie nearer to the smoothed
    estimator than that fraction of what smoothing changes. The other orders in which the
    in-place filter of the receiver may read its neighbours are computed beside it and not graded.
    """
    if [o2_shape[1], o2_shape[2] // reference.PILOTS_PER_PRB, o2_shape[0], 12] != list(o3_shape):
        raise NativeEstimatePlanError(f"an estimate of the shape {list(o3_shape)} and a comb payload of the shape "
                                      f"{list(o2_shape)} do not describe the same allocation")
    matched, shape = reference.matched(o2_values, o2_shape, smoothing=smoothed)
    pilot = reference.parity(o3_values, matched, shape, "pilot")
    every = reference.parity(o3_values, matched, shape, "all")
    raw = reference.matched(o2_values, o2_shape, smoothing=False)[0]
    effect = reference.parity(raw, matched, shape, "pilot")["nmse"] if smoothed else 0.0
    bound = dict(tolerance)
    if tolerance["smoothing_fraction"] is not None:
        bound["nmse"] = max(tolerance["nmse"], tolerance["smoothing_fraction"] * effect)
    in_place = {}
    for order in ("ascending", "descending") if smoothed else ():
        other = reference.parity(reference.matched(o2_values, o2_shape, order=order)[0], matched, shape, "pilot")
        in_place[order] = {"nmse": other["nmse"], "max_fp16_ulps": other["max_fp16_ulps"]}
    return {"passed": reference.within(pilot, bound) and pilot["samples"] > 0, "nmse_bound": bound["nmse"],
            "pilot_support": pilot, "all_elements": every, "smoothing_effect_nmse": effect,
            "unsmoothed_o2": reference.parity(o3_values, raw, shape, "pilot"), "in_place_model": in_place}


def parity_of(output, tolerance, race_free_ulps=1.0):
    """Compare every captured estimate of an audited run with its stage O2. Returns the o3-parity/v1 part of one run.

    The totals are sums over the records: numerator, denominator and sample count on the pilot
    support, so that the NMSE of the run can be recomputed from them.
    """
    check_tolerance(tolerance)
    if output.mode != "audit" or output.observer is None:
        raise NativeEstimatePlanError("the parity is taken on an audited run, which archives both stages")
    o2 = {record["host"]["opportunity_id"]: record for record in output.observer.observations}
    rows, total = [], {"numerator": 0.0, "denominator": 0.0, "samples": 0}
    for record in output.estimates:
        name = record["host"]["opportunity_id"]
        if name not in o2:
            raise NativeEstimatePlanError(f"the opportunity {name} has an estimate of the receiver and no stage O2")
        other = o2[name]
        allocation, wanted = record["allocation"], other["allocation"]
        if other["validity"]["holes"] or wanted["subcarrier_step"] != 2 \
                or wanted["first_subcarrier"] != allocation["first_subcarrier"] \
                or wanted["dmrs_symbols"] != allocation["dmrs_symbols"] or wanted["antenna_map"] != allocation["antenna_map"]:
            raise NativeEstimatePlanError(f"the two stages of {name} do not share one pilot support: O2 needs every "
                                          "second subcarrier of the same allocation, without a hole")
        try:
            result = compare(output.values(record), record["shape"]["shape"], output.observer.values(other),
                             other["shape"]["shape"], tolerance, smoothed="smoothing" in record["what"]["producer_steps"])
        except NativeEstimateError as error:
            raise NativeEstimatePlanError(f"{name}: {error}") from None
        pilot = result["pilot_support"]
        for key in total:
            total[key] += pilot[key]
        rows.append(dict(result, opportunity_id=name, opportunity_sequence=record["host"]["opportunity_sequence"],
                         shape=record["shape"]["shape"], launch=record["host"]["producer"]["launch"],
                         race_free=pilot["max_fp16_ulps"] <= race_free_ulps,
                         o3_sha256=record["payload"]["digest"], o2_sha256=other["payload"]["digest"]))
    total["nmse"] = total["numerator"] / total["denominator"] if total["denominator"] else None
    return {"run_id": output.run_id, "records": len(rows), "passed": bool(rows) and all(row["passed"] for row in rows),
            "race_free_records": sum(row["race_free"] for row in rows), "tolerance": tolerance,
            "race_free_ulps": race_free_ulps, "totals": total, "rows": rows}


def summary(output):
    """What a second reader recovers from a native output of version 3."""
    first = output.estimates[0] if output.estimates else None
    return {"run_id": output.run_id, "mode": output.mode, "estimate": None if output.estimate is None else {
                name: output.estimate[name] for name in ("tap", "stage", "layout", "backend", "copy_stream", "landing",
                                                         "outcomes", "landed", "submit_us", "resolve_us")},
            "account": account(output) if output.estimate else None,
            "first_record": None if first is None else {
                "opportunity_id": first["host"]["opportunity_id"], "shape": first["shape"]["shape"],
                "payload_sha256": first["payload"]["digest"], "producer": first["host"]["producer"],
                "element_0": [output.values(first)[0].real, output.values(first)[0].imag]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a plan of the receiver's estimate, or read a native output.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("plan", help="verify a replay plan of Sprint 2.4 and write a plan of version 3 on it")
    make.add_argument("--replay-plan", required=True, type=Path, help="plan.json of a receiver replay")
    make.add_argument("--run-id", required=True)
    make.add_argument("--mode", choices=sr.MODES, default="audit")
    make.add_argument("--no-estimate", action="store_true", help="the control of the patched build: estimate null")
    make.add_argument("--copy-stream", choices=COPY_STREAMS, default="producer")
    make.add_argument("--landing", choices=LANDINGS, default="slot")
    make.add_argument("--rings", action="store_true", help="publish both stages, each on its own ring")
    make.add_argument("--output", required=True, type=Path, help="the plan file to write")
    show = commands.add_parser("show", help="verify a native output and print its account and its first record")
    show.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            rings = args.rings or args.mode == "timed"
            estimate = None if args.no_estimate else estimate_spec(
                args.copy_stream, args.landing, ring=sr.ring_spec(3, 3, stream="o3") if rings else None)
            plan = build_plan(args.run_id, args.replay_plan, args.mode, sr.ring_spec(3, 3) if rings else None, estimate)
            write_plan(plan, args.output)
            print(json.dumps({"plan": str(args.output), "run_id": plan["run_id"], "evidence": plan["evidence"],
                              "estimate": plan["estimate"]}, indent=2))
            return 0
        print(json.dumps(summary(open_output(args.directory)), indent=2))
        return 0
    except (NativeEstimatePlanError, NativeEstimateError, ObsError) as error:
        parser.exit(2, f"native_estimate: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
