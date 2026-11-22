"""Observations on the ring: the plan, the reader and the account (Sprint 3.2, selector snapshot_ring).

The native tool pilot_ls (native/ocudu_adapter/observer/) reads a second plan version from this
sprint on. pilot-ls-plan/v2 is the plan of Sprint 3.1 with two more objects:

  evidence  the mode of the run. "audit" takes every digest in the loop, as Sprint 3.1 was
            accepted. "timed" takes none between the first fill and the drain of the ring.
  ring      null, or the observation ring the run publishes on: the records a window holds in
            place, the records of one uplink burst, the slot count that follows from the two, the
            wait mode of the consumer, whether the run waits for a consumer, and the drain cutoff.

This module is the offline side. It

  * writes and checks a plan of version 2 around a receiver replay plan of Sprint 2.4;
  * reads a native output of either mode: the radio side with the reader of Sprint 2.4, the
    observer side of an audited run with the reader of Sprint 3.1, and ring/ with its account;
  * balances the account: every eligible observation is published or dropped, and every
    published one is released, reclaimed with a closed epoch or pending at the cutoff;
  * joins the records that a consumer kept to the lines of the producer, and digests their
    payloads. A timed run publishes no digest: the evaluator takes it here, after the run.

Standard library only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import pilot_ls as pl
from . import replay_pusch as rp
from .pilot_ls import PilotLsPlanError
from .replay_pusch import ReplayPuschError
from ..replay import ContractError, load_json
from ..sensing import obs
from ..sensing import pilot_ls as reference
from ..sensing.obs import ObsError
from ..sensing.pilot_ls import PilotLsError

PLAN_SCHEMA = "pilot-ls-plan/v2"
NATIVE_SCHEMA = "pilot-ls-native/v2"
PUBLICATION_SCHEMA = "obs-publication/v1"
RING_SCHEMA = "obs-ring-report/v1"
RING_DIR = "ring"
PUBLICATIONS_FILE = "publications.jsonl"
RING_FILE = "ring.json"
MODES = ("audit", "timed")
WAITS = ("spin", "wait")
ATTACH = ("required", "none")
PLAN_FIELDS = pl.PLAN_FIELDS + ("evidence", "ring")
RING_FIELDS = ("stream", "window_records", "burst_records", "slots", "wait", "attach", "attach_timeout_ms", "drain_ms")
PUBLICATION_FIELDS = ("schema", "run_id", "mode", "opportunity_id", "opportunity_sequence", "outcome", "reason", "epoch",
                      "sequence", "depth", "bytes", "keys", "absolute_slot", "grid", "payload_bytes", "valid", "holes",
                      "t_read_ns", "t_ready_ns", "t_publish_ns", "t_done_ns", "service_us", "estimate_us", "publish_us",
                      "window_calls")
COUNT_FIELDS = ("admitted", "published", "dropped", "cancelled", "released", "reclaimed", "pending", "faults", "epochs",
                "refused", "wakes", "window_calls", "max_depth")
# The ceiling of the tap until Sprint 4.6 calibrates it, and what the breakdown of 2026-11-21 expects.
TAP_CEILING_US = 150.0
TAP_EXPLAIN_US = 20.0
# Measured on spark-ecf8 on 2026-11-21 at commit cf562d6, by a diagnostic on the storage model of the grid.
SPARK_BREAKDOWN_US = {"estimate": 6.7, "no_hash": 9.7, "payload_digest": 97.0, "grid_digest": 1356.4,
                      "accepted_observation": 1471.0}
# Published for the supervised ring of the OCUDU dApp platform on a GB10. Recorded, never graded.
PUBLISHED_RING_US = {"graded": False, "source": "O'Shea et al., arXiv:2609.07805v1",
                     "producer_p50": 1.97, "producer_p999_quiet": 2.34, "producer_p999_cell": 167.0,
                     "consumer_wake_p999_quiet": 99.0, "consumer_wake_p999_cell": 1620.0,
                     "system_calls_per_publication": 3, "payload_bytes": 68000}


class SnapshotRingError(ValueError):
    """A plan, a native output or an archive of the ring violates the documented interface."""


# ---------------------------------------------------------------- plan
def ring_spec(window_records, burst_records, wait="spin", attach="required", attach_timeout_ms=20000, drain_ms=2000,
              stream="o2"):
    """The ring object of a plan. The slot count follows from the window and the burst."""
    try:
        slots = obs.slots_for(window_records, burst_records)
    except ObsError as error:
        raise SnapshotRingError(str(error)) from None
    return {"stream": stream, "window_records": window_records, "burst_records": burst_records, "slots": slots,
            "wait": wait, "attach": attach, "attach_timeout_ms": attach_timeout_ms, "drain_ms": drain_ms}


def _first_version(plan):
    """The part of a plan that is a plan of Sprint 3.1."""
    inner = {key: plan[key] for key in pl.PLAN_FIELDS}
    inner["schema"] = pl.PLAN_SCHEMA
    return inner


def check_plan(plan):
    """Apply every rule of a plan of version 2 that needs no file."""
    if not isinstance(plan, dict) or set(plan) != set(PLAN_FIELDS):
        raise SnapshotRingError(f"a plan requires exactly {', '.join(PLAN_FIELDS)}")
    if plan["schema"] != PLAN_SCHEMA:
        raise SnapshotRingError(f"unsupported plan schema {plan['schema']!r}")
    try:
        pl.check_plan(_first_version(plan))
    except PilotLsPlanError as error:
        raise SnapshotRingError(str(error)) from None
    evidence, ring, tapped = plan["evidence"], plan["ring"], plan["observer"]["tap"] != "none"
    if not isinstance(evidence, dict) or set(evidence) != {"mode"} or evidence["mode"] not in MODES:
        raise SnapshotRingError(f"unknown evidence mode: evidence.mode is \"audit\" or \"timed\"")
    if ring is None:
        if evidence["mode"] == "timed" and tapped:
            raise SnapshotRingError("a timed observer writes its estimate into a slot of the ring: the plan names a tap "
                                    "and no ring")
        return plan
    if not isinstance(ring, dict) or set(ring) != set(RING_FIELDS):
        raise SnapshotRingError(f"ring requires exactly {', '.join(RING_FIELDS)}, or is null")
    if not tapped:
        raise SnapshotRingError("a run without a tap publishes nothing: the plan names a ring and no tap")
    for name, limit in (("window_records", 4096), ("burst_records", 4096), ("attach_timeout_ms", 600000),
                        ("drain_ms", 600000)):
        if type(ring[name]) is not int or not 0 <= ring[name] <= limit:
            raise SnapshotRingError(f"ring.{name} must be an integer from 0 to {limit}")
    if ring["slots"] != obs.slots_for(ring["window_records"], ring["burst_records"]):
        raise SnapshotRingError(f"ring.slots is {ring['slots']}, and a window of {ring['window_records']} records with a "
                                f"burst of {ring['burst_records']} needs "
                                f"{obs.slots_for(ring['window_records'], ring['burst_records'])}: the records a window "
                                "holds in place, plus one burst, plus one, rounded up to a power of two")
    if ring["wait"] not in WAITS:
        raise SnapshotRingError('ring.wait is "spin" or "wait"')
    if ring["attach"] not in ATTACH:
        raise SnapshotRingError('ring.attach is "required" or "none"')
    stream = ring["stream"]
    if not isinstance(stream, str) or not stream or len(stream) > 16 \
            or set(stream) - set("abcdefghijklmnopqrstuvwxyz0123456789"):
        raise SnapshotRingError("ring.stream is at most 16 lower-case letters and digits")
    return plan


def check_files(plan):
    """Verify everything a plan binds, as Sprint 3.1 does. Returns (replay plan, channel plan, sources)."""
    check_plan(plan)
    try:
        return pl.check_files(_first_version(plan))
    except PilotLsPlanError as error:
        raise SnapshotRingError(str(error)) from None


def build_plan(run_id, replay_plan_path, mode="timed", ring=None, tap="pilot_ls", antenna_map=None, epoch=1,
               source=pl.SOURCE):
    """A plan of version 2 around a replay plan file that was written before."""
    try:
        plan = pl.build_plan(run_id, replay_plan_path, tap, antenna_map, epoch, source)
    except PilotLsPlanError as error:
        raise SnapshotRingError(str(error)) from None
    plan.update(schema=PLAN_SCHEMA, evidence={"mode": mode}, ring=ring)
    return check_plan(plan)


def write_plan(plan, path):
    """Check a plan and write it. The file must not exist."""
    path = Path(path)
    if path.exists():
        raise SnapshotRingError(f"{path} already exists. A plan is never overwritten.")
    check_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return path


def load_plan(path):
    try:
        return check_plan(load_json(Path(path)))
    except (ContractError, OSError) as error:
        raise SnapshotRingError(f"cannot read the plan: {error}") from None


def socket_path(native_directory, plan):
    """Where a consumer of a run attaches."""
    return Path(native_directory) / RING_DIR / f"{plan['ring']['stream']}.sock"


# ---------------------------------------------------------------- native output
class Output:
    """A verified native output of a plan of version 2."""

    def __init__(self, directory, report, radio, observer, publications):
        self.directory, self.report, self.radio, self.observer = directory, report, radio, observer
        self.publications = publications
        self.records, self.descriptors = radio.records, radio.descriptors
        self.mode, self.tap, self.ring = report["evidence"]["mode"], report["observer"]["tap"], report["ring"]
        self.run_id = report["run_id"]
        self.published = [line for line in publications if line["outcome"] == "published"]
        self.dropped = [line for line in publications if line["outcome"] == "dropped"]

    def hash_in_window(self):
        return self.report["evidence"]["hash_in_window"]


def publication_violations(line):
    """What is wrong with one line of the producer's account on its own."""
    if not isinstance(line, dict) or set(line) != set(PUBLICATION_FIELDS):
        return ["the line does not hold exactly the fields of an obs-publication/v1 line"]
    found = []
    if line["schema"] != PUBLICATION_SCHEMA or line["mode"] not in MODES:
        found.append("another schema or an unknown evidence mode")
    keys = line["keys"]
    if not isinstance(keys, dict) or set(keys) != {"episode", "opportunity", "grant"} \
            or keys["opportunity"] != reference.opaque_key(line["opportunity_id"]):
        found.append("the opportunity key is not the key of the opportunity")
    published = line["outcome"] == "published"
    if line["outcome"] not in ("published", "dropped"):
        return found + [f"unknown outcome {line['outcome']!r}"]
    numbers = ("epoch", "sequence", "depth", "bytes", "valid", "holes", "t_ready_ns", "t_publish_ns", "estimate_us")
    if published:
        if line["reason"] is not None or any(line[name] is None for name in numbers) or not isinstance(line["grid"], dict):
            found.append("a published observation names its epoch, its sequence, its depth, its grid and its times")
        elif not line["t_read_ns"] <= line["t_ready_ns"] <= line["t_publish_ns"] <= line["t_done_ns"]:
            found.append("the times of a published observation are not in order: read, ready, publish, done")
    elif line["reason"] != "no_credit" or any(line[name] is not None for name in numbers) or line["grid"] is not None:
        found.append("a dropped observation names its reason and nothing of a record")
    if type(line["window_calls"]) is not int or line["window_calls"] < 0 or not line["service_us"] >= 0:
        found.append("the counts and the times are not negative")
    return found


def open_output(directory, schema=NATIVE_SCHEMA):
    """Verify a native output of a plan of version 2 and return it.

    schema is the report schema of this ticket unless a later plan version names its own (a run
    of Sprint 5.2 writes the same files under the schema of its plan version).

    The radio side is verified as Sprint 2.4 verifies it, in both modes: a timed run writes the
    same records after its last opportunity, with the digests that its evaluator side took after
    the drain. An audited run with the tap holds observer/ as Sprint 3.1 wrote it. A timed run
    holds no observer/: its estimates exist in the slots of the ring only. With a ring, ring/
    holds one account line per filled grid, in the order of the opportunities, and the report of
    the ring, whose counts must be those of the lines.
    """
    directory = Path(directory)
    try:
        report = load_json(directory / "native.json")
        mode, tap, ring = report["evidence"]["mode"], report["observer"]["tap"], report["ring"]
    except (ContractError, OSError, KeyError, TypeError) as error:
        raise SnapshotRingError(f"{directory} is not a complete native output of a plan of version 2: {error}") from None
    if report.get("schema") != schema or mode not in MODES or tap not in pl.TAPS:
        raise SnapshotRingError(f"{directory / 'native.json'} is not a {schema} report of a known mode and tap")
    observer = None
    try:
        if mode == "audit":
            observer = pl.open_output(directory, schema=schema)
            radio = observer.radio
        else:
            radio = rp.open_output(directory, schema=schema, taps=(tap,))
            if (directory / pl.OBSERVER_DIR).exists():
                raise SnapshotRingError("a timed run holds observer/: its estimates exist in the slots of the ring only")
            if report["receiver"]["tap"] != tap or any(record["tap"] != tap for record in radio.records):
                raise SnapshotRingError("the report and the radio records name different taps")
    except (PilotLsPlanError, PilotLsError, ReplayPuschError) as error:
        raise SnapshotRingError(str(error)) from None
    ring_dir = directory / RING_DIR
    if ring is None:
        if ring_dir.exists():
            raise SnapshotRingError("the report names no ring, and the output holds ring/")
        return Output(directory, report, radio, observer, [])
    try:
        lines = [json.loads(line) for line in (ring_dir / PUBLICATIONS_FILE).read_text(encoding="ascii").splitlines()
                 if line.strip()]
        written = load_json(ring_dir / RING_FILE)
    except (OSError, ValueError, ContractError) as error:
        raise SnapshotRingError(f"{ring_dir} holds no readable account of the ring: {error}") from None
    if written != ring or ring.get("schema") != RING_SCHEMA or set(ring.get("counts", ())) != set(COUNT_FIELDS):
        raise SnapshotRingError("ring.json is not the ring of the report, or is not an obs-ring-report/v1 record")
    if ring["definition"] != f"{obs.DEFINITION:016x}":
        raise SnapshotRingError(f"the ring was built from the record definition {ring['definition']}, and this reader "
                                f"from {obs.DEFINITION:016x}")
    filled = [item for item in report["opportunities"] if item.get("grid_index") is not None]
    if len(lines) != len(filled) or ring["eligible"] != len(lines) or report["observer"]["records"] != len(lines):
        raise SnapshotRingError(f"the account holds {len(lines)} lines for {len(filled)} filled grids")
    last = {}
    for position, (line, item) in enumerate(zip(lines, filled)):
        problems = publication_violations(line)
        if problems:
            raise SnapshotRingError(f"account line {position}: {problems[0]}")
        if line["run_id"] != report["run_id"] or line["mode"] != mode or line["opportunity_id"] != item["opportunity_id"] \
                or line["opportunity_sequence"] != item["sequence"]:
            raise SnapshotRingError(f"account line {position} does not belong to opportunity {item['sequence']} of the "
                                    "report")
        if line["outcome"] == "published":
            # First in, first out: within an epoch the sequences rise by one, from zero.
            if line["sequence"] != last.get(line["epoch"], -1) + 1:
                raise SnapshotRingError(f"account line {position} has the sequence {line['sequence']} of epoch "
                                        f"{line['epoch']}, where {last.get(line['epoch'], -1) + 1} follows")
            last[line["epoch"]] = line["sequence"]
            if line["depth"] >= ring["slot_count"] or line["grid"] != {"grid_index": item["grid_index"],
                                                                       "generation": item["generation"]}:
                raise SnapshotRingError(f"account line {position} names a depth beyond the ring or another grid than its "
                                        "opportunity was filled with")
    counts = ring["counts"]
    published = sum(line["outcome"] == "published" for line in lines)
    if counts["published"] != published or counts["dropped"] != len(lines) - published \
            or counts["admitted"] != published + counts["cancelled"]:
        raise SnapshotRingError("the counts of the ring are not those of its account lines")
    extra = sorted(path.name for path in ring_dir.iterdir() if path.name not in (PUBLICATIONS_FILE, RING_FILE))
    if extra:
        raise SnapshotRingError(f"ring/ holds {extra[0]} after the run: the producer removes its socket")
    return Output(directory, report, radio, observer, lines)


def account(output):
    """Where every eligible observation of a run ended. balanced is True when nothing is unaccounted for.

    Every eligible observation is published or dropped. Every published one is released by a
    consumer, reclaimed when its epoch closed before it was released, or pending at the cutoff.
    """
    counts, ring = output.ring["counts"], output.ring
    states = {"eligible": ring["eligible"], "published": counts["published"], "dropped": counts["dropped"],
              "released": counts["released"], "reclaimed": counts["reclaimed"], "pending": counts["pending"],
              "pending_at_cutoff": ring["pending_at_cutoff"], "cancelled": counts["cancelled"],
              "faults": counts["faults"], "epochs": counts["epochs"]}
    states["balanced"] = (states["eligible"] == states["published"] + states["dropped"]
                          and states["published"] == states["released"] + states["reclaimed"] + states["pending"]
                          and states["pending"] == states["pending_at_cutoff"] and states["cancelled"] == 0)
    return states


def join_archive(output, archive):
    """The records a consumer kept, against the lines of the producer. Returns (pairs, problems).

    A pair is (account line, record). A record must be the one its line describes: the epoch and
    the sequence, the opportunity key, the publication time, the valid count, the payload size,
    the ticket of the grid in its host block and the evidence mode of the run. In a timed run it
    carries no digest. In an audited run its digest is the SHA-256 of its payload.
    """
    by_place = {(line["epoch"], line["sequence"]): line for line in output.published}
    pairs, problems = [], []
    for record in archive.records:
        header = record.header
        line = by_place.get((header["epoch"], header["sequence"]))
        name = obs.record_name(header["epoch"], header["sequence"])
        if line is None:
            problems.append(f"{name}: the producer published no record at this place")
            continue
        try:
            host = obs.staged_host(record.host)
        except ObsError as error:
            problems.append(f"{name}: {error}")
            continue
        described = record.described()
        if f"{header['opportunity']:016x}" != line["keys"]["opportunity"] or f"{header['grant']:016x}" != line["keys"]["grant"] \
                or f"{header['episode']:016x}" != line["keys"]["episode"] or header["absolute_slot"] != line["absolute_slot"]:
            problems.append(f"{name}: the join keys or the slot are not those of its account line")
        if (header["t_publish_ns"], header["t_ready_ns"], header["valid"], header["payload_bytes"]) \
                != (line["t_publish_ns"], line["t_ready_ns"], line["valid"], line["payload_bytes"]):
            problems.append(f"{name}: the times, the valid count or the payload size are not those of its account line")
        if (host["grid_index"], host["grid_generation"], host["opportunity_sequence"]) \
                != (line["grid"]["grid_index"], line["grid"]["generation"], line["opportunity_sequence"]):
            problems.append(f"{name}: the host block names another grid ticket or another opportunity")
        if described["what"]["evidence"] != output.mode:
            problems.append(f"{name}: the record was published in the mode {described['what']['evidence']}")
        if output.mode == "timed" and (header["digest_kind"] != 0 or any(header["digest"])):
            problems.append(f"{name}: a timed record carries a digest")
        if output.mode == "audit" and (described["payload"]["digest_kind"] != "sha256"
                                       or described["payload"]["digest"] != hashlib.sha256(record.payload).hexdigest()):
            problems.append(f"{name}: the digest of an audited record is not the SHA-256 of its payload")
        if sum(bin(byte).count("1") for byte in record.mask) != header["valid"]:
            problems.append(f"{name}: the record counts other valid elements than its bits set")
        pairs.append((line, record))
    return pairs, problems


def payload_digests(pairs):
    """The SHA-256 of every kept payload and of its validity bits, by opportunity: taken offline, by the evaluator."""
    return {line["opportunity_id"]: (hashlib.sha256(record.payload).hexdigest(), hashlib.sha256(record.mask).hexdigest())
            for line, record in pairs}


def spread(values):
    """Count, mean, median, 99th and 99.9th percentile and largest of a list of numbers."""
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean": sum(values) / len(values), "p50": obs.percentile(values, 0.5),
            "p99": obs.percentile(values, 0.99), "p999": obs.percentile(values, 0.999), "max": max(values)}


def tap_times(output):
    """The service time of the tap and its two parts over the published observations, the first one excluded."""
    lines = output.published[1:]
    return {name: spread([line[name] for line in lines]) for name in ("service_us", "estimate_us", "publish_us")}


def summary(output):
    """What a second reader recovers from a native output of version 2: the mode, the account and the tap."""
    states = account(output) if output.ring else None
    return {"run_id": output.run_id, "mode": output.mode, "tap": output.tap, "radio_records": len(output.records),
            "hash_in_window": output.hash_in_window(), "account": states,
            "tap_us": tap_times(output) if output.ring else None,
            "ring": None if output.ring is None else {name: output.ring[name] for name in
                                                      ("slot_count", "slot_bytes", "segment_bytes", "wait", "definition")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a plan of the observation ring, or read a native output.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("plan", help="verify a replay plan of Sprint 2.4 and write a plan of version 2 on it")
    make.add_argument("--replay-plan", required=True, type=Path, help="plan.json of a receiver replay")
    make.add_argument("--run-id", required=True)
    make.add_argument("--mode", choices=MODES, default="timed")
    make.add_argument("--tap", choices=pl.TAPS, default="pilot_ls")
    make.add_argument("--no-ring", action="store_true", help="publish on no ring (an audited or an untapped run)")
    make.add_argument("--window-records", type=int, default=3)
    make.add_argument("--burst-records", type=int, default=3)
    make.add_argument("--wait", choices=WAITS, default="spin")
    make.add_argument("--attach", choices=ATTACH, default="required")
    make.add_argument("--output", required=True, type=Path, help="the plan file to write")
    show = commands.add_parser("show", help="verify a native output and print its mode, its account and its tap times")
    show.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            ring = None if args.no_ring or args.tap == "none" else ring_spec(
                args.window_records, args.burst_records, args.wait, args.attach)
            plan = build_plan(args.run_id, args.replay_plan, args.mode, ring, args.tap)
            write_plan(plan, args.output)
            print(json.dumps({"plan": str(args.output), "run_id": plan["run_id"], "evidence": plan["evidence"],
                              "ring": plan["ring"], "observer": plan["observer"]}, indent=2))
            return 0
        print(json.dumps(summary(open_output(args.directory)), indent=2))
        return 0
    except (SnapshotRingError, ObsError) as error:
        parser.exit(2, f"snapshot_ring: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
