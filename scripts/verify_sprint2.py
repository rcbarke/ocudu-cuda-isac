#!/usr/bin/env python3
"""Audit the chain of the four accepted Sprint 2 runs: one known waveform, from its record to the decoded bytes.

Each Sprint 2 gate graded its own evidence. This audit reads the four archived runs together and
checks what no single gate can: that the grant record of Sprint 2.1 is the one the pair of Sprint
2.2 was built on, that the channel application of Sprint 2.3 filled its grids from exactly those
transmit grids, channels and noise with the gain of the entry, that the receiver of Sprint 2.4
decoded those same grids, and that the bytes it returned are the transport block that Sprint 2.1
archived, on the CPU and on the resident CUDA path. It reads files and compares hashes and bytes.
It starts no native tool, needs the standard library only and writes nothing into the four runs.

    python scripts/verify_sprint2.py --s24-run results/runs/s24-spark-001 --output-dir results/runs/s2-spark-001/chain

The Sprint 2.3 and 2.2 runs are the ones the Sprint 2.4 run names, and the Sprint 2.1 run the one
the Sprint 2.2 run names. Name a run that was moved with --s23-run, --s22-run or --s21-run: it must
have the name that was recorded. The output directory must be new. The exit status is 0 only when
every check passed, 1 when one failed and 2 when the arguments are unusable.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shlex
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes import schedule as schedules  # noqa: E402
from ocudu_cuda_isac.episodes.episode import EpisodeError  # noqa: E402
from ocudu_cuda_isac.episodes.grant import GrantError  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
from ocudu_cuda_isac.episodes.replay_pusch import ReplayPuschError  # noqa: E402

GRADE_SCHEMA = "sprint-audit/s2/v1"
SPRINTS = ("2.1", "2.2", "2.3", "2.4")
SELECTORS = {"2.1": "grant_fixture", "2.2": "episode_pair", "2.3": "grid_apply", "2.4": "replay_pusch"}
STAGES = ("verdicts", "links", "build", "waveform", "channel", "receiver", "payload", "bad-input", "sources")
ENTRY = "episode_pair/reference_pair/store"
READ_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, GrantError, EpisodeError, GridApplyError,
               ReplayPuschError)


class AuditError(ValueError):
    """The arguments or a run directory are unusable. Nothing was graded."""


class Grader:
    def __init__(self):
        self.results, self.lines = [], []

    def check(self, stage, name, passed, detail):
        self.results.append({"stage": stage, "check": name, "passed": bool(passed), "detail": detail})
        line = ("PASS" if passed else "FAIL") + f"  S2 {stage} {name}: {detail}"
        self.lines.append(line)
        print(line)
        return bool(passed)

    def note(self, stage, text):
        line = f"NOTE  S2 {stage} {text}"
        self.lines.append(line)
        print(line)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return sha256(Path(path).read_bytes())


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def moment(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class Run:
    """One archived gate run: its verdict, its adapter receipt and the manifest and grade of its driver."""

    def __init__(self, sprint, directory):
        self.sprint, self.directory, self.selector = sprint, Path(directory), SELECTORS[sprint]
        self.name = self.directory.name
        self.evidence = self.directory / self.selector
        try:
            self.summary = (self.directory / "summary.txt").read_text(encoding="utf-8")
        except OSError:
            self.summary = ""

    def verdict(self):
        """(accepted, profile, host): the last line of the summary decides, and a failed line overrides it."""
        found = re.search(rf"^Sprint {re.escape(self.sprint)}: ACCEPTED on this (DGX Spark|host)", self.summary, re.M)
        opened = re.match(rf"Sprint {re.escape(self.sprint)} acceptance on ([^\s,]+)", self.summary)
        clean = not re.search(r"^FAIL  ", self.summary, re.M) and "NOT ACCEPTED" not in self.summary \
            and "REHEARS" not in self.summary
        return bool(found) and clean, found[1] if found else None, opened[1] if opened else None

    def manifest(self):
        return load(self.evidence / "manifest.json")

    def grade(self):
        return load(self.evidence / "grade.json")

    def receipt(self):
        return load(self.directory / "adapter-receipt.json")


def locate(args):
    """The four runs: the Sprint 2.4 run, and each earlier run from the record of the run that consumed it."""
    s24 = Run("2.4", args.s24_run.resolve())
    if not (s24.evidence / "manifest.json").is_file():
        raise AuditError(f"{s24.directory} holds no replay_pusch/manifest.json: it is not a Sprint 2.4 run")
    names, runs = {}, {"2.4": s24}
    try:
        manifest = s24.manifest()
        names["2.3"], names["2.2"] = Path(manifest["s23_run"]), Path(manifest["s22_run"])
    except READ_ERRORS as error:
        raise AuditError(f"the Sprint 2.4 run does not name the runs it consumed: {error}") from None
    for sprint, given in (("2.3", args.s23_run), ("2.2", args.s22_run)):
        runs[sprint] = Run(sprint, (given or names[sprint]).resolve())
    try:
        names["2.1"] = Path(runs["2.2"].manifest()["grant_store"]).parent
    except READ_ERRORS as error:
        raise AuditError(f"the Sprint 2.2 run {runs['2.2'].directory} does not name its grant store: {error}") from None
    runs["2.1"] = Run("2.1", (args.s21_run or names["2.1"]).resolve())
    try:
        names["2.3 stimulus"] = Path(runs["2.3"].manifest()["s22_run"])
    except READ_ERRORS as error:
        raise AuditError(f"the Sprint 2.3 run {runs['2.3'].directory} does not name its stimulus: {error}") from None
    return runs, names


# ---------------------------------------------------------------- verdicts, links, build
def check_verdicts(grader, runs):
    verdicts = {sprint: runs[sprint].verdict() for sprint in SPRINTS}
    profiles = {profile for _, profile, _ in verdicts.values()}
    hosts = {host for _, _, host in verdicts.values()}
    accepted_all = all(accepted for accepted, _, _ in verdicts.values())
    one_host = len(profiles) == 1 and len(hosts) == 1 and None not in hosts
    ok = accepted_all and one_host
    grader.check("verdicts", "four gates", ok,
                 "; ".join(f"S{sprint} {runs[sprint].name}: " + (f"ACCEPTED on this {profile}" if accepted else
                                                                 "no ACCEPTED verdict without a failed line")
                           for sprint, (accepted, profile, _) in verdicts.items())
                 + f"; host {', '.join(sorted(str(host) for host in hosts))}"
                 + (": the four runs are not of one host and profile" if accepted_all and not one_host else ""))
    shown, ok, gate = [], True, "unreadable"
    for sprint in SPRINTS:
        try:
            grade = runs[sprint].grade()
            if sprint == "2.4":
                gate = str(grade.get("hardware_gate")).split(":")[0]
            good = grade["judgment"] == "pass" and grade["cases"] >= 1 and grade["cases_passed"] == grade["cases"] \
                and grade["checks_failed"] == 0 and grade["checks_passed"] == grade["checks"] > 0 \
                and (sprint != "2.4" or grade.get("hardware_gate") == "closed")
            shown.append(f"S{sprint} {grade['cases_passed']} of {grade['cases']} cases and {grade['checks_passed']} of "
                         f"{grade['checks']} checks, judgment {grade['judgment']}")
        except READ_ERRORS as error:
            good = False
            shown.append(f"S{sprint} grade unreadable: {error}")
        ok = ok and good
    grader.check("verdicts", "grades", ok, "; ".join(shown) + f"; the hardware gate of S2.4 is {gate}"
                 + ("" if gate == "closed" else ": a rehearsal or a fallback is not an acceptance"))
    profile = next(iter(profiles)) if len(profiles) == 1 else None
    return profile


def check_links(grader, runs, names):
    wanted = (("S2.2 built its pair on the grant store of", "2.1", names["2.1"]),
              ("S2.3 applied the stimulus of", "2.2", names["2.3 stimulus"]),
              ("S2.4 decoded the stimulus of", "2.2", names["2.2"]),
              ("S2.4 compared its grids with those of", "2.3", names["2.3"]))
    wrong = [f"{what} {path.name}, and the audit reads {runs[sprint].name}" for what, sprint, path in wanted
             if path.name != runs[sprint].name]
    grader.check("links", "each gate consumed its predecessor", not wrong,
                 "; ".join(f"{what} {runs[sprint].name}" for what, sprint, _ in wanted) if not wrong else "; ".join(wrong))
    times, order = {}, []
    try:
        for sprint in SPRINTS:
            manifest = runs[sprint].manifest()
            times[sprint] = (moment(manifest["started_utc"]), moment(manifest["completed_utc"]))
        for producer, consumer in (("2.1", "2.2"), ("2.2", "2.3"), ("2.3", "2.4")):
            if times[consumer][0] < times[producer][1]:
                order.append(f"S{consumer} started before S{producer} completed")
        grader.check("links", "order", not order,
                     ", ".join(f"S{sprint} completed {times[sprint][1]:%Y-%m-%dT%H:%M:%SZ}" for sprint in SPRINTS)
                     + ": no gate started before the run it consumes completed" if not order else "; ".join(order))
    except READ_ERRORS as error:
        grader.check("links", "order", False, f"a manifest names no start or completion time: {error}")


def check_build(grader, runs):
    try:
        receipts = {sprint: runs[sprint].receipt() for sprint in SPRINTS}
    except READ_ERRORS as error:
        grader.check("build", "one receiver build", False, f"an adapter receipt is unreadable: {error}")
        grader.check("build", "unchanged tools", False, "an adapter receipt is unreadable")
        return {}
    facts = {sprint: (receipt.get("ocudu_build"), receipt.get("ocudu_receipt_sha256"),
                      receipt["pin"].get("link_txt_sha256"), receipt["pin"].get("flags_make_sha256"),
                      receipt["pin"].get("cuda")) for sprint, receipt in receipts.items()}
    same = len(set(facts.values())) == 1
    build, native, link, _, cuda = facts["2.4"]
    grader.check("build", "one receiver build", same and cuda is True,
                 f"the four adapters were compiled and linked against one OCUDU build ({build}): the same build "
                 f"receipt {str(native)[:12]}, the same link line {str(link)[:12]} and the same compile flags of the "
                 "pinned test, built with CUDA" if same and cuda is True else
                 "the adapters of the four runs were not built against one CUDA build: "
                 + "; ".join(f"S{sprint} {values[0]} receipt {str(values[1])[:12]} link {str(values[2])[:12]} "
                             f"cuda {values[4]}" for sprint, values in facts.items()))
    tools = {sprint: {name: entry["sha256"] for name, entry in receipt["executables"].items()}
             for sprint, receipt in receipts.items()}
    fixture = {values.get("grant_fixture") for values in tools.values()}
    apply = {tools[sprint].get("grid_apply") for sprint in ("2.3", "2.4")}
    warnings = {sprint: receipt.get("compiler_warnings") for sprint, receipt in receipts.items()}
    ok = len(fixture) == 1 and None not in fixture and len(apply) == 1 and None not in apply \
        and "replay_pusch" in tools["2.4"] and all(value == 0 for value in warnings.values())
    grader.check("build", "unchanged tools", ok,
                 f"grant_fixture {str(next(iter(fixture)))[:12]} is the same executable in the four runs and grid_apply "
                 f"{str(next(iter(apply)))[:12]} in S2.3 and S2.4, beside replay_pusch "
                 f"{str(tools['2.4'].get('replay_pusch'))[:12]}; no compiler warning in any adapter build" if ok else
                 "a tool changed between the runs, so an earlier grid was not made by the tool that S2.4 used: "
                 + "; ".join(f"S{sprint} " + ", ".join(f"{name} {value[:12]}" for name, value in sorted(values.items()))
                             + f", {warnings[sprint]} warnings" for sprint, values in tools.items())
                 + ". Run the four gates again in order: make accept-s2 RERUN=1")
    return tools["2.4"]


# ---------------------------------------------------------------- waveform and channel
def same_binding(plan, built):
    """Differences between an archived channel plan and the plan built here from the entry. Paths are not compared."""
    found = []
    if plan.get("gain", {}).get("value") != built["gain"]["value"] or plan.get("gain", {}).get("source") != "entry":
        found.append(f"gain {plan.get('gain')}")
    if plan.get("carrier") != built["carrier"]:
        found.append("carrier")
    ours = {item["grant_id"]: (item["record_sha256"], item["x"]["sha256"], [ref["sha256"] for ref in item["active"]])
            for item in built["grants"]}
    theirs = {item["grant_id"]: (item["record_sha256"], item["x"]["sha256"], [ref["sha256"] for ref in item["active"]])
              for item in plan.get("grants", [])}
    found += [f"grant {name}" for name in sorted(set(ours) | set(theirs)) if ours.get(name) != theirs.get(name)]
    ours = {item["realization_id"]: (item["h"]["sha256"], item["w"]["sha256"]) for item in built["sources"]}
    theirs = {item["realization_id"]: (item["h"]["sha256"], item["w"]["sha256"]) for item in plan.get("sources", [])}
    found += [f"realization {name}" for name in sorted(set(ours) | set(theirs)) if ours.get(name) != theirs.get(name)]
    fields = ("opportunity_id", "realization_id", "index", "grant_id", "sfn", "slot")
    if [[item[key] for key in fields] for item in plan.get("opportunities", [])] \
            != [[item[key] for key in fields] for item in built["opportunities"]]:
        found.append("opportunities")
    return found


def check_waveform(grader, runs, roots):
    """The grant record, the transmit grids and the pair. Returns what the later stages compare with."""
    entry_dir = runs["2.2"].directory / ENTRY
    search = [runs["2.2"].directory / "episode_pair/grant_fixture", runs["2.1"].directory / "grant_fixture", *roots]
    try:
        episode = episodes.open_episode(entry_dir)
        base = episode.grant
        store = grants.open_store(runs["2.1"].directory / "grant_fixture" / base.grant_id / "store")
        record = schedules.record_sha256(store.grant.record())
        bound = episode.manifest["grants"][base.grant_id]["record_sha256"]
        block = (store.directory / "tb.bin").read_bytes()
        ok = record == bound and block == base.tb == store.grant.tb
        grader.check("waveform", "grant record", ok,
                     f"the store entry of {base.grant_id} in the Sprint 2.1 run verifies against its manifest, and its "
                     f"record with its transport block has the hash {record[:16]} that the pair of Sprint 2.2 binds; "
                     f"the block is {len(block)} bytes, SHA-256 {sha256(block)[:16]}" if ok else
                     f"the record of {base.grant_id} in the Sprint 2.1 run has the hash {record[:16]}, and the pair of "
                     f"Sprint 2.2 was built on {str(bound)[:16]}")
    except READ_ERRORS as error:
        grader.check("waveform", "grant record", False, f"{type(error).__name__}: {error}")
        return None
    try:
        # entry_plan opens the store entry of every grant and refuses a grid whose hash the entry does not bind.
        built = replay.entry_plan(f"s2-audit-{runs['2.2'].name}", [entry_dir], search)
        stores = {item["grant_id"]: grants.open_store(Path(item["x"]["path"]).parent) for item in built["grants"]}
        slots = sorted(stores[name].grant.slot for name in stores)
        same_block = all(item.grant.tb == block for item in stores.values())
        files = episode.manifest["files"]
        ok = len(built["grants"]) == len(episode.schedule.grants) and same_block \
            and built["gain"]["value"] == episode.manifest["calibration"]["grid_gain"]
        grader.check("waveform", "transmit grids and pair", ok,
                     f"the entry {episode.episode_id} verifies against its manifest and its grant schedule: "
                     f"{len(episode.schedule.opportunities)} opportunities per case on {len(stores)} grants (slots "
                     f"{', '.join(str(slot) for slot in slots)}); the store entry of every grant verifies and holds "
                     f"the transmit grid whose hash the entry binds; H of both cases and W are "
                     + ", ".join(f"{name} {item['sha256'][:12]}" for name, item in sorted(files.items()))
                     + f"; every grant carries the block of {base.grant_id}; g = {built['gain']['value']:.4f}")
    except READ_ERRORS as error:
        grader.check("waveform", "transmit grids and pair", False, f"{type(error).__name__}: {error}")
        return None
    return {"episode": episode, "built": built, "stores": stores, "block": block, "base": base,
            "block_path": store.directory / "tb.bin"}


def check_channel(grader, runs, known):
    """The plan and the stored grids of Sprint 2.3. Returns the readback digest of every opportunity."""
    pair = runs["2.3"].evidence / "reference_pair"
    built = known["built"]
    try:
        plan = load(pair / "plan.json")
        output = replay.open_output(pair / "native")
        different = same_binding(plan, built)
        bound = output.report["plan"]["sha256"] == digest(pair / "plan.json")
        ok = not different and bound and output.report["gain"]["value"] == built["gain"]["value"]
        grader.check("channel", "plan of Sprint 2.3", ok,
                     f"the plan that grid_apply read ({output.report['plan']['sha256'][:16]}, the hash its report "
                     f"names) binds the transmit grid and the record of each of {len(built['grants'])} grants, H and W "
                     f"of {len(built['sources'])} realizations and {len(built['opportunities'])} opportunities exactly "
                     f"as the plan built here from the verified entry, with the entry's gain {built['gain']['value']:.4f}"
                     if ok else "the archived plan of Sprint 2.3 differs from the entry in: "
                     + (", ".join(different) or "the hash its report names or the gain"))
        stored = {item["opportunity_id"]: item["readback_sha256"] for item in output.report["opportunities"]}
        count = len(built["opportunities"])
        ok = len(stored) == count == len(output.descriptors) \
            and [item["opportunity_id"] for item in output.report["opportunities"]] \
            == [item["opportunity_id"] for item in built["opportunities"]]
        grader.check("channel", "stored grids", ok,
                     f"{len(output.descriptors)} stored grids of Sprint 2.3 verify against their readback digests, one "
                     f"per opportunity of the plan, in its order")
        return stored
    except READ_ERRORS as error:
        grader.check("channel", "plan of Sprint 2.3", False, f"{type(error).__name__}: {error}")
        return None


# ---------------------------------------------------------------- receiver
def native_runs(s24):
    """The native runs of the Sprint 2.4 driver, by name, each opened with the reader. A run that fails says why."""
    found = {}
    for item in s24.manifest()["runs"]:
        directory = s24.evidence / item["directory"]
        try:
            found[item["name"]] = (directory, rp.open_output(directory / "native"), None)
        except READ_ERRORS as error:
            found[item["name"]] = (directory, None, f"{type(error).__name__}: {error}")
    return found


def check_receiver(grader, runs, known, stored, opened, chain):
    built, block, base = known["built"], known["block"], known["base"]
    directory, output, problem = opened.get("reference-pair", (None, None, "the manifest lists no run reference-pair"))
    if output is None:
        for name in ("same grids", "original bytes", "executed path"):
            grader.check("receiver", name, False, f"the run of the pair does not verify: {problem}")
        return
    count = len(built["opportunities"])
    try:
        different = same_binding(load(directory / "channel-plan.json"), built)
        plan = load(directory / "plan.json")
        bound = plan["channel_plan"]["sha256"] == digest(directory / "channel-plan.json") \
            and output.report["plan"]["sha256"] == digest(directory / "plan.json")
        decoded = {item["opportunity_id"]: item["readback_sha256"] for item in output.report["opportunities"]}
        equal = sum(decoded.get(key) == value for key, value in (stored or {}).items())
        records = sum(record["ingress"]["readback_sha256"] == decoded.get(record["opportunity_id"])
                      for record in output.records)
        ok = not different and bound and stored is not None and equal == count == len(decoded) \
            and records == len(output.records)
        grader.check("receiver", "same grids", ok,
                     f"the receiver run read a channel plan that binds the same files as the entry, by the hash its "
                     f"plan names; {equal} of {count} grids it decoded have the readback digest of the grid that "
                     f"Sprint 2.3 stored for the same opportunity, and all {records} radio records name the digest of "
                     "their own opportunity" if ok else
                     f"{equal} of {count} decoded grids equal those of Sprint 2.3; plan differences: "
                     f"{', '.join(different) or 'none'}; plan hashes bound: {bound}")
    except READ_ERRORS as error:
        grader.check("receiver", "same grids", False, f"{type(error).__name__}: {error}")
    # The decoded bytes, read from the files of the run, against the block that Sprint 2.1 archived.
    backends = output.backends()
    returned = {backend: 0 for backend in backends}
    for record in output.records:
        data = output.decoded(record)
        good = record["outcome"] == "decoded" and record["crc_ok"] is True and data == block
        returned[record["backend"]] += good
        chain.setdefault(record["opportunity_id"], {}).update({
            f"decoded_sha256_{record['backend']}": record["decoded_sha256"], f"outcome_{record['backend']}": record["outcome"],
            f"executed_{record['backend']}": record["path"]["executed"] if isinstance(record["path"], dict) else None})
    ok = backends == list(rp.BACKENDS) and all(value == count for value in returned.values()) \
        and len(output.records) == count * len(backends) == len(output.by_key)
    grader.check("receiver", "original bytes", ok,
                 "; ".join(f"{backend} {value} of {count}" for backend, value in returned.items())
                 + f" decodes of the pair end as decoded with the decoder's CRC pass, and the bytes in the files of the "
                   f"run are the {len(block)} bytes of tb.bin in the Sprint 2.1 store of {base.grant_id}, compared here "
                   f"byte for byte; {len(output.records)} records under {len(output.by_key)} unique keys")
    # The path every decode executed, in every native run of the gate.
    paths, wrong, unbuilt = {backend: {} for backend in rp.BACKENDS}, [], []
    for name, (_, run, _) in sorted(opened.items()):
        if run is None:
            continue
        if run.report.get("built_with_cuda") is not True:
            unbuilt.append(name)
        for record in run.records:
            executed = record["path"]["executed"] if isinstance(record["path"], dict) else None
            paths.setdefault(record["backend"], {})
            paths[record["backend"]][executed] = paths[record["backend"]].get(executed, 0) + 1
            ports = len(record["allocation"]["rx_ports"])
            wanted = "resident" if record["backend"] == "cuda" and ports in rp.RESIDENT_PORTS else "host"
            if executed != wanted:
                wrong.append(f"{name} {record['key']} on the {executed} path")
    shown = "; ".join(f"{backend}: " + ", ".join(f"{number} on the {path} path" for path, number in sorted(found.items(), key=str))
                      for backend, found in paths.items())
    ok = not wrong and not unbuilt and paths["cuda"].get("resident", 0) > 0 and len(opened) > 0 \
        and all(run is not None for _, run, _ in opened.values())
    grader.check("receiver", "executed path", ok,
                 f"in the {len(opened)} native runs of the gate, {shown}: every decode requested from the CUDA processor "
                 "ran its resident path and every decode requested from the host processor the host path, on a tool "
                 "built with CUDA" if ok else
                 f"{shown}; {len(wrong)} decodes did not execute the requested path"
                 + (f", first {wrong[0]}" if wrong else "") + (f"; built without CUDA: {', '.join(unbuilt)}" if unbuilt else "")
                 + "".join(f"; {name} does not verify: {problem}" for name, (_, run, problem) in opened.items() if run is None))


def check_accounting(grader, opened):
    planned = records = anomalies = held = 0
    outcomes = {name: 0 for name in rp.OUTCOMES}
    unique, unread = True, []
    for name, (_, run, problem) in sorted(opened.items()):
        if run is None:
            unread.append(f"{name}: {problem}")
            continue
        planned += run.report["totals"]["planned"]
        records += len(run.records)
        anomalies += len(run.report["anomalies"])
        held += run.report["totals"]["grids_not_released"]
        unique = unique and len(run.by_key) == len(run.records)
        for record in run.records:
            outcomes[record["outcome"]] += 1
    ok = not unread and planned == records > 0 and unique and anomalies == 0 and held == 0 \
        and sum(outcomes.values()) == planned and outcomes["decoded"] + outcomes["crc_fail"] == planned
    grader.check("receiver", "every decode accounted for", ok,
                 f"{planned} decodes planned in {len(opened)} runs and {records} records, each under a key that is unique "
                 "in its run, each with one terminal outcome: "
                 + ", ".join(f"{number} {name}" for name, number in outcomes.items())
                 + f"; {anomalies} anomalies, {held} grids held" + "".join(f"; {text}" for text in unread))


def check_payload(grader, known, opened):
    directory, output, problem = opened.get("payload", (None, None, "the manifest lists no run payload"))
    if output is None:
        grader.check("payload", "a block per opportunity", False, f"the payload run does not verify: {problem}")
        return
    try:
        payload = load(directory / "plan.json")["payload"]
        seed, size = payload["seed"], len(known["block"])
        count = len(known["episode"].schedule.opportunities)
        own, blocks = 0, {backend: set() for backend in output.backends()}
        for record in output.records:
            index = int(record["opportunity_id"].rpartition(".")[2])
            data = output.decoded(record)
            own += record["outcome"] == "decoded" and data == rp.opportunity_block(seed, index, size)
            blocks[record["backend"]].add(record["decoded_sha256"])
        ok = payload["rule"] == "per_opportunity" and payload["generator"] == grants.TB_GENERATOR \
            and payload["blocks"] == count and own == len(output.records) == 2 * count * len(blocks) \
            and all(len(found) == count for found in blocks.values()) \
            and sha256(known["block"]) not in set().union(*blocks.values())
        grader.check("payload", "a block per opportunity", ok,
                     f"the payload rule {payload['rule']} with the generator {payload['generator']} and the seed "
                     f"'{seed}' is recorded in the plan; {own} of {len(output.records)} decodes return the block that "
                     f"the rule gives for their own opportunity, generated here again from the seed alone; "
                     + "; ".join(f"{backend} returned {len(found)} distinct blocks for {count} opportunities"
                                 for backend, found in blocks.items()) + "; none is the fixed block")
    except READ_ERRORS as error:
        grader.check("payload", "a block per opportunity", False, f"{type(error).__name__}: {error}")


def check_bad_input(grader, opened):
    names = sorted(name for name in opened if name.startswith("bad-input/"))
    planned = failed = kept = 0
    unread = []
    for name in names:
        directory, run, problem = opened[name]
        if run is None:
            unread.append(f"{name}: {problem}")
            continue
        planned += run.report["totals"]["planned"]
        failed += sum(record["outcome"] == "crc_fail" and record["crc_ok"] is False and record["payload_match"] is False
                      and record["decoded_file"] is None for record in run.records)
        kept += len([path for path in (directory / "native" / "radio").iterdir() if path.name != "radio.jsonl"])
    ok = len(names) >= 2 and not unread and planned == failed > 0 and kept == 0
    grader.check("bad-input", "kept in the denominator", ok,
                 f"{len(names)} runs on deliberately bad input ({', '.join(names)}): {failed} of {planned} planned decodes "
                 f"end as crc_fail by the decoder's own flag, each with its record and its key; {kept} files of decoded "
                 "bytes were kept, and no decode was acknowledged" + "".join(f"; {text}" for text in unread))


def check_sources(grader, runs):
    try:
        locks = {sprint: runs[sprint].manifest()["locked_sources"] for sprint in SPRINTS}
        current = {item["name"]: item["commit"] for item in load(ROOT / "ext/lock.json")["dependencies"]}
        ok = all(lock == current for lock in locks.values())
        grader.check("sources", "one pin", ok,
                     "the four runs and this checkout name the same locked sources: "
                     + ", ".join(f"{name} {commit[:8]}" for name, commit in sorted(current.items())) if ok else
                     "the locked sources differ: " + "; ".join(
                         f"S{sprint} " + ", ".join(f"{name} {str(commit)[:8]}" for name, commit in sorted(lock.items()))
                         for sprint, lock in locks.items()) + "; this checkout "
                     + ", ".join(f"{name} {commit[:8]}" for name, commit in sorted(current.items())))
        commits = {sprint: (re.search(r"^commit (\S+)", runs[sprint].summary, re.M) or [None, "not recorded"])[1]
                   for sprint in SPRINTS}
        grader.note("sources", "commits of the four gates, not graded: "
                    + ", ".join(f"S{sprint} {commit[:12]}" for sprint, commit in commits.items()))
    except READ_ERRORS as error:
        grader.check("sources", "one pin", False, f"{type(error).__name__}: {error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--s24-run", required=True, type=Path, help="run directory of the accepted Sprint 2.4 gate")
    parser.add_argument("--s23-run", type=Path, help="Sprint 2.3 run, when it was moved (default: the one S2.4 names)")
    parser.add_argument("--s22-run", type=Path, help="Sprint 2.2 run, when it was moved (default: the one S2.4 names)")
    parser.add_argument("--s21-run", type=Path, help="Sprint 2.1 run, when it was moved (default: the one S2.2 names)")
    parser.add_argument("--grant-store", action="append", type=Path, default=[],
                        help="further directory that holds <grant_id>/store (default: the paths the entry recorded)")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the audit")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"verify_sprint2: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        runs, names = locate(args)
    except AuditError as error:
        parser.exit(2, f"verify_sprint2: {error}\n")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    grader, chain = Grader(), {}

    def guarded(stage, action, default=None):
        before = len(grader.results)
        try:
            return action()
        except Exception as error:  # an unexpected fault fails its stage and keeps the remaining stages
            (out / f"audit-error-{stage}.txt").write_text(traceback.format_exc())
            grader.check(stage, "audit", False, f"{type(error).__name__}: {error} (audit-error-{stage}.txt)")
            return default
        finally:
            if len(grader.results) == before:
                grader.check(stage, "audit", False, "the stage produced no check")

    profile = guarded("verdicts", lambda: check_verdicts(grader, runs))
    guarded("links", lambda: check_links(grader, runs, names))
    tools = guarded("build", lambda: check_build(grader, runs), {})
    known = guarded("waveform", lambda: check_waveform(grader, runs, [path.resolve() for path in args.grant_store]))
    opened = {}
    try:
        opened = native_runs(runs["2.4"])
    except READ_ERRORS as error:
        grader.check("receiver", "native runs", False, f"the Sprint 2.4 manifest lists no native run: {error}")
    if known:
        stored = guarded("channel", lambda: check_channel(grader, runs, known))
        for item in known["built"]["opportunities"]:
            grant = next(entry for entry in known["built"]["grants"] if entry["grant_id"] == item["grant_id"])
            source = next(entry for entry in known["built"]["sources"] if entry["realization_id"] == item["realization_id"])
            chain[item["opportunity_id"]] = {
                "index": item["index"], "sfn": item["sfn"], "slot": item["slot"], "grant_id": item["grant_id"],
                "grant_record_sha256": grant["record_sha256"], "x_sha256": grant["x"]["sha256"],
                "h_sha256": source["h"]["sha256"], "w_sha256": source["w"]["sha256"],
                "tb_sha256": sha256(known["block"]), "y_sha256_s23": (stored or {}).get(item["opportunity_id"])}
        guarded("receiver", lambda: check_receiver(grader, runs, known, stored, opened, chain))
        guarded("receiver", lambda: check_accounting(grader, opened))
        guarded("payload", lambda: check_payload(grader, known, opened))
    else:
        for stage in ("channel", "receiver", "payload"):
            grader.check(stage, "waveform", False, "the grant record or the pair did not verify, so nothing follows from it")
    guarded("bad-input", lambda: check_bad_input(grader, opened))
    guarded("sources", lambda: check_sources(grader, runs))

    failed = [item for item in grader.results if not item["passed"]]
    stages = [stage for stage in STAGES if any(item["stage"] == stage for item in grader.results)]
    passed = [stage for stage in stages if all(item["passed"] for item in grader.results if item["stage"] == stage)]
    complete = not failed and len(passed) == len(STAGES)
    grade = {"schema": GRADE_SCHEMA, "run_id": out.name, "stages": len(STAGES), "stages_passed": len(passed),
             "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed),
             "checks_failed": len(failed), "judgment": "pass" if complete else "fail", "profile": profile,
             "reason": "every check of every stage passed" if complete else
                       f"{len(failed)} check(s) failed, first: {failed[0]['stage']} {failed[0]['check']}" if failed else
                       "a stage produced no check",
             "runs": {f"s{sprint.replace('.', '')}": runs[sprint].name for sprint in SPRINTS}, "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "chain.json", {
        "about": "one row per opportunity of the reference pair: the record and the transmit grid of its grant (S2.1, "
                 "S2.2), its channel and noise (S2.2), the grid stored for it (S2.3) and what each backend returned "
                 "from that grid (S2.4)", "opportunities": chain})
    write_json(out / "manifest.json", {
        "run_id": out.name, "selector": "sprint2_chain",
        "purpose": "Sprint 2 audit: the chain of the four accepted runs, from the grant record to the decoded bytes",
        "command": shlex.join([sys.executable, *sys.argv]), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "audit_sha256": digest(__file__), "runs": {f"s{sprint.replace('.', '')}": str(runs[sprint].directory)
                                                   for sprint in SPRINTS},
        "tools": tools,
        # For a reader with other tools: the archived block and the directory of the bytes decoded from the pair.
        "original_block": {"path": str(known["block_path"]), "sha256": sha256(known["block"]),
                           "bytes": len(known["block"])} if known else None,
        "pair_radio": str(opened["reference-pair"][0] / "native" / "radio") if "reference-pair" in opened else None,
        "pair_decodes": len(chain) * len(rp.BACKENDS),
        "judgment": grade["judgment"], "reason": grade["reason"]})
    summary = (f"sprint2: {len(passed)} of {len(STAGES)} stages passed, "
               f"{grade['checks_passed']} of {grade['checks']} checks")
    (out / "grade.txt").write_text("\n".join(grader.lines + [summary]) + "\n", encoding="ascii")
    print(summary)
    print(f"Audit artifacts: {out}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
