#!/usr/bin/env python3
"""Sprint 2.4 integration driver (selector replay_pusch). Hardware command, not part of unit discovery.

It runs the native receiver dispatch, built against the accepted OCUDU build, on the stimulus that
Sprint 2.2 archived: every grid is filled by the channel application of Sprint 2.3, handed to the
host and to the CUDA PUSCH processor of the pinned source, and the decoded bytes are compared with
the original transport block, by the tool and once more here. Eight cases, each graded on its own:

  unity            the six slot grants and the two other shipped grants through a unity channel
                   without noise: both backends return the original bytes, the host path with the
                   values Sprint 2.1 recorded
  direct           the direct path of the pair with the entry's gain and no noise, on all
                   opportunities: both backends return the original bytes
  reference-pair   both members of the pair with the entry's gain and noise: the grids are the
                   reference's and those Sprint 2.3 stored, every decode is retained under a unique
                   key, both members and both backends decode alike
  payload          a transport block per opportunity: the blocks, their transmit grids, and a
                   decode of the pair in which every opportunity returns its own block
  bad-input        nothing received and a channel that inverts the data against the pilots: the
                   decoder's own CRC failures, kept in the denominator, never acknowledged
  rerun            a failed and a decoded opportunity repeated exactly from the archived plan
  refusals         every plan rule, refused by the plan check and by the native tool before a file
                   is written
  backends         the path every decode executed, and the count of every planned decode

    python tests/integration/replay_pusch.py --config configs/integration/replay_pusch.json \\
        --s22-run results/runs/s22-spark-001 --s23-run results/runs/s23-spark-001 \\
        --output-dir results/runs/s24-spark-001

The output directory must be new. The exit status is 0 only when every check of every case passed.
On a host without a CUDA build, --host-rehearsal runs the same cases on the host path. Such a run
is graded as a rehearsal and cannot close the hardware gate.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes.episode import EpisodeError  # noqa: E402
from ocudu_cuda_isac.episodes.grant import GrantError  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
from ocudu_cuda_isac.episodes.replay_pusch import ReplayPuschError  # noqa: E402
from ocudu_cuda_isac.replay import ContractError, load_json  # noqa: E402
import native_adapter  # noqa: E402

CONFIG_SCHEMA = "integration-config/replay_pusch/v1"
GRADE_SCHEMA = "integration-grade/replay_pusch/v1"
CASES = ("unity", "direct", "reference-pair", "payload", "bad-input", "rerun", "refusals", "backends")
TOLERANCES = ("epre_db", "sinr_db", "storage_sqnr_db_min")
NATIVE_TIMEOUT_S = 1800


class ConfigError(ValueError):
    """The configuration or the run setup is unusable. Nothing was graded."""


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


apply_driver = load_script("grid_apply_driver", ROOT / "tests/integration/grid_apply.py")
fixture = apply_driver.fixture
digest, write_json, Grader = fixture.digest, fixture.write_json, fixture.Grader
reference, bf16_array, as_complex, grid_digest = (apply_driver.reference, apply_driver.bf16_array,
                                                  apply_driver.as_complex, apply_driver.grid_digest)
last_line, EVALUATOR_WORDS = apply_driver.last_line, apply_driver.EVALUATOR_WORDS


class Grade(Grader):
    """The Sprint 2.1 grader with this sprint's label."""

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S2.4 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, text):
        self._line(f"NOTE  S2.4 {case} {text}")


def load_config(path):
    try:
        config = load_json(Path(path))
    except (ContractError, OSError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    needed = {"schema", "selector", "adapter_build", "backends", "receiver", "tolerance", "pair", "unity", "payload"}
    if not isinstance(config, dict) or set(config) != needed or config["schema"] != CONFIG_SCHEMA \
            or config["selector"] != "replay_pusch":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object for the selector replay_pusch")
    if not isinstance(config["tolerance"], dict) or set(config["tolerance"]) != set(TOLERANCES):
        raise ConfigError(f"tolerance requires exactly {', '.join(TOLERANCES)}")
    receiver = config["receiver"]
    if not isinstance(receiver, dict) or set(receiver) != {"ldpc_iterations", "early_stop", "timeout_ms"}:
        raise ConfigError("receiver requires exactly ldpc_iterations, early_stop and timeout_ms")
    try:
        rp.check_receiver(dict(rp.RECEIVER, backends=config["backends"], **receiver))
    except (ReplayPuschError, TypeError) as error:
        raise ConfigError(str(error)) from None
    # Both backends decode every grid. A run with one of them answers another question.
    if list(config["backends"]) != list(rp.BACKENDS):
        raise ConfigError('backends must be ["cpu", "cuda"]: the gate decodes every grid with both')
    # The Sprint 1 gates measured the pinned test with these decoder settings.
    if (receiver["ldpc_iterations"], receiver["early_stop"]) != (rp.RECEIVER["ldpc_iterations"], rp.RECEIVER["early_stop"]):
        raise ConfigError("the decoder settings must be those of the pinned test: 6 iterations with early stop")
    pair = config["pair"]
    if not isinstance(pair, dict) or set(pair) != {"entry", "grant_stores", "expect"} \
            or not isinstance(pair["expect"], dict) or not isinstance(pair["grant_stores"], list):
        raise ConfigError("pair requires exactly entry, grant_stores and expect")
    unity = config["unity"]
    if not isinstance(unity, dict) or set(unity) != {"grants", "expect"} or not isinstance(unity["grants"], list) \
            or not isinstance(unity["expect"], dict) \
            or set(unity["expect"]) != {"bytes", "ldpc_iterations", "epre_db", "sinr_db"}:
        raise ConfigError("unity requires grants and expect with bytes, ldpc_iterations, epre_db and sinr_db")
    if not isinstance(config["payload"], dict) or set(config["payload"]) != {"seed"}:
        raise ConfigError("payload requires exactly seed")
    return config


# ---------------------------------------------------------------- native tool
def run_native(tool, plan_path, output, log, extra=()):
    """Run replay_pusch on one plan. Returns (exit status, report or None). output is created empty."""
    output.mkdir(parents=True)
    command = [str(tool), "--plan", str(plan_path), "--output-dir", str(output), *extra]
    with log.open("w") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        try:
            status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=NATIVE_TIMEOUT_S,
                                    check=False).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(f"{type(error).__name__}: {error}\n")
            return -1, None
    try:
        return status, json.loads((output / "native.json").read_text())
    except (OSError, ValueError):
        return status, None


def realize_grid(adapter, request, output, log):
    """Run the fixture adapter on one grant for its transmit grid alone. Returns (exit status, report or None).

    The adapter's own loopback is not run. Every grid made here is decoded by both backends through
    the channel of the pair, which is the stronger check, and the loopback of the accepted adapter
    destroys its processor before its workers are joined: on a loaded host that ended two of 24
    runs of a rehearsal with a segmentation fault after the decode (docs/integration/OCUDU.md).
    """
    output.mkdir(parents=True)
    command = [str(adapter), "--grant", str(request / "grant.json"), "--tb", str(request / "tb.bin"),
               "--output-dir", str(output), "--dmrs-reference", "crb0", "--no-loopback"]
    with log.open("w") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        try:
            status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=fixture.ADAPTER_TIMEOUT_S,
                                    check=False).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(f"{type(error).__name__}: {error}\n")
            return -1, None
    try:
        return status, json.loads((output / "native.json").read_text())
    except (OSError, ValueError):
        return status, None


class Run:
    """One native run: its plans, its verified output and the evaluator's join to the original blocks."""

    def __init__(self, name, directory):
        self.name, self.directory = name, directory
        self.plan = self.channel = self.report = self.output = None
        self.status, self.problem, self.rows, self.totals = None, None, [], {}

    @property
    def native(self):
        return self.directory / "native"

    def ok(self):
        return self.status == 0 and self.output is not None and self.problem is None

    def said(self):
        return self.problem or f"the native tool exited with status {self.status}: " \
                               f"{last_line(self.directory / 'native.log')}"

    def of(self, backend):
        return [row for row in self.rows if row["backend"] == backend]

    def counts(self):
        """Decoded and planned per backend, as text."""
        return ", ".join(f"{name} {total['decoded']} of {total['planned']} decoded" for name, total in self.totals.items())


def replay_run(runs, directory, name, tool, run_id, channel, payload, config, extra=()):
    """Write the channel plan and the replay plan, run the tool, verify its output and evaluate it."""
    run = Run(name, directory)
    runs.append(run)
    directory.mkdir(parents=True)
    run.channel = channel
    replay.write_plan(channel, directory / "channel-plan.json")
    run.plan = rp.build_plan(run_id, directory / "channel-plan.json", payload, config["backends"], config["receiver"])
    rp.write_plan(run.plan, directory / "plan.json")
    run.status, run.report = run_native(tool, directory / "plan.json", run.native, directory / "native.log", extra)
    if run.status != 0 or not run.report:
        return run
    try:
        run.output = rp.open_output(run.native)
        run.rows, run.totals = rp.evaluate(run.output, run.plan)
    except ReplayPuschError as error:
        run.problem = f"the native output does not verify: {error}"
    return run


def run_is_clean(run, opportunities, backends):
    """True when a run finished, verifies, holds one record per planned decode and nothing went wrong in it."""
    if not run.ok():
        return False
    totals = run.report["totals"]
    return len(run.output.records) == opportunities * len(backends) and totals["planned"] == len(run.output.records) \
        and not run.report["anomalies"] and totals["grids_not_released"] == 0 and totals["retained"] is True \
        and all(row["consistent"] for row in run.rows) \
        and len({record["key"] for record in run.output.records}) == len(run.output.records)


def all_decoded(run, backends):
    """True when every planned decode of every backend returned the original bytes with a CRC pass."""
    return run.ok() and set(run.totals) == set(backends) and all(
        total["decoded"] == total["planned"] == total["payload_matches"] > 0 for total in run.totals.values())


def reference_grids(np, channel):
    """The stored grid of every opportunity of a channel plan, from a complex128 reference: digests and precision."""
    gain = channel["gain"]["value"]
    x_of = {item["grant_id"]: np.load(item["x"]["path"], allow_pickle=False) for item in channel["grants"]}
    h_of = {item["realization_id"]: np.load(item["h"]["path"], mmap_mode="r", allow_pickle=False) for item in channel["sources"]}
    w_of = {item["realization_id"]: np.load(item["w"]["path"], mmap_mode="r", allow_pickle=False) for item in channel["sources"]}
    digests, signal, rounding = [], 0.0, 0.0
    for item in channel["opportunities"]:
        real, imag, _, _ = reference(np, x_of[item["grant_id"]], np.asarray(h_of[item["realization_id"]][item["index"]]),
                                     np.asarray(w_of[item["realization_id"]][item["index"]]), gain)
        kept_real, kept_imag = bf16_array(np, real.astype(np.float32)), bf16_array(np, imag.astype(np.float32))
        digests.append(grid_digest(as_complex(np, kept_real, kept_imag)))
        signal += float(np.mean(real ** 2 + imag ** 2))
        rounding += float(np.mean((kept_real.astype(np.float64) - real) ** 2 + (kept_imag.astype(np.float64) - imag) ** 2))
    sqnr = 10 * math.log10(signal / rounding) if rounding > 0 else math.inf
    return digests, sqnr


def host_values(run, backend, key):
    """One decoder value of every decode that a backend executed on the host path."""
    return [record["decoder"][key] for record in run.output.of(backend)
            if isinstance(record["decoder"], dict) and key in record["decoder"] and record["path"]["executed"] == "host"]


def mean(values):
    return sum(values) / len(values) if values else None


def text(value, digits=2):
    return "none" if value is None else f"{value:.{digits}f}"


# ---------------------------------------------------------------- stimulus
def locate(config, s22_run, roots):
    """Open the episode entry, build its channel plan and open the store entry of every slot grant."""
    entry = s22_run / config["pair"]["entry"]
    search = [s22_run / item for item in config["pair"]["grant_stores"]] + list(roots)
    channel = replay.entry_plan(f"s24-pair-{s22_run.name}", [entry], search)
    episode = episodes.open_episode(entry)
    stores = {item["grant_id"]: grants.open_store(Path(item["x"]["path"]).parent) for item in channel["grants"]}
    # The recorded store of the base grant lies in the Sprint 2.1 run, beside the other shipped grants.
    search = search + [Path(item["x"]["path"]).parents[2] for item in channel["grants"]]
    return entry, episode, channel, stores, search


def fixture_source(np, directory, name, carrier, h_value=None, h_ref=None, w_ref=None, count=1):
    """One channel realization of a fixture: a constant or an archived channel, and zero noise unless one is named."""
    rx, tx, symbols, subcarriers = carrier
    directory.mkdir(parents=True, exist_ok=True)
    if h_ref is None:
        h = np.empty((count, rx, tx, symbols, subcarriers), dtype="<c8")
        h[...] = h_value
        np.save(directory / "h.npy", h, allow_pickle=False)
        h_ref = replay.file_ref(directory / "h.npy")
    if w_ref is None:
        np.save(directory / "w.npy", np.zeros((count, rx, symbols, subcarriers), dtype="<c8"), allow_pickle=False)
        w_ref = replay.file_ref(directory / "w.npy")
    return {"realization_id": name, "h": h_ref, "w": w_ref}


# ---------------------------------------------------------------- unity
def run_unity(config, out, grader, np, tool, runs, channel, stores, search):
    name, expect, tolerance, backends = "unity", config["unity"]["expect"], config["tolerance"], config["backends"]
    case_dir = out / "unity"
    case_dir.mkdir()
    reason = "unity check of the archived transmit grids through the receiver: a fixture without an episode entry"
    measured, groups = {}, []
    # The six slot grants on the four receive planes of the demonstration.
    carrier = tuple(channel["carrier"][key] for key in replay.CARRIER_FIELDS)
    groups.append(("slot-grants", carrier, [(item, stores[item["grant_id"]]) for item in channel["grants"]]))
    # The other shipped grants, on the carrier of their own record.
    others = {}
    for grant_id in config["unity"]["grants"]:
        try:
            store = next(grants.open_store(root / grant_id / "store") for root in search
                         if (root / grant_id / "store" / "manifest.json").is_file())
        except (StopIteration, GrantError) as error:
            grader.check(name, grant_id, False, f"no verified store entry of the grant {grant_id} under "
                                                f"{', '.join(str(root) for root in search)}: {error}")
            continue
        grant = store.grant
        shape = (len(grant.rx_ports), grant.antenna_ports, grants.SYMBOLS_PER_SLOT, grant.grid_subcarriers)
        others.setdefault(shape, []).append((replay.grant_from_store(store), store))
    groups += [(f"shipped-{shape[0]}rx", shape, items) for shape, items in sorted(others.items())]

    clean, done = True, []
    for label, shape, items in groups:
        source = fixture_source(np, case_dir / label / "arrays", f"fx-unity-{label}", shape, h_value=1.0)
        opportunities = [{"opportunity_id": f"fx-unity-{label}.{number:04d}", "realization_id": source["realization_id"],
                          "index": 0, "grant_id": item["grant_id"], "sfn": 0, "slot": item["slot"]}
                         for number, (item, _) in enumerate(items)]
        plan = replay.fixture_plan(f"s24-unity-{label}", shape, 1.0, reason, [item for item, _ in items], [source],
                                   opportunities)
        run = replay_run(runs, case_dir / label / "run", f"unity/{label}", tool, f"s24-unity-{label}", plan, None,
                         config)
        clean = clean and run_is_clean(run, len(items), backends)
        done.append((run, shape, items))
    grader.check(name, "native run", clean and len(done) == len(groups),
                 "; ".join(f"{run.name}: {len(items)} grids on {shape[0]} receive plane(s), {run.counts()}"
                           if run.ok() else f"{run.name}: {run.said()}" for run, shape, items in done)
                 + "; a unity channel with zero noise and g = 1, an override recorded with its reason; one record per "
                   "planned decode under a unique key, no anomaly, every grid released")
    for run, shape, items in done:
        if not run.ok():
            continue
        rx = shape[0]
        for sequence, (item, store) in enumerate(items):
            grant_id, grant = item["grant_id"], store.grant
            payload = np.load(item["x"]["path"], allow_pickle=False).tobytes()
            records = [record for record in run.output.records if record["opportunity_sequence"] == sequence]
            rows = [row for row in run.rows if row["grant_id"] == grant_id]
            wanted_bytes = expect["bytes"].get(grant_id.split(".s")[0])
            ok = len(records) == len(backends) and all(
                record["outcome"] == "decoded" and record["tb"]["bytes"] == len(grant.tb) == wanted_bytes
                and record["decoded_sha256"] == hashlib.sha256(grant.tb).hexdigest()
                and record["ingress"]["readback_sha256"] == hashlib.sha256(payload * rx).hexdigest()
                for record in records) and all(row["payload_match"] and row["consistent"] for row in rows)
            shown = []
            for record in records:
                decoder, executed = record["decoder"] or {}, record["path"]["executed"]
                shown.append(f"{record['backend']} on the {executed} path: {record['outcome']}, LDPC iterations "
                             f"{text(decoder.get('ldpc_iterations_mean'), 1)}, EPRE {text(decoder.get('epre_db'))} dB, SINR "
                             f"{text(decoder.get('sinr_db'))} dB")
                if executed == "host":
                    # The values the accepted Sprint 2.1 loopback recorded for the host path.
                    ok = ok and decoder.get("ldpc_iterations_mean") == expect["ldpc_iterations"] \
                        and abs(decoder.get("epre_db", math.inf) - expect["epre_db"]) <= tolerance["epre_db"] \
                        and abs(decoder.get("sinr_db", math.inf) - expect["sinr_db"][str(rx)]) <= tolerance["sinr_db"]
            measured[grant_id] = {"records": [{key: record[key] for key in ("backend", "outcome", "decoder", "path")}
                                              for record in records], "tb_bytes": len(grant.tb)}
            grader.check(name, grant_id, ok,
                         f"slot {grant.slot}, {rx} receive port(s), CRB {grant.crb_start} to "
                         f"{grant.crb_start + grant.prb_count - 1}: every plane of the grid is x.npy "
                         f"({item['x']['sha256'][:16]}) bit for bit, and each backend returns the original "
                         f"{len(grant.tb)} bytes with a CRC pass; " + "; ".join(shown)
                         + f"; expected on the host path: {expect['ldpc_iterations']:.0f} iteration, EPRE "
                           f"{expect['epre_db']:.2f} dB, SINR {expect['sinr_db'][str(rx)]:.2f} dB")
    return measured


# ---------------------------------------------------------------- direct path
def run_direct(config, out, grader, np, tool, runs, entry, episode, channel):
    name, backends, tolerance = "direct", config["backends"], config["tolerance"]
    case_dir = out / "direct"
    case_dir.mkdir()
    manifest = episode.manifest
    case_of = {case["realization_id"]: case["id"] for case in manifest["cases"]}
    absent = next(item for item in channel["sources"] if case_of[item["realization_id"]] == "a")
    carrier = tuple(channel["carrier"][key] for key in replay.CARRIER_FIELDS)
    count = len(episode.schedule.opportunities)
    source = fixture_source(np, case_dir / "arrays", "fx-direct", carrier, h_ref=absent["h"], count=count)
    opportunities = [dict(item, opportunity_id=f"fx-direct.{item['index']:04d}", realization_id="fx-direct")
                     for item in channel["opportunities"] if item["realization_id"] == absent["realization_id"]]
    gain = channel["gain"]["value"]
    plan = replay.fixture_plan("s24-direct", carrier, gain,
                               "direct path of the reference pair without noise: a fixture, with the gain of the entry",
                               channel["grants"], [source], opportunities)
    run = replay_run(runs, case_dir / "run", "direct", tool, "s24-direct", plan, None, config)
    clean = run_is_clean(run, count, backends)
    grader.check(name, "native run", clean,
                 f"the target-absent channel of {episode.episode_id} ({absent['h']['sha256'][:16]}) on {count} "
                 f"opportunities with g = {gain:.4f} and zero noise, an override recorded with its reason: "
                 f"{run.counts()}; one record per planned decode, no anomaly, every grid released"
                 if run.ok() else run.said())
    if not run.ok():
        return {}
    digests, sqnr = reference_grids(np, plan)
    native = [item["readback_sha256"] for item in run.report["opportunities"]]
    equal = sum(a == b for a, b in zip(native, digests))
    grader.check(name, "grid precision", equal == count == len(native) and sqnr >= tolerance["storage_sqnr_db_min"],
                 f"{equal} of {count} readback digests equal the complex128 reference rounded to the storage format; "
                 f"without noise the grid holds the signal {sqnr:.1f} dB above its own rounding, the signal-to-noise "
                 f"ratio after storage (at least {tolerance['storage_sqnr_db_min']:.0f} dB required)")
    sinr = {backend: mean([record["decoder"]["sinr_db"] for record in run.output.of(backend)
                           if isinstance(record["decoder"], dict) and "sinr_db" in record["decoder"]])
            for backend in backends}
    grader.check(name, "decodes", all_decoded(run, backends),
                 f"high-SNR direct path: {run.counts()}, each with a CRC pass and all {len(episode.grant.tb)} bytes "
                 "equal to the original, by the tool and by the evaluator's own comparison; reported SINR, mean: "
                 + ", ".join(f"{backend} {text(value)} dB" for backend, value in sinr.items()))
    return {"storage_sqnr_db": sqnr, "sinr_db": sinr, "digests_equal": equal}


# ---------------------------------------------------------------- reference pair
def members_alike(run, case_of, backends):
    """Opportunities whose two members differ in outcome or in decoded bytes, per backend."""
    different = []
    for backend in backends:
        seen = {}
        for record in run.output.of(backend):
            realization, _, index = record["opportunity_id"].rpartition(".")
            seen.setdefault(index, {})[case_of.get(realization)] = (record["outcome"], record["decoded_sha256"])
        different += [f"{backend} opportunity {index}" for index, pair in sorted(seen.items())
                      if set(pair) != {"a", "b"} or pair["a"] != pair["b"]]
    return different


def run_pair(config, out, grader, np, tool, runs, entry, episode, channel, stores, s23_run):
    name, expect, backends = "reference-pair", config["pair"]["expect"], config["backends"]
    case_dir = out / "reference_pair"
    case_dir.mkdir()
    manifest = episode.manifest
    gain, count = channel["gain"]["value"], len(channel["opportunities"])
    case_of = {case["realization_id"]: case["id"] for case in manifest["cases"]}
    base_block = hashlib.sha256(episode.grant.tb).hexdigest()
    measured = {"gain": gain, "tb_sha256": base_block}

    run = replay_run(runs, case_dir / "run", "reference-pair", tool, channel["run_id"], channel, None, config)
    plan = run.plan
    grader.check(name, "plan", count == 2 * expect["opportunities"] and len(plan["grants"]) == expect["grants"]
                 and abs(gain - expect["grid_gain"]) <= 1e-9 * expect["grid_gain"]
                 and gain == manifest["calibration"]["grid_gain"] and plan["payload"] == {
                     "rule": "fixed", "generator": None, "seed": None, "blocks": 1}
                 and len(episode.grant.tb) == expect["tb_bytes"] and base_block == expect["tb_sha256"]
                 and all(item["tb"]["sha256"] == base_block for item in plan["grants"]),
                 f"from the verified entry {episode.episode_id}: {count} opportunities of {len(channel['sources'])} "
                 f"realizations on {len(plan['grants'])} grants with g = {gain:.4f} = calibration.grid_gain; payload "
                 f"rule fixed: every opportunity carries the block of {episode.grant.grant_id}, {len(episode.grant.tb)} "
                 f"bytes {base_block[:16]}; the record of each grant, with its block, has the hash that binds its grid")
    clean = run_is_clean(run, count, backends)
    grader.check(name, "native run", clean,
                 f"{count} grids filled, made ready and decoded by {' and '.join(backends)}: "
                 f"{len(run.output.records)} radio records under {len(run.output.by_key)} unique keys, "
                 f"{len(run.report['files'])} input files verified against their hashes, no anomaly, every grid "
                 "released after its last decoder callback (reference_pair/run/native)" if run.ok() else run.said())
    if not run.ok():
        return measured, run

    # ---- The grid the receiver decoded is the reference's, and the one Sprint 2.3 stored. ----
    digests, _ = reference_grids(np, channel)
    native = {item["opportunity_id"]: item["readback_sha256"] for item in run.report["opportunities"]}
    equal = sum(native[item["opportunity_id"]] == wanted for item, wanted in zip(channel["opportunities"], digests))
    run_digest = hashlib.sha256("\n".join(native[item["opportunity_id"]]
                                          for item in channel["opportunities"]).encode("ascii")).hexdigest()
    earlier, earlier_equal = None, None
    if s23_run is not None:
        try:
            earlier = replay.open_output(s23_run / "grid_apply" / "reference_pair" / "native")
            stored = {item["opportunity_id"]: item["readback_sha256"] for item in earlier.report["opportunities"]}
            earlier_equal = sum(native.get(key) == value for key, value in stored.items()) if len(stored) == count else -1
        except GridApplyError as error:
            earlier_equal, earlier = -1, str(error)
    intact = sum(record["ingress"].get("intact") is True for record in run.output.records)
    measured["ingress"] = {"digests_equal_to_reference": equal, "digests_equal_to_s23": earlier_equal,
                           "run_digest": run_digest, "intact_after_decode": intact}
    grader.check(name, "ingress digests", equal == count and intact == len(run.output.records)
                 and (s23_run is None or earlier_equal == count),
                 f"{equal} of {count} readback digests of the decoded grids equal the complex128 reference rounded to "
                 f"the storage format, and all {intact} decodes left their grid unchanged; "
                 + (f"{earlier_equal} of {count} equal the grids that the Sprint 2.3 run {s23_run.name} stored"
                    if s23_run is not None and earlier_equal != -1 else
                    "no Sprint 2.3 run was named, so its stored grids were not compared" if s23_run is None else
                    f"the Sprint 2.3 run {s23_run.name} holds no comparable output: {earlier}")
                 + f"; digest over the {count} grids {run_digest[:16]}")
    if expect.get("s23_run_digest"):
        grader.note(name, f"against the accepted Sprint 2.3 run (not graded): digest over the {count} decoded grids "
                          f"{'identical' if expect['s23_run_digest'] == run_digest else 'different'} "
                          f"({run_digest[:16]})")

    # ---- The observed outcome of every decode, retained, and the predeclared expectation. ----
    measured["totals"] = run.totals
    grader.check(name, "decodes", all_decoded(run, backends),
                 f"the physical pair at {manifest['noise']['snr_db']:g} dB: {run.counts()}; every decode is retained in "
                 f"radio/radio.jsonl with its outcome, its grid digest and the hash of its bytes, and the evaluator's "
                 f"own comparison with the original {len(episode.grant.tb)} bytes agrees on all "
                 f"{sum(row['consistent'] for row in run.rows)} records")
    different = members_alike(run, case_of, backends)
    grader.check(name, "members alike", not different,
                 f"target absent and target present decode alike on all {count // 2} opportunities and both backends: "
                 "the same outcome and the same bytes" if not different else
                 "a finding: the two members differ at " + ", ".join(different[:6]))
    by_key = run.output.by_key
    disagree = [item["opportunity_id"] for item in channel["opportunities"]
                if len({(by_key[f"{item['opportunity_id']}/{backend}"]["outcome"],
                         by_key[f"{item['opportunity_id']}/{backend}"]["decoded_sha256"]) for backend in backends}) != 1]
    grader.check(name, "backends agree", not disagree,
                 f"the host and the CUDA processor were handed the same {count} grids and return the same outcome and "
                 "the same bytes for each" if not disagree else
                 f"the backends differ on {len(disagree)} opportunities, first {disagree[0]}")

    # ---- Access: what the receiver side reads of its input names no channel, no noise, no block, no label. ----
    secrets = [item["sha256"] for item in manifest["files"].values()]
    secrets += [store.manifest["derived"]["tb_sha256"] for store in stores.values()]
    secrets += [store.manifest["files"]["tb.bin"]["sha256"] for store in stores.values()]
    secrets += list(EVALUATOR_WORDS) + [case["id"] + "_h" for case in manifest["cases"]]
    violations = replay.ingress_violations(run.native, secrets)
    radio_text = (run.native / "radio" / "radio.jsonl").read_text()
    leaked = [word for word in list(EVALUATOR_WORDS) + [item["sha256"] for item in manifest["files"].values()]
              if word in radio_text]
    measured["access"] = {"ingress_violations": violations, "radio_leaks": leaked}
    grader.check(name, "access", not violations and not leaked and run.report["receiver"]["tap"] == "none",
                 f"ingress/ holds the {count} stored grids and their {count} descriptors and nothing else, with no hash "
                 "or name of H, W or a transport block and no label; radio/ holds the records and the decoded bytes of "
                 "every CRC pass, with no name of H or W and no label; no observer is attached (tap none)"
                 if not violations and not leaked else
                 "the receiver side could read: " + "; ".join((violations + leaked)[:5]))

    # ---- A second reader: a separate process, standard library only. ----
    last, backend = count - 1, backends[-1]
    reader = subprocess.run([sys.executable, "-m", "ocudu_cuda_isac.episodes.replay_pusch", "show", str(run.native),
                             "--sequence", str(last), "--backend", backend], capture_output=True, text=True, check=False,
                            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    try:
        shown = json.loads(reader.stdout)
        write_json(case_dir / "second-reader.json", shown)
        grader.check(name, "second reader", reader.returncode == 0 and shown["verified_records"] == len(run.output.records)
                     and shown["ingress_sha256"] == digests[last] and shown["decoded_sha256"] == base_block
                     and shown["outcome"] == "decoded",
                     f"a separate process verifies the {shown['verified_records']} radio records, the decoded bytes "
                     f"and the stored grids against their hashes and recovers the decode {shown['key']}: "
                     f"{shown['outcome']} on the {shown['executed']} path, {shown['tb_bytes']} bytes starting "
                     f"{shown['decoded_first_bytes']}, from the grid {shown['ingress_sha256'][:16]} "
                     "(reference_pair/second-reader.json)")
    except (ValueError, KeyError, TypeError) as error:
        grader.check(name, "second reader", False, f"exit status {reader.returncode}: {error} {reader.stderr[-300:]}")
    return measured, run


# ---------------------------------------------------------------- payload
def run_payload(config, out, grader, np, tool, runs, entry, episode, channel, stores, search, adapter, receipt):
    name, backends, seed = "payload", config["backends"], config["payload"]["seed"]
    case_dir = out / "payload"
    case_dir.mkdir()
    schedule = episode.schedule
    planned = rp.payload_grants(schedule, seed)
    width, realized, failures, moved = schedule.base.grid_subcarriers, {}, [], []
    for item, grant in zip(schedule.opportunities, planned):
        directory = case_dir / "stores" / grant.grant_id
        directory.mkdir(parents=True)
        try:
            grants.export(grant, directory / "request")
            status, report = realize_grid(adapter, directory / "request", directory / "native", directory / "native.log")
            if status != 0 or not report or report.get("schema") != fixture.NATIVE_SCHEMA:
                failures.append(f"{grant.grant_id}: the fixture adapter exited with status {status}")
                continue
            native = fixture.read_grid(directory / "native" / "x_native.c64",
                                       grant.antenna_ports * grants.SYMBOLS_PER_SLOT * width)
            grants.check_reference(grant, native)
            if not fixture.pdu_matches(grant, report):
                failures.append(f"{grant.grant_id}: the fixture adapter did not build the PDU of the grant")
                continue
            grants.export(grant, directory / "store", {"x": {"values": native, "provenance": {
                "reference": "pinned OCUDU transmit chain at the Sprint 1 native boundary, read back from the "
                             "resource grid", "dmrs_reference": "crb0", "payload": "per_opportunity",
                "loopback": "not run: both backends decode this grid through the channel of the pair",
                "payload_seed": rp.block_seed(seed, item.index), "schedule": schedule.schedule_id,
                "adapter_sha256": receipt["executables"]["grant_fixture"]["sha256"],
                "ocudu_build": receipt["ocudu_build"], "built_with_cuda": report.get("built_with_cuda"),
                "storage": report["grid"]["storage"]}}})
            store = grants.open_store(directory / "store")
            same, changed = rp.pilots_equal(store, stores[item.grant.grant_id])
            if same != len(grant.pilot_index):
                failures.append(f"{grant.grant_id}: {len(grant.pilot_index) - same} pilots differ from the slot grant's")
            moved.append(changed)
            realized[grant.grant_id] = store
        except (GrantError, OSError, KeyError) as error:
            failures.append(f"{grant.grant_id}: {error}")
    payload, problem = None, None
    try:
        payload = rp.payload_set(schedule, seed, realized)
        write_json(case_dir / "payload.json", payload)
    except ReplayPuschError as error:
        problem = str(error)
    count = len(schedule.opportunities)
    blocks = {item["tb_sha256"] for item in payload["blocks"]} if payload else set()
    base_block = hashlib.sha256(schedule.base.tb).hexdigest()
    grader.check(name, "blocks", payload is not None and not failures and len(blocks) == count and base_block not in blocks
                 and bool(moved) and min(moved) > 0,
                 f"{len(blocks)} distinct transport blocks for {count} opportunities from the generator "
                 f"{grants.TB_GENERATOR} with the seed '{seed}'; the fixture adapter realized each on the grant of its "
                 f"slot: the record is the slot grant's apart from the identifier and the block, all pilots are those "
                 f"of the slot grant bit for bit, {min(moved) if moved else 0} to {max(moved) if moved else 0} of "
                 f"{len(schedule.base.data_index)} data symbols differ, and the adapter echoed the PDU of each grant; "
                 "the decode of every block follows in the run of the pair (payload/payload.json)"
                 if payload is not None and not failures else
                 "; ".join(([problem] if problem else []) + failures[:4]))
    if payload is None:
        return {}, None
    try:
        plan = rp.per_opportunity_channel_plan(f"s24-payload-{out.name}", [entry], payload, search)
    except (ReplayPuschError, GridApplyError) as error:
        grader.check(name, "plan", False, str(error))
        return {}, None
    run = replay_run(runs, case_dir / "run", "payload", tool, plan["run_id"], plan, payload, config)
    total = len(plan["opportunities"])
    grader.check(name, "plan", len(plan["grants"]) == count and total == 2 * count and run.plan["payload"] == {
        "rule": "per_opportunity", "generator": grants.TB_GENERATOR, "seed": seed, "blocks": count}
                 and plan["gain"] == channel["gain"] and plan["sources"] == channel["sources"],
                 f"payload rule per_opportunity, recorded in the plan and in the report: {total} opportunities of the "
                 f"pair on {len(plan['grants'])} grants, one per opportunity index, the same block for both members; "
                 "the gain, the channel and the noise are those of the entry; every block of the plan is the one the "
                 "rule generates for its opportunity")
    clean = run_is_clean(run, total, backends)
    grader.check(name, "native run", clean,
                 f"{total} grids decoded by {' and '.join(backends)}: {len(run.output.records)} radio records under "
                 f"unique keys, no anomaly, every grid released (payload/run/native)" if run.ok() else run.said())
    if not run.ok():
        return {"blocks": len(blocks)}, run
    # The evaluator generates each block again from the seed alone and compares the decoded bytes with it.
    own = 0
    for record in run.output.records:
        index = int(record["opportunity_id"].rpartition(".")[2])
        decoded = run.output.decoded(record)
        own += decoded is not None and decoded == rp.opportunity_block(seed, index, len(schedule.base.tb))
    hashes = {backend: {record["decoded_sha256"] for record in run.output.of(backend)} - {None} for backend in backends}
    grader.check(name, "decodes", all_decoded(run, backends) and own == len(run.output.records),
                 f"{run.counts()}; {own} of {len(run.output.records)} decodes return the block that the rule generates "
                 "for their own opportunity, compared byte for byte with a block built here from the seed alone")
    grader.check(name, "every opportunity its own block", all(len(found) == count for found in hashes.values())
                 and all(base_block not in found for found in hashes.values()),
                 "; ".join(f"{backend} returned {len(found)} distinct blocks for {count} opportunities"
                           for backend, found in hashes.items())
                 + ": bytes of another opportunity, or of the fixed block, would fail the comparison, which one block "
                   "for all opportunities cannot show")
    case_of = {case["realization_id"]: case["id"] for case in episode.manifest["cases"]}
    different = members_alike(run, case_of, backends)
    grader.check(name, "members alike", not different,
                 f"both members carry the same block at the same opportunity and decode alike on all {count} "
                 "opportunities and both backends" if not different else
                 "a finding: the two members differ at " + ", ".join(different[:6]))
    return {"blocks": len(blocks), "data_symbols_changed": [min(moved), max(moved)], "totals": run.totals}, run


# ---------------------------------------------------------------- bad input
def run_bad_input(config, out, grader, np, tool, runs, episode, channel):
    name, backends = "bad-input", config["backends"]
    case_dir = out / "bad_input"
    case_dir.mkdir()
    carrier = tuple(channel["carrier"][key] for key in replay.CARRIER_FIELDS)
    rx, tx, symbols, subcarriers = carrier
    count = len(channel["grants"])
    manifest = episode.manifest
    noise = next(item["w"] for item in channel["sources"])

    def opportunities(label):
        return [{"opportunity_id": f"fx-{label}.{number:04d}", "realization_id": f"fx-{label}", "index": number,
                 "grant_id": item["grant_id"], "sfn": 0, "slot": item["slot"]}
                for number, item in enumerate(channel["grants"])]

    # Nothing received: a zero channel leaves the noise of the entry, scaled by its gain.
    h_count = len(episode.schedule.opportunities)
    silent = fixture_source(np, case_dir / "noise_only" / "arrays", "fx-noise-only", carrier, h_value=0.0, w_ref=noise,
                            count=h_count)
    silent_plan = replay.fixture_plan(
        "s24-noise-only", carrier, channel["gain"]["value"],
        "bad input: a zero channel and the noise of the entry with the entry's gain, a fixture",
        channel["grants"], [silent], opportunities("noise-only"))
    # A bad channel: full power, and the data symbols inverted against the DM-RS symbols.
    h = np.full((count, rx, tx, symbols, subcarriers), -1.0, dtype="<c8")
    for number, item in enumerate(channel["grants"]):
        pilots = np.load(item["active"][0]["path"], allow_pickle=False)
        h[number, :, :, sorted(set(pilots[:, 0].tolist())), :] = 1.0
    (case_dir / "phase_break" / "arrays").mkdir(parents=True)
    np.save(case_dir / "phase_break" / "arrays" / "h.npy", h, allow_pickle=False)
    broken = fixture_source(np, case_dir / "phase_break" / "arrays", "fx-phase-break", carrier,
                            h_ref=replay.file_ref(case_dir / "phase_break" / "arrays" / "h.npy"), count=count)
    broken_plan = replay.fixture_plan(
        "s24-phase-break", carrier, 1.0,
        "bad channel: unity on the DM-RS symbols and minus one on the data symbols, no noise, a fixture",
        channel["grants"], [broken], opportunities("phase-break"))
    measured, done = {}, []
    for label, plan, what in (("noise_only", silent_plan, "noise only"), ("phase_break", broken_plan, "phase break")):
        run = replay_run(runs, case_dir / label / "run", f"bad-input/{label}", tool, plan["run_id"], plan, None, config)
        done.append((label, what, run))
    grader.check(name, "native run", all(run_is_clean(run, count, backends) for _, _, run in done),
                 "; ".join(f"{run.name}: exit status {run.status}, {len(run.output.records)} records for "
                           f"{count * len(backends)} planned decodes" if run.ok() else f"{run.name}: {run.said()}"
                           for _, _, run in done)
                 + "; a CRC failure is an observation of the decoder and not an error of the run")
    descriptions = {
        "noise only": f"a zero channel leaves g W, {20 * math.log10(channel['gain']['value'] * manifest['noise']['sigma']):.1f} "
                      "dB against unit signal power",
        "phase break": "unity on the DM-RS symbols and minus one on the data symbols, at full power without noise"}
    for label, what, run in done:
        if not run.ok():
            continue
        failed = sum(total["crc_fail"] for total in run.totals.values())
        planned = sum(total["planned"] for total in run.totals.values())
        kept = all(record["crc_ok"] is False and record["payload_match"] is False and record["decoded_file"] is None
                   and record["outcome"] == "crc_fail" for record in run.output.records)
        files = sorted(path.name for path in (run.native / "radio").iterdir())
        measured[label] = run.totals
        shares = " and ".join("%s %d of %d" % (backend, total["crc_fail"], total["planned"])
                              for backend, total in run.totals.items())
        grader.check(name, what, failed == planned == count * len(backends) and kept and files == ["radio.jsonl"],
                     f"{descriptions[what]}: {failed} of {planned} decodes end as crc_fail by the decoder's own "
                     f"callback, {shares}; each keeps its key and its place in the denominator, and no bytes are kept")
    clean = [run for _, _, run in done if run.ok()]
    acknowledged = sum(total["decoded"] + total["payload_mismatch"] for run in clean for total in run.totals.values())
    matches = sum(total["payload_matches"] for run in clean for total in run.totals.values())
    grader.check(name, "never acknowledged", len(clean) == len(done) and acknowledged == 0 and matches == 0,
                 f"{acknowledged} of {sum(len(run.output.records) for run in clean)} decodes of bad input report a CRC "
                 f"pass and {matches} match the original bytes: the CRC flag is the decoder's own and nothing forces "
                 "an acknowledgement")
    return measured, done[0][2]


# ---------------------------------------------------------------- rerun
def run_rerun(config, out, grader, tool, failed_run, decoded_run):
    name, backends = "rerun", config["backends"]
    case_dir = out / "rerun"
    case_dir.mkdir()
    measured = {}
    for label, what, source in (("failed", "a failed opportunity", failed_run), ("decoded", "a decoded opportunity", decoded_run)):
        if source is None or not source.ok():
            grader.check(name, what, False, "the run it repeats did not finish")
            continue
        # The last opportunity of the archived plan, alone, from the same plan file.
        chosen = source.report["opportunities"][-1]
        status, report = run_native(tool, source.directory / "plan.json", case_dir / label / "native",
                                    case_dir / label / "native.log", ("--only", chosen["opportunity_id"]))
        try:
            output = rp.open_output(case_dir / label / "native")
        except ReplayPuschError as error:
            grader.check(name, what, False, f"exit status {status}: {error}")
            continue
        earlier = {record["backend"]: record for record in source.output.records
                   if record["opportunity_id"] == chosen["opportunity_id"]}
        again = {record["backend"]: record for record in output.records}
        same = set(again) == set(earlier) == set(backends) and all(
            again[backend][key] == earlier[backend][key] for backend in backends
            for key in ("outcome", "crc_ok", "payload_match", "decoded_sha256", "grant_record_sha256", "sfn", "slot"))
        digests = {record["ingress"]["readback_sha256"] for record in list(again.values()) + list(earlier.values())}
        measured[label] = {"opportunity_id": chosen["opportunity_id"], "ingress_sha256": sorted(digests),
                           "outcomes": {backend: again[backend]["outcome"] for backend in again}}
        outcomes = ", ".join("%s %s" % (backend, again[backend]["outcome"]) for backend in again)
        grader.check(name, what, status == 0 and same and len(digests) == 1 and len(output.records) == len(backends)
                     and report["selected"] == [chosen["opportunity_id"]],
                     f"{chosen['opportunity_id']} of {source.name}, alone, from the archived plan and the files it "
                     f"binds: the same grid ({next(iter(digests))[:16]}), and on each backend the same outcome "
                     f"({outcomes}) and the same decoded bytes (rerun/{label}/native)" if len(digests) == 1 else
                     f"the repeated grid differs from the first run's: {sorted(digests)}")
    return measured


# ---------------------------------------------------------------- refusals
def run_refusals(config, out, grader, tool, pair_run):
    name = "refusals"
    case_dir = out / "refusals"
    case_dir.mkdir()
    if pair_run is None or pair_run.plan is None:
        grader.check(name, "plan rules", False, "the plan of the reference pair was not built")
        return {}
    plan, channel = pair_run.plan, pair_run.channel
    second = plan["grants"][1]

    def shorter_block(item, directory):
        block = Path(second["tb"]["path"]).read_bytes()[:-1]
        (directory / "tb.bin").write_bytes(block)
        item["grants"][1].update(tb={"path": str(directory / "tb.bin"), "sha256": hashlib.sha256(block).hexdigest()},
                                 tb_bytes=len(block))

    def edited_record(item, directory):
        record = json.loads(Path(second["record"]["path"]).read_text())
        record["rnti"] += 1
        record["tb"] = {"hex": Path(second["tb"]["path"]).read_bytes().hex()}
        (directory / "grant.json").write_text(json.dumps(record))
        item["grants"][1]["record"] = replay.file_ref(directory / "grant.json")

    def channel_change(change):
        def apply(item, directory):
            edited = copy.deepcopy(channel)
            change(edited)
            (directory / "channel-plan.json").write_text(json.dumps(edited, indent=2, sort_keys=True) + "\n")
            item["channel_plan"] = replay.file_ref(directory / "channel-plan.json")
        return apply

    def other_slot(edited):
        edited["opportunities"][1]["slot"] = edited["opportunities"][0]["slot"]

    def twice(edited):
        edited["opportunities"][1]["opportunity_id"] = edited["opportunities"][0]["opportunity_id"]

    # Each rule with the words its refusal must carry: in the plan check, then in the native tool.
    plan_rules = [
        ("a transport block of another size", shorter_block, "mismatched payload", "mismatched payload"),
        ("a transport block with another hash", lambda item, _: item["grants"][1]["tb"].update(sha256="0" * 64),
         "record binds", "record binds"),
        ("a grant record that was edited", edited_record, "changed allocation", "changed allocation"),
        ("a grant without a record", lambda item, _: item["grants"].pop(1), "names no record", "names no record"),
        ("an opportunity in another slot than its grant", channel_change(other_slot),
         "depends on the slot index", "depends on the slot index"),
        ("an opportunity delivered twice", channel_change(twice), "listed twice", "listed twice"),
        ("a backend that does not exist", lambda item, _: item["receiver"].update(backends=["cpu", "fpga"]),
         "unknown backend", "unknown backend"),
        ("an observer on the control run", lambda item, _: item["receiver"].update(tap="o2"),
         "without an observer", "without an observer"),
        ("a generated payload on fixed blocks",
         lambda item, _: item["payload"].update(rule="per_opportunity", generator=grants.TB_GENERATOR, seed="other"),
         "payload rule generates", None),
    ]
    channel_rules = [
        ("a channel plan with another hash", lambda item, _: item["channel_plan"].update(sha256="0" * 64),
         "channel plan does not have the hash", "channel plan does not have the hash"),
        ("a channel plan without a gain", channel_change(lambda edited: edited.pop("gain")),
         "names no grid gain", "names no grid gain"),
        ("a transmit grid with another hash",
         channel_change(lambda edited: edited["grants"][1]["x"].update(sha256=edited["grants"][0]["x"]["sha256"])),
         "record binds", "record binds"),
    ]
    measured = {}
    for label, rules in (("plan rules", plan_rules), ("channel rules", channel_rules)):
        refused, accepted = [], []
        for number, (what, change, checker_words, native_words) in enumerate(rules):
            directory = case_dir / f"{label.split()[0]}-{number}"
            directory.mkdir()
            mutated = copy.deepcopy(plan)
            mutated["run_id"] = f"s24-refusal-{label.split()[0]}-{number}"
            change(mutated, directory)
            try:
                rp.check_files(mutated)
                checker = None
            except ReplayPuschError as error:
                checker = str(error)
            # The plan is written as it is, checks bypassed, so that the native tool meets it.
            (directory / "plan.json").write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n")
            named = bool(checker) and checker_words in checker
            if native_words is None:
                # The rule needs the generator, which only the offline side holds.
                measured[what] = {"plan_check": checker, "native": "not applicable: the tool generates no block"}
                (refused if named else accepted).append(what)
                continue
            status, _ = run_native(tool, directory / "plan.json", directory / "native", directory / "native.log")
            said = last_line(directory / "native.log")
            untouched = not any((directory / "native").iterdir())
            measured[what] = {"plan_check": checker, "native_status": status, "native_message": said,
                              "no_file_written": untouched}
            (refused if named and native_words in said and status == 2 and untouched else accepted).append(what)
        grader.check(name, label, not accepted and len(refused) == len(rules),
                     "refused by the plan check and by the native tool, each naming the rule, with status 2 before a "
                     "processor is created or a file is written: " + ", ".join(refused)
                     + ("" if not accepted else "; NOT REFUSED AS REQUIRED: " + ", ".join(accepted)))
    return measured


# ---------------------------------------------------------------- backends
def run_backends(config, grader, runs, rehearsal):
    name, backends = "backends", config["backends"]
    finished = [run for run in runs if run.ok()]
    records = [record for run in finished for record in run.output.records]
    measured = {"runs": len(runs), "runs_finished": len(finished), "records": len(records)}
    if not finished:
        grader.check(name, "executed path", False, "no native run finished")
        grader.check(name, "accounting", False, "no native run finished")
        return measured
    built = all(run.report["built_with_cuda"] is True for run in finished)
    available = all(backend["acceleration_available"] is True for run in finished for backend in run.report["backends"]
                    if backend["name"] == "cuda")
    paths = {backend: {} for backend in backends}
    wrong = []
    for record in records:
        executed = record["path"]["executed"] if isinstance(record["path"], dict) else None
        paths[record["backend"]][executed] = paths[record["backend"]].get(executed, 0) + 1
        ports = len(record["allocation"]["rx_ports"])
        wanted = "resident" if record["backend"] == "cuda" and ports in rp.RESIDENT_PORTS else "host"
        if record["outcome"] in rp.COMPLETED and executed != wanted:
            wrong.append(record)
    fallbacks = sorted({record["path"]["fallback"] for record in wrong if record["path"].get("fallback")})
    measured["paths"], measured["built_with_cuda"], measured["acceleration_available"] = paths, built, available
    shown = "; ".join(f"{backend}: " + ", ".join(f"{count} on the {path} path" for path, count in sorted(found.items(), key=str))
                      for backend, found in paths.items())
    if rehearsal:
        host_only = not built and all(set(found) <= {"host"} for found in paths.values())
        grader.check(name, "executed path", host_only,
                     f"REHEARSAL on a build without CUDA, not a hardware result: {shown}; every decode requested from "
                     f"the CUDA processor fell back to the host path and says so ({'; '.join(fallbacks) or 'no reason'}). "
                     "The CUDA decodes of this gate are open." if host_only else
                     f"--host-rehearsal was given on a build with CUDA: {shown}. Run the gate without it.")
    else:
        grader.check(name, "executed path", built and available and not wrong,
                     f"{shown}; the adapter is built with CUDA, the device is usable, every decode requested from the "
                     "CUDA processor ran its resident path and every decode requested from the host processor the "
                     "host path" if built and available and not wrong else
                     f"{shown}; built with CUDA: {built}, device usable: {available}; {len(wrong)} decodes did not "
                     f"execute the requested path ({'; '.join(fallbacks) or 'no reason recorded'}). A fallback cannot "
                     "close this gate.")
    planned = sum(run.report["totals"]["planned"] for run in finished)
    outcomes = {outcome: sum(record["outcome"] == outcome for record in records) for outcome in rp.OUTCOMES}
    anomalies = sum(len(run.report["anomalies"]) for run in finished)
    held = sum(run.report["totals"]["grids_not_released"] for run in finished)
    unique = all(len(run.output.by_key) == len(run.output.records) for run in finished)
    measured["outcomes"], measured["planned"] = outcomes, planned
    grader.check(name, "accounting", len(finished) == len(runs) and planned == len(records) and unique
                 and sum(outcomes.values()) == planned and anomalies == 0 and held == 0
                 and outcomes["timeout"] == outcomes["not_dispatched"] == outcomes["decoder_error"] == 0
                 and outcomes["payload_mismatch"] == 0,
                 f"{planned} decodes planned in {len(runs)} runs: {len(records)} records, each under a key that is "
                 f"unique in its run, each with one terminal outcome: "
                 + ", ".join(f"{count} {outcome}" for outcome, count in outcomes.items())
                 + f"; {anomalies} anomalies, {held} grids held after the run")
    for backend in backends:
        times = [record["timing"]["service_us"] for record in records if record["backend"] == backend
                 and record["timing"]["service_us"] is not None and not record["timing"]["first_use"]
                 and len(record["allocation"]["rx_ports"]) == 4 and record["allocation"]["prb_count"] == 51]
        if times:
            times.sort()
            measured[f"service_us_{backend}"] = {"count": len(times), "mean": sum(times) / len(times),
                                                 "p50": times[len(times) // 2], "max": times[-1]}
            grader.note(name, f"PUSCH service time of {backend}, not graded: mean {sum(times) / len(times):.0f} us, "
                              f"median {times[len(times) // 2]:.0f} us, largest {times[-1]:.0f} us over {len(times)} "
                              "decodes of 51 blocks on four receive ports, from the start of the decode to the "
                              "decoder callback, the first decode of each run excluded; the host is not tuned")
    return measured


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the run")
    parser.add_argument("--s22-run", required=True, type=Path,
                        help="run directory of the accepted Sprint 2.2 gate: it holds the pair and the slot grants")
    parser.add_argument("--s23-run", type=Path,
                        help="run directory of the accepted Sprint 2.3 gate: its stored grids are compared with the "
                             "grids decoded here")
    parser.add_argument("--grant-store", action="append", type=Path, default=[],
                        help="further directory that holds <grant_id>/store (default: the paths the entry recorded)")
    parser.add_argument("--adapter-build", type=Path, help="adapter build directory (default: from the configuration)")
    parser.add_argument("--host-rehearsal", action="store_true",
                        help="development host without a CUDA build: grade the host path and leave the hardware gate open")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"replay_pusch: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        config = load_config(args.config)
        receipt = native_adapter.verified_receipt(args.adapter_build or ROOT / config["adapter_build"])
        for needed in ("replay_pusch", "grant_fixture"):
            if needed not in receipt["executables"]:
                raise ConfigError(f"the adapter build holds no {needed}. Rebuild it: python scripts/native_adapter.py build")
        import numpy as np
    except (ConfigError, native_adapter.AdapterError, ImportError) as error:
        parser.exit(2, f"replay_pusch: {error}\n")
    tool = Path(receipt["executables"]["replay_pusch"]["path"])
    adapter = Path(receipt["executables"]["grant_fixture"]["path"])
    s22_run = args.s22_run.resolve()
    s23_run = args.s23_run.resolve() if args.s23_run else None
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "adapter-receipt.json", receipt)

    grader, measured, runs = Grade(), {}, []

    def guarded(case, action, default=None):
        before = len(grader.results)
        try:
            return action()
        except Exception as error:  # an unexpected fault fails its case and keeps the remaining cases
            (out / f"driver-error-{case}.txt").write_text(traceback.format_exc())
            grader.check(case, "driver", False, f"{type(error).__name__}: {error} (driver-error-{case}.txt)")
            return default
        finally:
            if len(grader.results) == before:
                grader.check(case, "driver", False, "the case produced no check")

    located, problem = None, None
    try:
        located = locate(config, s22_run, [path.resolve() for path in args.grant_store])
    except (GridApplyError, EpisodeError, GrantError, OSError) as error:
        problem = f"the stimulus of the Sprint 2.2 run is unusable: {error}"
    if located:
        entry, episode, channel, stores, search = located
        measured["unity"] = guarded("unity", lambda: run_unity(config, out, grader, np, tool, runs, channel, stores, search))
        measured["direct"] = guarded("direct", lambda: run_direct(config, out, grader, np, tool, runs, entry, episode, channel))
        measured["reference-pair"], pair_run = guarded(
            "reference-pair", lambda: run_pair(config, out, grader, np, tool, runs, entry, episode, channel, stores,
                                               s23_run), ({}, None))
        measured["payload"], _ = guarded(
            "payload", lambda: run_payload(config, out, grader, np, tool, runs, entry, episode, channel, stores, search,
                                           adapter, receipt), ({}, None))
        measured["bad-input"], failed_run = guarded(
            "bad-input", lambda: run_bad_input(config, out, grader, np, tool, runs, episode, channel), ({}, None))
        measured["rerun"] = guarded("rerun", lambda: run_rerun(config, out, grader, tool, failed_run, pair_run))
        measured["refusals"] = guarded("refusals", lambda: run_refusals(config, out, grader, tool, pair_run))
        measured["backends"] = guarded("backends", lambda: run_backends(config, grader, runs, args.host_rehearsal))
    else:
        for case in CASES:
            grader.check(case, "stimulus", False, problem)

    failed = [item for item in grader.results if not item["passed"]]
    passed_cases = [case for case in CASES if any(item["case"] == case for item in grader.results)
                    and all(item["passed"] for item in grader.results if item["case"] == case)]
    complete = not failed and len(passed_cases) == len(CASES)
    grade = {
        "schema": GRADE_SCHEMA, "selector": "replay_pusch", "run_id": out.name, "cases": len(CASES),
        "cases_passed": len(passed_cases), "cases_failed": len(CASES) - len(passed_cases),
        "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
        "judgment": ("rehearsal" if args.host_rehearsal else "pass") if complete else "fail",
        "hardware_gate": "open: the CUDA processor did not execute, this is a host rehearsal" if args.host_rehearsal
                         else ("closed" if complete else "open"),
        "reason": "every check of every case passed" if not failed else
                  f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
        "backends": config["backends"], "payload_rules": list(rp.PAYLOAD_RULES), "tolerance": config["tolerance"],
        "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    lock = {item["name"]: item["commit"] for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]}
    write_json(out / "manifest.json", {
        "run_id": out.name, "selector": "replay_pusch",
        "purpose": "Sprint 2.4 integration: the grids of the accepted Sprint 2.2 stimulus, filled by the channel "
                   "application of Sprint 2.3, decoded by the host and the CUDA PUSCH processor of the pinned source "
                   "and compared with the original transport blocks",
        "command": shlex.join([sys.executable, *sys.argv]), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "numpy": np.__version__, "parent_commit": git_commit(), "locked_sources": lock,
        "inputs": {str(args.config): digest(args.config)}, "driver_sha256": digest(__file__),
        "reference_sha256": digest(ROOT / "src/ocudu_cuda_isac/episodes/replay_pusch.py"),
        "s22_run": str(s22_run), "s23_run": str(s23_run) if s23_run else None,
        "grant_stores": [str(path.resolve()) for path in args.grant_store],
        "axes": {"split": "staged_harness", "receiver_stage": "none: the untapped control", "ue_realization":
                 "replayed channel, no UE", "host_stack": "DeepSig-OCUDU", "grid_memory": "host resource grid"},
        "runs": [{"name": run.name, "directory": str(run.directory.relative_to(out)), "status": run.status,
                  "plan_sha256": digest(run.directory / "plan.json") if (run.directory / "plan.json").is_file() else None,
                  "payload": run.plan["payload"] if run.plan else None} for run in runs],
        "adapter": {"executable": str(tool), "sha256": receipt["executables"]["replay_pusch"]["sha256"],
                    "grant_fixture_sha256": receipt["executables"]["grant_fixture"]["sha256"],
                    "receipt": "adapter-receipt.json", "ocudu_build": receipt["ocudu_build"],
                    "built_with_cuda": receipt["pin"]["cuda"]},
        "host_rehearsal": args.host_rehearsal,
        "environment": {key: os.environ[key] for key in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES",
                                                         "OCUDU_PUSCH_ACCELERATION_TIMING") if key in os.environ},
        "judgment": grade["judgment"], "reason": grade["reason"]})
    summary = (f"replay_pusch: {grade['cases_passed']} of {grade['cases']} cases passed, "
               f"{grade['checks_passed']} of {grade['checks']} checks"
               + (", host rehearsal: the hardware gate stays open" if args.host_rehearsal else ""))
    (out / "grade.txt").write_text("\n".join(grader.lines + [summary]) + "\n", encoding="ascii")
    print(summary)
    print(f"Run artifacts: {out}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
