#!/usr/bin/env bash
# Run every Sprint 4.1 gate (mps_pacing) on this host and keep the evidence in a new run directory.
# The gate runs the receiver and the sensor as two processes and two MPS clients on one GPU. It
# rebuilds the adapter with the route to the receiver's estimate and with the pacing core of this
# sprint against the patched receiver that Sprint 5.2 built, which is not rebuilt. It verifies
# the build of the sensing core that Sprint 3.6 accepted, or builds it. It starts an MPS control
# daemon of its own under build/mps, runs the integration driver, which launches every arm with
# scripts/run_m1.py on validation entries of the episode set of the accepted Sprint 3.5 run with
# the engine of the accepted Sprint 3.6 run, and writes mps_pacing.json. Service times, tails and
# latencies are recorded and not graded: Sprint 4.6 calibrates on these arms. Every paced run
# decodes a lead-in ahead of its schedule, so that the first use of the receiver is no release:
# the cold start is recorded apart, in pacing/lead_in.jsonl, and the gate requires that no
# release of a schedule is the first use of its processor. The gate never
# rebuilds OCUDU, never traces an episode, never reads a held-out episode and never edits the
# sources, a build of the receiver, an episode set or an earlier run. It ends the MPS daemon when
# it started it.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s4_1.sh [--run-dir DIR] [--s36-run DIR] [--s35-run DIR] [--core-build DIR]"
    echo "Run it on the DGX Spark that passed 'make accept-s3.6': it needs the accepted Sprint 3.6 run (default:"
    echo "the newest) and the Sprint 3.5 run it names, with the episode set of that run where it was, the patched"
    echo "receiver in build/ocudu-spark-o3 ('make native-build-o3'), NumPy and PyTorch in isac ('make env-full'),"
    echo "ONNX, ONNX Runtime and pybind11 in isac ('make core-env'), TensorRT, and nvidia-cuda-mps-control on PATH."
    echo "The core is verified in build/sense-core unless --core-build names another directory."
}
run_dir=""
s36_arg=""
s35_arg=""
core_dir=""
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --s36-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s36_arg=$2; shift 2 ;;
        --s35-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s35_arg=$2; shift 2 ;;
        --core-build)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            core_dir=$2; shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s4_1: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s41-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s4_1: $run_dir already exists. Evidence is never overwritten." >&2
    exit 2
fi
mkdir -p -- "$run_dir" || exit 2
run_dir=$(cd -- "$run_dir" && pwd)
core_dir=${core_dir:-"$PWD/build/sense-core"}
summary="$run_dir/summary.txt"
failed=0
note() { echo "$*" | tee -a "$summary"; }
pass() { note "PASS  $*"; }
fail() { note "FAIL  $*"; failed=1; }
# detail FILE: show the end of a failed gate's output under its FAIL line, on screen and in the summary.
detail() {
    [[ -s "$1" ]] || return 0
    grep -v '^[[:space:]]*$' "$1" | tail -n 5 | cut -c 1-300 | sed 's/^/      /' | tee -a "$summary"
}
# step LABEL LOG COMMAND...: run one gate, keep its output, record the result.
step() {
    local label=$1 log=$2
    shift 2
    if "$@" > "$run_dir/$log" 2>&1; then
        pass "$label"
    else
        fail "$label (see $log)"
        detail "$run_dir/$log"
    fi
}
# tests MODE LABEL LOG COMMAND...: run a unittest command. An empty run cannot close a gate, and
# with MODE "complete" neither can a run that skipped a test.
tests() {
    local mode=$1 label=$2 log=$3 status ran verdict skipped
    shift 3
    "$@" > "$run_dir/$log" 2>&1
    status=$?
    ran=$(grep -E '^Ran [0-9]+ tests? in ' "$run_dir/$log" | tail -n 1 | awk '{print $2}')
    verdict=$(grep -E '^(OK|FAILED)' "$run_dir/$log" | tail -n 1)
    skipped=$(echo "$verdict" | sed -n 's/.*skipped=\([0-9][0-9]*\).*/\1/p')
    ran=${ran:-0}
    skipped=${skipped:-0}
    if (( status != 0 )) || [[ "$verdict" != OK* ]]; then
        fail "$label ($ran ran; see $log)"
        grep -E '^(FAIL|ERROR): ' "$run_dir/$log" | head -n 10 | cut -c 1-300 | sed 's/^/      /' | tee -a "$summary"
    elif (( ran == 0 )); then
        fail "$label (none ran)"
    elif [[ "$mode" == complete ]] && (( skipped > 0 )); then
        fail "$label ($ran ran, $skipped skipped: the gate needs cmake, a C++17 compiler, NumPy and PyTorch in isac," \
            "'make env-full', and ONNX, ONNX Runtime and pybind11, 'make core-env'; see $log)"
    else
        pass "$label ($ran ran, $skipped skipped)"
    fi
}
# accepted RUN SPRINT: true when RUN holds the verdict of that sprint on this DGX Spark, without a failed line.
accepted() {
    grep -qE "^Sprint $2: ACCEPTED on this DGX Spark" "$1/summary.txt" 2>/dev/null && ! grep -qE '^FAIL  ' "$1/summary.txt"
}
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 4.1 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ -e .git ]]; then
    note "commit $(git rev-parse HEAD)"
    git status --short > "$run_dir/git-status.txt" 2>&1
    if [[ -s "$run_dir/git-status.txt" ]]; then
        fail "committed tree (uncommitted changes are listed in git-status.txt)"
    else
        pass "committed tree"
    fi
else
    note "source export without Git metadata"
fi
"$python" -m pip freeze --all > "$run_dir/python-packages.txt" 2>&1

# The portable suite, then the card's unit command exactly as the plan names it, then the driver
# on a stand-in control. The unit gate builds the pacing core on its own and runs its cases in
# CTest, and the driver test runs the real sensor of the host twin: neither may skip a test.
tests any "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
tests complete "S4.1 unit gate, run directly: python tests/unit/test_mps_pacing.py" mps-pacing-unit.log \
    "$python" tests/unit/test_mps_pacing.py -v
tests complete "S4.1 driver on a stand-in control, run directly: python tests/unit/test_mps_pacing_driver.py" \
    mps-pacing-driver.log "$python" tests/unit/test_mps_pacing_driver.py -v

# The adapter with the O3 route and the pacing core, out of tree against the patched receiver of
# Sprint 5.2, which is not rebuilt. The build itself starts no CUDA work.
adapter_ok=0
if [[ -d build/ocudu-spark-o3 ]]; then
    if "$python" scripts/native_adapter.py build --o3 --run-dir "$run_dir/adapter-build" > "$run_dir/adapter-build.log" 2>&1 \
            && "$python" scripts/native_adapter.py verify --o3 > "$run_dir/adapter-receipt.json" \
                2> "$run_dir/adapter-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
names, route, patch = receipt["executables"], receipt.get("o3_route") or {}, receipt.get("patch") or {}
pacing = [path for path in receipt["sources"] if "/pacing/" in path]
if not receipt["pin"]["cuda"] or route.get("compiled") is not True or not patch.get("series_sha256") or len(pacing) < 4:
    sys.exit("the receipt names no route, no patch, no CUDA or no pacing core")
print("pilot_ls %s with the route and the pacing core (%d source files of native/ocudu_adapter/pacing), obs_tail %s, "
      "series %s, with ENABLE_CUDA, %d adapter tests, %d compiler warnings"
      % (names["pilot_ls"]["sha256"][:12], len(pacing), names["obs_tail"]["sha256"][:12], patch["series_sha256"][:16],
         receipt["tests_run"], receipt["compiler_warnings"]))' "$run_dir/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        said=$(build/ocudu-adapter-o3/observer/pilot_ls --pacing 2>> "$run_dir/adapter-verify.log" | head -n 1)
        if [[ -n "$facts" && "$said" == "pilot_ls pacing: pilot-ls-plan/v4"*"lead-in ahead of the schedule"* ]]; then
            adapter_ok=1
            pass "S4.1 adapter: built against the patched receiver of Sprint 5.2 and verified, $facts; its unit cases" \
                "ran the pacing core in CTest; $said (adapter-receipt.json)"
        else
            fail "S4.1 adapter: the receipt names no route, no patch, no CUDA or no pacing core, or pilot_ls does not" \
                "read a plan of version 4 with a lead-in (see adapter-verify.log)"
            detail "$run_dir/adapter-verify.log"
        fi
    else
        fail "S4.1 adapter: build or verification failed (see adapter-build.log, adapter-verify.log)"
        detail "$run_dir/adapter-build.log"
        detail "$run_dir/adapter-verify.log"
    fi
else
    fail "S4.1 adapter: build/ocudu-spark-o3 holds no patched receiver. Build it once: make native-build-o3"
fi

# The sensing core is the one Sprint 3.6 accepted: this sprint changes no source of it. A build
# that no longer verifies against the tree is built again, with its TensorRT backend.
core_ok=0
if "$python" scripts/native_core.py verify --build-dir "$core_dir" > "$run_dir/isac-core.json" 2> "$run_dir/core-verify.log"; then
    built="verified as it was built"
else
    built="built again"
    "$python" scripts/native_core.py build --tensorrt on --build-dir "$core_dir" --run-dir "$run_dir/core-build" \
        > "$run_dir/core-build.log" 2>&1
    "$python" scripts/native_core.py verify --build-dir "$core_dir" > "$run_dir/isac-core.json" 2>> "$run_dir/core-verify.log"
fi
facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
tools = receipt["executables"]
if receipt["device"] != "cuda" or not receipt["tensorrt"] or "sense_host" not in tools or "sense_feed" not in tools:
    sys.exit("the receipt names no CUDA device, no TensorRT, no sense_host or no sense_feed")
print("sources %s, CUDA architecture %s, TensorRT %s, sense_host %s, sense_feed %s"
      % (receipt["sources_sha256"][:16], receipt["cuda_architectures"], receipt["tensorrt"]["version"],
         tools["sense_host"]["sha256"][:12], tools["sense_feed"]["sha256"][:12]))' "$run_dir/isac-core.json" \
    2>> "$run_dir/core-verify.log")
if [[ -n "$facts" ]]; then
    core_ok=1
    pass "S4.1 sensing core: $core_dir $built against its receipt and the tree: $facts (isac-core.json)"
    ldd "$core_dir/sense_host" > "$run_dir/sense-host-ldd.txt" 2>&1
    if awk '{print $1}' "$run_dir/sense-host-ldd.txt" | sed 's|.*/||' | grep -qiE 'python|ocudu|srsran|aerial|cuphy'; then
        fail "S4.1 sensor process: sense_host resolves a library of an interpreter or of a receiver (sense-host-ldd.txt)"
    else
        pass "S4.1 sensor process: the loader resolves $(grep -c '=>' "$run_dir/sense-host-ldd.txt") libraries for" \
            "sense_host, none of an interpreter, of a receiver or of Aerial (sense-host-ldd.txt)"
    fi
else
    fail "S4.1 sensing core: $core_dir is no build for this GPU with TensorRT (see core-verify.log, core-build.log)"
    detail "$run_dir/core-verify.log"
fi

# The MPS control interface of this host. The driver starts a daemon of its own under build/mps.
mps_ok=0
started_here=0
if command -v nvidia-cuda-mps-control > /dev/null 2>&1; then
    "$python" scripts/run_m1.py mps status > "$run_dir/mps-before.json" 2> "$run_dir/mps-before.log"
    if grep -q '"alive": true' "$run_dir/mps-before.json" 2>/dev/null; then
        note "NOTE  S4.1 an MPS control daemon answers on build/mps/pipe already: the gate uses it and leaves it running"
    else
        started_here=1
    fi
    mps_ok=1
    pass "S4.1 MPS: nvidia-cuda-mps-control is $(command -v nvidia-cuda-mps-control); the daemon of this gate keeps its" \
        "pipe and its log under build/mps, apart from any daemon of the system"
else
    fail "S4.1 MPS: nvidia-cuda-mps-control is not on PATH. It ships with the NVIDIA driver: without it no arm can be" \
        "labeled MPS"
fi

# The control is the run that Sprint 3.6 accepted on this DGX Spark: the newest one, or the one
# named. It holds the engine and names the Sprint 3.5 run with the artifact and the episode set.
s36_run=""
if [[ -n "$s36_arg" ]]; then
    candidates=("$s36_arg")
else
    candidates=(results/runs/s36-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if accepted "$candidate" '3\.6' && [[ -f "$candidate/sense_core/grade.json" \
            && -f "$candidate/sense_core/engine/engine/engine.json" ]]; then
        s36_run=$candidate
    fi
done
if [[ -n "$s36_run" ]]; then
    s36_run=$(cd -- "$s36_run" && pwd)
    pass "S4.1 control: the Sprint 3.6 run $(basename -- "$s36_run") is ACCEPTED on this DGX Spark and holds its engine" \
        "(sense_core/engine/engine/engine.json)"
else
    fail "S4.1 control: no Sprint 3.6 verdict ACCEPTED on this DGX Spark with its engine in ${s36_arg:-results/runs/s36-*}"
    note "      Run 'make accept-s3.6' on this host first, or name its run with --s36-run."
fi

grade_ok=0
if (( adapter_ok && core_ok && mps_ok )) && [[ -n "$s36_run" ]]; then
    note "Receiver and sensor as two MPS clients, on the controls of $(basename -- "$s36_run"), with" \
        "configs/integration/mps_pacing.json."
    "$python" tests/integration/mps_pacing.py --config configs/integration/mps_pacing.json --s36-run "$s36_run" \
        ${s35_arg:+--s35-run "$s35_arg"} --core-build "$core_dir" --output-dir "$run_dir/mps_pacing" \
        > "$run_dir/mps-pacing.log" 2>&1
    status=$?
    out="$run_dir/mps_pacing"
    graded=$(grep -cE '^(PASS|FAIL)  ' "$out/grade.txt" 2>/dev/null)
    graded=${graded:-0}
    grep -E '^(PASS|FAIL|NOTE)  ' "$out/grade.txt" 2>/dev/null | tee -a "$summary"
    if grep -qE '^FAIL  ' "$out/grade.txt" 2>/dev/null; then failed=1; fi
    counts=$("$python" -c 'import json, sys
grade = json.load(open(sys.argv[1], encoding="utf-8"))
ok = grade["judgment"] == "pass" and grade["cases"] >= 1 and grade["cases_passed"] == grade["cases"] \
    and grade["checks_failed"] == 0 and grade["hardware_gate"] == "closed" and grade["rehearsal"] is False
print("%d of %d cases, %d of %d checks, judgment %s, hardware gate %s, control in effect %s"
      % (grade["cases_passed"], grade["cases"], grade["checks_passed"], grade["checks"], grade["judgment"],
         grade["hardware_gate"].split(":")[0], grade["control_in_effect"]))
sys.exit(0 if ok else 1)' "$out/grade.json" 2> "$run_dir/grade-read.log")
    counted=$?
    if (( status == 0 && counted == 0 && graded > 0 )); then
        grade_ok=1
        pass "S4.1 integration grade: $counts (mps_pacing/grade.json)"
    elif (( graded == 0 )); then
        fail "S4.1 integration driver produced no graded checks (see mps-pacing.log)"
        detail "$run_dir/mps-pacing.log"
    else
        fail "S4.1 integration grade: ${counts:-unreadable}, driver exit status $status (mps_pacing/grade.json)"
    fi

    # The record of the card, read here with the standard library alone. The gate requires that it
    # exists, that it is not graded as a latency, that every timed arm carries the label MPS on a
    # proof, and that the load is stated. What the figures are is recorded and decides nothing.
    record=$("$python" -c 'import json, sys
record = json.load(open(sys.argv[1] + "/mps_pacing.json", encoding="utf-8"))
if record["graded"] is not False or record["load"]["scripted"] is not True or record["rehearsal"] is not False:
    sys.exit("the record is graded, names no scripted load, or is that of a rehearsal")
arms = record["arms"]
for name in ("dsp_only", "capture_only", "ai_only", "concurrent"):
    runs = [item for item in arms.get(name, []) if not item["warmup"] and item["control"] == "mps_default"
            and item["fault"] is None and item["sharing"]["requested"] == "mps"]
    if not runs or any(item["sharing"]["label"] != "mps" or not item["sharing"]["verified"] for item in runs):
        sys.exit("the arm %s holds no repetition, or one without a proven label MPS" % name)
service = record["contention"]["service"]
shift = record["contention"]["shifts"]["mps_default"]
cold = record["cold_start"]
if cold["lead_in"] is None:
    apart = "no lead-in: the first release of a run is the first use of its receiver"
else:
    if cold["first_use_in_schedule"] != 0 or cold["skipped_first_releases"] != 0 or cold["runs"] < 1:
        sys.exit("a release of a schedule is the first use of its receiver, or the cold start cost a release")
    apart = ("the cold start apart, in a lead-in of %d decodes and in no statistic: the first decode of a receiver "
             "%.1f us in the median of %d runs, the first release of a schedule %.1f us"
             % (cold["lead_in"]["releases"], cold["by_decode_us"][0]["p50_us"], cold["runs"],
                cold["first_release_us"]["p50_us"]))
print("mps_pacing.json holds %d launches; PUSCH service time, median / 99th percentile in us: tap off %.1f / %.1f, "
      "capture only %.1f / %.1f, concurrent %.1f / %.1f; the %s moved by %.1f us against %.1f us allowed by the %s "
      "rule; control in effect %s; %s; load scripted, no core network attached; \"graded\": false"
      % (sum(len(items) for items in arms.values()), service["dsp_only"]["p50_us"], service["dsp_only"]["p99_us"],
         service["capture_only"]["p50_us"], service["capture_only"]["p99_us"], service["concurrent"]["p50_us"],
         service["concurrent"]["p99_us"], shift["quantile"], shift["shift_us"], shift["allowed_us"],
         record["contention"]["rule"]["status"], record["contention"]["control_in_effect"], apart))' "$out" \
        2> "$run_dir/records-read.log")
    if [[ -n "$record" ]]; then
        pass "S4.1 record, written and not graded: $record"
    else
        fail "S4.1 record: mps_pacing.json is missing, graded, incomplete, holds a timed arm without a proven label, or" \
            "names a release as the first use of its receiver (see records-read.log)"
        detail "$run_dir/records-read.log"
    fi

    # A second reader, with tools that are not the driver's: the command-line reader of the
    # reference on what sense_host wrote beside the receiver, and grep on the releases of that arm.
    arm=$(ls -d "$out"/arms/r0-concurrent 2>/dev/null | head -n 1)
    if [[ -n "$arm" ]] && "$python" -m ocudu_cuda_isac.sensing.cusense show "$arm/sensor" --window 0 \
            > "$run_dir/recovered.json" 2> "$run_dir/recovered.log"; then
        releases=$(grep -cE '"schema": ?"pace-release/v1"' "$arm/native/pacing/releases.jsonl" 2>> "$run_dir/recovered.log")
        radio=$(grep -cE '"outcome": ?"decoded"' "$arm/native/radio/radio.jsonl" 2>> "$run_dir/recovered.log")
        skipped=$(grep -cE '"outcome": ?"skipped"' "$arm/native/pacing/releases.jsonl" 2>> "$run_dir/recovered.log")
        # The lead-in of that arm, when its plan names one: its decodes are in a file of their own, and
        # with it no radio record is the first use of its processor.
        ahead=$(grep -cE '"schema": ?"pace-lead-in/v1"' "$arm/native/pacing/lead_in.jsonl" 2>/dev/null)
        ahead=${ahead:-0}
        cold=$(grep -cE '"first_use": ?true' "$arm/native/radio/radio.jsonl" 2>> "$run_dir/recovered.log")
        cold=${cold:-0}
        shown=$("$python" -c 'import json, sys
shown = json.load(open(sys.argv[1], encoding="utf-8"))
window = shown["window"]
print("the reader of the reference verifies the output of sense_host beside the receiver: the model %s in the runtime "
      "%s on %s scored %d windows; window %s has the logit %.4f, %s the threshold"
      % (shown["model_id"], shown["runtime"], shown["device"], shown["scored_windows"], window["id"], window["logit"],
         "above" if window["present"] else "at or below"))' "$run_dir/recovered.json" 2>> "$run_dir/recovered.log")
        if [[ -n "$shown" ]] && (( releases > 0 && radio + skipped == releases && (ahead == 0 || cold == 0) )); then
            pass "S4.1 a second reader recovers a decision and the releases of one concurrent arm: $shown; grep counts" \
                "$releases releases in pacing/releases.jsonl, $skipped skipped, and $radio decoded radio records;" \
                "$ahead decodes of a lead-in in pacing/lead_in.jsonl, none of them a release or a radio record, and" \
                "$cold radio records that are the first use of their processor (recovered.json)"
        else
            fail "S4.1 a second reader: $releases releases, $skipped skipped and $radio decoded radio records in" \
                "$(basename -- "$arm"), $cold of them the first use of their processor beside $ahead decodes of a" \
                "lead-in, or no decision (see recovered.log)"
        fi
    else
        fail "S4.1 a second reader recovers a decision of a concurrent arm (see recovered.log)"
        detail "$run_dir/recovered.log"
    fi
else
    note "SKIP  S4.1 integration driver, record and second reader: no adapter, no core, no MPS or no control"
fi

# The daemon this gate started is ended. One that answered before the gate is left as it was.
if (( started_here )); then
    if "$python" scripts/run_m1.py mps stop > "$run_dir/mps-stop.log" 2>&1; then
        note "NOTE  S4.1 the MPS control daemon of this gate was ended (mps-stop.log)"
    else
        note "NOTE  S4.1 the MPS control daemon of this gate did not answer a quit: python scripts/run_m1.py mps stop"
    fi
fi
step "S4.1 sources at the locked revisions after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees

note "Evidence: $run_dir"
if (( failed || ! grade_ok )); then
    note "Sprint 4.1: NOT ACCEPTED on this host"
    exit 1
fi
note "Sprint 4.1: ACCEPTED on this DGX Spark. The receiver and the sensor run as two processes and two clients of one" \
    "MPS server on one GPU. The staged harness is paced on absolute release times, with nothing of its own allocated" \
    "per decode, and every arm decodes the same realization on the same offered schedule: tap off, capture only," \
    "buffered records to the sensor alone, and both together. Both process identifiers were read from the client list" \
    "in one reading, the sensor's context existed before the receiver, and a sensor that was killed did not reach the" \
    "receiver. The first use of the receiver is a decode of the lead-in ahead of each schedule: the cold start is" \
    "recorded apart and is a sample of no statistic. Service times, tails and latencies are recorded in" \
    "mps_pacing.json and are not graded. These arms are the timed half of the Sprint 3.6 Integrate step. Next: S4.6," \
    "which calibrates on them."
