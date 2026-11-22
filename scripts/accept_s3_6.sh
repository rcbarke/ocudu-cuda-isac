#!/usr/bin/env bash
# Run every Sprint 3.6 gate (sense_core) on this host and keep the evidence in a new run directory.
# The gate builds the native sensing core, native/sense_core, and runs it beside the reference of
# Sprint 3.5 on the held-out records of the accepted Sprint 3.5 run, with the artifact that run
# froze. It starts with checkpoint D0: a one-layer TensorRT engine is built and run on this GPU
# before anything else. Then the core is built with its CUDA device and its TensorRT backend, its
# unit cases run in CTest with the kernels on the device, the artifact is exported to ONNX and
# its engine is built on this GPU, and the integration driver replays the held-out archives
# through sense_host and through the reference, publishes them again on an observation ring for
# both sensors, and writes runtime_parity.json, runtime_latency.json and cusense_reference.json.
# No receiver runs here and no launcher: the gate never rebuilds OCUDU, never traces an episode
# and never edits the sources, a build of the receiver, an episode set or an earlier run. The
# latency beside the published figure is recorded and not graded. The timed arms with the
# receiver beside the sensor on one GPU are Sprint 4.1's, and this ticket closes with checkpoint D4.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s3_6.sh [--run-dir DIR] [--s35-run DIR] [--build-dir DIR]"
    echo "Run it on the DGX Spark that passed 'make accept-s3.5': it needs the accepted Sprint 3.5 run (default:"
    echo "the newest), NumPy and PyTorch in isac ('make env-full'), ONNX, ONNX Runtime and pybind11 in isac"
    echo "('make core-env'), the CUDA toolkit with nvcc,"
    echo "and TensorRT with its headers: a package installation, TENSORRT_ROOT, or the wheel tensorrt_cu13 in isac"
    echo "with the headers that 'python scripts/native_core.py headers' fetches. The core is built into"
    echo "build/sense-core unless --build-dir names another directory. A development host without a GPU"
    echo "rehearses the driver on the host twin: see docs/interfaces/SENSE_CORE.md."
}
run_dir=""
s35_arg=""
build_dir=""
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --s35-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s35_arg=$2; shift 2 ;;
        --build-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            build_dir=$2; shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s3_6: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s36-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s3_6: $run_dir already exists. Evidence is never overwritten." >&2
    exit 2
fi
mkdir -p -- "$run_dir" || exit 2
run_dir=$(cd -- "$run_dir" && pwd)
build_dir=${build_dir:-"$PWD/build/sense-core"}
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

note "Sprint 3.6 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

# Checkpoint D0, before anything is built: TensorRT with its headers on this host, a one-layer
# engine built on the GPU, executed on a stream and read back. The demonstration needs this backend.
probe_ok=0
if "$python" scripts/native_core.py probe --build-dir "$build_dir" --output "$run_dir/sense-probe.json" \
        > "$run_dir/probe.log" 2>&1; then
    probe_ok=1
    pass "S3.6 D0 probe: $(tail -n 1 "$run_dir/probe.log" | sed 's/^native_core: //' | cut -c 1-600)"
else
    fail "S3.6 D0 probe: this host has no usable TensorRT backend (see probe.log, sense-probe.json)"
    detail "$run_dir/probe.log"
    note "      The declared fallback is the LibTorch backend, and a run on it records the native showing as blocked."
    note "      A wheel without headers: python scripts/native_core.py headers. Another place: export TENSORRT_ROOT."
fi

# The portable suite, then the card's unit command exactly as the plan names it, then the two
# rules of the core: what its Python package may import and what its binaries may link. The unit
# gate builds the host twin and runs the unit cases of the core in CTest, so none may be skipped.
tests any "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
tests complete "S3.6 unit gate, run directly: python tests/unit/test_sense_core.py" sense-core-unit.log \
    "$python" tests/unit/test_sense_core.py -v
tests complete "S3.6 link rule, run directly: python tests/unit/test_core_link.py" core-link.log \
    "$python" tests/unit/test_core_link.py -v
tests complete "S3.6 import rule, run directly: python tests/unit/test_core_imports.py" core-imports.log \
    "$python" tests/unit/test_core_imports.py -v

# The core with its CUDA device and its TensorRT backend, out of tree, with its unit cases in CTest.
build_ok=0
if (( probe_ok )); then
    if "$python" scripts/native_core.py build --tensorrt on --build-dir "$build_dir" --run-dir "$run_dir/core-build" \
            > "$run_dir/core-build.log" 2>&1 \
            && "$python" scripts/native_core.py verify --build-dir "$build_dir" > "$run_dir/isac-core.json" \
                2> "$run_dir/core-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
tests, tools = receipt["tests"], receipt["executables"]
if receipt["device"] != "cuda" or not receipt["tensorrt"] or "sense_engine" not in tools or not receipt["binding"]:
    sys.exit("the receipt names no CUDA device, no TensorRT, no sense_engine or no binding")
if tests["device_vectors"] != "ran":
    sys.exit("the unit case of the kernels on the device " + tests["device_vectors"])
print("%d source files %s, CUDA architecture %s by %s, TensorRT %s (%s), %d unit cases with the kernels on the device, "
      "%d compiler warnings, sense_host %s, sense_feed %s, sense_engine %s, the binding %s"
      % (receipt["source_files"], receipt["sources_sha256"][:16], receipt["cuda_architectures"],
         receipt["toolchain"]["cuda_compiler"], receipt["tensorrt"]["version"], receipt["tensorrt"]["origin"],
         tests["ran"], receipt["compiler_warnings"], tools["sense_host"]["sha256"][:12],
         tools["sense_feed"]["sha256"][:12], tools["sense_engine"]["sha256"][:12], receipt["binding"]["sha256"][:12]))' \
            "$run_dir/isac-core.json" 2>> "$run_dir/core-verify.log")
        if [[ -n "$facts" ]]; then
            build_ok=1
            pass "S3.6 core: built out of tree and verified against its receipt: $facts (isac-core.json)"
        else
            fail "S3.6 core: the receipt is not that of a build for this GPU with TensorRT (see core-verify.log)"
            detail "$run_dir/core-verify.log"
        fi
    else
        fail "S3.6 core: build or verification failed (see core-build.log, core-verify.log)"
        detail "$run_dir/core-build.log"
        detail "$run_dir/core-verify.log"
    fi
    # The sensor is a native executable: what the loader resolves for it names no interpreter.
    if (( build_ok )); then
        ldd "$build_dir/sense_host" > "$run_dir/sense-host-ldd.txt" 2>&1
        if awk '{print $1}' "$run_dir/sense-host-ldd.txt" | sed 's|.*/||' | grep -qiE 'python|ocudu|srsran|aerial|cuphy'; then
            fail "S3.6 sensor process: sense_host resolves a library of an interpreter or of a receiver (sense-host-ldd.txt)"
        else
            pass "S3.6 sensor process: the loader resolves $(grep -c '=>' "$run_dir/sense-host-ldd.txt") libraries for" \
                "sense_host, none of an interpreter, of a receiver or of Aerial (sense-host-ldd.txt)"
        fi
    fi
else
    note "SKIP  S3.6 core build with the TensorRT backend: the probe failed"
fi

# The control is the run that Sprint 3.5 accepted on this DGX Spark: the newest one, or the one
# named. It holds the frozen artifact, the output of the reference sensor on the held-out
# records, and the records that sensor kept.
s35_run=""
if [[ -n "$s35_arg" ]]; then
    candidates=("$s35_arg")
else
    candidates=(results/runs/s35-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if accepted "$candidate" '3\.5' && [[ -f "$candidate/cusense_frontend/recover.json" \
            && -f "$candidate/cusense_frontend/grade.json" ]]; then
        s35_run=$candidate
    fi
done
if [[ -n "$s35_run" ]]; then
    s35_run=$(cd -- "$s35_run" && pwd)
    pass "S3.6 control: the Sprint 3.5 run $(basename -- "$s35_run") is ACCEPTED on this DGX Spark and holds its frozen" \
        "artifact and its held-out records (cusense_frontend/recover.json)"
else
    fail "S3.6 control: no Sprint 3.5 verdict ACCEPTED on this DGX Spark with its artifact and its held-out records in" \
        "${s35_arg:-results/runs/s35-*}"
    note "      Run 'make accept-s3.5' on this host first, or name its run with --s35-run."
fi

grade_ok=0
if (( build_ok )) && [[ -n "$s35_run" ]]; then
    note "Native sensing core on the held-out records of $(basename -- "$s35_run"), with" \
        "configs/integration/sense_core.json."
    "$python" tests/integration/sense_core.py --config configs/integration/sense_core.json --s35-run "$s35_run" \
        --core-build "$build_dir" --output-dir "$run_dir/sense_core" > "$run_dir/sense-core.log" 2>&1
    status=$?
    out="$run_dir/sense_core"
    graded=$(grep -cE '^(PASS|FAIL)  ' "$out/grade.txt" 2>/dev/null)
    graded=${graded:-0}
    grep -E '^(PASS|FAIL|NOTE)  ' "$out/grade.txt" 2>/dev/null | tee -a "$summary"
    if grep -qE '^FAIL  ' "$out/grade.txt" 2>/dev/null; then failed=1; fi
    counts=$("$python" -c 'import json, sys
grade = json.load(open(sys.argv[1], encoding="utf-8"))
ok = grade["judgment"] == "pass" and grade["cases"] >= 1 and grade["cases_passed"] == grade["cases"] \
    and grade["checks_failed"] == 0 and grade["hardware_gate"] == "closed" and grade["rehearsal"] is False
print("%d of %d cases, %d of %d checks, judgment %s, hardware gate %s"
      % (grade["cases_passed"], grade["cases"], grade["checks_passed"], grade["checks"], grade["judgment"],
         grade["hardware_gate"].split(":")[0]))
sys.exit(0 if ok else 1)' "$out/grade.json" 2> "$run_dir/grade-read.log")
    counted=$?
    if (( status == 0 && counted == 0 && graded > 0 )); then
        grade_ok=1
        pass "S3.6 integration grade: $counts (sense_core/grade.json)"
    elif (( graded == 0 )); then
        fail "S3.6 integration driver produced no graded checks (see sense-core.log)"
        detail "$run_dir/sense-core.log"
    else
        fail "S3.6 integration grade: ${counts:-unreadable}, driver exit status $status (sense_core/grade.json)"
    fi

    # The three records of the card, read here with the standard library alone. The gate requires
    # that they exist, that the comparison with the published figure is not graded and that every
    # figure has its count. What the figures are is recorded and decides nothing.
    record=$("$python" -c 'import json, sys
out = sys.argv[1]
parity = json.load(open(out + "/runtime_parity.json", encoding="utf-8"))
latency = json.load(open(out + "/runtime_latency.json", encoding="utf-8"))
reference = json.load(open(out + "/cusense_reference.json", encoding="utf-8"))
totals, native, other = parity["totals"], latency["native"], latency["reference"]
if reference["graded"] is not False or latency["graded"] is not False:
    sys.exit("a record of the latency is marked as graded")
if totals["problems"] != 0 or totals["windows"] < 1:
    sys.exit("the parity record holds a problem or no window")
for name in ("end_to_end", "post_window", "cusense_boundary"):
    if native[name]["count"] < 1 or native[name]["p99_us"] is None:
        sys.exit("the native latency %s has no count or no tail" % name)
if not native["timing_valid"] or native["engine"] != "tensorrt" or len(reference["differences"]) < 3:
    sys.exit("the native latency is not that of the CUDA device with TensorRT, or the differences are not stated")
boundary, published = native["cusense_boundary"], reference["published"]["live_inference_us"]["value"]
beside = "" if other is None else (
    "; the reference runtime on %s by the same boundary: %.0f us in the median and %.0f us at the 99th percentile over "
    "%d windows" % (other["device"], other["cusense_boundary"]["p50_us"], other["cusense_boundary"]["p99_us"],
                    other["cusense_boundary"]["count"]))
print("runtime_parity.json holds %d windows scored by both runtimes: %d decisions agree, %d flip(s), the largest logit "
      "difference %.2e; runtime_latency.json holds the three latency definitions with counts and tails: native, by the "
      "cuSense boundary, %.1f us in the median, %.1f us at the 99th percentile and %.1f us at most over %d windows on %s "
      "with %s%s; cusense_reference.json puts that beside the published %d us with \"graded\": false and lists %d "
      "differences"
      % (totals["windows"], totals["decisions_agree"], len(totals["flips"]), totals["largest_logit_difference"],
         boundary["p50_us"], boundary["p99_us"], boundary["max_us"], boundary["count"], native["device"],
         native["engine_detail"], beside, published, len(reference["differences"])))' "$out" 2> "$run_dir/records-read.log")
    if [[ -n "$record" ]]; then
        pass "S3.6 records, written and not graded: $record"
    else
        fail "S3.6 records: runtime_parity.json, runtime_latency.json or cusense_reference.json is missing, graded or" \
            "incomplete (see records-read.log)"
        detail "$run_dir/records-read.log"
    fi

    # A second reader, with tools that are not the driver's: the command-line reader of the
    # reference on what sense_host wrote, and sha256sum on the engine against its manifest.
    native_out=$(ls -d "$out"/parity/*-native 2>/dev/null | head -n 1)
    frozen=$("$python" -c 'import json, sys
found = json.load(open(sys.argv[1] + "/recover.json", encoding="utf-8"))
print(json.load(open(sys.argv[1] + "/" + found["freeze"], encoding="utf-8"))["artifact"]["sha256"])' \
        "$s35_run/cusense_frontend" 2> "$run_dir/recovered.log")
    if [[ -n "$native_out" && -n "$frozen" ]] \
            && "$python" -m ocudu_cuda_isac.sensing.cusense show "$native_out" --window 0 > "$run_dir/recovered.json" \
                2>> "$run_dir/recovered.log"; then
        shown=$("$python" -c 'import json, sys
shown, engine = json.load(open(sys.argv[1], encoding="utf-8")), json.load(open(sys.argv[2], encoding="utf-8"))
window = shown["window"]
print(shown["artifact_sha256"])
print(engine["sha256"])
print(engine["artifact_sha256"])
print("the reader of the reference verifies the output of sense_host: the model %s, loaded %d time(s) on %s in the "
      "runtime %s, scored %d windows; window %s from %d observations has the logit %.4f, %s the threshold, with its "
      "largest value %.4f in the cell %s"
      % (shown["model_id"], shown["loads"], shown["device"], shown["runtime"], shown["scored_windows"], window["id"],
         window["contributing"], window["logit"], "above" if window["present"] else "at or below", window["map_peak"],
         window["cell"]))' "$run_dir/recovered.json" "$out/engine/engine/engine.json" 2>> "$run_dir/recovered.log")
        used=$(echo "$shown" | sed -n 1p)
        named=$(echo "$shown" | sed -n 2p)
        built_from=$(echo "$shown" | sed -n 3p)
        text=$(echo "$shown" | sed -n 4p)
        plan=$(sha256sum "$out/engine/engine/model.plan" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
        if [[ -n "$text" && "$used" == "$frozen" && "$built_from" == "$frozen" && "$plan" == "$named" ]]; then
            pass "S3.6 a second reader recovers a native decision and the engine: $text; the artifact ${frozen:0:16} is" \
                "the one the freeze record of the Sprint 3.5 run names, and the engine built from it has the SHA-256" \
                "${plan:0:16} by sha256sum, as its manifest says (recovered.json)"
        else
            fail "S3.6 a second reader: the freeze record names the artifact ${frozen:-none}, sense_host scored with" \
                "${used:-none}, the engine names ${built_from:-none}, and model.plan has ${plan:-unreadable} against" \
                "${named:-none} in its manifest"
        fi
    else
        fail "S3.6 a second reader recovers a native decision and the engine (see recovered.log)"
        detail "$run_dir/recovered.log"
    fi
else
    note "SKIP  S3.6 integration driver, records and second reader: no build of the core with TensorRT, or no control"
fi

# This sprint adds a directory of its own and changes nothing of the receiver side: the adapter
# of Sprint 5.2, where this host still holds its build, must verify against the tree as it is.
if [[ -f build/ocudu-adapter-o3/isac-adapter.json ]]; then
    step "S3.6 receiver side untouched: the adapter build of Sprint 5.2 verifies against the tree (adapter-receipt.json)" \
        adapter-receipt.json "$python" scripts/native_adapter.py verify --o3
fi
step "S3.6 sources at the locked revisions after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees

note "Evidence: $run_dir"
if (( failed || ! grade_ok )); then
    note "Sprint 3.6: NOT ACCEPTED on this host"
    exit 1
fi
note "Sprint 3.6: ACCEPTED on this DGX Spark. The native sensing core runs the cuSense-like branch of Sprint 3.5: the" \
    "ingest and the front end as CUDA kernels and the model as a TensorRT engine built on this GPU, one submission per" \
    "window on one stream. On the held-out records of the accepted Sprint 3.5 run it gives the windows, the scores and" \
    "the decisions of the reference within the declared tolerance, and its latency by the cuSense boundary is recorded" \
    "beside the published figure and is not graded. This is the unit acceptance of the ticket: the timed arms with the" \
    "receiver beside the sensor on one GPU are Sprint 4.1's, and the ticket closes with checkpoint D4. Next: S4.1."
