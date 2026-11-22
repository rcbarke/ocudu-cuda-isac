#!/usr/bin/env bash
# Run every Sprint 5.2 gate (native_estimate) on this host and keep the evidence in a new run directory.
# The gate exports the receiver's own channel estimate, the stage O3: what the resident CUDA kernels
# of the pinned receiver leave on the device after pilot least squares, frequency interpolation and
# frequency smoothing. It checks the patch series of patches/ocudu/ against the pinned commit, builds
# the patched receiver from a separate tree into its own build directory, builds the adapter twice,
# against the accepted Sprint 1.1 build without the route and against the patched build with it, and
# decodes the plans of the accepted Sprint 3.1 run once more with the patched receiver. Every decode
# is compared with the one the unpatched build wrote without an observer, the stage O2 with the
# accepted one, and every exported estimate with the matched estimator on the stage O2 of its slot.
# A host stub, an estimate that was never smoothed and a record that was published before its copy
# completed cannot close it. It never edits the locked sources, the accepted build or an earlier run:
# the submodule stays pristine and the links in isac/bin keep naming the unpatched build.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s5_2.sh [--run-dir DIR] [--s31-run DIR] [--jobs N]"
    echo "Run it on a DGX Spark that passed 'make accept-s3.1': it needs the accepted OCUDU build behind"
    echo "isac/bin, NumPy in isac and the accepted Sprint 3.1 run (default: the newest one). That run names"
    echo "the plans this gate decodes and holds the accepted stage O2, and the Sprint 2.4 run those plans lie"
    echo "in holds the decodes of the unpatched build. Both runs must still be where they were."
    echo "The first run builds the patched receiver in build/ocudu-spark-o3, which takes as long as the"
    echo "Sprint 1.1 build. A later run compiles only what changed. --jobs sets the jobs of that build."
    echo "This gate has no host rehearsal: the estimate exists only on a CUDA device."
}
run_dir=""
s31_arg=""
jobs=""
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --s31-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s31_arg=$2; shift 2 ;;
        --jobs)
            if (( $# < 2 )) || ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then usage >&2; exit 2; fi
            jobs=$2; shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s5_2: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s52-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s5_2: $run_dir already exists. Evidence is never overwritten." >&2
    exit 2
fi
mkdir -p -- "$run_dir" || exit 2
run_dir=$(cd -- "$run_dir" && pwd)
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
        fail "$label ($ran ran, $skipped skipped: the native unit cases need cmake, ctest and a C++17 compiler; see $log)"
    else
        pass "$label ($ran ran, $skipped skipped)"
    fi
}
# verdict RUN: true when RUN holds the accepted verdict of Sprint 3.1 on a DGX Spark, without a failed line.
verdict() {
    grep -qE '^Sprint 3\.1: ACCEPTED on this DGX Spark' "$1/summary.txt" 2>/dev/null \
        && ! grep -qE '^FAIL  ' "$1/summary.txt"
}
# The native tools must be found the way an activated shell finds them.
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 5.2 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

# The portable suite, then the card's unit command exactly as the plan names it. The unit gate
# builds the capture core on its own and runs its cases, so none of its tests may be skipped here.
tests any "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
tests complete "S5.2 unit gate, run directly: python tests/unit/test_native_estimate.py" native-estimate-unit.log \
    "$python" tests/unit/test_native_estimate.py -v

# The series: every patch with the hash its manifest names, on the commit the lock pins.
if "$python" scripts/receiver_patch.py check > "$run_dir/patch-series.json" 2> "$run_dir/patch-series.log"; then
    facts=$("$python" -c 'import json, sys
series = json.load(open(sys.argv[1], encoding="utf-8"))
print("%s on %s: %s; %d files touched"
      % (series["series_sha256"][:16], series["base_commit"][:12],
         ", ".join("%s %s" % (item["file"], item["sha256"][:16]) for item in series["patches"]), len(series["files"])))' \
        "$run_dir/patch-series.json" 2>> "$run_dir/patch-series.log")
    pass "S5.2 patch series: patches/ocudu/series.json ${facts:-unreadable} (patch-series.json)"
else
    fail "S5.2 patch series: the manifest disagrees with its patch files or with the lock (see patch-series.log)"
    detail "$run_dir/patch-series.log"
fi

# The control is the run that Sprint 3.1 accepted on this DGX Spark: the newest one, or the one named.
s31_run=""
s31_id=""
if [[ -n "$s31_arg" ]]; then
    candidates=("$s31_arg")
else
    candidates=(results/runs/s31-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if verdict "$candidate" && [[ -f "$candidate/pilot_ls/grade.json" && -f "$candidate/pilot_ls/manifest.json" ]]; then
        s31_run=$candidate
    fi
done
if [[ -n "$s31_run" ]]; then
    s31_run=$(cd -- "$s31_run" && pwd)
    s31_id=$(basename -- "$s31_run")
    pass "S5.2 control: the Sprint 3.1 run $s31_id is ACCEPTED on this DGX Spark and holds the plans it tapped and" \
        "the accepted stage O2 (pilot_ls/)"
else
    fail "S5.2 control: no Sprint 3.1 verdict ACCEPTED on this DGX Spark with its plans in ${s31_arg:-results/runs/s31-*}"
    note "      Run 'make accept-s3.1' on this host first, or name its run with --s31-run."
fi

# The unpatched receiver is the pinned one: the OCUDU build that Sprint 1.1 published into the
# environment. It stays the hook-free control, and nothing here writes into it.
suffix=tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test
target=$(readlink -f -- "$venv_bin/pusch_e2e_pipeline_test" 2>/dev/null)
build_dir=${target%"/$suffix"}
grade_ok=0
if [[ -n "$s31_run" && -L "$venv_bin/pusch_e2e_pipeline_test" && -n "$target" && "$build_dir" != "$target" \
        && -f "$build_dir/isac-native.json" ]]; then
    step "S5.2 references: the Sprint 1.1 build, from the project interpreter (environment.json)" \
        environment.log "$python" scripts/verify_environment.py --profile core --with-native \
        --report "$run_dir/environment.json"
    control_sha=$(sha256sum "$build_dir/isac-native.json" | cut -d ' ' -f 1)

    # The patched receiver: a separate tree of the pinned commit with the series applied, built into
    # its own directory. Its output, including everything CMake prints, is shown while it runs.
    evidence="$run_dir/patched-build"
    build_args=()
    if [[ -n "$jobs" ]]; then build_args=(--jobs "$jobs"); fi
    note "Building the patched receiver. The first build takes as long as the Sprint 1.1 build."
    "$python" scripts/native_ocudu.py build --patched --run-dir "$evidence" \
        ${build_args[@]+"${build_args[@]}"} 2>&1 | tee "$run_dir/patched-build.log"
    built=${PIPESTATUS[0]}
    patched_ok=0
    if (( built == 0 )); then
        pass "S5.2 patched receiver: build command (patched-build.log, evidence in patched-build/)"
        "$python" scripts/native_ocudu.py grade "$evidence" > "$run_dir/patched-grade.txt" 2>&1
        graded=$?
        grep -E '^(PASS|FAIL|NOTE)  ' "$run_dir/patched-grade.txt" | tee -a "$summary"
        if (( graded == 0 )) && ! grep -qE '^FAIL  ' "$run_dir/patched-grade.txt" \
                && grep -qE '^PASS  S5\.2 patched receiver patch series ' "$run_dir/patched-grade.txt"; then
            patched_ok=1
        else
            failed=1
            if ! grep -qE '^(PASS|FAIL)  ' "$run_dir/patched-grade.txt"; then
                fail "S5.2 patched receiver: the build evidence could not be graded (see patched-grade.txt)"
            fi
        fi
    else
        fail "S5.2 patched receiver: build command exited with status $built (see patched-build.log)"
        detail "$run_dir/patched-build.log"
    fi
    # The accepted build and its links are what they were before the patched build.
    after_sha=$(sha256sum "$build_dir/isac-native.json" 2>/dev/null | cut -d ' ' -f 1)
    after_target=$(readlink -f -- "$venv_bin/pusch_e2e_pipeline_test" 2>/dev/null)
    if [[ "$after_sha" == "$control_sha" && "$after_target" == "$target" ]]; then
        pass "S5.2 hook-free control: the accepted Sprint 1.1 build $build_dir keeps its receipt ${control_sha:0:12}, and" \
            "isac/bin still links it"
    else
        fail "S5.2 hook-free control: the receipt of the accepted build or the link in isac/bin changed while the patched" \
            "receiver was built"
    fi

    # The adapter, twice. Against the accepted build it has no route to the estimate. Against the
    # patched build it has. Each build runs its own CTest cases and starts no CUDA work.
    adapters_ok=0
    if "$python" scripts/native_adapter.py build --ocudu-build-dir "$build_dir" \
            --run-dir "$run_dir/adapter-build" > "$run_dir/adapter-build.log" 2>&1 \
            && "$python" scripts/native_adapter.py verify > "$run_dir/adapter-receipt.json" 2> "$run_dir/adapter-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
names = receipt["executables"]
if receipt.get("patch") or receipt.get("o3_route"):
    sys.exit("the control adapter carries a patch or a route")
print("pilot_ls %s without the route, %s, %d adapter tests, %d compiler warnings"
      % (names["pilot_ls"]["sha256"][:12], "with ENABLE_CUDA" if receipt["pin"]["cuda"] else "without CUDA",
         receipt["tests_run"], receipt["compiler_warnings"]))' "$run_dir/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        ctest_log="$run_dir/adapter-build/adapter-test.log"
        cases=$(grep -cE 'estimate_core_[a-z0-9_]+ \.+ +Passed' "$ctest_log" 2>/dev/null)
        cases=${cases:-0}
        if [[ -n "$facts" ]] && (( cases >= 11 )) && grep -qE 'pilot_ls_o3_route \.+ +Passed' "$ctest_log"; then
            pass "S5.2 control adapter: built against the Sprint 1.1 build and verified, $facts; $cases cases of the" \
                "capture core passed, and pilot_ls reports no route (adapter-receipt.json)"
            adapters_ok=1
        else
            fail "S5.2 control adapter: ${facts:-the receipt is unusable}; $cases of 11 cases of the capture core passed," \
                "or pilot_ls did not report its route (adapter-build/adapter-test.log, adapter-verify.log)"
        fi
    else
        fail "S5.2 control adapter: build or verification failed (see adapter-build.log, adapter-verify.log)"
        detail "$run_dir/adapter-build.log"
        detail "$run_dir/adapter-verify.log"
    fi
    if (( patched_ok )) && "$python" scripts/native_adapter.py build --o3 \
            --run-dir "$run_dir/o3-adapter-build" > "$run_dir/o3-adapter-build.log" 2>&1 \
            && "$python" scripts/native_adapter.py verify --o3 > "$run_dir/o3-adapter-receipt.json" \
                2> "$run_dir/o3-adapter-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
names, route, patch = receipt["executables"], receipt["o3_route"], receipt["patch"]
if not receipt["pin"]["cuda"]:
    sys.exit("the patched build has no CUDA")
print("pilot_ls %s with the route, tap interface version %d, series %s built from a %s, %d adapter tests, %d compiler warnings"
      % (names["pilot_ls"]["sha256"][:12], route["tap_interface_version"], patch["series_sha256"][:16], patch["tree_kind"],
         receipt["tests_run"], receipt["compiler_warnings"]))' "$run_dir/o3-adapter-receipt.json" \
            2>> "$run_dir/o3-adapter-verify.log")
        if [[ -n "$facts" ]] && grep -qE 'pilot_ls_o3_route \.+ +Passed' "$run_dir/o3-adapter-build/adapter-test.log"; then
            pass "S5.2 O3 adapter: built against the patched receiver and verified, $facts (o3-adapter-receipt.json)"
            pass "S5.2 hardware: the O3 adapter is linked against the CUDA build of the patched receiver, so its resident" \
                "path can execute and leave an estimate"
        else
            fail "S5.2 O3 adapter: ${facts:-the receipt names no route, no patch or no CUDA} (o3-adapter-verify.log)"
            adapters_ok=0
        fi
    else
        adapters_ok=0
        if (( patched_ok )); then
            fail "S5.2 O3 adapter: build or verification failed (see o3-adapter-build.log, o3-adapter-verify.log)"
            detail "$run_dir/o3-adapter-build.log"
            detail "$run_dir/o3-adapter-verify.log"
        else
            note "SKIP  S5.2 O3 adapter: no patched receiver"
        fi
    fi

    if (( adapters_ok )); then
        # Integration driver. It grades its own evidence, and its lines are copied here.
        note "Exporting the estimate of the patched receiver on the plans of $s31_id with" \
            "configs/integration/native_estimate.json."
        "$python" tests/integration/native_estimate.py --config configs/integration/native_estimate.json \
            --s31-run "$s31_run" --output-dir "$run_dir/native_estimate" > "$run_dir/native-estimate.log" 2>&1
        status=$?
        grade="$run_dir/native_estimate/grade.txt"
        graded=$(grep -cE '^(PASS|FAIL)  ' "$grade" 2>/dev/null)
        graded=${graded:-0}
        grep -E '^(PASS|FAIL|NOTE)  ' "$grade" 2>/dev/null | tee -a "$summary"
        if grep -qE '^FAIL  ' "$grade" 2>/dev/null; then failed=1; fi
        counts=$("$python" -c 'import json, sys
grade = json.load(open(sys.argv[1], encoding="utf-8"))
ok = grade["judgment"] == "pass" and grade["cases"] >= 1 and grade["cases_passed"] == grade["cases"] \
    and grade["checks_failed"] == 0 and grade["hardware_gate"] == "closed"
print("%d of %d cases, %d of %d checks, judgment %s, hardware gate %s"
      % (grade["cases_passed"], grade["cases"], grade["checks_passed"], grade["checks"], grade["judgment"],
         grade["hardware_gate"].split(":")[0]))
sys.exit(0 if ok else 1)' "$run_dir/native_estimate/grade.json" 2> "$run_dir/grade-read.log")
        counted=$?
        if (( status == 0 && counted == 0 && graded > 0 )); then
            grade_ok=1
            pass "S5.2 integration grade: $counts (native_estimate/grade.json)"
        elif (( graded == 0 )); then
            fail "S5.2 integration driver produced no graded checks (see native-estimate.log)"
            detail "$run_dir/native-estimate.log"
        else
            fail "S5.2 integration grade: ${counts:-unreadable}, driver exit status $status (native_estimate/grade.json)"
        fi

        # The counts of the parity, as the acceptance names them.
        parity=$("$python" -c 'import json, sys
parity = json.load(open(sys.argv[1], encoding="utf-8"))
total = parity["totals"]
rows = sum(len(item["rows"]) for item in parity["runs"])
print("%d estimates of %d runs on the pilot support: numerator %.6e, denominator %.6e, %d samples, NMSE %.3e, against "
      "the series %s" % (rows, len(parity["runs"]), total["numerator"], total["denominator"], total["samples"],
                         total["nmse"], parity["patch"]["series_sha256"][:16]))
sys.exit(0 if parity["passed"] and rows and parity["schema"] == "o3-parity/v1" else 1)' \
            "$run_dir/native_estimate/o3_parity.json" 2> "$run_dir/parity-read.log")
        if (( $? == 0 )); then
            pass "S5.2 parity record: $parity (native_estimate/o3_parity.json)"
        else
            fail "S5.2 parity record: ${parity:-o3_parity.json is missing or unreadable} (see parity-read.log)"
        fi

        # A second reader, with tools that are not the driver's: the command-line reader on the
        # audited run of the noisy pair, and sha256sum on the payload file of its first estimate.
        native="$run_dir/native_estimate/noisy/run/native"
        if "$python" -m ocudu_cuda_isac.episodes.native_estimate show "$native" \
                > "$run_dir/recovered.json" 2> "$run_dir/recovered.log"; then
            shown=$("$python" -c 'import json, sys
found = json.load(open(sys.argv[1], encoding="utf-8"))
record, account = found["first_record"], found["account"]
producer = record["producer"]
print(record["payload_sha256"])
print("%d estimates of %d decodes of the CUDA processor captured; the first is the estimate of %s in the shape %s, launch "
      "%d of the producer, from resource block %d on DM-RS symbols %s, after %s, scale %.7f, element 0 = %+.7f%+.7fj"
      % (account["captured"] + account["published"], account["eligible"], record["opportunity_id"], record["shape"],
         producer["launch"], producer["start_prb"], producer["dmrs_symbols"], ", ".join(producer["steps"]),
         producer["scale"], record["element_0"][0], record["element_0"][1]))' "$run_dir/recovered.json" \
                2>> "$run_dir/recovered.log")
            wanted=$(echo "$shown" | sed -n 1p)
            words=$(echo "$shown" | sed -n 2p)
            first=$(ls "$native/estimate" 2>/dev/null | grep -E '\.o3\.fp16$' | head -n 1)
            found=$(sha256sum "$native/estimate/$first" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            if [[ -n "$wanted" && -n "$first" && "$found" == "$wanted" ]]; then
                pass "S5.2 a second reader recovers an estimate: $words; SHA-256 of $first ${found:0:16} by sha256sum, the" \
                    "digest its record names (recovered.json)"
            else
                fail "S5.2 a second reader recovers an estimate: ${first:-no payload file} has the digest" \
                    "${found:-unreadable} by sha256sum, and its record names ${wanted:-none}"
            fi
        else
            fail "S5.2 a second reader recovers an estimate (see recovered.log)"
            detail "$run_dir/recovered.log"
        fi
    else
        note "SKIP  S5.2 integration driver, parity record and record recovery: no pair of adapters"
    fi
elif [[ -n "$s31_run" ]]; then
    fail "S5.2 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a build with a receipt"
    note "      Run 'make accept-s1.1' on this host first."
    note "SKIP  S5.2 patched receiver, adapters, integration driver and record recovery: no Sprint 1.1 build"
else
    note "SKIP  S5.2 patched receiver, adapters, integration driver and record recovery: no control"
fi

# The upstream sources stay at the locked commits with no edited file, and the patched tree is the
# pinned commit with exactly the series applied.
step "S5.2 sources at the locked revisions after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees
if [[ -d build/ocudu-src-o3 ]]; then
    step "S5.2 patched tree: build/ocudu-src-o3 is the pinned commit with exactly the series applied (patched-tree.json)" \
        patched-tree.json "$python" scripts/receiver_patch.py verify build/ocudu-src-o3
fi

note "Evidence: $run_dir"
if (( failed || ! grade_ok )); then
    note "Sprint 5.2: NOT ACCEPTED on this host"
    exit 1
fi
note "Sprint 5.2: ACCEPTED on this DGX Spark. The receiver's own CUDA channel estimate leaves the patched receiver" \
    "as the stage O3: copied behind the kernels that wrote it, published on the completion event of that copy, and" \
    "equal to the matched estimator on the stage O2 within the declared tolerance, while every decode is the one the" \
    "unpatched build wrote on the plans of the accepted Sprint 3.1 run $s31_id. Next: S3.5 cuSense-like pipeline on" \
    "O3 records."
