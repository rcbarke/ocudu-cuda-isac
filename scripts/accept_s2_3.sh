#!/usr/bin/env bash
# Run every Sprint 2.3 gate (grid_apply) on this host and keep the evidence in a new run directory.
# The gate builds the native channel application against the accepted OCUDU build, applies the
# stimulus that Sprint 2.2 archived, Y = g (sum over tx of H X + W) with the grid gain of the entry,
# into the pinned resource grid, reads every grid back and compares it with a complex128 reference
# in Python. It uses the CPU only. It links no decoder: the receiver is Sprint 2.4.
# It never rebuilds OCUDU and never edits the sources, a build or an earlier run.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s2_3.sh [--run-dir DIR] [--s22-run DIR] [--grant-store DIR]..."
    echo "Run it on a host that passed 'make accept-s1.1', 'make accept-s2.1' and 'make accept-s2.2': it needs"
    echo "the accepted OCUDU build behind isac/bin, NumPy in isac and the stimulus of the accepted Sprint 2.2"
    echo "run (default: the newest one). --grant-store names a directory that holds <grant_id>/store, for a"
    echo "host where the runs were moved after the pair recorded them."
}
run_dir=""
s22_arg=""
stores=()
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --s22-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s22_arg=$2; shift 2 ;;
        --grant-store)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            stores+=(--grant-store "$2"); shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s2_3: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s23-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s2_3: $run_dir already exists. Evidence is never overwritten." >&2
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
# The native tools must be found the way an activated shell finds them.
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 2.3 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
# builds the host-free core and runs its cases, so none of its tests may be skipped here.
tests any "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
tests complete "S2.3 unit gate, run directly: python tests/unit/test_grid_apply.py" grid-apply-unit.log \
    "$python" tests/unit/test_grid_apply.py -v

# The stimulus comes from the accepted Sprint 2.2 run: the newest one, or the one named.
s22_run=""
s22_id=""
if [[ -n "$s22_arg" ]]; then
    candidates=("$s22_arg")
else
    candidates=(results/runs/s22-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if grep -qE '^Sprint 2\.2: ACCEPTED on this ' "$candidate/summary.txt" 2>/dev/null \
            && ! grep -qE '^FAIL  ' "$candidate/summary.txt" \
            && [[ -f "$candidate/episode_pair/reference_pair/store/manifest.json" ]]; then
        s22_run=$candidate
    fi
done
profile="host"
if [[ -n "$s22_run" ]]; then
    s22_run=$(cd -- "$s22_run" && pwd)
    s22_id=$(basename -- "$s22_run")
    if grep -qE '^Sprint 2\.2: ACCEPTED on this DGX Spark' "$s22_run/summary.txt"; then profile="DGX Spark"; fi
    pass "S2.3 stimulus: the accepted Sprint 2.2 run $s22_id holds the pair and the slot grants (episode_pair/)"
else
    fail "S2.3 stimulus: no Sprint 2.2 ACCEPTED verdict with a stored pair in ${s22_arg:-results/runs/s22-*}"
    note "      Run 'make accept-s2.2' on this host first, or name its run with --s22-run."
fi

# The receiver grid must be the pinned one: the OCUDU build that Sprint 1.1 published into the
# environment, which the adapter is compiled and linked against.
suffix=tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test
target=$(readlink -f -- "$venv_bin/pusch_e2e_pipeline_test" 2>/dev/null)
build_dir=${target%"/$suffix"}
if [[ -n "$s22_run" && -L "$venv_bin/pusch_e2e_pipeline_test" && -n "$target" && "$build_dir" != "$target" \
        && -f "$build_dir/isac-native.json" ]]; then
    step "S2.3 references: the Sprint 1.1 build, from the project interpreter (environment.json)" \
        environment.log "$python" scripts/verify_environment.py --profile core --with-native \
        --report "$run_dir/environment.json"

    # The adapter is built out of tree with the compile flags and the link line of the pinned test.
    # Its own CTest cases run in that build: the host-free core and the unit vectors through the pinned grid.
    if "$python" scripts/native_adapter.py build --ocudu-build-dir "$build_dir" \
            --run-dir "$run_dir/adapter-build" > "$run_dir/adapter-build.log" 2>&1 \
            && "$python" scripts/native_adapter.py verify > "$run_dir/adapter-receipt.json" 2> "$run_dir/adapter-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
names = receipt["executables"]
print("grid_apply %s and grant_fixture %s, %s, %d adapter tests, %d compiler warnings"
      % (names["grid_apply"]["sha256"][:12], names["grant_fixture"]["sha256"][:12],
         "with ENABLE_CUDA" if receipt["pin"]["cuda"] else "without CUDA", receipt["tests_run"], receipt["compiler_warnings"]))' \
            "$run_dir/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        if [[ -n "$facts" ]]; then
            pass "S2.3 adapter: built against the Sprint 1.1 build and verified, $facts (adapter-receipt.json)"
        else
            fail "S2.3 adapter: the receipt names no grid_apply (see adapter-verify.log)"
            detail "$run_dir/adapter-verify.log"
        fi
        ctest_log="$run_dir/adapter-build/adapter-test.log"
        core=$(grep -cE 'replay_core_[a-z0-9_]+ \.+ +Passed' "$ctest_log" 2>/dev/null)
        core=${core:-0}
        if (( core >= 14 )) && grep -qE 'grid_apply_pinned_grid \.+ +Passed' "$ctest_log" 2>/dev/null; then
            pass "S2.3 native unit cases: $core cases of the host-free core and the unit vectors through the" \
                "pinned resource grid passed in the adapter build (adapter-build/adapter-test.log)"
        else
            fail "S2.3 native unit cases: $core cases of the core passed and the pinned-grid self-test did not," \
                "or the log is missing (adapter-build/adapter-test.log)"
        fi

        # Integration driver. It grades its own evidence, and its lines are copied here.
        note "Applying the stimulus of $s22_id with configs/integration/grid_apply.json."
        "$python" tests/integration/grid_apply.py --config configs/integration/grid_apply.json \
            --s22-run "$s22_run" ${stores[@]+"${stores[@]}"} --output-dir "$run_dir/grid_apply" \
            > "$run_dir/grid-apply.log" 2>&1
        status=$?
        grade="$run_dir/grid_apply/grade.txt"
        graded=$(grep -cE '^(PASS|FAIL)  ' "$grade" 2>/dev/null)
        graded=${graded:-0}
        grep -E '^(PASS|FAIL|NOTE)  ' "$grade" 2>/dev/null | tee -a "$summary"
        if grep -qE '^FAIL  ' "$grade" 2>/dev/null; then failed=1; fi
        counts=$("$python" -c 'import json, sys
grade = json.load(open(sys.argv[1], encoding="utf-8"))
ok = grade["judgment"] == "pass" and grade["cases"] >= 1 and grade["cases_passed"] == grade["cases"] \
    and grade["checks_failed"] == 0
print("%d of %d cases, %d of %d checks" % (grade["cases_passed"], grade["cases"], grade["checks_passed"], grade["checks"]))
sys.exit(0 if ok else 1)' "$run_dir/grid_apply/grade.json" 2> "$run_dir/grade-read.log")
        counted=$?
        if (( status == 0 && counted == 0 && graded > 0 )); then
            pass "S2.3 integration grade: $counts passed (grid_apply/grade.json)"
        elif (( graded == 0 )); then
            fail "S2.3 integration driver produced no graded checks (see grid-apply.log)"
            detail "$run_dir/grid-apply.log"
        else
            fail "S2.3 integration grade: ${counts:-unreadable} passed, driver exit status $status (grid_apply/grade.json)"
        fi

        # A second reader: the command-line reader on the stored grids of the pair, with the standard
        # library only, and the digest of the first grid once more with sha256sum.
        native="$run_dir/grid_apply/reference_pair/native"
        if "$python" -m ocudu_cuda_isac.episodes.grid_apply show "$native" --sequence 0 --rx 0 --symbol 2 --subcarrier 0 \
                > "$run_dir/recovered.json" 2> "$run_dir/recovered.log"; then
            shown=$("$python" -c 'import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
value, gain = record["element"], record["gain"]
print(record["readback_sha256"], record["grid_file"], "%d stored grids verified; Y of opportunity %s (slot %d, grant %s) at port %d, symbol %d, subcarrier %d = %+.6f%+.6fj with g = %.4f"
      % (record["verified_grids"], record["opportunity_id"], record["slot"], record["grant_id"], value["rx_port"],
         value["symbol"], value["subcarrier"], value["value"][0], value["value"][1], gain["value"]))' \
                "$run_dir/recovered.json" 2>> "$run_dir/recovered.log")
            digest=${shown%% *}
            rest=${shown#* }
            file=${rest%% *}
            independent=$(sha256sum "$native/ingress/$file" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            if [[ -n "$independent" && "$digest" == "$independent" ]]; then
                pass "S2.3 a second reader recovers a stored grid: ${rest#* }; readback SHA-256 ${independent:0:16}" \
                    "by the reader and by sha256sum (recovered.json)"
            else
                fail "S2.3 a second reader recovers a stored grid: its digest is ${digest:-unreadable} by the reader and" \
                    "${independent:-unreadable} by sha256sum"
            fi
        else
            fail "S2.3 a second reader recovers a stored grid (see recovered.log)"
            detail "$run_dir/recovered.log"
        fi
    else
        fail "S2.3 adapter: build or verification failed (see adapter-build.log, adapter-verify.log)"
        detail "$run_dir/adapter-build.log"
        detail "$run_dir/adapter-verify.log"
        note "SKIP  S2.3 native unit cases, integration driver and grid recovery: no adapter"
    fi
elif [[ -n "$s22_run" ]]; then
    fail "S2.3 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a build with a receipt"
    note "      Run 'make accept-s1.1' on this host first."
    note "SKIP  S2.3 reference check, adapter build, integration driver and grid recovery: no Sprint 1.1 build"
else
    note "SKIP  S2.3 reference check, adapter build, integration driver and grid recovery: no Sprint 2.2 stimulus"
fi

# The upstream sources stay at the locked commits with no edited file.
step "S2.3 sources at the locked revisions after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees

note "Evidence: $run_dir"
if (( failed )); then
    note "Sprint 2.3: NOT ACCEPTED on this host"
    exit 1
fi
note "Sprint 2.3: ACCEPTED on this $profile. The native channel application with the grid gain fills the pinned" \
    "resource grid from the archived stimulus (S2.2 run $s22_id). Next: S2.4 replay_pusch."
