#!/usr/bin/env bash
# Run every Sprint 1.2 gate on this DGX Spark and keep the evidence in a new run directory.
# The gate measures the build that Sprint 1.1 accepted and published into isac/bin: first the
# DeepSig Table II row (200 CPU and 200 GPU decodes of the same frames on the paper's CPU set),
# then the four-RX smoke. It never builds, and it never edits the sources, the build or an
# earlier run.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s1_2.sh [--run-dir DIR]"
    echo "Run it on the DGX Spark that passed 'make accept-s1.1', with no gNB, training or"
    echo "inference process on the GPU. The Table II row is assigned to a GB10, so this gate"
    echo "has no local profile."
}
run_dir=""
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s1_2: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s12-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s1_2: $run_dir already exists. Evidence is never overwritten." >&2
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
# measure PROFILE: run one profile of the native test into its own evidence directory, then grade
# what it archived. A run that reached the test is graded even when it failed, so the FAIL lines
# name what the test reported.
measure() {
    local profile=$1 evidence="$run_dir/$1" stage reason status graded
    if "$python" scripts/native_ocudu.py run --profile "$profile" --skip-build --build-dir "$build_dir" \
            --run-dir "$evidence" > "$run_dir/$profile-run.log" 2>&1; then
        pass "S1.2 $profile run command ($profile-run.log, evidence in $profile/)"
    else
        stage=$(sed -n 's/^stage=//p' "$evidence/status.txt" 2>/dev/null)
        reason=$(head -n 1 "$evidence/error.txt" 2>/dev/null | cut -c 1-300)
        fail "S1.2 $profile run command stopped in stage ${stage:-unknown}: ${reason:-no evidence written}" \
            "(see $profile-run.log)"
        if [[ ! -f "$evidence/pusch.log" ]]; then
            sed -n '2,12p' "$evidence/error.txt" 2>/dev/null | cut -c 1-300 | sed 's/^/      /' | tee -a "$summary"
            note "SKIP  S1.2 $profile evidence grade: the run stopped before the native test wrote a transcript"
            return
        fi
    fi
    "$python" scripts/native_ocudu.py grade-run "$evidence" > "$run_dir/$profile-grade.txt" 2>&1
    status=$?
    graded=$(grep -cE '^(PASS|FAIL)  ' "$run_dir/$profile-grade.txt")
    grep -E '^(PASS|FAIL|NOTE)  ' "$run_dir/$profile-grade.txt" | tee -a "$summary"
    if (( status != 0 )); then
        failed=1
        if ! grep -qE '^FAIL  ' "$run_dir/$profile-grade.txt"; then
            fail "S1.2 $profile evidence could not be graded (see $profile-grade.txt)"
            detail "$run_dir/$profile-grade.txt"
        fi
    elif (( graded == 0 )); then
        fail "S1.2 $profile evidence produced no graded checks (see $profile-grade.txt)"
    fi
}
# The native tools must be found the way an activated shell finds them.
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 1.2 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

# The portable suite holds the transcript and evidence graders' own tests. An empty run cannot close the gate.
PYTHONPATH=src "$python" -m unittest discover -s tests -v > "$run_dir/unit-tests.log" 2>&1
status=$?
ran=$(grep -E '^Ran [0-9]+ tests? in ' "$run_dir/unit-tests.log" | tail -n 1 | awk '{print $2}')
verdict=$(grep -E '^(OK|FAILED)' "$run_dir/unit-tests.log" | tail -n 1)
skipped=$(echo "$verdict" | sed -n 's/.*skipped=\([0-9][0-9]*\).*/\1/p')
ran=${ran:-0}
skipped=${skipped:-0}
if (( status != 0 )) || [[ "$verdict" != OK* ]]; then
    fail "unit tests ($ran ran; see unit-tests.log)"
    grep -E '^(FAIL|ERROR): ' "$run_dir/unit-tests.log" | head -n 10 | cut -c 1-300 | sed 's/^/      /' | tee -a "$summary"
elif (( ran == 0 )); then
    fail "unit tests (none ran)"
else
    pass "unit tests ($ran ran, $skipped skipped)"
fi

# The measured build is the one Sprint 1.1 published into the environment. Its directory is read
# from the link, so this gate cannot measure one build while isac/bin points at another.
suffix=tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test
target=$(readlink -f -- "$venv_bin/pusch_e2e_pipeline_test" 2>/dev/null)
build_dir=${target%"/$suffix"}
s11_id=""
if [[ -L "$venv_bin/pusch_e2e_pipeline_test" && -n "$target" && "$build_dir" != "$target" \
        && -f "$build_dir/isac-native.json" ]]; then
    step "S1.2 the Sprint 1.1 build is intact: gnb and pusch_e2e_pipeline_test in isac/bin match its receipt (native-env.json)" \
        native-env.log "$python" scripts/verify_environment.py --profile core --with-native \
        --report "$run_dir/native-env.json"
    # The receipt names the evidence of the build it describes. That run must hold the S1.1 verdict.
    accepted=$("$python" -c 'import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["build_evidence"])' \
        "$build_dir/isac-native.json" 2> "$run_dir/s11-record.log")
    s11_run=$(dirname -- "${accepted:-/nonexistent/native-build}")
    if grep -qE '^Sprint 1\.1: ACCEPTED on this DGX Spark' "$s11_run/summary.txt" 2>/dev/null \
            && ! grep -qE '^FAIL  ' "$s11_run/summary.txt"; then
        s11_id=$(basename -- "$s11_run")
        pass "S1.2 the measured build is the one Sprint 1.1 accepted: run $s11_id"
    else
        fail "S1.2 the measured build is the one Sprint 1.1 accepted: no ACCEPTED verdict in $s11_run/summary.txt"
        note "      The build was made or rebuilt outside the Sprint 1.1 gate. Run 'make accept-s1.1', then this gate again."
    fi

    note "Measuring the Table II row: 200 CPU and 200 GPU decodes of the same frames, pinned to CPUs 5-9,15-19."
    measure paper
    note "Measuring the four-RX smoke: 20 CPU and 20 GPU decodes."
    measure smoke
else
    fail "S1.2 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a build with a receipt"
    note "      Run 'make accept-s1.1' on this host first."
    note "SKIP  S1.2 build checks, paper run and smoke run: no Sprint 1.1 build"
fi

# The upstream control stays at the locked commit with no edited file.
step "S1.2 OCUDU source at the locked revision after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees --only ocudu

note "Evidence: $run_dir"
if (( failed )); then
    note "Sprint 1.2: NOT ACCEPTED on this host"
    exit 1
fi
note "Sprint 1.2: ACCEPTED on this DGX Spark. Table II correctness is reproduced. The timing comparison" \
    "is recorded in paper/table-ii.json and is not a gate."
note "Sprint 1: ACCEPTED on this DGX Spark (S1.1 run $s11_id, S1.2 run $(basename -- "$run_dir"))." \
    "Next: S2.1 grant_fixture."
