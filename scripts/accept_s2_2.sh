#!/usr/bin/env bash
# Run every Sprint 2.2 gate (episode_pair) on this host and keep the evidence in a new run directory.
# The gate builds the uplink grant fixture of an episode (one legal grant per scheduled uplink slot,
# bound to the record that Sprint 2.1 archived and realized by the accepted native fixture adapter),
# traces the matched target-absent and target-present pair on it with Sionna RT, compares the stored
# pair with the analytic model at every grant coordinate, and exports a small demonstration set.
# It uses the CPU only. It never rebuilds OCUDU and never edits the sources, a build or an earlier run.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s2_2.sh [--run-dir DIR] [--s21-run DIR]"
    echo "Run it on a host that passed 'make accept-s0' with the cpu backend, 'make accept-s1.1' and"
    echo "'make accept-s2.1': it needs the pinned Sionna RT in isac, the accepted OCUDU build behind"
    echo "isac/bin and the grant record of the accepted Sprint 2.1 run (default: the newest one)."
}
run_dir=""
s21_arg=""
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --s21-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s21_arg=$2; shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s2_2: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s22-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s2_2: $run_dir already exists. Evidence is never overwritten." >&2
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
# tests LABEL LOG COMMAND...: run a unittest command. An empty run cannot close a gate.
tests() {
    local label=$1 log=$2 status ran verdict skipped
    shift 2
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
    else
        pass "$label ($ran ran, $skipped skipped)"
    fi
}
# The native tools must be found the way an activated shell finds them.
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 2.2 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

# The portable suite, then the card's unit command exactly as the plan names it.
tests "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
tests "S2.2 unit gate, run directly: python tests/unit/test_episode_pair.py" episode-pair-unit.log \
    "$python" tests/unit/test_episode_pair.py -v

# The grant record comes from the accepted Sprint 2.1 run: the newest one, or the one named.
s21_run=""
s21_id=""
if [[ -n "$s21_arg" ]]; then
    candidates=("$s21_arg")
else
    candidates=(results/runs/s21-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if grep -qE '^Sprint 2\.1: ACCEPTED on this ' "$candidate/summary.txt" 2>/dev/null \
            && ! grep -qE '^FAIL  ' "$candidate/summary.txt" && [[ -d "$candidate/grant_fixture" ]]; then
        s21_run=$candidate
    fi
done
profile="host"
if [[ -n "$s21_run" ]]; then
    s21_run=$(cd -- "$s21_run" && pwd)
    s21_id=$(basename -- "$s21_run")
    if grep -qE '^Sprint 2\.1: ACCEPTED on this DGX Spark' "$s21_run/summary.txt"; then profile="DGX Spark"; fi
    pass "S2.2 grant record: the accepted Sprint 2.1 run $s21_id holds it (grant_fixture/)"
else
    fail "S2.2 grant record: no Sprint 2.1 ACCEPTED verdict with a grant_fixture directory in ${s21_arg:-results/runs/s21-*}"
    note "      Run 'make accept-s2.1' on this host first, or name its run with --s21-run."
fi

# Both references must be the pinned ones: the Sionna files of the locked trees and the OCUDU build
# that Sprint 1.1 published into the environment, which the fixture adapter is built against.
suffix=tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test
target=$(readlink -f -- "$venv_bin/pusch_e2e_pipeline_test" 2>/dev/null)
build_dir=${target%"/$suffix"}
if [[ -n "$s21_run" && -L "$venv_bin/pusch_e2e_pipeline_test" && -n "$target" && "$build_dir" != "$target" \
        && -f "$build_dir/isac-native.json" ]]; then
    step "S2.2 references: the pinned Sionna and Sionna RT on the CPU and the Sprint 1.1 build, from one interpreter (environment.json)" \
        environment.log "$python" scripts/verify_environment.py --profile simulation --backend cpu --with-native \
        --report "$run_dir/environment.json"

    # The adapter is built out of tree with the compile flags and the link line of the pinned test.
    if "$python" scripts/native_adapter.py build --ocudu-build-dir "$build_dir" \
            --run-dir "$run_dir/adapter-build" > "$run_dir/adapter-build.log" 2>&1 \
            && "$python" scripts/native_adapter.py verify > "$run_dir/adapter-receipt.json" 2> "$run_dir/adapter-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
print("grant_fixture %s, %s, %d adapter tests, %d compiler warnings"
      % (receipt["executables"]["grant_fixture"]["sha256"][:12], "with ENABLE_CUDA" if receipt["pin"]["cuda"] else "without CUDA",
         receipt["tests_run"], receipt["compiler_warnings"]))' "$run_dir/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        pass "S2.2 fixture adapter: built against the Sprint 1.1 build and verified, ${facts:-unreadable} (adapter-receipt.json)"

        # Integration driver. It grades its own evidence, and its lines are copied here.
        note "Building the grant fixture, tracing the pair and the set of configs/integration/episode_pair.json."
        "$python" tests/integration/episode_pair.py --config configs/integration/episode_pair.json \
            --grant-store "$s21_run/grant_fixture" --output-dir "$run_dir/episode_pair" > "$run_dir/episode-pair.log" 2>&1
        status=$?
        grade="$run_dir/episode_pair/grade.txt"
        graded=$(grep -cE '^(PASS|FAIL)  ' "$grade" 2>/dev/null)
        graded=${graded:-0}
        grep -E '^(PASS|FAIL|NOTE)  ' "$grade" 2>/dev/null | tee -a "$summary"
        if grep -qE '^FAIL  ' "$grade" 2>/dev/null; then failed=1; fi
        counts=$("$python" -c 'import json, sys
grade = json.load(open(sys.argv[1], encoding="utf-8"))
ok = grade["judgment"] == "pass" and grade["cases"] >= 1 and grade["cases_passed"] == grade["cases"] \
    and grade["checks_failed"] == 0
print("%d of %d cases, %d of %d checks" % (grade["cases_passed"], grade["cases"], grade["checks_passed"], grade["checks"]))
sys.exit(0 if ok else 1)' "$run_dir/episode_pair/grade.json" 2> "$run_dir/grade-read.log")
        counted=$?
        if (( status == 0 && counted == 0 && graded > 0 )); then
            pass "S2.2 integration grade: $counts passed (episode_pair/grade.json)"
        elif (( graded == 0 )); then
            fail "S2.2 integration driver produced no graded checks (see episode-pair.log)"
            detail "$run_dir/episode-pair.log"
        else
            fail "S2.2 integration grade: ${counts:-unreadable} passed, driver exit status $status (episode_pair/grade.json)"
        fi

        # A second reader: the command-line reader on the stored pair, with the standard library only,
        # and the hash of the target-present channel once more with sha256sum.
        store="$run_dir/episode_pair/reference_pair/store"
        if "$python" -m ocudu_cuda_isac.episodes.episode show "$store" --case b --opportunity 0 --rx 0 \
                > "$run_dir/recovered.json" 2> "$run_dir/recovered.log"; then
            shown=$("$python" -c 'import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
value, steps = record["coefficient"], record["phase_advance"]
print(record["sha256"]["b_h.npy"], "H at symbol %d, subcarrier %d (%+.2f MHz) = %+.6e%+.6ej, %.1e from the analytic value; phase steps of the return %+.5f rad in %.2f us and %+.5f rad in %.2f us"
      % (value["symbol"], value["subcarrier"], value["offset_hz"] / 1e6, value["value"][0], value["value"][1],
         value["analytic_relative_error"], steps[0]["measured_rad"], 1e6 * steps[0]["interval_s"],
         steps[1]["measured_rad"], 1e6 * steps[1]["interval_s"]))' "$run_dir/recovered.json" 2>> "$run_dir/recovered.log")
            independent=$(sha256sum "$store/b_h.npy" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            if [[ -n "$independent" && "${shown%% *}" == "$independent" ]]; then
                pass "S2.2 a second reader recovers the pair: ${shown#* }; b_h.npy SHA-256 ${independent:0:16}" \
                    "by the reader and by sha256sum (recovered.json)"
            else
                fail "S2.2 a second reader recovers the pair: b_h.npy is ${shown%% *} by the reader and" \
                    "${independent:-unreadable} by sha256sum"
            fi
        else
            fail "S2.2 a second reader recovers the pair (see recovered.log)"
            detail "$run_dir/recovered.log"
        fi
    else
        fail "S2.2 fixture adapter: build or verification failed (see adapter-build.log, adapter-verify.log)"
        detail "$run_dir/adapter-build.log"
        detail "$run_dir/adapter-verify.log"
        note "SKIP  S2.2 integration driver and pair recovery: no adapter"
    fi
elif [[ -n "$s21_run" ]]; then
    fail "S2.2 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a build with a receipt"
    note "      Run 'make accept-s1.1' on this host first."
    note "SKIP  S2.2 reference check, adapter build, integration driver and pair recovery: no Sprint 1.1 build"
else
    note "SKIP  S2.2 reference check, adapter build, integration driver and pair recovery: no Sprint 2.1 record"
fi

# The upstream sources stay at the locked commits with no edited file.
step "S2.2 sources at the locked revisions after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees

note "Evidence: $run_dir"
if (( failed )); then
    note "Sprint 2.2: NOT ACCEPTED on this host"
    exit 1
fi
note "Sprint 2.2: ACCEPTED on this $profile. The uplink grant fixture, the matched pair on it and the episode set" \
    "are archived (S2.1 run $s21_id). Next: S2.3 grid_apply."
