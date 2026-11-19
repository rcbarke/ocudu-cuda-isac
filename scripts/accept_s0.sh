#!/usr/bin/env bash
# Run every Sprint 0 gate on this host and keep the evidence in a new run directory.
# Each gate is an existing project command. This script only runs them, saves their output and
# reports one verdict. It never changes the environment, the sources or an earlier run.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s0.sh [--run-dir DIR] [cpu] [cuda]"
    echo "Name the simulation backends this host must pass. With none, the core gates run alone."
}
run_dir=""
backends=()
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        cpu|cuda) backends+=("$1"); shift ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s0: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s0-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s0: $run_dir already exists. Evidence is never overwritten." >&2
    exit 2
fi
mkdir -p -- "$run_dir" || exit 2
run_dir=$(cd -- "$run_dir" && pwd)
summary="$run_dir/summary.txt"
failed=0
note() { echo "$*" | tee -a "$summary"; }
pass() { note "PASS  $*"; }
fail() { note "FAIL  $*"; failed=1; }
# step LABEL LOG COMMAND...: run one gate, keep its output, record the result.
step() {
    local label=$1 log=$2
    shift 2
    if "$@" > "$run_dir/$log" 2>&1; then pass "$label"; else fail "$label (see $log)"; fi
}

note "Sprint 0 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

step "S0.1 environment report (core-env.json)" core-env.log \
    "$python" scripts/verify_environment.py --profile core --report "$run_dir/core-env.json"
step "S0.2 source identities (sources.txt)" sources.txt \
    "$python" scripts/check_pins.py --worktrees
for backend in ${backends[@]+"${backends[@]}"}; do
    step "S0.2 simulation on $backend (simulation-$backend.json)" "simulation-$backend.log" \
        "$python" scripts/verify_environment.py --profile simulation --backend "$backend" \
        --report "$run_dir/simulation-$backend.json"
done

step "S0.3 replay fixture (smoke-a)" smoke-a.log \
    env PYTHON="$python" bash scripts/run_smoke.sh --output-dir "$run_dir/smoke-a"
step "S0.3 repeat of the fixture (smoke-b)" smoke-b.log \
    env PYTHON="$python" bash scripts/run_smoke.sh --output-dir "$run_dir/smoke-b"
if "$python" -c '
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"on_time": 1, "late": 1, "dropped": 1, "missing": 1}
sys.exit(0 if (summary["scheduled_updates"], summary["counts"]) == (4, expected) else 1)
' "$run_dir/smoke-a/timing_summary.json" > "$run_dir/smoke-counts.log" 2>&1; then
    pass "S0.3 four scheduled: one on time, one late, one dropped, one missing"
else
    fail "S0.3 four scheduled: one on time, one late, one dropped, one missing (see smoke-a/timing_summary.json)"
fi
files=$(ls "$run_dir/smoke-a" 2>/dev/null | wc -l)
if (( files == 3 )) && diff -r "$run_dir/smoke-a" "$run_dir/smoke-b" > "$run_dir/smoke-diff.txt" 2>&1; then
    (cd "$run_dir/smoke-a" && sha256sum -- *) > "$run_dir/smoke-sha256.txt"
    pass "S0.3 three output files byte-identical on repeat (smoke-sha256.txt)"
else
    fail "S0.3 three output files byte-identical on repeat (see smoke-diff.txt)"
fi

# The whole portable suite. An empty run or a skipped simulation test cannot close the gate.
PYTHONPATH=src "$python" -m unittest discover -s tests -v > "$run_dir/unit-tests.log" 2>&1
status=$?
ran=$(grep -E '^Ran [0-9]+ tests? in ' "$run_dir/unit-tests.log" | tail -n 1 | awk '{print $2}')
verdict=$(grep -E '^(OK|FAILED)' "$run_dir/unit-tests.log" | tail -n 1)
skipped=$(echo "$verdict" | sed -n 's/.*skipped=\([0-9][0-9]*\).*/\1/p')
ran=${ran:-0}
skipped=${skipped:-0}
if (( status != 0 )) || [[ "$verdict" != OK* ]]; then
    fail "unit tests ($ran ran; see unit-tests.log)"
elif (( ran == 0 )); then
    fail "unit tests (none ran)"
elif (( skipped > 0 && ${#backends[@]} > 0 )); then
    fail "unit tests ($skipped skipped on a simulation host; see unit-tests.log)"
else
    pass "unit tests ($ran ran, $skipped skipped)"
fi

note "Evidence: $run_dir"
if (( failed )); then
    note "Sprint 0: NOT ACCEPTED on this host"
    exit 1
fi
if (( ${#backends[@]} > 0 )); then
    note "Sprint 0: ACCEPTED on this host, including simulation on ${backends[*]}"
else
    note "Sprint 0: ACCEPTED as a development host. The simulation check still needs a provisioned scene host."
fi
