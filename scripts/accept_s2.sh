#!/usr/bin/env bash
# Close Sprint 2 on this host: audit the four accepted runs as one chain and keep the evidence in a
# new run directory. Each gate of Sprint 2 graded its own step. This one checks what links them:
# the grant record of Sprint 2.1 is the one the pair of Sprint 2.2 stands on, the channel
# application of Sprint 2.3 filled its grids from those transmit grids, channels and noise, the
# receiver of Sprint 2.4 decoded those same grids on the host and on the resident CUDA path, and
# the bytes it returned are the transport block that Sprint 2.1 archived.
# It reads the archived runs and starts no native tool. With --rerun it first runs the four gates
# again, in order, each on the run of the one before. It never edits a source, a build or a run.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s2.sh [--run-dir DIR] [--rerun] [--s24-run DIR] [--s23-run DIR] [--s22-run DIR]"
    echo "                                 [--s21-run DIR] [--grant-store DIR]..."
    echo "Run it on a host that passed 'make accept-s2.1' to 'make accept-s2.4'. It audits the newest accepted"
    echo "Sprint 2.4 run and the runs that one names. Name a run that was moved with its option: it must have"
    echo "the name that was recorded. --grant-store names a directory that holds <grant_id>/store."
    echo "--rerun runs accept_s2_1.sh to accept_s2_4.sh first and audits the four new runs."
}
run_dir=""
rerun=0
named=()
s24_arg=""
stores=()
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --rerun) rerun=1; shift ;;
        --s24-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s24_arg=$2; shift 2 ;;
        --s23-run|--s22-run|--s21-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            named+=("$1" "$2"); shift 2 ;;
        --grant-store)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            stores+=(--grant-store "$2"); shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if (( rerun )) && [[ -n "$s24_arg" || ${#named[@]} -gt 0 ]]; then
    echo "accept_s2: --rerun audits the runs it makes. Name no earlier run with it." >&2
    exit 2
fi
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s2: $python is missing. Run 'make env' first." >&2
    exit 2
fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s2-$(hostname)-$stamp"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s2: $run_dir already exists. Evidence is never overwritten." >&2
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
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 2 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

# With --rerun the four gates run again, each on the run of the one before. A gate that fails stops
# the sequence: the later gates would only repeat its failure.
s24_run=""
if (( rerun )); then
    note "Running the four gates of Sprint 2 again, in order."
    prefix="results/runs"
    host=$(hostname)
    previous=""
    for gate in 1 2 3 4; do
        gate_dir="$prefix/s2$gate-$host-$stamp"
        chained=()
        case "$gate" in
            2) chained=(--s21-run "$previous") ;;
            3) chained=(--s22-run "$previous") ;;
            4) chained=(--s23-run "$previous") ;;
        esac
        if bash "scripts/accept_s2_$gate.sh" --run-dir "$gate_dir" ${chained[@]+"${chained[@]}"} \
                > "$run_dir/rerun-s2$gate.log" 2>&1; then
            verdict=$(sed -n "s/^Sprint 2\\.$gate: \\(ACCEPTED on this [A-Za-z ]*\\)\\..*/\\1/p" "$run_dir/rerun-s2$gate.log" | tail -n 1)
            pass "S2 rerun: accept_s2_$gate.sh, ${verdict:-no verdict line} ($(basename -- "$gate_dir"), rerun-s2$gate.log)"
            previous=$gate_dir
        else
            fail "S2 rerun: accept_s2_$gate.sh did not accept (see rerun-s2$gate.log and $gate_dir/summary.txt)"
            grep -E '^FAIL  ' "$run_dir/rerun-s2$gate.log" | head -n 5 | cut -c 1-300 | sed 's/^/      /' | tee -a "$summary"
            previous=""
            break
        fi
    done
    if [[ -n "$previous" ]]; then s24_arg=$previous; fi
fi

# The portable suite, then the four unit gates exactly as their cards name them, each graded as its
# own gate grades it. The last two build a host-free native core, so none of their tests may be
# skipped here.
tests any "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
for selector in grant_fixture episode_pair; do
    tests any "S2 unit gate, run directly: python tests/unit/test_$selector.py" "unit-$selector.log" \
        "$python" "tests/unit/test_$selector.py" -v
done
for selector in grid_apply replay_pusch; do
    tests complete "S2 unit gate, run directly: python tests/unit/test_$selector.py" "unit-$selector.log" \
        "$python" "tests/unit/test_$selector.py" -v
done

# The chain starts at the accepted Sprint 2.4 run: the newest one, or the one named or just made.
if [[ -n "$s24_arg" ]]; then
    candidates=("$s24_arg")
elif (( rerun )); then
    candidates=()
else
    candidates=(results/runs/s24-*/)
fi
for candidate in ${candidates[@]+"${candidates[@]}"}; do
    candidate=${candidate%/}
    if grep -qE '^Sprint 2\.4: ACCEPTED on this ' "$candidate/summary.txt" 2>/dev/null \
            && ! grep -qE '^FAIL  ' "$candidate/summary.txt" \
            && [[ -f "$candidate/replay_pusch/manifest.json" ]]; then
        s24_run=$candidate
    fi
done
profile="host"
audited=0
if [[ -n "$s24_run" ]]; then
    s24_run=$(cd -- "$s24_run" && pwd)
    pass "S2 last gate: the accepted Sprint 2.4 run $(basename -- "$s24_run") holds the decodes (replay_pusch/)"

    # The audit grades its own evidence, and its lines are copied here.
    "$python" scripts/verify_sprint2.py --s24-run "$s24_run" ${named[@]+"${named[@]}"} ${stores[@]+"${stores[@]}"} \
        --output-dir "$run_dir/chain" > "$run_dir/chain.log" 2>&1
    status=$?
    grade="$run_dir/chain/grade.txt"
    graded=$(grep -cE '^(PASS|FAIL)  ' "$grade" 2>/dev/null)
    graded=${graded:-0}
    grep -E '^(PASS|FAIL|NOTE)  ' "$grade" 2>/dev/null | tee -a "$summary"
    if grep -qE '^FAIL  ' "$grade" 2>/dev/null; then failed=1; fi
    counts=$("$python" -c 'import json, sys
grade = json.load(open(sys.argv[1], encoding="utf-8"))
ok = grade["judgment"] == "pass" and grade["stages"] >= 1 and grade["stages_passed"] == grade["stages"] \
    and grade["checks_failed"] == 0 and grade["checks"] > 0
runs = grade["runs"]
print(grade["profile"] or "host")
print("%s, %s, %s and %s" % (runs["s21"], runs["s22"], runs["s23"], runs["s24"]))
print("%d of %d stages, %d of %d checks" % (grade["stages_passed"], grade["stages"], grade["checks_passed"], grade["checks"]))
sys.exit(0 if ok else 1)' "$run_dir/chain/grade.json" 2> "$run_dir/grade-read.log")
    counted=$?
    chain_runs=$(echo "$counts" | sed -n 2p)
    if (( status == 0 && counted == 0 && graded > 0 )); then
        audited=1
        profile=$(echo "$counts" | sed -n 1p)
        pass "S2 chain audit: $(echo "$counts" | sed -n 3p) passed on the runs $chain_runs (chain/grade.json)"
    elif (( graded == 0 )); then
        fail "S2 chain audit produced no graded checks (see chain.log)"
        detail "$run_dir/chain.log"
    else
        fail "S2 chain audit: $(echo "$counts" | sed -n 3p) passed, exit status $status (chain/grade.json)"
    fi

    # Other tools on the same files: the hash of the archived transport block and of every file of
    # bytes that the receiver decoded from the pair, by sha256sum.
    if [[ -f "$run_dir/chain/manifest.json" ]]; then
        facts=$("$python" -c 'import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
block = manifest["original_block"]
print(block["path"])
print(block["sha256"])
print(manifest["pair_radio"])
print(manifest["pair_decodes"])' "$run_dir/chain/manifest.json" 2> "$run_dir/recovered.log")
        block=$(echo "$facts" | sed -n 1p)
        recorded=$(echo "$facts" | sed -n 2p)
        radio=$(echo "$facts" | sed -n 3p)
        planned=$(echo "$facts" | sed -n 4p)
        original=$(sha256sum "$block" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
        total=0
        equal=0
        for file in "$radio"/*.bin; do
            [[ -f "$file" ]] || continue
            total=$(( total + 1 ))
            if [[ "$(sha256sum "$file" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)" == "$original" ]]; then
                equal=$(( equal + 1 ))
            fi
        done
        echo "$original  $block" > "$run_dir/recovered.txt"
        echo "$equal of $total files in $radio have this hash; $planned decodes of the pair were planned" >> "$run_dir/recovered.txt"
        if [[ -n "$original" && "$original" == "$recorded" ]] && (( total > 0 && equal == total && total == ${planned:-0} )); then
            pass "S2 a second reader follows the block: tb.bin of the Sprint 2.1 store has the SHA-256 ${original:0:16}" \
                "by sha256sum, and $equal of $total files of decoded bytes of the pair have the same one, one for" \
                "each of the $planned decodes on both backends (recovered.txt)"
        else
            fail "S2 a second reader follows the block: tb.bin has the SHA-256 ${original:-unreadable} by sha256sum and" \
                "${recorded:-unreadable} by the audit, and $equal of $total files of decoded bytes have it, for" \
                "${planned:-an unreadable count of} planned decodes (recovered.txt)"
        fi
    else
        fail "S2 a second reader follows the block: the audit left no manifest (see chain.log)"
    fi
elif (( rerun )); then
    note "SKIP  S2 chain audit and block recovery: the four gates did not all accept"
else
    fail "S2 last gate: no Sprint 2.4 ACCEPTED verdict with decodes in ${s24_arg:-results/runs/s24-*}"
    note "      Run 'make accept-s2.4' on this host first, or name its run with --s24-run. A rehearsal is not one."
    note "SKIP  S2 chain audit and block recovery: no Sprint 2.4 run"
fi

# The upstream sources stay at the locked commits with no edited file.
step "S2 sources at the locked revisions (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees

note "Evidence: $run_dir"
if (( failed || ! audited )); then
    note "Sprint 2: NOT ACCEPTED on this host"
    exit 1
fi
note "Sprint 2: ACCEPTED on this $profile. A known waveform reaches the native receiver: the grant record, the" \
    "matched pair, the channel application and the host and CUDA decodes are one chain, and both backends return" \
    "the original transport block (runs $chain_runs). Sprint 2 is complete. Next: S3.1 pilot_ls."
