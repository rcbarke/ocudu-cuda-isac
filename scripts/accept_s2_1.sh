#!/usr/bin/env bash
# Run every Sprint 2.1 gate (grant_fixture) on this host and keep the evidence in a new run directory.
# The gate builds the project's native fixture adapter out of tree against the OCUDU build that
# Sprint 1.1 accepted, feeds the same known transport block to both transmit references (the
# pinned OCUDU chain and Sionna NR), compares them and archives the grant record. It uses the CPU
# only. It never rebuilds OCUDU and never edits the sources, the OCUDU build or an earlier run.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s2_1.sh [--run-dir DIR]"
    echo "Run it on a host that passed 'make accept-s0' with the cpu backend and 'make accept-s1.1':"
    echo "it needs the pinned Sionna in isac and the accepted OCUDU build behind isac/bin."
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
    echo "accept_s2_1: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s21-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s2_1: $run_dir already exists. Evidence is never overwritten." >&2
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

note "Sprint 2.1 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
tests "S2.1 unit gate, run directly: python tests/unit/test_grant_fixture.py" grant-fixture-unit.log \
    "$python" tests/unit/test_grant_fixture.py -v

# Both transmit references must be the pinned ones: the Sionna files of the locked tree and the
# OCUDU build that Sprint 1.1 published into the environment.
suffix=tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test
target=$(readlink -f -- "$venv_bin/pusch_e2e_pipeline_test" 2>/dev/null)
build_dir=${target%"/$suffix"}
s11_id=""
profile="host"
if [[ -L "$venv_bin/pusch_e2e_pipeline_test" && -n "$target" && "$build_dir" != "$target" \
        && -f "$build_dir/isac-native.json" ]]; then
    step "S2.1 references: the pinned Sionna on the CPU and the Sprint 1.1 build, from one interpreter (environment.json)" \
        environment.log "$python" scripts/verify_environment.py --profile simulation --backend cpu --with-native \
        --report "$run_dir/environment.json"
    accepted=$("$python" -c 'import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["build_evidence"])' \
        "$build_dir/isac-native.json" 2> "$run_dir/s11-record.log")
    s11_run=$(dirname -- "${accepted:-/nonexistent/native-build}")
    if grep -qE '^Sprint 1\.1: ACCEPTED on this ' "$s11_run/summary.txt" 2>/dev/null \
            && ! grep -qE '^FAIL  ' "$s11_run/summary.txt"; then
        s11_id=$(basename -- "$s11_run")
        if grep -qE '^Sprint 1\.1: ACCEPTED on this DGX Spark' "$s11_run/summary.txt"; then profile="DGX Spark"; fi
        pass "S2.1 native boundary: the OCUDU build is the one Sprint 1.1 accepted, run $s11_id"
    else
        fail "S2.1 native boundary: no Sprint 1.1 ACCEPTED verdict in $s11_run/summary.txt"
        note "      The build was made or rebuilt outside the Sprint 1.1 gate. Run 'make accept-s1.1', then this gate again."
    fi

    # The adapter is built out of tree with the compile flags and the link line of the pinned test.
    if "$python" scripts/native_adapter.py build --ocudu-build-dir "$build_dir" \
            --run-dir "$run_dir/adapter-build" > "$run_dir/adapter-build.log" 2>&1; then
        pass "S2.1 adapter build command (adapter-build.log, evidence in adapter-build/)"
        if "$python" scripts/native_adapter.py verify > "$run_dir/adapter-receipt.json" 2> "$run_dir/adapter-verify.log"; then
            facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
print("grant_fixture %s, compiled and linked like pusch_e2e_pipeline_test (%s, %d OCUDU archives), %d adapter tests, %d compiler warnings"
      % (receipt["executables"]["grant_fixture"]["sha256"][:12], "with ENABLE_CUDA" if receipt["pin"]["cuda"] else "without CUDA",
         receipt["pin"]["archives"], receipt["tests_run"], receipt["compiler_warnings"]))' \
                "$run_dir/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
            pass "S2.1 adapter receipt: ${facts:-unreadable} (adapter-receipt.json)"
        else
            fail "S2.1 adapter receipt (see adapter-verify.log)"
            detail "$run_dir/adapter-verify.log"
        fi

        # Integration driver. It grades its own evidence, and its lines are copied here.
        note "Comparing the transmit references for every grant of configs/integration/grant_fixture.json."
        "$python" tests/integration/grant_fixture.py --config configs/integration/grant_fixture.json \
            --output-dir "$run_dir/grant_fixture" > "$run_dir/grant-fixture.log" 2>&1
        status=$?
        grade="$run_dir/grant_fixture/grade.txt"
        graded=$(grep -cE '^(PASS|FAIL)  ' "$grade" 2>/dev/null)
        graded=${graded:-0}
        grep -E '^(PASS|FAIL|NOTE)  ' "$grade" 2>/dev/null | tee -a "$summary"
        if grep -qE '^FAIL  ' "$grade" 2>/dev/null; then failed=1; fi
        counts=$("$python" -c 'import json, sys
grade = json.load(open(sys.argv[1], encoding="utf-8"))
ok = grade["judgment"] == "pass" and grade["cases"] >= 1 and grade["cases_passed"] == grade["cases"] \
    and grade["checks_failed"] == 0
print("%d of %d cases, %d of %d checks" % (grade["cases_passed"], grade["cases"], grade["checks_passed"], grade["checks"]))
sys.exit(0 if ok else 1)' "$run_dir/grant_fixture/grade.json" 2> "$run_dir/grade-read.log")
        counted=$?
        if (( status == 0 && counted == 0 && graded > 0 )); then
            pass "S2.1 integration grade: $counts passed (grant_fixture/grade.json)"
        elif (( graded == 0 )); then
            fail "S2.1 integration driver produced no graded checks (see grant-fixture.log)"
            detail "$run_dir/grant-fixture.log"
        else
            fail "S2.1 integration grade: ${counts:-unreadable} passed, driver exit status $status (grant_fixture/grade.json)"
        fi

        # A second reader: the command-line reader on the archived record of the first case, and
        # the block hash once more with sha256sum, which shares no code with this project.
        store=$("$python" -c 'import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
grant = json.load(open(config["cases"][0]["grant"], encoding="utf-8"))
print("%s/%s/store %s" % (sys.argv[2], grant["grant_id"], config["cases"][0]["expect"]["tb_sha256"]))' \
            configs/integration/grant_fixture.json "$run_dir/grant_fixture" 2> "$run_dir/recovered.log")
        expected=${store##* }
        store=${store% *}
        if [[ -n "$store" ]] && "$python" -m ocudu_cuda_isac.episodes.grant show "$store" --pilot 0 --data 0 \
                > "$run_dir/recovered.json" 2>> "$run_dir/recovered.log"; then
            shown=$("$python" -c 'import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
print(record["tb_sha256"], "pilot 0 at symbol %d, subcarrier %d; data symbol 0 at symbol %d, subcarrier %d = %+.6f%+.6fj"
      % (record["pilot"]["symbol"], record["pilot"]["subcarrier"], record["data"]["symbol"], record["data"]["subcarrier"],
         record["data"]["value"][0], record["data"]["value"][1]))' "$run_dir/recovered.json" 2>> "$run_dir/recovered.log")
            independent=$(sha256sum "$store/tb.bin" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            if [[ -n "$expected" && "${shown%% *}" == "$expected" && "$independent" == "$expected" ]]; then
                pass "S2.1 a second reader recovers the record: ${shown#* }; transport block SHA-256 ${expected:0:16}" \
                    "by the reader and by sha256sum (recovered.json)"
            else
                fail "S2.1 a second reader recovers the record: the transport block hash is ${shown%% *} by the reader," \
                    "${independent:-unreadable} by sha256sum, $expected expected"
            fi
        else
            fail "S2.1 a second reader recovers the record (see recovered.log)"
            detail "$run_dir/recovered.log"
        fi
    else
        fail "S2.1 adapter build command (see adapter-build.log)"
        detail "$run_dir/adapter-build.log"
        note "SKIP  S2.1 adapter receipt, integration driver and record recovery: no adapter"
    fi
else
    fail "S2.1 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a build with a receipt"
    note "      Run 'make accept-s1.1' on this host first."
    note "SKIP  S2.1 reference check, adapter build, integration driver and record recovery: no Sprint 1.1 build"
fi

# The upstream control stays at the locked commit with no edited file.
step "S2.1 OCUDU source at the locked revision after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees --only ocudu

note "Evidence: $run_dir"
if (( failed )); then
    note "Sprint 2.1: NOT ACCEPTED on this host"
    exit 1
fi
note "Sprint 2.1: ACCEPTED on this $profile. One legal uplink grant and its transmit reference are archived," \
    "and both transmit references agree (S1.1 run $s11_id). Next: S2.2 episode_pair and S2.3 grid_apply."
