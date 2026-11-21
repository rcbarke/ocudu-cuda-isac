#!/usr/bin/env bash
# Run every Sprint 2.4 gate (replay_pusch) on this host and keep the evidence in a new run directory.
# The gate builds the native receiver dispatch against the accepted OCUDU build, fills every grid of
# the stimulus that Sprint 2.2 archived with the channel application of Sprint 2.3, hands the same
# grid to the host and to the CUDA PUSCH processor of the pinned source and compares the decoded
# bytes with the original transport block. A decode that falls back to the host path cannot close
# it. It never rebuilds OCUDU and never edits the sources, a build or an earlier run.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s2_4.sh [--run-dir DIR] [--s23-run DIR] [--s22-run DIR] [--grant-store DIR]..."
    echo "                                   [--host-rehearsal]"
    echo "Run it on a host that passed 'make accept-s1.2' and 'make accept-s2.1' to 'make accept-s2.3': it needs"
    echo "the accepted OCUDU build behind isac/bin, NumPy in isac, the accepted Sprint 2.3 run (default: the"
    echo "newest one) and the Sprint 2.2 run whose stimulus that run applied (default: the one it names)."
    echo "--grant-store names a directory that holds <grant_id>/store, for a host where the runs were moved."
    echo "--host-rehearsal grades the host path on a build without CUDA. A rehearsal is never an acceptance."
}
run_dir=""
s22_arg=""
s23_arg=""
rehearsal=0
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
        --s23-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s23_arg=$2; shift 2 ;;
        --grant-store)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            stores+=(--grant-store "$2"); shift 2 ;;
        --host-rehearsal) rehearsal=1; shift ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s2_4: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s24-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s2_4: $run_dir already exists. Evidence is never overwritten." >&2
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
# accepted RUN SPRINT: true when RUN holds an ACCEPTED verdict of SPRINT (written 2\.3) without a failed line.
accepted() {
    grep -qE "^Sprint $2: ACCEPTED on this " "$1/summary.txt" 2>/dev/null && ! grep -qE '^FAIL  ' "$1/summary.txt"
}
# The native tools must be found the way an activated shell finds them.
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 2.4 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if (( rehearsal )); then
    note "HOST REHEARSAL: the host path is graded on a build without CUDA. This run cannot accept the sprint."
fi
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
# builds the host-free dispatch core and runs its cases, so none of its tests may be skipped here.
tests any "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
tests complete "S2.4 unit gate, run directly: python tests/unit/test_replay_pusch.py" replay-pusch-unit.log \
    "$python" tests/unit/test_replay_pusch.py -v

# The grids come from the channel application that Sprint 2.3 accepted: the newest accepted run, or
# the one named. Its stored grids are compared with the grids decoded here.
s23_run=""
s23_id=""
if [[ -n "$s23_arg" ]]; then
    candidates=("$s23_arg")
else
    candidates=(results/runs/s23-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if accepted "$candidate" '2\.3' && [[ -f "$candidate/grid_apply/reference_pair/native/native.json" ]]; then
        s23_run=$candidate
    fi
done
profile="host"
if [[ -n "$s23_run" ]]; then
    s23_run=$(cd -- "$s23_run" && pwd)
    s23_id=$(basename -- "$s23_run")
    if grep -qE '^Sprint 2\.3: ACCEPTED on this DGX Spark' "$s23_run/summary.txt"; then profile="DGX Spark"; fi
    pass "S2.4 grids: the accepted Sprint 2.3 run $s23_id holds the stored grids of the pair (grid_apply/)"
else
    fail "S2.4 grids: no Sprint 2.3 ACCEPTED verdict with stored grids in ${s23_arg:-results/runs/s23-*}"
    note "      Run 'make accept-s2.3' on this host first, or name its run with --s23-run."
fi

# The stimulus is the one that run applied: the Sprint 2.2 run its manifest names, or the run named
# here, which must then be that one.
s22_run=""
s22_id=""
if [[ -n "$s23_run" ]]; then
    applied=$("$python" -c 'import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["s22_run"])' "$s23_run/grid_apply/manifest.json" 2> "$run_dir/stimulus.log")
    applied_id=$(basename -- "${applied:-unknown}")
    candidate=${s22_arg:-$applied}
    candidate=${candidate%/}
    if [[ -z "$applied" ]]; then
        fail "S2.4 stimulus: the Sprint 2.3 run $s23_id does not name the Sprint 2.2 run it applied (see stimulus.log)"
    elif [[ "$(basename -- "$candidate")" != "$applied_id" ]]; then
        fail "S2.4 stimulus: the Sprint 2.3 run $s23_id applied the stimulus of $applied_id, and --s22-run names" \
            "$(basename -- "$candidate")"
    elif accepted "$candidate" '2\.2' && [[ -f "$candidate/episode_pair/reference_pair/store/manifest.json" ]]; then
        s22_run=$(cd -- "$candidate" && pwd)
        s22_id=$applied_id
        pass "S2.4 stimulus: the accepted Sprint 2.2 run $s22_id holds the pair and the slot grants, and it is the" \
            "stimulus that $s23_id applied"
    else
        fail "S2.4 stimulus: no Sprint 2.2 ACCEPTED verdict with a stored pair in $candidate"
        note "      The Sprint 2.3 run names it. If the runs were moved, name the directory with --s22-run."
    fi
fi

# The receiver must be the pinned one: the OCUDU build that Sprint 1.1 published into the
# environment, which the adapter is compiled and linked against.
suffix=tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test
target=$(readlink -f -- "$venv_bin/pusch_e2e_pipeline_test" 2>/dev/null)
build_dir=${target%"/$suffix"}
grade_ok=0
if [[ -n "$s22_run" && -L "$venv_bin/pusch_e2e_pipeline_test" && -n "$target" && "$build_dir" != "$target" \
        && -f "$build_dir/isac-native.json" ]]; then
    step "S2.4 references: the Sprint 1.1 build, from the project interpreter (environment.json)" \
        environment.log "$python" scripts/verify_environment.py --profile core --with-native \
        --report "$run_dir/environment.json"

    # The adapter is built out of tree with the compile flags and the link line of the pinned test.
    # Its own CTest cases run in that build: the host-free dispatch core, and one decode on the host
    # path through the pinned PUSCH processor. The build itself starts no CUDA work.
    if "$python" scripts/native_adapter.py build --ocudu-build-dir "$build_dir" \
            --run-dir "$run_dir/adapter-build" > "$run_dir/adapter-build.log" 2>&1 \
            && "$python" scripts/native_adapter.py verify > "$run_dir/adapter-receipt.json" 2> "$run_dir/adapter-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
names = receipt["executables"]
print("%s|replay_pusch %s beside grid_apply %s and grant_fixture %s, %s, %d adapter tests, %d compiler warnings"
      % ("cuda" if receipt["pin"]["cuda"] else "host", names["replay_pusch"]["sha256"][:12],
         names["grid_apply"]["sha256"][:12], names["grant_fixture"]["sha256"][:12],
         "with ENABLE_CUDA" if receipt["pin"]["cuda"] else "without CUDA", receipt["tests_run"],
         receipt["compiler_warnings"]))' "$run_dir/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        built=${facts%%|*}
        if [[ -n "$facts" ]]; then
            pass "S2.4 adapter: built against the Sprint 1.1 build and verified, ${facts#*|} (adapter-receipt.json)"
        else
            fail "S2.4 adapter: the receipt names no replay_pusch (see adapter-verify.log)"
            detail "$run_dir/adapter-verify.log"
        fi
        ctest_log="$run_dir/adapter-build/adapter-test.log"
        core=$(grep -cE 'dispatch_core_[a-z0-9_]+ \.+ +Passed' "$ctest_log" 2>/dev/null)
        core=${core:-0}
        if (( core >= 12 )) && grep -qE 'replay_pusch_host_decode \.+ +Passed' "$ctest_log" 2>/dev/null; then
            pass "S2.4 native unit cases: $core cases of the host-free dispatch core and one decode through the" \
                "pinned host PUSCH processor passed in the adapter build (adapter-build/adapter-test.log)"
        else
            fail "S2.4 native unit cases: $core cases of the dispatch core passed and the host decode self-test did" \
                "not, or the log is missing (adapter-build/adapter-test.log)"
        fi
        # The hardware gate needs the CUDA processor. A build without it can only be rehearsed.
        driver_mode=()
        if (( rehearsal )); then
            driver_mode=(--host-rehearsal)
            note "NOTE  S2.4 hardware: host rehearsal on an adapter built for the ${built:-unknown} path. The CUDA" \
                "decodes of this gate stay open."
        elif [[ "$built" == cuda ]]; then
            pass "S2.4 hardware: the adapter is linked against the CUDA build, so the CUDA processor can execute"
        elif [[ -n "$facts" ]]; then
            fail "S2.4 hardware: the accepted build has no CUDA, so the CUDA processor cannot execute here. A host" \
                "path decode cannot close this gate. Rehearse with --host-rehearsal, accept on the DGX Spark."
        fi

        # Integration driver. It grades its own evidence, and its lines are copied here.
        note "Decoding the stimulus of $s22_id on the grids of $s23_id with configs/integration/replay_pusch.json."
        "$python" tests/integration/replay_pusch.py --config configs/integration/replay_pusch.json \
            --s22-run "$s22_run" --s23-run "$s23_run" ${stores[@]+"${stores[@]}"} \
            ${driver_mode[@]+"${driver_mode[@]}"} --output-dir "$run_dir/replay_pusch" \
            > "$run_dir/replay-pusch.log" 2>&1
        status=$?
        grade="$run_dir/replay_pusch/grade.txt"
        graded=$(grep -cE '^(PASS|FAIL)  ' "$grade" 2>/dev/null)
        graded=${graded:-0}
        grep -E '^(PASS|FAIL|NOTE)  ' "$grade" 2>/dev/null | tee -a "$summary"
        if grep -qE '^FAIL  ' "$grade" 2>/dev/null; then failed=1; fi
        wanted=pass
        if (( rehearsal )); then wanted=rehearsal; fi
        counts=$("$python" -c 'import json, sys
grade = json.load(open(sys.argv[1], encoding="utf-8"))
ok = grade["judgment"] == sys.argv[2] and grade["cases"] >= 1 and grade["cases_passed"] == grade["cases"] \
    and grade["checks_failed"] == 0 and (sys.argv[2] != "pass" or grade["hardware_gate"] == "closed")
print("%d of %d cases, %d of %d checks, judgment %s, hardware gate %s"
      % (grade["cases_passed"], grade["cases"], grade["checks_passed"], grade["checks"], grade["judgment"],
         grade["hardware_gate"].split(":")[0]))
sys.exit(0 if ok else 1)' "$run_dir/replay_pusch/grade.json" "$wanted" 2> "$run_dir/grade-read.log")
        counted=$?
        if (( status == 0 && counted == 0 && graded > 0 )); then
            grade_ok=1
            pass "S2.4 integration grade: $counts (replay_pusch/grade.json)"
        elif (( graded == 0 )); then
            fail "S2.4 integration driver produced no graded checks (see replay-pusch.log)"
            detail "$run_dir/replay-pusch.log"
        else
            fail "S2.4 integration grade: ${counts:-unreadable}, driver exit status $status (replay_pusch/grade.json)"
        fi

        # A second reader: the command-line reader on the radio records of the pair, with the standard
        # library only. The bytes it names are compared once more with the original transport block
        # by cmp, and their hash is taken once more by sha256sum.
        pair="$run_dir/replay_pusch/reference_pair/run"
        if "$python" -m ocudu_cuda_isac.episodes.replay_pusch show "$pair/native" --sequence 0 --backend cuda \
                > "$run_dir/recovered.json" 2> "$run_dir/recovered.log"; then
            shown=$("$python" -c 'import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
plan = json.load(open(sys.argv[2], encoding="utf-8"))
block = [grant["tb"]["path"] for grant in plan["grants"] if grant["grant_id"] == record["grant_id"]][0]
print(record["decoded_sha256"])
print(record["decoded_file"])
print(block)
print("%d radio records verified; the decode %s (slot %d, grant %s) is %s on the %s path with %d bytes from the grid %s"
      % (record["verified_records"], record["key"], record["slot"], record["grant_id"], record["outcome"],
         record["executed"], record["tb_bytes"], record["ingress_sha256"][:16]))' \
                "$run_dir/recovered.json" "$pair/plan.json" 2>> "$run_dir/recovered.log")
            digest=$(echo "$shown" | sed -n 1p)
            file=$(echo "$shown" | sed -n 2p)
            block=$(echo "$shown" | sed -n 3p)
            text=$(echo "$shown" | sed -n 4p)
            independent=$(sha256sum "$pair/native/radio/$file" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            if [[ -n "$independent" && "$digest" == "$independent" ]] \
                    && cmp -s -- "$pair/native/radio/$file" "$block" 2>> "$run_dir/recovered.log"; then
                pass "S2.4 a second reader recovers a decode: $text; the decoded bytes are the original transport" \
                    "block byte for byte by cmp, SHA-256 ${independent:0:16} by the reader and by sha256sum (recovered.json)"
            else
                fail "S2.4 a second reader recovers a decode: the decoded bytes have the digest ${digest:-unreadable} by" \
                    "the reader and ${independent:-unreadable} by sha256sum, or cmp finds them different from the" \
                    "original transport block ${block:-of an unreadable plan}"
            fi
        else
            fail "S2.4 a second reader recovers a decode (see recovered.log)"
            detail "$run_dir/recovered.log"
        fi
    else
        fail "S2.4 adapter: build or verification failed (see adapter-build.log, adapter-verify.log)"
        detail "$run_dir/adapter-build.log"
        detail "$run_dir/adapter-verify.log"
        note "SKIP  S2.4 native unit cases, integration driver and decode recovery: no adapter"
    fi
elif [[ -n "$s22_run" ]]; then
    fail "S2.4 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a build with a receipt"
    note "      Run 'make accept-s1.1' on this host first."
    note "SKIP  S2.4 reference check, adapter build, integration driver and decode recovery: no Sprint 1.1 build"
else
    note "SKIP  S2.4 reference check, adapter build, integration driver and decode recovery: no stimulus"
fi

# The upstream sources stay at the locked commits with no edited file.
step "S2.4 sources at the locked revisions after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees

note "Evidence: $run_dir"
if (( failed || ! grade_ok )); then
    note "Sprint 2.4: NOT ACCEPTED on this host"
    exit 1
fi
if (( rehearsal )); then
    note "Sprint 2.4: REHEARSED on this host, NOT ACCEPTED. The host path returned the original transport blocks" \
        "(S2.2 run $s22_id, S2.3 run $s23_id). The CUDA decodes are open: run 'make accept-s2.4' on the DGX Spark."
    exit 0
fi
note "Sprint 2.4: ACCEPTED on this $profile. The host and the CUDA PUSCH processor of the pinned source decode the" \
    "replayed grids and return the original transport blocks (S2.2 run $s22_id, S2.3 run $s23_id)." \
    "Next: 'make accept-s2' closes Sprint 2, then S3.1 pilot_ls."
