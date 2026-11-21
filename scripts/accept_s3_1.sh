#!/usr/bin/env bash
# Run every Sprint 3.1 gate (pilot_ls) on this host and keep the evidence in a new run directory.
# The gate builds the tapped receiver replay against the accepted OCUDU build, decodes the plans
# that the accepted Sprint 2.4 run archived once more, with an observer that exports the pilot
# least squares of the grant's sensing source from each ready grid (the stage O2), and compares the
# radio records with those of that run, which the accepted replay_pusch wrote without an observer.
# A decode that falls back to the host path cannot close it. It never rebuilds OCUDU and never
# edits the sources, a build or an earlier run.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s3_1.sh [--run-dir DIR] [--s24-run DIR] [--host-rehearsal]"
    echo "Run it on a host that passed 'make accept-s2': it needs the accepted OCUDU build behind isac/bin,"
    echo "NumPy in isac and the accepted Sprint 2.4 run (default: the newest one), whose plans it decodes"
    echo "again and whose radio records are the control. The runs that the plans of that run name must"
    echo "still be where they were."
    echo "--host-rehearsal grades the host path on a build without CUDA against a rehearsed Sprint 2.4"
    echo "run. A rehearsal is never an acceptance."
}
run_dir=""
s24_arg=""
rehearsal=0
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --s24-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s24_arg=$2; shift 2 ;;
        --host-rehearsal) rehearsal=1; shift ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s3_1: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s31-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s3_1: $run_dir already exists. Evidence is never overwritten." >&2
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
# verdict RUN: true when RUN holds the verdict of Sprint 2.4 that this gate needs, without a failed line.
verdict() {
    local wanted='^Sprint 2\.4: ACCEPTED on this '
    if (( rehearsal )); then wanted='^Sprint 2\.4: REHEARSED on this host, NOT ACCEPTED'; fi
    grep -qE "$wanted" "$1/summary.txt" 2>/dev/null && ! grep -qE '^FAIL  ' "$1/summary.txt"
}
# The native tools must be found the way an activated shell finds them.
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 3.1 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
# builds the host-free observer core and runs its cases, so none of its tests may be skipped here.
tests any "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
tests complete "S3.1 unit gate, run directly: python tests/unit/test_pilot_ls.py" pilot-ls-unit.log \
    "$python" tests/unit/test_pilot_ls.py -v

# The control is the run that Sprint 2.4 accepted: the newest one, or the one named. Its plans are
# decoded again here, and its radio records were written by the accepted replay_pusch without a tap.
s24_run=""
s24_id=""
if [[ -n "$s24_arg" ]]; then
    candidates=("$s24_arg")
else
    candidates=(results/runs/s24-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if verdict "$candidate" && [[ -f "$candidate/replay_pusch/grade.json" \
            && -f "$candidate/replay_pusch/reference_pair/run/plan.json" ]]; then
        s24_run=$candidate
    fi
done
profile="host"
kind="ACCEPTED"
if (( rehearsal )); then kind="REHEARSED"; fi
if [[ -n "$s24_run" ]]; then
    s24_run=$(cd -- "$s24_run" && pwd)
    s24_id=$(basename -- "$s24_run")
    if grep -qE '^Sprint 2\.4: ACCEPTED on this DGX Spark' "$s24_run/summary.txt"; then profile="DGX Spark"; fi
    pass "S3.1 control: the Sprint 2.4 run $s24_id is $kind and holds the plans and the radio records of the" \
        "untapped decodes (replay_pusch/)"
else
    fail "S3.1 control: no Sprint 2.4 $kind verdict with archived plans in ${s24_arg:-results/runs/s24-*}"
    if (( rehearsal )); then
        note "      Run 'make accept-s2.4 REHEARSAL=1' on this host first, or name its run with --s24-run."
    else
        note "      Run 'make accept-s2.4' on this host first, or name its run with --s24-run."
    fi
fi

# The receiver must be the pinned one: the OCUDU build that Sprint 1.1 published into the
# environment, which the adapter is compiled and linked against.
suffix=tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test
target=$(readlink -f -- "$venv_bin/pusch_e2e_pipeline_test" 2>/dev/null)
build_dir=${target%"/$suffix"}
grade_ok=0
if [[ -n "$s24_run" && -L "$venv_bin/pusch_e2e_pipeline_test" && -n "$target" && "$build_dir" != "$target" \
        && -f "$build_dir/isac-native.json" ]]; then
    step "S3.1 references: the Sprint 1.1 build, from the project interpreter (environment.json)" \
        environment.log "$python" scripts/verify_environment.py --profile core --with-native \
        --report "$run_dir/environment.json"

    # The adapter is built out of tree with the compile flags and the link line of the pinned test.
    # Its own CTest cases run in that build: the host-free observer core, and one grant through the
    # pinned grid and the pinned host PUSCH processor with the tap off and on. The build itself
    # starts no CUDA work.
    if "$python" scripts/native_adapter.py build --ocudu-build-dir "$build_dir" \
            --run-dir "$run_dir/adapter-build" > "$run_dir/adapter-build.log" 2>&1 \
            && "$python" scripts/native_adapter.py verify > "$run_dir/adapter-receipt.json" 2> "$run_dir/adapter-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
names = receipt["executables"]
print("%s|pilot_ls %s beside replay_pusch %s, grid_apply %s and grant_fixture %s, %s, %d adapter tests, %d compiler warnings"
      % ("cuda" if receipt["pin"]["cuda"] else "host", names["pilot_ls"]["sha256"][:12],
         names["replay_pusch"]["sha256"][:12], names["grid_apply"]["sha256"][:12], names["grant_fixture"]["sha256"][:12],
         "with ENABLE_CUDA" if receipt["pin"]["cuda"] else "without CUDA", receipt["tests_run"],
         receipt["compiler_warnings"]))' "$run_dir/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        built=${facts%%|*}
        if [[ -n "$facts" ]]; then
            pass "S3.1 adapter: built against the Sprint 1.1 build and verified, ${facts#*|} (adapter-receipt.json)"
        else
            fail "S3.1 adapter: the receipt names no pilot_ls (see adapter-verify.log)"
            detail "$run_dir/adapter-verify.log"
        fi
        # The accepted executables are not edited by this sprint: the build of this tree gives the
        # control, the channel application and the fixture adapter the bytes of the Sprint 2.4 run.
        same=$("$python" -c 'import json, sys
now = json.load(open(sys.argv[1], encoding="utf-8"))["executables"]
then = json.load(open(sys.argv[2], encoding="utf-8"))["executables"]
names = ("replay_pusch", "grid_apply", "grant_fixture")
changed = [name for name in names if now[name]["sha256"] != then[name]["sha256"]]
print("%s|%s" % ("same" if not changed else "changed",
                 ", ".join("%s %s" % (name, then[name]["sha256"][:12]) for name in names) if not changed
                 else ", ".join("%s is %s and was %s" % (name, now[name]["sha256"][:12], then[name]["sha256"][:12])
                                for name in changed)))' "$run_dir/adapter-receipt.json" "$s24_run/adapter-receipt.json" \
            2>> "$run_dir/adapter-verify.log")
        if [[ "${same%%|*}" == same ]]; then
            pass "S3.1 control executables: this tree builds the executables that the Sprint 2.4 run used, byte for" \
                "byte: ${same#*|}"
        else
            fail "S3.1 control executables: an executable that Sprint 2 accepted has other bytes in this build:" \
                "${same#*|} (adapter-receipt.json against the receipt of $s24_id)"
        fi
        ctest_log="$run_dir/adapter-build/adapter-test.log"
        core=$(grep -cE 'observer_core_[a-z0-9_]+ \.+ +Passed' "$ctest_log" 2>/dev/null)
        core=${core:-0}
        if (( core >= 11 )) && grep -qE 'pilot_ls_host_tap \.+ +Passed' "$ctest_log" 2>/dev/null; then
            pass "S3.1 native unit cases: $core cases of the host-free observer core passed, and one grant through the" \
                "pinned grid and the pinned host PUSCH processor reads 1.00119 on every pilot and decodes to the" \
                "original bytes with the tap on (adapter-build/adapter-test.log)"
        else
            fail "S3.1 native unit cases: $core cases of the observer core passed and the tap self-test on the pinned" \
                "grid did not, or the log is missing (adapter-build/adapter-test.log)"
        fi
        # The hardware gate needs the CUDA processor. A build without it can only be rehearsed.
        driver_mode=()
        if (( rehearsal )); then
            driver_mode=(--host-rehearsal)
            note "NOTE  S3.1 hardware: host rehearsal on an adapter built for the ${built:-unknown} path. The CUDA" \
                "decodes of this gate stay open."
        elif [[ "$built" == cuda ]]; then
            pass "S3.1 hardware: the adapter is linked against the CUDA build, so the CUDA processor can execute"
        elif [[ -n "$facts" ]]; then
            fail "S3.1 hardware: the accepted build has no CUDA, so the CUDA processor cannot execute here. A host" \
                "path decode cannot close this gate. Rehearse with --host-rehearsal, accept on the DGX Spark."
        fi

        # Integration driver. It grades its own evidence, and its lines are copied here.
        note "Tapping the plans of $s24_id with configs/integration/pilot_ls.json."
        "$python" tests/integration/pilot_ls.py --config configs/integration/pilot_ls.json \
            --s24-run "$s24_run" ${driver_mode[@]+"${driver_mode[@]}"} --output-dir "$run_dir/pilot_ls" \
            > "$run_dir/pilot-ls.log" 2>&1
        status=$?
        grade="$run_dir/pilot_ls/grade.txt"
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
sys.exit(0 if ok else 1)' "$run_dir/pilot_ls/grade.json" "$wanted" 2> "$run_dir/grade-read.log")
        counted=$?
        if (( status == 0 && counted == 0 && graded > 0 )); then
            grade_ok=1
            pass "S3.1 integration grade: $counts (pilot_ls/grade.json)"
        elif (( graded == 0 )); then
            fail "S3.1 integration driver produced no graded checks (see pilot-ls.log)"
            detail "$run_dir/pilot-ls.log"
        else
            fail "S3.1 integration grade: ${counts:-unreadable}, driver exit status $status (pilot_ls/grade.json)"
        fi

        # A second reader: the command-line reader on the observations of the pair, with the standard
        # library only. The hash of the payload it names and of the grid that payload was computed
        # from are taken once more by sha256sum, and the grid digest is read from the first radio
        # record of the Sprint 2.4 run with a third reader.
        pair="$run_dir/pilot_ls/reference_pair/run"
        if "$python" -m ocudu_cuda_isac.episodes.pilot_ls show "$pair/native" --sequence 0 --rx 0 --row 0 --pilot 0 \
                > "$run_dir/recovered.json" 2> "$run_dir/recovered.log"; then
            shown=$("$python" -c 'import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
control = json.loads(open(sys.argv[2], encoding="utf-8").readline())
element = record["element"]
print(record["payload_sha256"])
print(record["payload_file"])
print(record["ingress_sha256"])
print(control["ingress"]["readback_sha256"] if control["opportunity_id"] == record["opportunity_id"] else "another opportunity")
print("%d observations verified; the %s estimate of %s by the %s on %s (slot %d, grant %s) has the shape %s with %d valid "
      "elements, and its first element, symbol %d, subcarrier %d of receive plane %d, is %.6f%+.6fj"
      % (record["verified_observations"], record["stage"], record["source"], record["producer"], record["opportunity_id"],
         record["slot"], record["grant_id"], record["shape"], record["valid"], element["symbol"], element["subcarrier"],
         element["rx_plane"], element["value"][0], element["value"][1]))' \
                "$run_dir/recovered.json" "$s24_run/replay_pusch/reference_pair/run/native/radio/radio.jsonl" \
                2>> "$run_dir/recovered.log")
            digest=$(echo "$shown" | sed -n 1p)
            file=$(echo "$shown" | sed -n 2p)
            ingress=$(echo "$shown" | sed -n 3p)
            control=$(echo "$shown" | sed -n 4p)
            text=$(echo "$shown" | sed -n 5p)
            independent=$(sha256sum "$pair/native/observer/$file" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            grid=$(sha256sum "$pair/native/ingress/0000.c64" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            if [[ -n "$independent" && "$digest" == "$independent" && -n "$grid" && "$grid" == "$ingress" \
                    && "$grid" == "$control" ]]; then
                pass "S3.1 a second reader recovers an observation: $text; SHA-256 of the payload ${independent:0:16} by" \
                    "the reader and by sha256sum; the grid it was computed from is ${grid:0:16} by sha256sum, in the" \
                    "observation and in the radio record of $s24_id (recovered.json)"
            else
                fail "S3.1 a second reader recovers an observation: the payload has the digest ${digest:-unreadable} by" \
                    "the reader and ${independent:-unreadable} by sha256sum, or its grid is ${grid:-unreadable} by" \
                    "sha256sum, ${ingress:-unreadable} in the observation and ${control:-unreadable} in the radio" \
                    "record of $s24_id"
            fi
        else
            fail "S3.1 a second reader recovers an observation (see recovered.log)"
            detail "$run_dir/recovered.log"
        fi
    else
        fail "S3.1 adapter: build or verification failed (see adapter-build.log, adapter-verify.log)"
        detail "$run_dir/adapter-build.log"
        detail "$run_dir/adapter-verify.log"
        note "SKIP  S3.1 native unit cases, integration driver and observation recovery: no adapter"
    fi
elif [[ -n "$s24_run" ]]; then
    fail "S3.1 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a build with a receipt"
    note "      Run 'make accept-s1.1' on this host first."
    note "SKIP  S3.1 reference check, adapter build, integration driver and observation recovery: no Sprint 1.1 build"
else
    note "SKIP  S3.1 reference check, adapter build, integration driver and observation recovery: no control"
fi

# The upstream sources stay at the locked commits with no edited file.
step "S3.1 sources at the locked revisions after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees

note "Evidence: $run_dir"
if (( failed || ! grade_ok )); then
    note "Sprint 3.1: NOT ACCEPTED on this host"
    exit 1
fi
if (( rehearsal )); then
    note "Sprint 3.1: REHEARSED on this host, NOT ACCEPTED. The observer exported the pilot least squares from the" \
        "grids of the rehearsed Sprint 2.4 run $s24_id and the host path decodes were unchanged. The CUDA decodes" \
        "are open: run 'make accept-s3.1' on the DGX Spark."
    exit 0
fi
note "Sprint 3.1: ACCEPTED on this $profile. The observer exports the pilot least squares of the PUSCH DM-RS, the" \
    "stage O2, from the grid that the host and the CUDA PUSCH processor of the pinned source decode, and every" \
    "decode is the one of the accepted Sprint 2.4 run $s24_id. Next: S3.2 snapshot_ring."
