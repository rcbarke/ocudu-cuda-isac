#!/usr/bin/env bash
# Run every Sprint 3.2 gate (snapshot_ring) on this host and keep the evidence in a new run directory.
# The gate builds the observation ring and the tapped receiver replay against the accepted OCUDU
# build and decodes the reference pair of the accepted Sprint 3.1 run once more, in the evidence
# modes audit and timed, with every observation published on the ring obs-ring/v1 to a consumer
# in a second process. The decodes are compared with those of the accepted Sprint 2.4 run, which
# the accepted replay_pusch wrote without an observer, and the estimates with those of the
# accepted Sprint 3.1 run. A timed run that computes a checksum between its first fill and its
# drain, or whose producer makes a system call between an admission and its publication, cannot
# close it, and neither can a decode that falls back to the host path. It never rebuilds OCUDU
# and never edits the sources, a build or an earlier run.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s3_2.sh [--run-dir DIR] [--s31-run DIR] [--host-rehearsal]"
    echo "Run it on a host that passed 'make accept-s3.1': it needs the accepted OCUDU build behind isac/bin,"
    echo "NumPy in isac and the accepted Sprint 3.1 run (default: the newest one). That run names the plan of"
    echo "the reference pair and holds the accepted estimates, and the Sprint 2.4 run that plan lies in holds"
    echo "the decodes without an observer. Both runs must still be where they were."
    echo "--host-rehearsal grades the host path on a build without CUDA against a rehearsed Sprint 3.1"
    echo "run. A rehearsal is never an acceptance."
}
run_dir=""
s31_arg=""
rehearsal=0
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --s31-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s31_arg=$2; shift 2 ;;
        --host-rehearsal) rehearsal=1; shift ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s3_2: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s32-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s3_2: $run_dir already exists. Evidence is never overwritten." >&2
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
# verdict RUN: true when RUN holds the verdict of Sprint 3.1 that this gate needs, without a failed line.
verdict() {
    local wanted='^Sprint 3\.1: ACCEPTED on this '
    if (( rehearsal )); then wanted='^Sprint 3\.1: REHEARSED on this host, NOT ACCEPTED'; fi
    grep -qE "$wanted" "$1/summary.txt" 2>/dev/null && ! grep -qE '^FAIL  ' "$1/summary.txt"
}
# The native tools must be found the way an activated shell finds them.
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 3.2 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
# builds the ring, the producer glue and the cores it joins and runs their cases, and lets both
# readers read one native producer, so none of its tests may be skipped here.
tests any "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
tests complete "S3.2 unit gate, run directly: python tests/unit/test_snapshot_ring.py" snapshot-ring-unit.log \
    "$python" tests/unit/test_snapshot_ring.py -v

# The control is the run that Sprint 3.1 accepted: the newest one, or the one named. It tapped the
# reference pair of the accepted Sprint 2.4 run, whose plan this gate decodes again.
s31_run=""
s31_id=""
if [[ -n "$s31_arg" ]]; then
    candidates=("$s31_arg")
else
    candidates=(results/runs/s31-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if verdict "$candidate" && [[ -f "$candidate/pilot_ls/grade.json" \
            && -f "$candidate/pilot_ls/reference_pair/run/plan.json" ]]; then
        s31_run=$candidate
    fi
done
profile="host"
kind="ACCEPTED"
if (( rehearsal )); then kind="REHEARSED"; fi
if [[ -n "$s31_run" ]]; then
    s31_run=$(cd -- "$s31_run" && pwd)
    s31_id=$(basename -- "$s31_run")
    if grep -qE '^Sprint 3\.1: ACCEPTED on this DGX Spark' "$s31_run/summary.txt"; then profile="DGX Spark"; fi
    pass "S3.2 control: the Sprint 3.1 run $s31_id is $kind and holds the plan of the reference pair and the" \
        "accepted estimates (pilot_ls/)"
else
    fail "S3.2 control: no Sprint 3.1 $kind verdict with the plan of the reference pair in ${s31_arg:-results/runs/s31-*}"
    if (( rehearsal )); then
        note "      Run 'make accept-s3.1 REHEARSAL=1' on this host first, or name its run with --s31-run."
    else
        note "      Run 'make accept-s3.1' on this host first, or name its run with --s31-run."
    fi
fi

# The receiver must be the pinned one: the OCUDU build that Sprint 1.1 published into the
# environment, which the adapter is compiled and linked against.
suffix=tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test
target=$(readlink -f -- "$venv_bin/pusch_e2e_pipeline_test" 2>/dev/null)
build_dir=${target%"/$suffix"}
grade_ok=0
if [[ -n "$s31_run" && -L "$venv_bin/pusch_e2e_pipeline_test" && -n "$target" && "$build_dir" != "$target" \
        && -f "$build_dir/isac-native.json" ]]; then
    step "S3.2 references: the Sprint 1.1 build, from the project interpreter (environment.json)" \
        environment.log "$python" scripts/verify_environment.py --profile core --with-native \
        --report "$run_dir/environment.json"

    # The adapter is built out of tree with the compile flags and the link line of the pinned test,
    # and the observation ring with it, from native/obs_ring/ and with its own flags. Their CTest
    # cases run in that build. The build itself starts no CUDA work.
    if "$python" scripts/native_adapter.py build --ocudu-build-dir "$build_dir" \
            --run-dir "$run_dir/adapter-build" > "$run_dir/adapter-build.log" 2>&1 \
            && "$python" scripts/native_adapter.py verify > "$run_dir/adapter-receipt.json" 2> "$run_dir/adapter-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
names = receipt["executables"]
print("%s|pilot_ls %s, obs_tail %s and obs_feed %s, %s, %d adapter tests, %d compiler warnings"
      % ("cuda" if receipt["pin"]["cuda"] else "host", names["pilot_ls"]["sha256"][:12], names["obs_tail"]["sha256"][:12],
         names["obs_feed"]["sha256"][:12], "with ENABLE_CUDA" if receipt["pin"]["cuda"] else "without CUDA",
         receipt["tests_run"], receipt["compiler_warnings"]))' "$run_dir/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        built=${facts%%|*}
        if [[ -n "$facts" ]]; then
            pass "S3.2 adapter: built against the Sprint 1.1 build and verified, ${facts#*|} (adapter-receipt.json)"
        else
            fail "S3.2 adapter: the receipt names no pilot_ls, obs_tail and obs_feed (see adapter-verify.log)"
            detail "$run_dir/adapter-verify.log"
        fi
        # This sprint amends two cores that Sprint 2 accepted, so the tools that link them have new
        # bytes: the driver below shows that they decode and estimate what the accepted ones did.
        # The fixture adapter links neither core and must keep the bytes of the accepted run.
        same=$("$python" -c 'import json, sys
now = json.load(open(sys.argv[1], encoding="utf-8"))["executables"]
then = json.load(open(sys.argv[2], encoding="utf-8"))["executables"]
kept = now["grant_fixture"]["sha256"] == then["grant_fixture"]["sha256"]
print("%s|grant_fixture %s; rebuilt from the amended cores: %s"
      % ("same" if kept else "changed",
         then["grant_fixture"]["sha256"][:12] if kept else "is %s and was %s" % (now["grant_fixture"]["sha256"][:12],
                                                                                 then["grant_fixture"]["sha256"][:12]),
         ", ".join("%s %s (accepted: %s)" % (name, now[name]["sha256"][:12], then[name]["sha256"][:12])
                   for name in ("replay_pusch", "grid_apply", "pilot_ls") if name in now and name in then)))' \
            "$run_dir/adapter-receipt.json" "$s31_run/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        if [[ "${same%%|*}" == same ]]; then
            pass "S3.2 accepted executables: the fixture adapter, which links no amended core, has the bytes of the" \
                "Sprint 3.1 run: ${same#*|}"
        else
            fail "S3.2 accepted executables: the fixture adapter links no core that this sprint amended and must keep" \
                "its bytes: ${same#*|} (adapter-receipt.json against the receipt of $s31_id)"
        fi
        ctest_log="$run_dir/adapter-build/adapter-test.log"
        ring_cases=$(grep -cE 'obs_ring_[a-z0-9_]+ \.+ +Passed' "$ctest_log" 2>/dev/null)
        glue_cases=$(grep -cE 'obs_publisher_[a-z0-9_]+ \.+ +Passed' "$ctest_log" 2>/dev/null)
        ring_cases=${ring_cases:-0}
        glue_cases=${glue_cases:-0}
        amended=0
        for name in replay_core_evidence dispatch_core_timed observer_core_timed pilot_ls_host_tap; do
            if grep -qE "$name \.+ +Passed" "$ctest_log" 2>/dev/null; then amended=$((amended + 1)); fi
        done
        if (( ring_cases >= 16 && glue_cases >= 6 && amended == 4 )); then
            pass "S3.2 native unit cases: $ring_cases cases of the ring and $glue_cases of the producer glue passed, the" \
                "timed cases of the three cores passed, and one grant through the pinned grid and the pinned host PUSCH" \
                "processor decodes in the mode timed without one SHA-256 from its fill to its release" \
                "(adapter-build/adapter-test.log)"
        else
            fail "S3.2 native unit cases: $ring_cases of 16 cases of the ring and $glue_cases of 6 of the producer glue" \
                "passed, and $amended of the 4 timed cases of the cores and of the tool, or the log is missing" \
                "(adapter-build/adapter-test.log)"
        fi
        # The hardware gate needs the CUDA processor. A build without it can only be rehearsed.
        driver_mode=()
        if (( rehearsal )); then
            driver_mode=(--host-rehearsal)
            note "NOTE  S3.2 hardware: host rehearsal on an adapter built for the ${built:-unknown} path. The CUDA" \
                "decodes of this gate stay open."
        elif [[ "$built" == cuda ]]; then
            pass "S3.2 hardware: the adapter is linked against the CUDA build, so the CUDA processor can execute"
        elif [[ -n "$facts" ]]; then
            fail "S3.2 hardware: the accepted build has no CUDA, so the CUDA processor cannot execute here. A host" \
                "path decode cannot close this gate. Rehearse with --host-rehearsal, accept on the DGX Spark."
        fi

        # Integration driver. It grades its own evidence, and its lines are copied here.
        note "Publishing the reference pair of $s31_id with configs/integration/snapshot_ring.json."
        "$python" tests/integration/snapshot_ring.py --config configs/integration/snapshot_ring.json \
            --s31-run "$s31_run" ${driver_mode[@]+"${driver_mode[@]}"} --output-dir "$run_dir/snapshot_ring" \
            > "$run_dir/snapshot-ring.log" 2>&1
        status=$?
        grade="$run_dir/snapshot_ring/grade.txt"
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
sys.exit(0 if ok else 1)' "$run_dir/snapshot_ring/grade.json" "$wanted" 2> "$run_dir/grade-read.log")
        counted=$?
        if (( status == 0 && counted == 0 && graded > 0 )); then
            grade_ok=1
            pass "S3.2 integration grade: $counts (snapshot_ring/grade.json)"
        elif (( graded == 0 )); then
            fail "S3.2 integration driver produced no graded checks (see snapshot-ring.log)"
            detail "$run_dir/snapshot-ring.log"
        else
            fail "S3.2 integration grade: ${counts:-unreadable}, driver exit status $status (snapshot_ring/grade.json)"
        fi

        # A second reader, with tools that are not the driver's: the command-line reader on what the
        # consumer of the timed run kept. The payload of its first record is the tail of the record
        # file. sha256sum takes its digest and the digest of the payload file that the audited run
        # and the accepted Sprint 3.1 run wrote for the same opportunity. All three must be equal.
        timed="$run_dir/snapshot_ring/timed/run"
        if "$python" -m ocudu_cuda_isac.sensing.obs show "$timed/consumer-0" --sequence 0 \
                > "$run_dir/recovered.json" 2> "$run_dir/recovered.log" \
                && "$python" -m ocudu_cuda_isac.episodes.snapshot_ring show "$timed/native" \
                > "$run_dir/recovered-account.json" 2>> "$run_dir/recovered.log"; then
            shown=$("$python" -c 'import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
account = json.load(open(sys.argv[2], encoding="utf-8"))
host, window = record["staged_host"], account["hash_in_window"]
print(record["file"])
print(record["payload"]["bytes"])
print("%04d.o2.c64" % host["opportunity_sequence"])
print("%d records kept by the %s consumer in the mode %s; record 0 of epoch %d is the %s estimate of %s in the layout %s, "
      "shape %s, %d valid elements, published in the mode %s with %s digest, from grid %d of generation %d; the run "
      "computed %d digests over %d bytes between its first fill and its drain"
      % (record["records"], record["reader"], record["wait"], record["keys"]["epoch"], record["what"]["stage"],
         record["what"]["source"], record["what"]["layout"], record["shape"]["shape"], record["validity"]["valid"],
         record["what"]["evidence"], "a" if record["payload"]["digest_kind"] else "no", host["grid_index"],
         host["grid_generation"], window["digests"], window["bytes"]))' \
                "$run_dir/recovered.json" "$run_dir/recovered-account.json" 2>> "$run_dir/recovered.log")
            file=$(echo "$shown" | sed -n 1p)
            bytes=$(echo "$shown" | sed -n 2p)
            payload=$(echo "$shown" | sed -n 3p)
            text=$(echo "$shown" | sed -n 4p)
            kept=$(tail -c "${bytes:-0}" "$timed/consumer-0/$file" 2>> "$run_dir/recovered.log" | sha256sum | cut -d ' ' -f 1)
            audited=$(sha256sum "$run_dir/snapshot_ring/audit/run/native/observer/$payload" 2>> "$run_dir/recovered.log" \
                | cut -d ' ' -f 1)
            accepted=$(sha256sum "$s31_run/pilot_ls/reference_pair/run/native/observer/$payload" \
                2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            if [[ -n "$file" && -n "$audited" && "$kept" == "$audited" && "$kept" == "$accepted" ]]; then
                pass "S3.2 a second reader recovers a timed record: $text; SHA-256 of its payload ${kept:0:16} by" \
                    "sha256sum, the digest of $payload of the audited run and of the accepted run $s31_id" \
                    "(recovered.json, recovered-account.json)"
            else
                fail "S3.2 a second reader recovers a timed record: its payload has the digest ${kept:-unreadable} by" \
                    "sha256sum, $payload of the audited run ${audited:-unreadable} and of the accepted run $s31_id" \
                    "${accepted:-unreadable}"
            fi
        else
            fail "S3.2 a second reader recovers a timed record (see recovered.log)"
            detail "$run_dir/recovered.log"
        fi
    else
        fail "S3.2 adapter: build or verification failed (see adapter-build.log, adapter-verify.log)"
        detail "$run_dir/adapter-build.log"
        detail "$run_dir/adapter-verify.log"
        note "SKIP  S3.2 native unit cases, integration driver and record recovery: no adapter"
    fi
elif [[ -n "$s31_run" ]]; then
    fail "S3.2 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a build with a receipt"
    note "      Run 'make accept-s1.1' on this host first."
    note "SKIP  S3.2 reference check, adapter build, integration driver and record recovery: no Sprint 1.1 build"
else
    note "SKIP  S3.2 reference check, adapter build, integration driver and record recovery: no control"
fi

# The upstream sources stay at the locked commits with no edited file.
step "S3.2 sources at the locked revisions after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees

note "Evidence: $run_dir"
if (( failed || ! grade_ok )); then
    note "Sprint 3.2: NOT ACCEPTED on this host"
    exit 1
fi
if (( rehearsal )); then
    note "Sprint 3.2: REHEARSED on this host, NOT ACCEPTED. The observations of the rehearsed Sprint 3.1 run $s31_id" \
        "were published on the ring in both evidence modes and the host path decodes were unchanged. The CUDA" \
        "decodes are open: run 'make accept-s3.2' on the DGX Spark."
    exit 0
fi
note "Sprint 3.2: ACCEPTED on this $profile. Every observation leaves the receiver as one obs/v1 record on the ring" \
    "obs-ring/v1, first in, first out, or is dropped and counted. The timed path computes no checksum and makes no" \
    "system call in a publication, and its estimates and its decodes are those of the audited path and of the" \
    "accepted Sprint 3.1 run $s31_id. Next: S3.3 radar cube, features and known return."
