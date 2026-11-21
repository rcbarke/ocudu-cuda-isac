#!/usr/bin/env bash
# Run every Sprint 3.3 gate (pilot_features) on this host and keep the evidence in a new run directory.
# The gate computes the radar cube, the delay-Doppler view and the causal features of the reference
# pair with the reference of the sensing core: from the records that the consumer of the accepted
# Sprint 3.2 run kept from the ring, from the files that its audited observer wrote for the same
# estimates, and from a ring that the tapped receiver replay fills once more while the reference
# sensor follows it in a second process. The three must agree, the return of the target-present
# member must lie within one resolvable bin of the delay and the Doppler shift that the geometry
# of the episode gives, and the decodes with the sensor attached must be those of the control.
# This sprint changes no native source: the adapter is rebuilt and must have the bytes of the
# accepted Sprint 3.2 run. It never rebuilds OCUDU and never edits the sources, a build or an
# earlier run. Every feature is computed in NumPy on the CPU with the runtime reference, and no
# latency of this gate is a result.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s3_3.sh [--run-dir DIR] [--s32-run DIR] [--host-rehearsal]"
    echo "Run it on a host that passed 'make accept-s3.2': it needs the accepted OCUDU build behind isac/bin,"
    echo "NumPy in isac and the accepted Sprint 3.2 run (default: the newest one). That run holds the records"
    echo "of the ring and the files of the audited observer, and its plans name the Sprint 2 runs by absolute"
    echo "path. All of them must still be where they were."
    echo "--host-rehearsal grades the host path on a build without CUDA against a rehearsed Sprint 3.2"
    echo "run. A rehearsal is never an acceptance."
}
run_dir=""
s32_arg=""
rehearsal=0
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --s32-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s32_arg=$2; shift 2 ;;
        --host-rehearsal) rehearsal=1; shift ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s3_3: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s33-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s3_3: $run_dir already exists. Evidence is never overwritten." >&2
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
        fail "$label ($ran ran, $skipped skipped: the reference needs NumPy in isac, and its live case cmake and a" \
            "C++17 compiler; see $log)"
    else
        pass "$label ($ran ran, $skipped skipped)"
    fi
}
# verdict RUN: true when RUN holds the verdict of Sprint 3.2 that this gate needs, without a failed line.
verdict() {
    local wanted='^Sprint 3\.2: ACCEPTED on this '
    if (( rehearsal )); then wanted='^Sprint 3\.2: REHEARSED on this host, NOT ACCEPTED'; fi
    grep -qE "$wanted" "$1/summary.txt" 2>/dev/null && ! grep -qE '^FAIL  ' "$1/summary.txt"
}
# The native tools must be found the way an activated shell finds them.
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 3.3 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

# The portable suite, then the card's unit command exactly as the plan names it, then the import
# rule that holds from this sprint on. The unit gate needs NumPy for its vectors and attaches the
# reference sensor to a native producer of the ring, so none of its tests may be skipped here.
tests any "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
tests complete "S3.3 unit gate, run directly: python tests/unit/test_pilot_features.py" pilot-features-unit.log \
    "$python" tests/unit/test_pilot_features.py -v
tests complete "S3.3 import rule, run directly: python tests/unit/test_core_imports.py" core-imports.log \
    "$python" tests/unit/test_core_imports.py -v

# The control is the run that Sprint 3.2 accepted: the newest one, or the one named. It holds the
# records that its native consumer kept from the ring in the mode timed and the files of its
# audited observer.
s32_run=""
s32_id=""
if [[ -n "$s32_arg" ]]; then
    candidates=("$s32_arg")
else
    candidates=(results/runs/s32-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if verdict "$candidate" && [[ -f "$candidate/snapshot_ring/grade.json" \
            && -f "$candidate/snapshot_ring/timed/run/consumer-0/consumer.json" \
            && -f "$candidate/snapshot_ring/audit/run/native/observer/o2.jsonl" ]]; then
        s32_run=$candidate
    fi
done
profile="host"
kind="ACCEPTED"
if (( rehearsal )); then kind="REHEARSED"; fi
if [[ -n "$s32_run" ]]; then
    s32_run=$(cd -- "$s32_run" && pwd)
    s32_id=$(basename -- "$s32_run")
    if grep -qE '^Sprint 3\.2: ACCEPTED on this DGX Spark' "$s32_run/summary.txt"; then profile="DGX Spark"; fi
    pass "S3.3 control: the Sprint 3.2 run $s32_id is $kind and holds the records its consumer kept from the ring" \
        "and the files of its audited observer (snapshot_ring/)"
else
    fail "S3.3 control: no Sprint 3.2 $kind verdict with the records of the ring and the files of the audited" \
        "observer in ${s32_arg:-results/runs/s32-*}"
    if (( rehearsal )); then
        note "      Run 'make accept-s3.2 REHEARSAL=1' on this host first, or name its run with --s32-run."
    else
        note "      Run 'make accept-s3.2' on this host first, or name its run with --s32-run."
    fi
fi

# The receiver must be the pinned one: the OCUDU build that Sprint 1.1 published into the
# environment, which the adapter is compiled and linked against.
suffix=tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test
target=$(readlink -f -- "$venv_bin/pusch_e2e_pipeline_test" 2>/dev/null)
build_dir=${target%"/$suffix"}
grade_ok=0
if [[ -n "$s32_run" && -L "$venv_bin/pusch_e2e_pipeline_test" && -n "$target" && "$build_dir" != "$target" \
        && -f "$build_dir/isac-native.json" ]]; then
    step "S3.3 references: the Sprint 1.1 build, from the project interpreter (environment.json)" \
        environment.log "$python" scripts/verify_environment.py --profile core --with-native \
        --report "$run_dir/environment.json"

    # The adapter is built out of tree as in Sprint 3.2. The live case runs its pilot_ls. The build
    # itself starts no CUDA work.
    if "$python" scripts/native_adapter.py build --ocudu-build-dir "$build_dir" \
            --run-dir "$run_dir/adapter-build" > "$run_dir/adapter-build.log" 2>&1 \
            && "$python" scripts/native_adapter.py verify > "$run_dir/adapter-receipt.json" 2> "$run_dir/adapter-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
names = receipt["executables"]
print("%s|pilot_ls %s, %s, %d adapter tests, %d compiler warnings"
      % ("cuda" if receipt["pin"]["cuda"] else "host", names["pilot_ls"]["sha256"][:12],
         "with ENABLE_CUDA" if receipt["pin"]["cuda"] else "without CUDA", receipt["tests_run"],
         receipt["compiler_warnings"]))' "$run_dir/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        built=${facts%%|*}
        if [[ -n "$facts" ]]; then
            pass "S3.3 adapter: built against the Sprint 1.1 build and verified, ${facts#*|} (adapter-receipt.json)"
        else
            fail "S3.3 adapter: the receipt names no pilot_ls (see adapter-verify.log)"
            detail "$run_dir/adapter-verify.log"
        fi
        # This sprint adds Python on the sensing side and changes no native source, so every tool
        # must have the bytes that the accepted Sprint 3.2 run was measured with.
        same=$("$python" -c 'import json, sys
now = json.load(open(sys.argv[1], encoding="utf-8"))["executables"]
then = json.load(open(sys.argv[2], encoding="utf-8"))["executables"]
changed = sorted(name for name in then if name not in now or now[name]["sha256"] != then[name]["sha256"])
print("%s|%s" % ("changed" if changed or not then else "same",
                 ", ".join("%s is %s and was %s" % (name, now.get(name, {}).get("sha256", "missing")[:12],
                                                    then[name]["sha256"][:12]) for name in changed)
                 or ", ".join("%s %s" % (name, then[name]["sha256"][:12]) for name in sorted(then))))' \
            "$run_dir/adapter-receipt.json" "$s32_run/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        if [[ "${same%%|*}" == same ]]; then
            pass "S3.3 accepted executables: this sprint changes no native source, and every tool has the bytes of" \
                "the Sprint 3.2 run: ${same#*|}"
        else
            fail "S3.3 accepted executables: this sprint changes no native source, so every tool must keep the bytes" \
                "of the Sprint 3.2 run: ${same#*|} (adapter-receipt.json against the receipt of $s32_id)"
        fi
        # The hardware gate needs the CUDA processor. A build without it can only be rehearsed.
        driver_mode=()
        if (( rehearsal )); then
            driver_mode=(--host-rehearsal)
            note "NOTE  S3.3 hardware: host rehearsal on an adapter built for the ${built:-unknown} path. The CUDA" \
                "decodes of the live case stay open."
        elif [[ "$built" == cuda ]]; then
            pass "S3.3 hardware: the adapter is linked against the CUDA build, so the CUDA processor can execute"
        elif [[ -n "$facts" ]]; then
            fail "S3.3 hardware: the accepted build has no CUDA, so the CUDA processor cannot execute here. A host" \
                "path decode cannot close this gate. Rehearse with --host-rehearsal, accept on the DGX Spark."
        fi

        # Integration driver. It grades its own evidence, and its lines are copied here.
        note "Features of the reference pair of $s32_id with configs/integration/pilot_features.json."
        "$python" tests/integration/pilot_features.py --config configs/integration/pilot_features.json \
            --s32-run "$s32_run" ${driver_mode[@]+"${driver_mode[@]}"} --output-dir "$run_dir/pilot_features" \
            > "$run_dir/pilot-features.log" 2>&1
        status=$?
        grade="$run_dir/pilot_features/grade.txt"
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
sys.exit(0 if ok else 1)' "$run_dir/pilot_features/grade.json" "$wanted" 2> "$run_dir/grade-read.log")
        counted=$?
        if (( status == 0 && counted == 0 && graded > 0 )); then
            grade_ok=1
            pass "S3.3 integration grade: $counts (pilot_features/grade.json)"
        elif (( graded == 0 )); then
            fail "S3.3 integration driver produced no graded checks (see pilot-features.log)"
            detail "$run_dir/pilot-features.log"
        else
            fail "S3.3 integration grade: ${counts:-unreadable}, driver exit status $status (pilot_features/grade.json)"
        fi

        # A second reader, with tools that are not the driver's: the command-line reader on what the
        # live sensor wrote. It prints the target-present window. sha256sum takes the digest of the
        # view file of that window and of the view that the recorded stream of the accepted run
        # gave for the same window. Both must be the digest that the line of the sensor names.
        out="$run_dir/pilot_features"
        place=$("$python" -c 'import json, sys
found = json.load(open(sys.argv[1], encoding="utf-8"))
print(found["sensor"])
print(found["window"])
print(found["stream_view"])' "$out/recover.json" 2> "$run_dir/recovered.log")
        sensor=$(echo "$place" | sed -n 1p)
        window=$(echo "$place" | sed -n 2p)
        stream_view=$(echo "$place" | sed -n 3p)
        if [[ -n "$sensor" && -n "$window" ]] && "$python" -m ocudu_cuda_isac.sensing.features show "$out/$sensor" \
                --window "$window" > "$run_dir/recovered.json" 2>> "$run_dir/recovered.log"; then
            shown=$("$python" -c 'import json, sys
shown = json.load(open(sys.argv[1], encoding="utf-8"))
window, values = shown["window"], shown["window"]["features"]
print(window["view_file"])
print(window["view_sha256"])
print("%d complete windows from a %s in the runtime %s; window %s holds %d observations over %.2f ms, with %s; its "
      "strongest moving cell lies at delay bin %d and Doppler bin %d, %.1f ns and %+.1f Hz, %.1f dB above the median, "
      "with bins of %.2f ns and %.2f Hz"
      % (shown["complete_windows"], shown["source"], shown["runtime"], window["id"], window["contributing"],
         window["acquisition_us"] / 1000.0, ", ".join("%s %s" % item for item in sorted(shown["steps"].items())),
         window["peak"]["delay_bin"], window["peak"]["doppler_bin"], window["peak"]["delay_ns"],
         window["peak"]["doppler_hz"], values["peak_to_median_db"], window["delay_resolution_ns"],
         window["doppler_resolution_hz"]))' "$run_dir/recovered.json" 2>> "$run_dir/recovered.log")
            file=$(echo "$shown" | sed -n 1p)
            named=$(echo "$shown" | sed -n 2p)
            text=$(echo "$shown" | sed -n 3p)
            live=$(sha256sum "$out/$sensor/$file" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            recorded=$(sha256sum "$out/$stream_view" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            if [[ -n "$file" && -n "$named" && "$live" == "$named" && "$recorded" == "$named" ]]; then
                pass "S3.3 a second reader recovers the target-present window of the live sensor: $text; SHA-256 of its" \
                    "view ${live:0:16} by sha256sum, the digest its line names and that of the same window from the" \
                    "records of the accepted run $s32_id (recovered.json)"
            else
                fail "S3.3 a second reader recovers the target-present window of the live sensor: its view has the" \
                    "digest ${live:-unreadable} by sha256sum, its line names ${named:-none}, and the view of the" \
                    "recorded stream has ${recorded:-unreadable}"
            fi
        else
            fail "S3.3 a second reader recovers the target-present window of the live sensor (see recovered.log)"
            detail "$run_dir/recovered.log"
        fi
    else
        fail "S3.3 adapter: build or verification failed (see adapter-build.log, adapter-verify.log)"
        detail "$run_dir/adapter-build.log"
        detail "$run_dir/adapter-verify.log"
        note "SKIP  S3.3 integration driver and window recovery: no adapter"
    fi
elif [[ -n "$s32_run" ]]; then
    fail "S3.3 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a build with a receipt"
    note "      Run 'make accept-s1.1' on this host first."
    note "SKIP  S3.3 reference check, adapter build, integration driver and window recovery: no Sprint 1.1 build"
else
    note "SKIP  S3.3 reference check, adapter build, integration driver and window recovery: no control"
fi

# The upstream sources stay at the locked commits with no edited file.
step "S3.3 sources at the locked revisions after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees

note "Evidence: $run_dir"
if (( failed || ! grade_ok )); then
    note "Sprint 3.3: NOT ACCEPTED on this host"
    exit 1
fi
if (( rehearsal )); then
    note "Sprint 3.3: REHEARSED on this host, NOT ACCEPTED. The features of the rehearsed Sprint 3.2 run $s32_id" \
        "agree from the recorded ring, from the files of the observer and from a live ring, and the host path" \
        "decodes were unchanged. The CUDA decodes are open: run 'make accept-s3.3' on the DGX Spark."
    exit 0
fi
note "Sprint 3.3: ACCEPTED on this $profile. The reference of the sensing core forms the radar cube, the" \
    "delay-Doppler view and the causal features from the records of the ring, one record at a time. The known" \
    "return of the reference pair lies within one resolvable bin of its geometry, and the features agree from the" \
    "ring of the accepted Sprint 3.2 run $s32_id, from the files of its observer and from a live ring. Next: S3.4" \
    "presence head."
