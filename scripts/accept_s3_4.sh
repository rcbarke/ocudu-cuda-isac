#!/usr/bin/env bash
# Run every Sprint 3.4 gate (presence_head) on this host and keep the evidence in a new run directory.
# The gate trains one linear presence head on the delay-Doppler features of Sprint 3.3 and proves
# its artifact. The tapped receiver replay publishes the training and the validation entries of an
# episode set, a native consumer keeps every record, and the causal stream gives one window per
# realization. scripts/train_presence.py fits the normalizer and the weights on the training
# windows, sets the threshold on the target-absent validation windows and freezes one artifact of
# two files. Only then are the held-out entries replayed, with the presence sensor on the ring in
# a second process on the CUDA device, and the records it kept are scored again offline on the CPU:
# the same features, the same input digests and the same scores within the declared tolerance.
# This sprint changes no native source: the adapter is rebuilt and must have the bytes of the
# accepted Sprint 3.3 run. It never rebuilds OCUDU, never traces an episode and never edits the
# sources, a build, an episode set or an earlier run. Every number of the head carries the runtime
# reference, and neither a latency nor the quality on the held-out episodes is graded here.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s3_4.sh [--run-dir DIR] [--s33-run DIR] [--episode-set DIR] [--host-rehearsal]"
    echo "Run it on a host that passed 'make accept-s3.3': it needs the accepted OCUDU build behind isac/bin,"
    echo "NumPy and PyTorch in isac ('make env-full'), the accepted Sprint 3.3 run (default: the newest one)"
    echo "and an episode set under data/episodes (default: the newest demo-set-*; 'make episode-set' traces"
    echo "one). The plans of that run name files of the Sprint 2 runs by absolute path. All of them must still"
    echo "be where they were."
    echo "--host-rehearsal grades the host path on a build without CUDA against a rehearsed Sprint 3.3"
    echo "run and scores on the CPU. A rehearsal is never an acceptance."
}
run_dir=""
s33_arg=""
set_arg=""
rehearsal=0
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --s33-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s33_arg=$2; shift 2 ;;
        --episode-set)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            set_arg=$2; shift 2 ;;
        --host-rehearsal) rehearsal=1; shift ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s3_4: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s34-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s3_4: $run_dir already exists. Evidence is never overwritten." >&2
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
        fail "$label ($ran ran, $skipped skipped: the head needs NumPy and PyTorch in isac, 'make env-full', and its" \
            "live case cmake and a C++17 compiler; see $log)"
    else
        pass "$label ($ran ran, $skipped skipped)"
    fi
}
# verdict RUN: true when RUN holds the verdict of Sprint 3.3 that this gate needs, without a failed line.
verdict() {
    local wanted='^Sprint 3\.3: ACCEPTED on this '
    if (( rehearsal )); then wanted='^Sprint 3\.3: REHEARSED on this host, NOT ACCEPTED'; fi
    grep -qE "$wanted" "$1/summary.txt" 2>/dev/null && ! grep -qE '^FAIL  ' "$1/summary.txt"
}
# The native tools must be found the way an activated shell finds them.
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 3.4 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if (( rehearsal )); then
    note "HOST REHEARSAL: the host path is graded on a build without CUDA and the head scores on the CPU. This run" \
        "cannot accept the sprint."
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
# rule of the sensing package, which now covers the head. The unit gate needs NumPy and PyTorch
# and attaches the presence sensor to a native producer of the ring, so none of its tests may be
# skipped here.
tests any "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
tests complete "S3.4 unit gate, run directly: python tests/unit/test_presence_head.py" presence-head-unit.log \
    "$python" tests/unit/test_presence_head.py -v
tests complete "S3.4 import rule, run directly: python tests/unit/test_core_imports.py" core-imports.log \
    "$python" tests/unit/test_core_imports.py -v

# The control is the run that Sprint 3.3 accepted: the newest one, or the one named. Its grade
# names the feature specification, and the plan of its live run names the replay of the pair,
# whose entry binds the transmit grid of every grant.
s33_run=""
s33_id=""
if [[ -n "$s33_arg" ]]; then
    candidates=("$s33_arg")
else
    candidates=(results/runs/s33-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if verdict "$candidate" && [[ -f "$candidate/pilot_features/grade.json" \
            && -f "$candidate/pilot_features/live/run/plan.json" && -f "$candidate/adapter-receipt.json" ]]; then
        s33_run=$candidate
    fi
done
profile="host"
kind="ACCEPTED"
if (( rehearsal )); then kind="REHEARSED"; fi
if [[ -n "$s33_run" ]]; then
    s33_run=$(cd -- "$s33_run" && pwd)
    s33_id=$(basename -- "$s33_run")
    if grep -qE '^Sprint 3\.3: ACCEPTED on this DGX Spark' "$s33_run/summary.txt"; then profile="DGX Spark"; fi
    pass "S3.4 control: the Sprint 3.3 run $s33_id is $kind and holds its grade and the plan of its live run" \
        "(pilot_features/)"
else
    fail "S3.4 control: no Sprint 3.3 $kind verdict with its grade and the plan of its live run in" \
        "${s33_arg:-results/runs/s33-*}"
    if (( rehearsal )); then
        note "      Run 'make accept-s3.3 REHEARSAL=1' on this host first, or name its run with --s33-run."
    else
        note "      Run 'make accept-s3.3' on this host first, or name its run with --s33-run."
    fi
fi

# The stimulus is an episode set that the exporter of Sprint 2.2 wrote: the newest demo set under
# data/episodes, or the one named. This gate reads it and never traces one.
episode_set=""
if [[ -n "$set_arg" ]]; then
    candidates=("$set_arg")
else
    candidates=(data/episodes/demo-set-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if [[ -f "$candidate/manifest.json" && -d "$candidate/episodes" ]]; then
        episode_set=$candidate
    fi
done
if [[ -n "$episode_set" ]]; then
    episode_set=$(cd -- "$episode_set" && pwd)
    pass "S3.4 stimulus: the episode set $(basename -- "$episode_set") holds a manifest and its episodes ($episode_set)"
else
    fail "S3.4 stimulus: no episode set with a manifest in ${set_arg:-data/episodes/demo-set-*}"
    note "      Trace the demonstration set once with 'make episode-set' (about 7 GB, CPU only), or name a set" \
        "with --episode-set."
fi

# The receiver must be the pinned one: the OCUDU build that Sprint 1.1 published into the
# environment, which the adapter is compiled and linked against.
suffix=tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test
target=$(readlink -f -- "$venv_bin/pusch_e2e_pipeline_test" 2>/dev/null)
build_dir=${target%"/$suffix"}
grade_ok=0
if [[ -n "$s33_run" && -n "$episode_set" && -L "$venv_bin/pusch_e2e_pipeline_test" && -n "$target" \
        && "$build_dir" != "$target" && -f "$build_dir/isac-native.json" ]]; then
    step "S3.4 references: the Sprint 1.1 build, from the project interpreter (environment.json)" \
        environment.log "$python" scripts/verify_environment.py --profile core --with-native \
        --report "$run_dir/environment.json"

    # The adapter is built out of tree as in Sprint 3.3. The gate runs its pilot_ls and its obs_tail.
    # The build itself starts no CUDA work.
    if "$python" scripts/native_adapter.py build --ocudu-build-dir "$build_dir" \
            --run-dir "$run_dir/adapter-build" > "$run_dir/adapter-build.log" 2>&1 \
            && "$python" scripts/native_adapter.py verify > "$run_dir/adapter-receipt.json" 2> "$run_dir/adapter-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
names = receipt["executables"]
print("%s|pilot_ls %s, obs_tail %s, %s, %d adapter tests, %d compiler warnings"
      % ("cuda" if receipt["pin"]["cuda"] else "host", names["pilot_ls"]["sha256"][:12],
         names["obs_tail"]["sha256"][:12], "with ENABLE_CUDA" if receipt["pin"]["cuda"] else "without CUDA",
         receipt["tests_run"], receipt["compiler_warnings"]))' "$run_dir/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        built=${facts%%|*}
        if [[ -n "$facts" ]]; then
            pass "S3.4 adapter: built against the Sprint 1.1 build and verified, ${facts#*|} (adapter-receipt.json)"
        else
            fail "S3.4 adapter: the receipt names no pilot_ls and obs_tail (see adapter-verify.log)"
            detail "$run_dir/adapter-verify.log"
        fi
        # This sprint adds Python on the sensing side and changes no native source, so every tool
        # must have the bytes that the accepted Sprint 3.3 run was measured with.
        same=$("$python" -c 'import json, sys
now = json.load(open(sys.argv[1], encoding="utf-8"))["executables"]
then = json.load(open(sys.argv[2], encoding="utf-8"))["executables"]
changed = sorted(name for name in then if name not in now or now[name]["sha256"] != then[name]["sha256"])
print("%s|%s" % ("changed" if changed or not then else "same",
                 ", ".join("%s is %s and was %s" % (name, now.get(name, {}).get("sha256", "missing")[:12],
                                                    then[name]["sha256"][:12]) for name in changed)
                 or ", ".join("%s %s" % (name, then[name]["sha256"][:12]) for name in sorted(then))))' \
            "$run_dir/adapter-receipt.json" "$s33_run/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        if [[ "${same%%|*}" == same ]]; then
            pass "S3.4 accepted executables: this sprint changes no native source, and every tool has the bytes of" \
                "the Sprint 3.3 run: ${same#*|}"
        else
            fail "S3.4 accepted executables: this sprint changes no native source, so every tool must keep the bytes" \
                "of the Sprint 3.3 run: ${same#*|} (adapter-receipt.json against the receipt of $s33_id)"
        fi
        # The hardware gate needs the CUDA processor of the receiver and a CUDA device for the head.
        # A build or an interpreter without one can only be rehearsed.
        driver_mode=()
        device=""
        if "$python" -m ocudu_cuda_isac.sensing.presence device > "$run_dir/device.json" 2> "$run_dir/device.log"; then
            device=$("$python" -c 'import json, sys
facts = json.load(open(sys.argv[1], encoding="utf-8"))
print("%s|PyTorch %s%s" % (facts["type"], facts["torch"],
                           " reaches %s (%s, capability %s, CUDA runtime %s)"
                           % (facts["device"], facts.get("name"), facts.get("capability"), facts["cuda_runtime"])
                           if facts["type"] == "cuda" else " reaches no CUDA device in isac"))' \
                "$run_dir/device.json" 2>> "$run_dir/device.log")
        fi
        device=${device:-"none|PyTorch does not load in isac"}
        if (( rehearsal )); then
            driver_mode=(--host-rehearsal)
            note "NOTE  S3.4 hardware: host rehearsal on an adapter built for the ${built:-unknown} path; ${device#*|}." \
                "The CUDA decodes and the device of the head stay open."
        else
            if [[ "$built" == cuda ]]; then
                pass "S3.4 hardware: the adapter is linked against the CUDA build, so the CUDA processor can execute"
            elif [[ -n "$facts" ]]; then
                fail "S3.4 hardware: the accepted build has no CUDA, so the CUDA processor cannot execute here. A host" \
                    "path decode cannot close this gate. Rehearse with --host-rehearsal, accept on the DGX Spark."
            fi
            if [[ "${device%%|*}" == cuda ]]; then
                pass "S3.4 device: ${device#*|}: the head is scored in a second CUDA process (device.json)"
            else
                fail "S3.4 device: ${device#*|}, so the head cannot be scored on the selected device (see device.log)." \
                    "Install the full profile with 'make env-full', or rehearse with --host-rehearsal."
                detail "$run_dir/device.log"
            fi
        fi

        # Integration driver. It grades its own evidence, and its lines are copied here.
        note "Presence head on the episode set $(basename -- "$episode_set"), on the control $s33_id, with" \
            "configs/integration/presence_head.json."
        "$python" tests/integration/presence_head.py --config configs/integration/presence_head.json \
            --s33-run "$s33_run" --episode-set "$episode_set" ${driver_mode[@]+"${driver_mode[@]}"} \
            --output-dir "$run_dir/presence_head" > "$run_dir/presence-head.log" 2>&1
        status=$?
        grade="$run_dir/presence_head/grade.txt"
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
sys.exit(0 if ok else 1)' "$run_dir/presence_head/grade.json" "$wanted" 2> "$run_dir/grade-read.log")
        counted=$?
        if (( status == 0 && counted == 0 && graded > 0 )); then
            grade_ok=1
            pass "S3.4 integration grade: $counts (presence_head/grade.json)"
        elif (( graded == 0 )); then
            fail "S3.4 integration driver produced no graded checks (see presence-head.log)"
            detail "$run_dir/presence-head.log"
        else
            fail "S3.4 integration grade: ${counts:-unreadable}, driver exit status $status (presence_head/grade.json)"
        fi

        # A second reader, with tools that are not the driver's: the command-line reader on the
        # frozen artifact and on what the sensor wrote. The freeze record names the hash of the
        # artifact and of its weights. The reader must find that artifact, the output of the
        # sensor must name it, and sha256sum must give the hash of the weights file.
        out="$run_dir/presence_head"
        place=$("$python" -c 'import json, sys
found = json.load(open(sys.argv[1], encoding="utf-8"))
freeze = json.load(open(sys.argv[2] + "/" + found["freeze"], encoding="utf-8"))
print(found["artifact"])
print(found["sensor"])
print(found["window"])
print(freeze["artifact"]["sha256"])
print(freeze["artifact"]["weights_sha256"])
print(freeze["frozen_utc"])' "$out/recover.json" "$out" 2> "$run_dir/recovered.log")
        artifact=$(echo "$place" | sed -n 1p)
        sensor=$(echo "$place" | sed -n 2p)
        window=$(echo "$place" | sed -n 3p)
        frozen=$(echo "$place" | sed -n 4p)
        frozen_weights=$(echo "$place" | sed -n 5p)
        frozen_at=$(echo "$place" | sed -n 6p)
        if [[ -n "$artifact" && -n "$sensor" && -n "$frozen" ]] \
                && "$python" -m ocudu_cuda_isac.sensing.presence inspect "$out/$artifact" --sha256 "$frozen" \
                    > "$run_dir/recovered-artifact.json" 2>> "$run_dir/recovered.log" \
                && "$python" -m ocudu_cuda_isac.sensing.presence show "$out/$sensor" --window "$window" \
                    > "$run_dir/recovered.json" 2>> "$run_dir/recovered.log"; then
            shown=$("$python" -c 'import json, sys
head = json.load(open(sys.argv[1], encoding="utf-8"))
shown = json.load(open(sys.argv[2], encoding="utf-8"))
window, threshold = shown["window"], head["threshold"]
print(shown["artifact_sha256"])
print(head["weights_sha256"])
print("the head %s reads %d of %d values, %s, from the files %s; its threshold is the logit %.4f, from %d "
      "target-absent validation windows with %d of %d allowed above it; the sensor loaded it %d time(s) on %s in the "
      "runtime %s and scored %d windows: window %s from %d observations has the logit %.4f and the probability %.4f, "
      "%s the threshold"
      % (head["head_id"], len(head["columns"]), head["vector"], ", ".join(head["columns"]), " and ".join(head["files"]),
         threshold["logit"], threshold["negatives"], threshold["exceeding"], threshold["allowed"], shown["loads"],
         shown["device"], shown["runtime"], shown["scored_windows"], window["id"], window["contributing"],
         window["logit"], window["probability"], "above" if window["present"] else "at or below"))' \
                "$run_dir/recovered-artifact.json" "$run_dir/recovered.json" 2>> "$run_dir/recovered.log")
            used=$(echo "$shown" | sed -n 1p)
            named=$(echo "$shown" | sed -n 2p)
            text=$(echo "$shown" | sed -n 3p)
            weights=$(sha256sum "$out/$artifact/weights.f32" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            if [[ -n "$text" && "$used" == "$frozen" && "$weights" == "$frozen_weights" && "$named" == "$frozen_weights" ]]; then
                pass "S3.4 a second reader recovers the frozen head and a held-out decision: $text; the artifact" \
                    "${frozen:0:16} is the one the freeze record of $frozen_at names, and its weights have the SHA-256" \
                    "${weights:0:16} by sha256sum (recovered-artifact.json, recovered.json)"
            else
                fail "S3.4 a second reader recovers the frozen head and a held-out decision: the freeze record names" \
                    "the artifact ${frozen:-none} with the weights ${frozen_weights:-none}, the sensor scored with" \
                    "${used:-none}, and the weights file has ${weights:-unreadable} by sha256sum"
            fi
        else
            fail "S3.4 a second reader recovers the frozen head and a held-out decision (see recovered.log)"
            detail "$run_dir/recovered.log"
        fi
    else
        fail "S3.4 adapter: build or verification failed (see adapter-build.log, adapter-verify.log)"
        detail "$run_dir/adapter-build.log"
        detail "$run_dir/adapter-verify.log"
        note "SKIP  S3.4 integration driver and artifact recovery: no adapter"
    fi
elif [[ -n "$s33_run" && -n "$episode_set" ]]; then
    fail "S3.4 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a build with a receipt"
    note "      Run 'make accept-s1.1' on this host first."
    note "SKIP  S3.4 reference check, adapter build, integration driver and artifact recovery: no Sprint 1.1 build"
else
    note "SKIP  S3.4 reference check, adapter build, integration driver and artifact recovery: no control or no" \
        "episode set"
fi

# The upstream sources stay at the locked commits with no edited file.
step "S3.4 sources at the locked revisions after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees

note "Evidence: $run_dir"
if (( failed || ! grade_ok )); then
    note "Sprint 3.4: NOT ACCEPTED on this host"
    exit 1
fi
if (( rehearsal )); then
    note "Sprint 3.4: REHEARSED on this host, NOT ACCEPTED. One head was trained on the records of the rehearsed" \
        "chain behind $s33_id, frozen, and scored on held-out records in a second process and offline, on the CPU." \
        "The CUDA decodes and the CUDA device of the head are open: run 'make accept-s3.4' on the DGX Spark."
    exit 0
fi
note "Sprint 3.4: ACCEPTED on this $profile. One linear presence head on the delay-Doppler features was trained on" \
    "receiver-derived records of the training trajectories, its threshold was set on target-absent validation" \
    "windows, and its artifact of two files was frozen before a held-out entry was replayed. The same artifact scores" \
    "the same held-out feature in a second CUDA process and offline within the declared tolerance. Quality and" \
    "timing on the held-out episodes are recorded here and graded in Sprint 4. Next: S5.2 native estimate."
