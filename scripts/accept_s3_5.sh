#!/usr/bin/env bash
# Run every Sprint 3.5 gate (cusense_frontend) on this host and keep the evidence in a new run directory.
# The gate runs the cuSense-like branch on the receiver's own estimate, the stage O3 of Sprint 5.2,
# with the pilot comb of the stage O2 as the second input entry. The tapped receiver replay
# publishes both stages of the training and of the validation entries of an episode set, each on
# a ring of its own, and a native consumer on each ring keeps every record. The window stream cuts
# one window per TDD period. scripts/train_cusense.py fits the background template and the
# statistics of the z-score on the training windows, trains the location map and the presence
# logit, sets the threshold on the target-absent validation windows and freezes one artifact of
# four files per input entry. Only then are the held-out entries replayed, with the sensor on the
# O3 ring in a second process on the CUDA device, and the records it kept are scored again offline
# on the CPU. The linear head of the accepted Sprint 3.4 run is scored on the same records.
# This sprint changes no native source: the adapter with the O3 route is rebuilt against the
# patched receiver that Sprint 5.2 built, and must have the bytes of that run. The gate never
# rebuilds OCUDU, never traces an episode and never edits the sources, a build, an episode set or
# an earlier run. Every number of the branch carries the runtime reference. The presence counts
# and the position error on the held-out episodes are recorded beside the published figures and
# are not graded, and the latency comparison belongs to Sprint 3.6.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s3_5.sh [--run-dir DIR] [--s34-run DIR] [--s52-run DIR] [--episode-set DIR]"
    echo "Run it on the DGX Spark that passed 'make accept-s3.4' and 'make accept-s5.2': it needs the patched"
    echo "receiver of Sprint 5.2 in build/ocudu-spark-o3, NumPy and PyTorch in isac ('make env-full'), the"
    echo "accepted Sprint 3.4 run and the accepted Sprint 5.2 run (default: the newest of each) and the episode"
    echo "set that the Sprint 3.4 run was graded on (default: the directory that run names). The plans of the"
    echo "Sprint 3.4 run name files of earlier runs by absolute path. All of them must still be where they were."
    echo "There is no host rehearsal: the receiver's estimate exists only on a CUDA device."
}
run_dir=""
s34_arg=""
s52_arg=""
set_arg=""
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --s34-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s34_arg=$2; shift 2 ;;
        --s52-run)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            s52_arg=$2; shift 2 ;;
        --episode-set)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            set_arg=$2; shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s3_5: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s35-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s3_5: $run_dir already exists. Evidence is never overwritten." >&2
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
        fail "$label ($ran ran, $skipped skipped: the branch needs NumPy and PyTorch in isac, 'make env-full', and its" \
            "live case cmake and a C++17 compiler; see $log)"
    else
        pass "$label ($ran ran, $skipped skipped)"
    fi
}
# accepted RUN SPRINT: true when RUN holds the verdict of that sprint on this DGX Spark, without a failed line.
accepted() {
    grep -qE "^Sprint $2: ACCEPTED on this DGX Spark" "$1/summary.txt" 2>/dev/null && ! grep -qE '^FAIL  ' "$1/summary.txt"
}
# The native tools must be found the way an activated shell finds them.
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 3.5 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
# rule of the sensing package, which now covers the branch and its model entry. The unit gate
# needs NumPy and PyTorch and attaches the sensor to a native producer of the ring, so none of its
# tests may be skipped here.
tests any "unit tests" unit-tests.log env PYTHONPATH=src "$python" -m unittest discover -s tests -v
tests complete "S3.5 unit gate, run directly: python tests/unit/test_cusense_frontend.py" cusense-frontend-unit.log \
    "$python" tests/unit/test_cusense_frontend.py -v
tests complete "S3.5 import rule, run directly: python tests/unit/test_core_imports.py" core-imports.log \
    "$python" tests/unit/test_core_imports.py -v

# The series of the receiver patch: every patch with the hash its manifest names, on the commit the
# lock pins. The adapter of this gate must be linked against a receiver built with exactly it.
if "$python" scripts/receiver_patch.py check > "$run_dir/patch-series.json" 2> "$run_dir/patch-series.log"; then
    facts=$("$python" -c 'import json, sys
series = json.load(open(sys.argv[1], encoding="utf-8"))
print("%s on %s, %d patch file(s), %d files touched"
      % (series["series_sha256"][:16], series["base_commit"][:12], len(series["patches"]), len(series["files"])))' \
        "$run_dir/patch-series.json" 2>> "$run_dir/patch-series.log")
    pass "S3.5 patch series: patches/ocudu/series.json ${facts:-unreadable} (patch-series.json)"
else
    fail "S3.5 patch series: the manifest disagrees with its patch files or with the lock (see patch-series.log)"
    detail "$run_dir/patch-series.log"
fi

# The first control is the run that Sprint 3.4 accepted on this DGX Spark: the newest one, or the
# one named. It holds the frozen linear head, the episode set it was graded on, and the plans whose
# channel plan binds the transmit grid of every grant.
s34_run=""
s34_id=""
if [[ -n "$s34_arg" ]]; then
    candidates=("$s34_arg")
else
    candidates=(results/runs/s34-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if accepted "$candidate" '3\.4' && [[ -f "$candidate/presence_head/grade.json" \
            && -f "$candidate/presence_head/manifest.json" && -f "$candidate/presence_head/spec.json" \
            && -f "$candidate/presence_head/train/artifact/head.json" ]]; then
        s34_run=$candidate
    fi
done
if [[ -n "$s34_run" ]]; then
    s34_run=$(cd -- "$s34_run" && pwd)
    s34_id=$(basename -- "$s34_run")
    pass "S3.5 control: the Sprint 3.4 run $s34_id is ACCEPTED on this DGX Spark and holds its grade, its manifest" \
        "and its frozen head (presence_head/)"
else
    fail "S3.5 control: no Sprint 3.4 verdict ACCEPTED on this DGX Spark with its grade, its manifest and its frozen" \
        "head in ${s34_arg:-results/runs/s34-*}"
    note "      Run 'make accept-s3.4' on this host first, or name its run with --s34-run."
fi

# The second control is the run that Sprint 5.2 accepted: it built the patched receiver and the
# adapter with the O3 route, and its receipt names the bytes of every tool.
s52_run=""
s52_id=""
if [[ -n "$s52_arg" ]]; then
    candidates=("$s52_arg")
else
    candidates=(results/runs/s52-*/)
fi
for candidate in "${candidates[@]}"; do
    candidate=${candidate%/}
    if accepted "$candidate" '5\.2' && [[ -f "$candidate/o3-adapter-receipt.json" ]]; then
        s52_run=$candidate
    fi
done
if [[ -n "$s52_run" ]]; then
    s52_run=$(cd -- "$s52_run" && pwd)
    s52_id=$(basename -- "$s52_run")
    pass "S3.5 estimate: the Sprint 5.2 run $s52_id is ACCEPTED on this DGX Spark and holds the receipt of the adapter" \
        "with the O3 route (o3-adapter-receipt.json)"
else
    fail "S3.5 estimate: no Sprint 5.2 verdict ACCEPTED on this DGX Spark with the receipt of its O3 adapter in" \
        "${s52_arg:-results/runs/s52-*}"
    note "      Run 'make accept-s5.2' on this host first, or name its run with --s52-run."
fi

# The stimulus is the episode set that the Sprint 3.4 run was graded on, so that both gates score
# the same held-out episodes: the directory that run names, or the one named here. The driver
# checks that it is that set by the hashes of its configuration and of its assignment. This gate
# reads the set and never traces one.
episode_set=""
if [[ -n "$set_arg" ]]; then
    candidate=$set_arg
elif [[ -n "$s34_run" ]]; then
    candidate=$("$python" -c 'import json, sys
print((json.load(open(sys.argv[1], encoding="utf-8")).get("episode_set") or {}).get("directory") or "")' \
        "$s34_run/presence_head/manifest.json" 2> "$run_dir/episode-set.log")
else
    candidate=""
fi
candidate=${candidate%/}
if [[ -n "$candidate" && -f "$candidate/manifest.json" && -d "$candidate/episodes" ]]; then
    episode_set=$(cd -- "$candidate" && pwd)
    pass "S3.5 stimulus: the episode set $(basename -- "$episode_set") holds a manifest and its episodes ($episode_set)"
elif [[ -n "$s34_run" || -n "$set_arg" ]]; then
    fail "S3.5 stimulus: no episode set with a manifest in ${candidate:-the manifest of the Sprint 3.4 run}"
    note "      The set that the Sprint 3.4 run was graded on must still be where it was, or be named with" \
        "--episode-set. This gate traces none."
fi

# The patched receiver is the one Sprint 5.2 built into build/ocudu-spark-o3. It is not rebuilt.
grade_ok=0
if [[ -n "$s34_run" && -n "$s52_run" && -n "$episode_set" && -d build/ocudu-spark-o3 ]]; then
    step "S3.5 references: the Sprint 1.1 build, from the project interpreter (environment.json)" \
        environment.log "$python" scripts/verify_environment.py --profile core --with-native \
        --report "$run_dir/environment.json"

    # The adapter with the O3 route is built out of tree against the patched receiver, as in Sprint
    # 5.2. The gate runs its pilot_ls and its obs_tail. The build itself starts no CUDA work.
    if "$python" scripts/native_adapter.py build --o3 --run-dir "$run_dir/adapter-build" \
            > "$run_dir/adapter-build.log" 2>&1 \
            && "$python" scripts/native_adapter.py verify --o3 > "$run_dir/adapter-receipt.json" \
                2> "$run_dir/adapter-verify.log"; then
        facts=$("$python" -c 'import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
names, route, patch = receipt["executables"], receipt.get("o3_route") or {}, receipt.get("patch") or {}
if not receipt["pin"]["cuda"] or route.get("compiled") is not True or not patch.get("series_sha256"):
    sys.exit("the receipt names no route, no patch or no CUDA")
print("pilot_ls %s with the route, tap interface version %s, obs_tail %s, series %s, with ENABLE_CUDA, %d adapter "
      "tests, %d compiler warnings"
      % (names["pilot_ls"]["sha256"][:12], route.get("tap_interface_version"), names["obs_tail"]["sha256"][:12],
         patch["series_sha256"][:16], receipt["tests_run"], receipt["compiler_warnings"]))' \
            "$run_dir/adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        if [[ -n "$facts" ]]; then
            pass "S3.5 adapter: built against the patched receiver of Sprint 5.2 and verified, $facts" \
                "(adapter-receipt.json)"
            pass "S3.5 hardware: the adapter is linked against the CUDA build of the patched receiver, so its resident" \
                "path can execute and leave an estimate"
        else
            fail "S3.5 adapter: the receipt names no route to the receiver's estimate, no patch, no CUDA or no pilot_ls" \
                "and obs_tail (see adapter-verify.log)"
            detail "$run_dir/adapter-verify.log"
        fi
        # This sprint adds Python on the sensing side and changes no native source, so every tool
        # must have the bytes that the accepted Sprint 5.2 run was measured with.
        same=$("$python" -c 'import json, sys
now = json.load(open(sys.argv[1], encoding="utf-8"))["executables"]
then = json.load(open(sys.argv[2], encoding="utf-8"))["executables"]
changed = sorted(name for name in then if name not in now or now[name]["sha256"] != then[name]["sha256"])
print("%s|%s" % ("changed" if changed or not then else "same",
                 ", ".join("%s is %s and was %s" % (name, now.get(name, {}).get("sha256", "missing")[:12],
                                                    then[name]["sha256"][:12]) for name in changed)
                 or ", ".join("%s %s" % (name, then[name]["sha256"][:12]) for name in sorted(then))))' \
            "$run_dir/adapter-receipt.json" "$s52_run/o3-adapter-receipt.json" 2>> "$run_dir/adapter-verify.log")
        if [[ "${same%%|*}" == same ]]; then
            pass "S3.5 accepted executables: this sprint changes no native source, and every tool has the bytes of" \
                "the Sprint 5.2 run: ${same#*|}"
        else
            fail "S3.5 accepted executables: this sprint changes no native source, so every tool must keep the bytes" \
                "of the Sprint 5.2 run: ${same#*|} (adapter-receipt.json against the O3 receipt of $s52_id)"
        fi
        # The model is scored in a second CUDA process: the interpreter of the sensor must reach a device.
        device=""
        if "$python" -m ocudu_cuda_isac.sensing.cusense device > "$run_dir/device.json" 2> "$run_dir/device.log"; then
            device=$("$python" -c 'import json, sys
facts = json.load(open(sys.argv[1], encoding="utf-8"))
print("%s|PyTorch %s%s" % (facts["type"], facts["torch"],
                           " reaches %s (%s, capability %s, CUDA runtime %s)"
                           % (facts["device"], facts.get("name"), facts.get("capability"), facts["cuda_runtime"])
                           if facts["type"] == "cuda" else " reaches no CUDA device in isac"))' \
                "$run_dir/device.json" 2>> "$run_dir/device.log")
        fi
        device=${device:-"none|PyTorch does not load in isac"}
        if [[ "${device%%|*}" == cuda ]]; then
            pass "S3.5 device: ${device#*|}: the model is trained there and scored in a second CUDA process (device.json)"
        else
            fail "S3.5 device: ${device#*|}, so the model cannot be scored on the selected device (see device.log)." \
                "Install the full profile with 'make env-full'."
            detail "$run_dir/device.log"
        fi

        # Integration driver. It grades its own evidence, and its lines are copied here.
        note "cuSense-like branch on the episode set $(basename -- "$episode_set"), on the control $s34_id, with" \
            "configs/integration/cusense_frontend.json."
        "$python" tests/integration/cusense_frontend.py --config configs/integration/cusense_frontend.json \
            --s34-run "$s34_run" --episode-set "$episode_set" \
            --output-dir "$run_dir/cusense_frontend" > "$run_dir/cusense-frontend.log" 2>&1
        status=$?
        grade="$run_dir/cusense_frontend/grade.txt"
        graded=$(grep -cE '^(PASS|FAIL)  ' "$grade" 2>/dev/null)
        graded=${graded:-0}
        grep -E '^(PASS|FAIL|NOTE)  ' "$grade" 2>/dev/null | tee -a "$summary"
        if grep -qE '^FAIL  ' "$grade" 2>/dev/null; then failed=1; fi
        counts=$("$python" -c 'import json, sys
grade = json.load(open(sys.argv[1], encoding="utf-8"))
ok = grade["judgment"] == "pass" and grade["cases"] >= 1 and grade["cases_passed"] == grade["cases"] \
    and grade["checks_failed"] == 0 and grade["hardware_gate"] == "closed"
print("%d of %d cases, %d of %d checks, judgment %s, hardware gate %s"
      % (grade["cases_passed"], grade["cases"], grade["checks_passed"], grade["checks"], grade["judgment"],
         grade["hardware_gate"].split(":")[0]))
sys.exit(0 if ok else 1)' "$run_dir/cusense_frontend/grade.json" 2> "$run_dir/grade-read.log")
        counted=$?
        if (( status == 0 && counted == 0 && graded > 0 )); then
            grade_ok=1
            pass "S3.5 integration grade: $counts (cusense_frontend/grade.json)"
        elif (( graded == 0 )); then
            fail "S3.5 integration driver produced no graded checks (see cusense-frontend.log)"
            detail "$run_dir/cusense-frontend.log"
        else
            fail "S3.5 integration grade: ${counts:-unreadable}, driver exit status $status (cusense_frontend/grade.json)"
        fi

        # A second reader, with tools that are not the driver's: the command-line reader on the
        # frozen artifact and on what the sensor wrote. The freeze record names the hash of the
        # artifact, of its weights and of its template. The reader must find that artifact, the
        # output of the sensor must name it, and sha256sum must give the hashes of both files.
        out="$run_dir/cusense_frontend"
        place=$("$python" -c 'import json, sys
found = json.load(open(sys.argv[1], encoding="utf-8"))
freeze = json.load(open(sys.argv[2] + "/" + found["freeze"], encoding="utf-8"))
print(found["artifact"])
print(found["sensor"])
print(found["window"])
print(freeze["artifact"]["sha256"])
print(freeze["artifact"]["weights_sha256"])
print(freeze["artifact"]["template_sha256"])
print(freeze["frozen_utc"])' "$out/recover.json" "$out" 2> "$run_dir/recovered.log")
        artifact=$(echo "$place" | sed -n 1p)
        sensor=$(echo "$place" | sed -n 2p)
        window=$(echo "$place" | sed -n 3p)
        frozen=$(echo "$place" | sed -n 4p)
        frozen_weights=$(echo "$place" | sed -n 5p)
        frozen_template=$(echo "$place" | sed -n 6p)
        frozen_at=$(echo "$place" | sed -n 7p)
        if [[ -n "$artifact" && -n "$sensor" && -n "$frozen" ]] \
                && "$python" -m ocudu_cuda_isac.sensing.cusense inspect "$out/$artifact" --sha256 "$frozen" \
                    > "$run_dir/recovered-artifact.json" 2>> "$run_dir/recovered.log" \
                && "$python" -m ocudu_cuda_isac.sensing.cusense show "$out/$sensor" --window "$window" \
                    > "$run_dir/recovered.json" 2>> "$run_dir/recovered.log"; then
            shown=$("$python" -c 'import json, sys
model = json.load(open(sys.argv[1], encoding="utf-8"))
shown = json.load(open(sys.argv[2], encoding="utf-8"))
window, threshold, grid = shown["window"], model["threshold"], model["grid"]
print(shown["artifact_sha256"])
print(model["weights_sha256"])
print("the model %s reads the input entry %s of shape %s and answers on a grid of %d by %d cells, from the files %s; "
      "its threshold is the logit %.4f, from %d target-absent validation windows with %d of %d allowed above it; the "
      "sensor loaded it %d time(s) on %s in the runtime %s and scored %d windows: window %s from %d observations has "
      "the logit %.4f and the probability %.4f, %s the threshold, and a map that sums to %.4f with its largest value "
      "%.4f in the cell %s at %.2f, %.2f m"
      % (model["model_id"], model["entries"]["input"], model["input_shape"], grid["rows"], grid["cols"],
         ", ".join(model["files"]), threshold["logit"], threshold["negatives"], threshold["exceeding"],
         threshold["allowed"], shown["loads"], shown["device"], shown["runtime"], shown["scored_windows"], window["id"],
         window["contributing"], window["logit"], window["probability"], "above" if window["present"] else "at or below",
         window["map_sum"], window["map_peak"], window["cell"], window["position_m"][0], window["position_m"][1]))' \
                "$run_dir/recovered-artifact.json" "$run_dir/recovered.json" 2>> "$run_dir/recovered.log")
            used=$(echo "$shown" | sed -n 1p)
            named=$(echo "$shown" | sed -n 2p)
            text=$(echo "$shown" | sed -n 3p)
            weights=$(sha256sum "$out/$artifact/weights.f32" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            template=$(sha256sum "$out/$artifact/template.c64" 2>> "$run_dir/recovered.log" | cut -d ' ' -f 1)
            if [[ -n "$text" && "$used" == "$frozen" && "$weights" == "$frozen_weights" && "$named" == "$frozen_weights" \
                    && "$template" == "$frozen_template" ]]; then
                pass "S3.5 a second reader recovers the frozen model and a held-out decision: $text; the artifact" \
                    "${frozen:0:16} is the one the freeze record of $frozen_at names, and its weights and its template" \
                    "have the SHA-256 ${weights:0:16} and ${template:0:16} by sha256sum (recovered-artifact.json," \
                    "recovered.json)"
            else
                fail "S3.5 a second reader recovers the frozen model and a held-out decision: the freeze record names" \
                    "the artifact ${frozen:-none} with the weights ${frozen_weights:-none} and the template" \
                    "${frozen_template:-none}, the sensor scored with ${used:-none}, and the files have" \
                    "${weights:-unreadable} and ${template:-unreadable} by sha256sum"
            fi
        else
            fail "S3.5 a second reader recovers the frozen model and a held-out decision (see recovered.log)"
            detail "$run_dir/recovered.log"
        fi

        # The two records of the card, read here with the standard library alone. The gate requires
        # that they exist, that neither is graded and that every figure has its denominator. What
        # the figures are is recorded and decides nothing.
        record=$("$python" -c 'import json, sys
made = json.load(open(sys.argv[1] + "/cusense_eval.json", encoding="utf-8"))
reference = json.load(open(sys.argv[1] + "/cusense_reference.json", encoding="utf-8"))
if made["graded"] is not False or reference["graded"] is not False:
    sys.exit("a record of the held-out episodes is marked as graded")
arm = made["arms"]["o3"]
episodes, position = arm["presence"]["episodes"], arm["position"]
raw, kalman = position["raw"], position["kalman"]
if position["unit"] != "m" or raw["eligible"] < 1 or None in (raw["mean_m"], raw["median_m"], raw["p90_m"]):
    sys.exit("the position error has no mean, median and 90th percentile in metres with a count")
published = reference["published"]
print("cusense_eval.json holds, for the input entry %s, %d of %d target-present held-out episodes detected and %d of %d "
      "target-absent alarmed, and a position error with a mean of %.3f m, a median of %.3f m and a 90th percentile of "
      "%.3f m over %d windows, %.3f m in the mean after the Kalman filter; cusense_reference.json puts that beside the "
      "published mean of %.1f cm with \"graded\": false and lists %d differences; the latency carries the runtime %s and "
      "is not compared with the published %d us"
      % (arm["input"], episodes["target_present"]["detected"], episodes["target_present"]["total"],
         episodes["target_absent"]["false_alarm"], episodes["target_absent"]["total"], raw["mean_m"], raw["median_m"],
         raw["p90_m"], raw["eligible"], kalman["mean_m"] if kalman["mean_m"] is not None else float("nan"),
         published["position_error_cm"]["mean"], len(reference["differences"]),
         reference["measured"]["latency"]["runtime"], published["live_inference_us"]["value"]))' \
            "$out" 2> "$run_dir/records-read.log")
        if [[ -n "$record" ]]; then
            pass "S3.5 records, written and not graded: $record"
        else
            fail "S3.5 records: cusense_eval.json or cusense_reference.json is missing, graded or incomplete (see" \
                "records-read.log)"
            detail "$run_dir/records-read.log"
        fi
    else
        fail "S3.5 adapter: build or verification failed (see adapter-build.log, adapter-verify.log)"
        detail "$run_dir/adapter-build.log"
        detail "$run_dir/adapter-verify.log"
        note "SKIP  S3.5 integration driver, artifact recovery and records: no adapter"
    fi
elif [[ -n "$s34_run" && -n "$s52_run" && -n "$episode_set" ]]; then
    fail "S3.5 needs the patched receiver of Sprint 5.2: build/ocudu-spark-o3 is missing"
    note "      Run 'make accept-s5.2' on this host first: it builds the patched receiver from its own tree."
    note "SKIP  S3.5 reference check, adapter build, integration driver, artifact recovery and records: no patched" \
        "receiver"
else
    note "SKIP  S3.5 reference check, adapter build, integration driver, artifact recovery and records: no control or" \
        "no episode set"
fi

# The upstream sources stay at the locked commits with no edited file, and the patched tree is the
# pinned commit with exactly the series applied.
step "S3.5 sources at the locked revisions after the runs (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees
if [[ -d build/ocudu-src-o3 ]]; then
    step "S3.5 patched tree: build/ocudu-src-o3 is the pinned commit with exactly the series applied (patched-tree.json)" \
        patched-tree.json "$python" scripts/receiver_patch.py verify build/ocudu-src-o3
fi

note "Evidence: $run_dir"
if (( failed || ! grade_ok )); then
    note "Sprint 3.5: NOT ACCEPTED on this host"
    exit 1
fi
note "Sprint 3.5: ACCEPTED on this DGX Spark. The cuSense-like branch runs on the receiver's own estimate: a" \
    "background template and the statistics of the z-score were fitted on the training trajectories, the location map" \
    "and the presence logit were trained on receiver-derived windows of one TDD period, the threshold was set on" \
    "target-absent validation windows, and one artifact of four files per input entry was frozen before a held-out" \
    "entry was replayed. The same artifact gives the same map and score for the same held-out window in a second CUDA" \
    "process and offline within the declared tolerance, and the pilot comb and the linear head of $s34_id were scored" \
    "on the records of the same run. The position error and the presence counts are recorded beside the published" \
    "figures and are not graded. Next: S3.6 native sensing core."
