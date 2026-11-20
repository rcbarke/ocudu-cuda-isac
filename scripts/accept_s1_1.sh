#!/usr/bin/env bash
# Run every Sprint 1.1 gate on this host and keep the evidence in a new run directory.
# The gate builds the locked OCUDU Release with CUDA, grades the build evidence against the
# DeepSig reference environment and starts both executables from the isac environment.
# It never edits the sources or an earlier run. A second run reuses the build directory.
set -uo pipefail
usage() {
    echo "Usage: bash scripts/accept_s1_1.sh [--run-dir DIR] [--jobs N] [--build-dir DIR]"
    echo "                                   [--machine spark|local] [--mcpu TARGET] [cpu] [cuda]"
    echo "Name the simulation backends this host passed in Sprint 0. The native executables are"
    echo "then verified together with Sionna and Sionna RT from one interpreter. With none, the"
    echo "native gates run alone. The default profile is the DGX Spark. Another GPU host uses"
    echo "--machine local."
}
run_dir=""
build_args=()
machine=spark
backends=()
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            run_dir=$2; shift 2 ;;
        --jobs|--build-dir|--mcpu)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            build_args+=("$1" "$2"); shift 2 ;;
        --machine)
            if (( $# < 2 )) || [[ "$2" != spark && "$2" != local ]]; then usage >&2; exit 2; fi
            machine=$2; shift 2 ;;
        cpu|cuda) backends+=("$1"); shift ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="$PWD/isac/bin/python"
if [[ ! -x "$python" ]]; then
    echo "accept_s1_1: $python is missing. Run 'make env' first." >&2
    exit 2
fi
if [[ -z "$run_dir" ]]; then
    run_dir="results/runs/s11-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$run_dir" ]]; then
    echo "accept_s1_1: $run_dir already exists. Evidence is never overwritten." >&2
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
# The native tools must be found the way an activated shell finds them.
venv_bin="$PWD/isac/bin"
export VIRTUAL_ENV="$PWD/isac"
export PATH="$venv_bin:$PATH"

note "Sprint 1.1 acceptance on $(hostname), $(date -u +%Y-%m-%dT%H:%M:%SZ), $machine profile"
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

# The portable suite holds the S1.1 build contract tests. An empty run cannot close the gate.
PYTHONPATH=src "$python" -m unittest discover -s tests -v > "$run_dir/unit-tests.log" 2>&1
status=$?
ran=$(grep -E '^Ran [0-9]+ tests? in ' "$run_dir/unit-tests.log" | tail -n 1 | awk '{print $2}')
verdict=$(grep -E '^(OK|FAILED)' "$run_dir/unit-tests.log" | tail -n 1)
skipped=$(echo "$verdict" | sed -n 's/.*skipped=\([0-9][0-9]*\).*/\1/p')
ran=${ran:-0}
skipped=${skipped:-0}
if (( status != 0 )) || [[ "$verdict" != OK* ]]; then
    fail "unit tests ($ran ran; see unit-tests.log)"
    grep -E '^(FAIL|ERROR): ' "$run_dir/unit-tests.log" | head -n 10 | cut -c 1-300 | sed 's/^/      /' | tee -a "$summary"
elif (( ran == 0 )); then
    fail "unit tests (none ran)"
else
    pass "unit tests ($ran ran, $skipped skipped)"
fi

# The upstream control must be pristine before the build starts and unchanged after it.
source_dir=$("$python" scripts/check_pins.py --path ocudu 2> "$run_dir/source-path.log")
source_git=0
if [[ -z "$source_dir" ]]; then
    fail "S1.1 OCUDU source directory could not be resolved (see source-path.log)"
elif [[ -e "$source_dir/.git" ]]; then
    source_git=1
    if git -C "$source_dir" status --porcelain --untracked-files=all > "$run_dir/source-before.txt" 2>&1 \
            && [[ ! -s "$run_dir/source-before.txt" ]]; then
        pass "S1.1 OCUDU source is pristine before the build"
    else
        fail "S1.1 OCUDU source is pristine before the build (see source-before.txt)"
        note "      Remove the listed files, for example a build directory made by hand inside $source_dir"
    fi
fi

# Configure, build and inspect. The builder writes its own evidence directory inside this run.
# Its output, including everything CMake prints, is shown while it runs and kept in native-build.log.
evidence="$run_dir/native-build"
note "Building the locked OCUDU Release. CMake output follows. A first build takes a while."
"$python" scripts/native_ocudu.py build --run-dir "$evidence" --machine "$machine" \
    ${build_args[@]+"${build_args[@]}"} 2>&1 | tee "$run_dir/native-build.log"
if (( PIPESTATUS[0] == 0 )); then
    built=1
    pass "S1.1 native build command (native-build.log, evidence in native-build/)"
else
    built=0
    stage=$(sed -n 's/^stage=//p' "$evidence/status.txt" 2>/dev/null)
    reason=$(head -n 1 "$evidence/error.txt" 2>/dev/null | cut -c 1-300)
    fail "S1.1 native build command stopped in stage ${stage:-unknown}: ${reason:-no evidence written}"
    # The failing tool's own words and the remedy are on the screen above. Keep them in the summary too.
    sed -n '2,60p' "$evidence/error.txt" 2>/dev/null | cut -c 1-300 | sed 's/^/      /' >> "$summary"
fi

if (( built )); then
    # Grade the archived evidence: completion, source, platform, configuration, CUDA compilation, hashes.
    "$python" scripts/native_ocudu.py grade "$evidence" > "$run_dir/native-grade.txt" 2>&1
    status=$?
    graded=$(grep -cE '^(PASS|FAIL)  ' "$run_dir/native-grade.txt")
    grep -E '^(PASS|FAIL|NOTE)  ' "$run_dir/native-grade.txt" | tee -a "$summary"
    if (( status != 0 )); then
        failed=1
        if ! grep -qE '^FAIL  ' "$run_dir/native-grade.txt"; then
            fail "S1.1 build evidence could not be graded (see native-grade.txt)"
            detail "$run_dir/native-grade.txt"
        fi
    elif (( graded == 0 )); then
        fail "S1.1 build evidence produced no graded checks (see native-grade.txt)"
    fi

    # Both programs resolve inside isac/bin and are the files the build receipt hashed.
    resolved=1
    for tool in gnb pusch_e2e_pipeline_test; do
        found=$(command -v "$tool" 2>/dev/null)
        echo "$tool -> ${found:-not found} -> $(readlink -f -- "${found:-/nonexistent}" 2>/dev/null)" \
            >> "$run_dir/native-tools.txt"
        [[ "$found" == "$venv_bin/$tool" && -L "$found" && -x "$found" ]] || resolved=0
    done
    if (( resolved )); then
        pass "S1.1 gnb and pusch_e2e_pipeline_test resolve inside isac/bin (native-tools.txt)"
    else
        fail "S1.1 gnb and pusch_e2e_pipeline_test resolve inside isac/bin (see native-tools.txt)"
    fi
    if gnb --version > "$run_dir/gnb-version.txt" 2>&1 \
            && banner=$(grep -E '^OCUDU 5G gNB version [0-9.]+ \([0-9a-f]*\)$' "$run_dir/gnb-version.txt"); then
        pass "S1.1 gnb --version from the environment: $banner"
    else
        fail "S1.1 gnb --version from the environment (see gnb-version.txt)"
        detail "$run_dir/gnb-version.txt"
    fi
    if pusch_e2e_pipeline_test --help > "$run_dir/pusch-help.txt" 2>&1 \
            && grep -q -- '--require-gpu-resident' "$run_dir/pusch-help.txt" \
            && grep -q -- '--rx-device-grid' "$run_dir/pusch-help.txt" \
            && grep -q -- '--gpu-output-buffer' "$run_dir/pusch-help.txt"; then
        pass "S1.1 pusch_e2e_pipeline_test --help lists the resident, device-grid and output-buffer options"
    else
        fail "S1.1 pusch_e2e_pipeline_test --help (see pusch-help.txt)"
        detail "$run_dir/pusch-help.txt"
    fi

    step "S1.1 native environment report (native-env.json)" native-env.log \
        "$python" scripts/verify_environment.py --profile core --with-native --report "$run_dir/native-env.json"
    for backend in ${backends[@]+"${backends[@]}"}; do
        step "S1.1 simulation on $backend and native from one interpreter (collocated-$backend.json)" \
            "collocated-$backend.log" "$python" scripts/verify_environment.py --profile simulation \
            --backend "$backend" --with-native --report "$run_dir/collocated-$backend.json"
    done
else
    # These gates read what a completed build publishes. Without one they could only repeat the failure.
    note "SKIP  S1.1 evidence grade, isac/bin links, gnb --version, pusch_e2e_pipeline_test --help" \
        "and environment reports: no completed build"
fi

# The upstream control stays pristine: locked commit, no edited file and nothing left by the build.
step "S1.1 OCUDU source at the locked revision after the build (sources-after.txt)" sources-after.txt \
    "$python" scripts/check_pins.py --worktrees --only ocudu
if (( source_git )); then
    if git -C "$source_dir" status --porcelain --untracked-files=all > "$run_dir/source-after.txt" 2>&1 \
            && cmp -s "$run_dir/source-before.txt" "$run_dir/source-after.txt"; then
        pass "S1.1 out-of-tree build left no file in the OCUDU source"
    else
        fail "S1.1 out-of-tree build left no file in the OCUDU source (compare source-before.txt with source-after.txt)"
    fi
fi

note "Evidence: $run_dir"
if (( failed )); then
    note "Sprint 1.1: NOT ACCEPTED on this host"
    exit 1
fi
if [[ "$machine" == spark ]]; then
    note "Sprint 1.1: ACCEPTED on this DGX Spark. Next gate: make paper-baseline (S1.2)."
else
    note "Sprint 1.1: ACCEPTED on this host for the local profile. The DeepSig Table II row stays assigned to a GB10."
fi
