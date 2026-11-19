#!/usr/bin/env bash
# Build and run the pinned high-SINR OCUDU receiver check on a DGX Spark.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash scripts/run_spark_baseline.sh [--build-dir DIR] [--jobs N] [--output-root DIR]
Defaults: build/ocudu-spark, 4 build jobs, results/runs/. Paths are relative to the repository root.
This initializes the pinned OCUDU submodule if needed and reuses the selected CMake build directory.
USAGE
}
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
build_dir=build/ocudu-spark
output_root=results/runs
jobs=4
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --build-dir|--output-root|--jobs)
            if (( $# < 2 )) || [[ -z "$2" ]]; then usage >&2; exit 2; fi
            case "$1" in
                --build-dir) build_dir=$2 ;;
                --output-root) output_root=$2 ;;
                --jobs) jobs=$2 ;;
            esac
            shift 2
            ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo '--jobs requires a positive integer' >&2
    exit 2
fi

mkdir -p -- "$output_root"
run_dir=$(mktemp -d -- "$output_root/spark-pusch.XXXXXXXX")
echo "Run artifacts: $run_dir"
stage=preflight
trap 'status=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$status" > "$run_dir/status.txt"' EXIT
for tool in git python3 cmake c++ nvcc nvidia-smi timeout; do
    command -v "$tool" > /dev/null || { echo "Missing required command: $tool" >&2; exit 1; }
done
{
    date -u
    uname -a
    git rev-parse HEAD
    git status --short --untracked-files=no
    nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version --format=csv
    nvcc --version
    cmake --version
    c++ --version
} > "$run_dir/platform.txt" 2>&1
gpu=$(nvidia-smi -i 0 --query-gpu=name,compute_cap --format=csv,noheader)
if [[ "$gpu" != *GB10* || "${gpu##*, }" != 12.1 ]]; then
    echo "This recipe targets GB10 compute capability 12.1. Found $gpu" >&2
    exit 1
fi

stage=source
python3 scripts/check_pins.py > "$run_dir/pins.txt"
if [[ ! -e deps/ocudu/.git ]]; then
    git submodule update --init --checkout -- deps/ocudu 2>&1 | tee "$run_dir/hydrate.log"
fi
native_pin=$(git rev-parse HEAD:deps/ocudu)
if [[ "$(git -C deps/ocudu rev-parse HEAD)" != "$native_pin" || -n "$(git -C deps/ocudu status --porcelain)" ]]; then
    echo "The native baseline requires a clean deps/ocudu checkout at $native_pin." >&2
    exit 1
fi

stage=configure
cmake -S deps/ocudu -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=121 -DMCPU=neoverse-v2 \
    -DBUILD_TESTING=ON -DDISABLE_MBEDTLS=OFF -DDISABLE_SCTP=OFF \
    -DENABLE_UHD=OFF -DENABLE_ZEROMQ=OFF 2>&1 | tee "$run_dir/configure.log"
cp "$build_dir/CMakeCache.txt" "$run_dir/CMakeCache.txt"
if ! grep -Fqx 'CUDA_ACCEL_FOUND:BOOL=TRUE' "$run_dir/CMakeCache.txt"; then
    echo 'CMake did not activate the in-tree CUDA backend. See configure.log.' >&2
    exit 1
fi
stage=build
cmake --build "$build_dir" --target pusch_e2e_pipeline_test --parallel "$jobs" 2>&1 | tee "$run_dir/build.log"

stage=run
native_test="$build_dir/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
args=(--prb 51 --sinr 30 --mcs 10 --mcs-table qam64
      --layers 1 --ports 4 --dmrs-type type1 --dmrs-symbols 2,11
      --iterations 20 --warmup 5 --cpu-every 1
      --rx-device-grid direct-managed --require-gpu-resident --quiet)
export OCUDU_PUSCH_ACCELERATION_TIMING=1
{
    printf 'OCUDU_PUSCH_ACCELERATION_TIMING=1 '
    printf '%q ' "$native_test" "${args[@]}"
    printf '\n'
} > "$run_dir/command.txt"
timeout --kill-after=10s 5m "$native_test" "${args[@]}" 2>&1 | tee "$run_dir/pusch.log"

stage=verify
python3 scripts/verify_pusch_log.py "$run_dir/pusch.log" --iterations 20 2>&1 | tee "$run_dir/verification.txt"
stage=completed
