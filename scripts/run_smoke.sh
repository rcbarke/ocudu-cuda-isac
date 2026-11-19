#!/usr/bin/env sh
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ "$#" -eq 0 ]; then
    set -- --output-dir "$project_root/build/smoke"
fi
PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    exec python3 -m ocudu_cuda_isac --episode "$project_root/configs/smoke_replay.json" \
        --trace "$project_root/configs/smoke_timing.json" "$@"
