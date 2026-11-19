#!/usr/bin/env bash
# Run the original smoke workload or the DeepSig Table II comparison on a DGX Spark.
set -eu
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec python scripts/native_ocudu.py run "$@"
