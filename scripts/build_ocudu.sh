#!/usr/bin/env bash
# Configure and build the locked native dependency with the active project environment.
set -eu
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec python scripts/native_ocudu.py build "$@"
