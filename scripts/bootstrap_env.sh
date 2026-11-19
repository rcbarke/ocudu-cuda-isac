#!/usr/bin/env bash
# Provision project Python dependencies only. Host drivers/toolkits are provisioned per machine.
set -euo pipefail
usage() {
    echo "Usage: bash scripts/bootstrap_env.sh [core|full]"
    echo "Default: full. PYTHON chooses the base interpreter (Python 3.11+)."
}
fail() {
    echo "bootstrap_env: $*" >&2
    exit 1
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# > 1 )); then usage >&2; exit 2; fi
profile=${1:-full}
case "$profile" in core|full) ;; *) usage >&2; exit 2 ;; esac
# Use the physical path so the location recorded inside the venv is the same on every retry.
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
venv="$PWD/isac"
python="$venv/bin/python"
if [[ ! -f "$venv/pyvenv.cfg" ]]; then
    if [[ -e "$venv" || -L "$venv" ]]; then
        fail "Refusing to reuse $venv without pyvenv.cfg; choose a clean checkout."
    fi
    base_python=${PYTHON:-python3}
    command -v "$base_python" >/dev/null 2>&1 \
        || fail "Base interpreter '$base_python' not found. Select Python 3.11+ with PYTHON=/path/to/python"
    # Check the base interpreter before anything is written so a refusal leaves the checkout untouched.
    "$base_python" -c '
import sys
if sys.version_info < (3, 11):
    sys.exit("bootstrap_env: Select Python 3.11+ with PYTHON=/path/to/python")
try:
    import ensurepip, venv
except ImportError as error:
    sys.exit(f"bootstrap_env: {sys.executable} cannot create a venv with pip ({error}). "
             "On Ubuntu run: sudo apt-get install python3-venv")
'
    # Only this invocation created the directory, so a failed creation may remove it again.
    if ! "$base_python" -m venv --prompt isac "$venv"; then
        rm -rf -- "$venv"
        fail "Could not create $venv; the partial directory was removed. Fix the error above and rerun."
    fi
fi
if [[ ! -x "$python" || ! -f "$venv/bin/activate" ]] || ! "$python" -m pip --version >/dev/null 2>&1; then
    fail "$venv is incomplete (interrupted creation, missing pip, or copied from another host)." \
         "Remove it with 'rm -rf isac' in $PWD and rerun."
fi
(cd scripts && "$python" -c '
import sys
from verify_environment import ROOT, check_isolation, check_search_path
try:
    check_isolation(ROOT / "isac")
    check_search_path(ROOT / "isac", ROOT)
except ValueError as error:
    sys.exit(f"bootstrap_env: {error}")
')

if [[ "$profile" == full ]]; then
    "$python" scripts/check_pins.py --worktrees
    # ext/lock.json owns both checkout paths. Pass both local projects to pip simultaneously
    # so Sionna resolves this exact RT source.
    sionna=$("$python" scripts/check_pins.py --path sionna)
    sionna_rt=$("$python" scripts/check_pins.py --path sionna-rt)
    "$python" -m pip install -e . "$sionna" "$sionna_rt"
else
    "$python" -m pip install -e .
fi
"$python" -m pip check
echo "Activate with: source isac/bin/activate"
echo "Run the environment checks in docs/dev/ENVIRONMENT.md."
