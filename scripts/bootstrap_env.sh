#!/usr/bin/env bash
# Provision project Python dependencies only. Host drivers/toolkits are provisioned per machine.
set -euo pipefail
usage() {
    echo "Usage: bash scripts/bootstrap_env.sh [core|full]"
    echo "Default: full. PYTHON chooses the base interpreter (Python 3.11+)."
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# > 1 )); then usage >&2; exit 2; fi
profile=${1:-full}
case "$profile" in core|full) ;; *) usage >&2; exit 2 ;; esac
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
venv="$PWD/isac"
if [[ ! -f "$venv/pyvenv.cfg" ]]; then
    if [[ -e "$venv" ]]; then
        echo "Refusing to reuse $venv without pyvenv.cfg; choose a clean checkout." >&2
        exit 1
    fi
    base_python=${PYTHON:-python3}
    "$base_python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Select Python 3.11+ with PYTHON=/path/to/python")'
    "$base_python" -m venv --prompt isac "$venv"
fi
python="$venv/bin/python"
(cd scripts && "$python" -c 'from verify_environment import ROOT, check_isolation; check_isolation(ROOT / "isac")')

if [[ "$profile" == full ]]; then
    "$python" scripts/check_pins.py --worktrees
    # Pass both local projects to pip simultaneously so Sionna resolves this exact RT source.
    "$python" -m pip install -e . ./ext/sionna ./ext/sionna-rt
else
    "$python" -m pip install -e .
fi
"$python" -m pip check
echo "Activate with: source isac/bin/activate"
echo "Run the environment checks in docs/dev/ENVIRONMENT.md."
