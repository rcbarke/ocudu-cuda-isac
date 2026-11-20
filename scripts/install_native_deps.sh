#!/usr/bin/env bash
# Install the Ubuntu development libraries the pinned OCUDU configure links. The list is the
# library part of OCUDU's own docker/scripts/install_dependencies.sh.
# Safe to run again: it asks dpkg what is installed, installs only what is missing and, with
# nothing missing, changes nothing and never calls sudo.
# The compiler, CMake, pkg-config and the CUDA toolkit belong to the host. The builder's
# preflight checks them.
set -euo pipefail
packages=(libfftw3-dev libmbedtls-dev libsctp-dev libyaml-cpp-dev libgtest-dev)
check_only=0
case "${1:-}" in
    "") ;;
    --check) check_only=1 ;;
    *) echo "Usage: bash scripts/install_native_deps.sh [--check]" >&2; exit 2 ;;
esac
if ! command -v dpkg-query > /dev/null || ! command -v apt-get > /dev/null; then
    echo "native prerequisites: no dpkg and apt-get on this host. Install the equivalents of: ${packages[*]}"
    exit 0
fi
# Fill 'missing' with the packages dpkg does not report as fully installed.
find_missing() {
    missing=()
    local package
    for package in "${packages[@]}"; do
        [[ "$(dpkg-query -W -f='${Status}' "$package" 2> /dev/null)" == "install ok installed" ]] \
            || missing+=("$package")
    done
}
find_missing
if (( ${#missing[@]} == 0 )); then
    echo "native prerequisites: all ${#packages[@]} packages are installed (${packages[*]})"
    exit 0
fi
if (( check_only )); then
    echo "native prerequisites: missing ${missing[*]}. Install with: bash scripts/install_native_deps.sh"
    exit 1
fi
echo "native prerequisites: installing ${missing[*]} with apt-get"
# apt-get as root, without a question on the terminal.
run_apt() {
    if [[ "$(id -u)" == 0 ]]; then
        DEBIAN_FRONTEND=noninteractive apt-get "$@"
    else
        sudo DEBIAN_FRONTEND=noninteractive apt-get "$@"
    fi
}
manual="sudo apt-get update && sudo apt-get install ${missing[*]}"
if [[ "$(id -u)" != 0 ]] && ! command -v sudo > /dev/null; then
    echo "native prerequisites: sudo is missing. As root run: apt-get install ${missing[*]}" >&2
    exit 1
fi
if ! run_apt update || ! run_apt install -y "${missing[@]}"; then
    echo "native prerequisites: apt-get failed. Fix the cause above, or run: $manual" >&2
    exit 1
fi
find_missing
if (( ${#missing[@]} > 0 )); then
    echo "native prerequisites: still missing after apt-get: ${missing[*]}. Run: $manual" >&2
    exit 1
fi
echo "native prerequisites: installed"
