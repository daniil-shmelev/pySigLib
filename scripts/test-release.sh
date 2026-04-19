#!/usr/bin/env bash
# Reinstalls the latest pysiglib (including RCs) from TestPyPI and runs the
# repo's test suite. Works on Linux and macOS.

read -r -p "Install from production PyPI? [y/n] (n = TestPyPI) " pypi
read -r -p "Install with CUDA support? [y/n] " cuda

python3 -m pip uninstall -y pysiglib pysiglib-cuda || true

packages=(pysiglib)
if [[ "$cuda" =~ ^[yY] ]]; then
    packages+=(pysiglib-cuda)
fi

index_args=()
if [[ ! "$pypi" =~ ^[yY] ]]; then
    index_args=(--index-url https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/)
fi

python3 -m pip install --pre "${index_args[@]}" "${packages[@]}"

repo_dir="${TMPDIR:-/tmp}/pysiglib-rc-tests"
rm -rf "$repo_dir"
git clone --depth 1 --branch main https://github.com/daniil-shmelev/pySigLib.git "$repo_dir"
python3 -m pip install pytest
python3 -m pytest "$repo_dir/tests" -v
read -r -p "Press Enter to close..."
