#!/usr/bin/env bash
set -euo pipefail

git config --global --add safe.directory /workspace
nvidia-smi
cuda_arch="$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader)"
cuda_arch="${cuda_arch//./}"
if [[ ! "$cuda_arch" =~ ^[0-9]+$ ]]; then
    echo "Could not determine the CUDA architecture: $cuda_arch" >&2
    exit 1
fi

cmake -S /workspace -B /tmp/build-codspeed-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYSIGLIB_CUDA=ON \
    -DPYSIGLIB_CUDA_ARCH="$cuda_arch" \
    -DPYSIGLIB_BENCHMARKS=ON \
    -DPYSIGLIB_JAX_FFI=OFF \
    -DCODSPEED_MODE=walltime
cmake --build /tmp/build-codspeed-cuda --target bench_cusig --parallel 1

# CodSpeed 5.1.0 configures profiling sysctls during setup even without a profiler.
codspeed run --mode walltime --skip-setup "$@" -- /tmp/build-codspeed-cuda/benchmarks/bench_cusig --benchmark_min_warmup_time=0.2
