#!/usr/bin/env bash
set -e

: "${CUDA_PATH:?CUDA_PATH environment variable is not set}"
NVCC_EXE="${CUDA_PATH}/bin/nvcc"

SIGLIB_DIR="$(pwd)/siglib"

cd "${SIGLIB_DIR}/cusig"

echo "*** Detecting supported GPU architectures ***"

# Query nvcc for supported architectures (available since CUDA 11.5)
ARCHS=$("${NVCC_EXE}" --list-gpu-arch 2>/dev/null | sed -n 's/^compute_//p' | sort -n -u)

if [ -z "${ARCHS}" ]; then
    echo "Error: failed to detect supported GPU architectures from nvcc."
    exit 1
fi

# Build -gencode flags for each supported architecture
GENCODE_FLAGS=""
LAST_ARCH=""
for arch in ${ARCHS}; do
    GENCODE_FLAGS="${GENCODE_FLAGS} -gencode=arch=compute_${arch},code=sm_${arch}"
    LAST_ARCH="${arch}"
done

# Add PTX for the highest architecture (forward compatibility)
GENCODE_FLAGS="${GENCODE_FLAGS} -gencode=arch=compute_${LAST_ARCH},code=compute_${LAST_ARCH}"

echo "Architectures: ${ARCHS}"
echo "PTX forward-compat: compute_${LAST_ARCH}"

echo "*** Compile CUDA files with nvcc + Linking ***"

mkdir -p "${SIGLIB_DIR}/x64/Release"

"${NVCC_EXE}" \
${GENCODE_FLAGS} \
--std c++17 -shared -Xcompiler -fPIC -DNDEBUG -DCUSIG_EXPORTS \
    cu_sig_kernel.cu cu_path_transforms.cu cu_tensor_poly.cu cu_signature.cu cu_log_signature.cu cu_log_sig_cache.cu cu_sig_coef.cu \
    -o "${SIGLIB_DIR}/x64/Release/libcusig.so"

echo "*** Build complete. ***"
