# Copyright 2026 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

cmake_minimum_required(VERSION 3.24)
project(endpoint_gemm_experiment LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
get_filename_component(REPO "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CPU_BUILD "${REPO}/out/build/windows-cpu" CACHE PATH "Built CPU dependencies")
set(TBB_HEADERS "${REPO}/build/py3-none-win_amd64/_deps/onetbb-src/include" CACHE PATH "oneTBB headers")
set(HWY_HEADERS "${REPO}/build/py3-none-win_amd64/_deps/highway-src" CACHE PATH "Highway headers")
find_library(TBB_LIBRARY tbb12 PATHS "${CPU_BUILD}/msvc_19.44_cxx20_64_mt_release" REQUIRED NO_DEFAULT_PATH)
find_library(BLAS_LIBRARY openblas PATHS "${CPU_BUILD}/pysiglib-openblas/install/lib" REQUIRED NO_DEFAULT_PATH)
add_library(endpoint_cpu SHARED cpu.cpp "${REPO}/siglib/cpsig/multithreading.cpp")
target_include_directories(endpoint_cpu PRIVATE "${REPO}/siglib/cpsig" "${REPO}/siglib/shared"
    "${CPU_BUILD}/pysiglib-openblas/install/include/openblas" "${TBB_HEADERS}" "${HWY_HEADERS}")
target_compile_definitions(endpoint_cpu PRIVATE CPSIG_STATIC_TARGET CPSIG_EXPORTS)
target_compile_options(endpoint_cpu PRIVATE /O2 /arch:AVX2)
target_link_libraries(endpoint_cpu PRIVATE "${BLAS_LIBRARY}" "${TBB_LIBRARY}" "${CPU_BUILD}/_deps/highway-build/Release/hwy.lib")
find_package(CUDAToolkit REQUIRED)
set(CUBLAS_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" CACHE PATH "cuBLAS installation")
find_path(CUBLAS_INCLUDE cublas_v2.h PATHS "${CUBLAS_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
find_library(CUBLAS_LIBRARY cublas PATHS "${CUBLAS_ROOT}/lib/x64" REQUIRED NO_DEFAULT_PATH)
add_library(endpoint_cuda SHARED cuda.cu)
target_include_directories(endpoint_cuda PRIVATE "${REPO}/siglib/shared" "${CUBLAS_INCLUDE}")
target_link_libraries(endpoint_cuda PRIVATE CUDA::cudart "${CUBLAS_LIBRARY}")
target_compile_options(endpoint_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
