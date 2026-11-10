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
project(reversal_experiments LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CUDA_ARCHITECTURES native)
get_filename_component(REPO "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CPU_BUILD "${REPO}/out/build/windows-cpu" CACHE PATH "Existing Release CPU build")
set(DEPS_SOURCE "${REPO}/build/py3-none-win_amd64/_deps" CACHE PATH "Existing dependency sources")
set(CUBLAS_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" CACHE PATH "cuBLAS installation")
set(CUDA_cublas_LIBRARY "${CUBLAS_ROOT}/lib/x64/cublas.lib" CACHE FILEPATH "cuBLAS library")
set(CUDA_cublasLt_LIBRARY "${CUBLAS_ROOT}/lib/x64/cublasLt.lib" CACHE FILEPATH "cuBLAS Lt library")
find_package(CUDAToolkit REQUIRED)
add_library(reversal_cpu SHARED cpu.cpp "${REPO}/siglib/cpsig/multithreading.cpp")
target_compile_definitions(reversal_cpu PRIVATE CPSIG_STATIC_TARGET CPSIG_EXPORTS NOMINMAX)
target_compile_options(reversal_cpu PRIVATE /arch:AVX2 /bigobj)
target_include_directories(reversal_cpu PRIVATE "${REPO}/siglib/cpsig" "${REPO}/siglib/shared"
    "${DEPS_SOURCE}/highway-src" "${DEPS_SOURCE}/onetbb-src/include"
    "${CPU_BUILD}/pysiglib-openblas/install/include/openblas64")
target_link_libraries(reversal_cpu PRIVATE "${CPU_BUILD}/_deps/highway-build/Release/hwy.lib"
    "${CPU_BUILD}/msvc_19.44_cxx20_64_mt_release/tbb12.lib"
    "${CPU_BUILD}/pysiglib-openblas/install/lib/openblas_64.lib")
add_custom_command(TARGET reversal_cpu POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${REPO}/pysiglib/tbb12.dll" "$<TARGET_FILE_DIR:reversal_cpu>")
add_library(reversal_cuda SHARED cuda.cu)
target_include_directories(reversal_cuda PRIVATE "${REPO}/siglib/cusig" "${REPO}/siglib/shared" "${CUBLAS_ROOT}/include")
target_compile_definitions(reversal_cuda PRIVATE CUSIG_EXPORTS NOMINMAX)
target_compile_options(reversal_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(reversal_cuda PRIVATE CUDA::cudart CUDA::cublas)
