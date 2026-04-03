cmake_minimum_required(VERSION 3.24)
project(pysiglib LANGUAGES CXX)

include(CheckLanguage)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(PYSIGLIB_CUDA "Build CUDA support (cusig library)" ON)
option(PYSIGLIB_AVX  "Enable AVX/NEON vectorization"      ON)
set(PYSIGLIB_CUDA_ARCH "native" CACHE STRING
    "CUDA architectures (native, all, all-major, or semicolon-separated list)")

# ---------------------------------------------------------------------------
# Match the original B2 / batch-script build: static CRT (/MT) on MSVC
# ---------------------------------------------------------------------------
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Threading — B2 uses <threading>multi which adds -pthread on Linux
find_package(Threads REQUIRED)

# macOS has no CUDA support
if(APPLE AND PYSIGLIB_CUDA)
    message(STATUS "CUDA is not supported on macOS — disabling")
    set(PYSIGLIB_CUDA OFF)
endif()

# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------
set(_cuda_enabled FALSE)
set(_cuda_version "")
if(PYSIGLIB_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(_cuda_enabled TRUE)
        set(_cuda_version "${CUDAToolkit_VERSION}")
    else()
        message(STATUS "CUDA compiler not found — building without CUDA")
        message(STATUS "  CUDA_PATH env var  = '$ENV{CUDA_PATH}'")
        find_program(_nvcc_probe nvcc
            HINTS "$ENV{CUDA_PATH}/bin" "$ENV{CUDA_PATH}/Library/bin")
        if(_nvcc_probe)
            message(STATUS "  nvcc found at: ${_nvcc_probe}")
            message(STATUS "  nvcc was found but failed to compile a test program.")
            message(STATUS "  Common causes:")
            message(STATUS "    - cl.exe (MSVC) is not on PATH — install Visual Studio Build Tools")
            message(STATUS "    - CUDA Visual Studio integration files are missing — install the")
            message(STATUS "      CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
            message(STATUS "      (the conda cuda-toolkit package does not include VS integration)")
        else()
            message(STATUS "  nvcc was not found on PATH or in CUDA_PATH")
            message(STATUS "  Install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
        endif()
        message(STATUS "  CMAKE_GENERATOR = ${CMAKE_GENERATOR}")
    endif()
endif()

# ---------------------------------------------------------------------------
# AVX / vectorization detection
# ---------------------------------------------------------------------------
set(_enable_vec FALSE)
set(_has_avx 0)
set(_has_avx2 0)
set(_has_avx512f 0)

if(PYSIGLIB_AVX)
    if(APPLE)
        # ARM NEON is always available on Apple Silicon; on Intel Macs
        # -march=native enables whatever the CPU supports.
        set(_enable_vec TRUE)
    else()
        # Compile and run the existing avx_info utility to probe the CPU.
        try_run(_avx_run_result _avx_compile_result
            "${CMAKE_CURRENT_BINARY_DIR}/_avx_detect"
            "${CMAKE_CURRENT_SOURCE_DIR}/avx_info/avx_info.cpp"
        )
        if(_avx_compile_result AND NOT _avx_run_result STREQUAL "FAILED_TO_RUN")
            math(EXPR _has_avx     "${_avx_run_result} & 1")
            math(EXPR _has_avx2    "${_avx_run_result} & 2")
            math(EXPR _has_avx512f "${_avx_run_result} & 4")
            if(_has_avx2)
                set(_enable_vec TRUE)
            else()
                set(PYSIGLIB_AVX OFF)
            endif()
        else()
            message(WARNING "AVX detection failed — disabling vectorization")
            set(PYSIGLIB_AVX OFF)
        endif()
    endif()
endif()

message(STATUS "")
message(STATUS "  ---- pysiglib build summary ----")
message(STATUS "  System        : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Generator     : ${CMAKE_GENERATOR}")
message(STATUS "  CXX compiler  : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "  Build configs : ${CMAKE_CONFIGURATION_TYPES}")
else()
    message(STATUS "  Build type    : ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "  CUDA enabled  : ${_cuda_enabled}")
if(_cuda_enabled)
    message(STATUS "  CUDA compiler : ${CMAKE_CUDA_COMPILER}")
    message(STATUS "  CUDA version  : ${_cuda_version}")
    message(STATUS "  CUDA arch     : ${PYSIGLIB_CUDA_ARCH}")
endif()
message(STATUS "  AVX enabled   : ${_enable_vec}")
if(_enable_vec AND NOT APPLE)
    if(_has_avx2)
        set(_avx2_str "yes")
    else()
        set(_avx2_str "no")
    endif()
    if(_has_avx512f)
        set(_avx512_str "yes")
    else()
        set(_avx512_str "no")
    endif()
    message(STATUS "  AVX2          : ${_avx2_str}")
    message(STATUS "  AVX-512F      : ${_avx512_str}")
endif()
message(STATUS "  --------------------------------")
message(STATUS "")

# ---------------------------------------------------------------------------
# Generate _config.py  (consumed by pysiglib.load_siglib at runtime)
# ---------------------------------------------------------------------------
if(_cuda_enabled)
    set(_cuda_py "True")
else()
    set(_cuda_py "False")
endif()
if(_enable_vec)
    set(_vec_py "True")
else()
    set(_vec_py "False")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_config.py"
"# This file is automatically generated by CMake and should not be edited.\n\
# It contains information about how the package was built.\n\
SYSTEM = '${CMAKE_SYSTEM_NAME}'\n\
BUILT_WITH_CUDA = ${_cuda_py}\n\
BUILT_WITH_AVX = ${_vec_py}\n\
CUDA_VERSION = '${_cuda_version}'\n\
CXX_COMPILER = '${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}'\n\
CMAKE_GENERATOR = '${CMAKE_GENERATOR}'\n")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_config.py" DESTINATION pysiglib)

# ---------------------------------------------------------------------------
# Sub-projects
# ---------------------------------------------------------------------------
add_subdirectory(siglib/cpsig)

if(_cuda_enabled)
    add_subdirectory(siglib/cusig)
endif()
