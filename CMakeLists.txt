cmake_minimum_required(VERSION 3.24)
project(pysiglib LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global C++ toolchain requirements
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Error for old GNU toolchains.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
    message(FATAL_ERROR
        "\n"
        "pysiglib requires GCC/G++ >= 10 for source builds.\n"
        "Detected: ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_VERSION})\n"
        "\n"
        "Please use a newer compiler, for example:\n"
        "  CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 uv pip install -e . --no-build-isolation --reinstall -v\n"
    )
endif()

include(CheckLanguage)
find_package(Python COMPONENTS Interpreter REQUIRED)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(PYSIGLIB_AVX  "Enable AVX/NEON vectorization"      ON)
option(PYSIGLIB_JAX_FFI "Build JAX FFI support when headers are available" ON)
option(PYSIGLIB_BENCHMARKS "Build CodSpeed / Google Benchmark harness" OFF)
option(PYSIGLIB_TESTS "Build C++ unit tests (Google Test)" OFF)

# Static CRT (/MT) on MSVC, to match the original B2 build.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# AVX / vectorization detection
# ---------------------------------------------------------------------------
set(_enable_vec FALSE)
set(_has_avx 0)
set(_has_avx2 0)
set(_has_avx512f 0)
set(JAXLIB_FOUND FALSE)
set(JAXLIB_INCLUDE_DIR "")
set(JAXLIB_PROBE_LOG "")

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

# ---------------------------------------------------------------------------
# JAX FFI detection (optional). Probe the Python env for jaxlib's XLA FFI
# headers; enable the JAX FFI subproject if found. Probe logic lives in
# cmake/FindJaxlib.cmake so the plugins/cuda build can reuse it.
# ---------------------------------------------------------------------------
if(PYSIGLIB_JAX_FFI)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindJaxlib.cmake)
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
message(STATUS "  CUDA          : extracted to pysiglib-cuda plugin (install via pip install pysiglib[cuda])")
message(STATUS "  JAX FFI opt   : ${PYSIGLIB_JAX_FFI}")
message(STATUS "  JAX FFI       : ${JAXLIB_FOUND}")
if(JAXLIB_FOUND)
    message(STATUS "  JAX include   : ${JAXLIB_INCLUDE_DIR}")
endif()
if(PYSIGLIB_JAX_FFI AND JAXLIB_PROBE_LOG)
    message(STATUS "  JAX probe log :")
    # Print each line of the probe log as a separate STATUS message
    string(REPLACE "\n" ";" _jax_log_lines "${JAXLIB_PROBE_LOG}")
    foreach(_line IN LISTS _jax_log_lines)
        message(STATUS "  ${_line}")
    endforeach()
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
if(_enable_vec)
    set(_vec_py "True")
else()
    set(_vec_py "False")
endif()
if(JAXLIB_FOUND)
    set(_jax_py "True")
else()
    set(_jax_py "False")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_config.py"
"# This file is automatically generated by CMake and should not be edited.\n\
# It contains information about how the package was built.\n\
SYSTEM = '${CMAKE_SYSTEM_NAME}'\n\
BUILT_WITH_JAX_FFI = ${_jax_py}\n\
BUILT_WITH_AVX = ${_vec_py}\n\
CXX_COMPILER = '${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}'\n\
CMAKE_GENERATOR = '${CMAKE_GENERATOR}'\n")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_config.py" DESTINATION pysiglib)

# ---------------------------------------------------------------------------
# Sub-projects
# ---------------------------------------------------------------------------
add_subdirectory(siglib/cpsig)

if(JAXLIB_FOUND)
    add_subdirectory(siglib/jax_ffi)
endif()

if(PYSIGLIB_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(PYSIGLIB_TESTS)
    enable_testing()
    add_subdirectory(siglib/cpsig_unit_testing)
endif()
