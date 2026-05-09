cmake_minimum_required(VERSION 3.24)
project(pysiglib_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
    message(FATAL_ERROR
        "pysiglib-cuda requires GCC/G++ >= 10. Detected: "
        "${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_VERSION})")
endif()

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(PYSIGLIB_CUDA_ARCH "native" CACHE STRING
    "CUDA architectures (native, all, all-major, or semicolon-separated list)")
option(PYSIGLIB_TESTS "Build CUDA C++ unit tests (Google Test)" OFF)

set(CUSIG_INSTALL_DESTINATION "pysiglib_cuda" CACHE INTERNAL "")

# Binary dir override is required because siglib/cusig is outside this project.
add_subdirectory(
    "${CMAKE_SOURCE_DIR}/../../siglib/cusig"
    "${CMAKE_BINARY_DIR}/cusig_build"
)

if(PYSIGLIB_TESTS)
    enable_testing()
    add_subdirectory(
        "${CMAKE_SOURCE_DIR}/../../siglib/cusig_unit_testing"
        "${CMAKE_BINARY_DIR}/cusig_unit_testing"
    )
endif()

# ---------------------------------------------------------------------------
# JAX FFI CUDA library. Built from the shared pysiglib_jax_ffi.cc source with
# PYSIGLIB_JAX_WITH_CUDA defined, so it exposes the CUDA FFI handlers that
# jax's XLA runtime registers when targeting platform="CUDA". The CPU handlers
# are also compiled in (a side-effect of sharing the source with the base
# wheel's CPU-only build); they are redundant here because pysiglib's Python
# loader pulls CPU handlers from the base wheel's pysiglib_jax_ffi_cpu.dll.
# Skipped if jaxlib headers are not present so the plugin still builds for
# users who don't need JAX.
# ---------------------------------------------------------------------------
include("${CMAKE_SOURCE_DIR}/../../cmake/FindJaxlib.cmake")

set(_jax_ffi_cuda_enabled FALSE)
if(JAXLIB_FOUND)
    # cpsig is the CPU core shipped by the base pysiglib wheel. The plugin
    # wheel must not install another copy; CPSIG_SKIP_INSTALL prevents cpsig's
    # CMakeLists from registering its install() rule. The build still produces
    # the cpsig.lib / libcpsig.so that pysiglib_jax_ffi_cuda links against;
    # runtime resolution falls back to the base wheel's cpsig that
    # pysiglib.load_siglib pre-loads.
    set(CPSIG_SKIP_INSTALL TRUE)
    add_subdirectory(
        "${CMAKE_SOURCE_DIR}/../../siglib/cpsig"
        "${CMAKE_BINARY_DIR}/cpsig_build_for_linkage"
        EXCLUDE_FROM_ALL
    )

    add_library(pysiglib_jax_ffi_cuda SHARED
        "${CMAKE_SOURCE_DIR}/../../siglib/jax_ffi/pysiglib_jax_ffi.cc"
    )
    target_compile_definitions(pysiglib_jax_ffi_cuda PRIVATE PYSIGLIB_JAX_WITH_CUDA)
    set_target_properties(pysiglib_jax_ffi_cuda PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
        MACOSX_RPATH TRUE
    )
    target_include_directories(pysiglib_jax_ffi_cuda PRIVATE
        "${JAXLIB_INCLUDE_DIR}"
        "${CMAKE_SOURCE_DIR}/../../siglib/jax_ffi"
        "${CMAKE_SOURCE_DIR}/../../siglib/cpsig"
        "${CMAKE_SOURCE_DIR}/../../siglib/cusig"
        "${CMAKE_SOURCE_DIR}/../../siglib/shared"
    )
    target_link_libraries(pysiglib_jax_ffi_cuda PRIVATE cpsig cusig CUDA::cudart)
    install(TARGETS pysiglib_jax_ffi_cuda
        LIBRARY DESTINATION ${CUSIG_INSTALL_DESTINATION}
        RUNTIME DESTINATION ${CUSIG_INSTALL_DESTINATION}
        ARCHIVE DESTINATION ${CUSIG_INSTALL_DESTINATION}
    )
    set(_jax_ffi_cuda_enabled TRUE)

    # Plugin-side config marker, read by pysiglib/jax_api/_ffi.py to decide
    # whether to attempt loading the CUDA FFI library.
    configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/_config.py.in"
        "${CMAKE_BINARY_DIR}/_config.py"
        @ONLY
    )
    install(FILES "${CMAKE_BINARY_DIR}/_config.py"
            DESTINATION ${CUSIG_INSTALL_DESTINATION})
endif()

message(STATUS "")
message(STATUS "  ---- pysiglib-cuda build summary ----")
message(STATUS "  System        : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  CXX compiler  : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA compiler : ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA version  : ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA arch     : ${PYSIGLIB_CUDA_ARCH}")
message(STATUS "  JAX FFI CUDA  : ${_jax_ffi_cuda_enabled}")
if(_jax_ffi_cuda_enabled)
    message(STATUS "  JAX include   : ${JAXLIB_INCLUDE_DIR}")
endif()
message(STATUS "  Install dest  : ${CUSIG_INSTALL_DESTINATION}/")
message(STATUS "  -------------------------------------")
message(STATUS "")
