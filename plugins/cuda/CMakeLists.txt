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

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(PYSIGLIB_CUDA_ARCH "native" CACHE STRING
    "CUDA architectures (native, all, all-major, or semicolon-separated list)")

set(CUSIG_INSTALL_DESTINATION "pysiglib_cuda" CACHE INTERNAL "")

# Binary dir override is required because siglib/cusig is outside this project.
add_subdirectory(
    "${CMAKE_SOURCE_DIR}/../../siglib/cusig"
    "${CMAKE_BINARY_DIR}/cusig_build"
)

message(STATUS "")
message(STATUS "  ---- pysiglib-cuda build summary ----")
message(STATUS "  System        : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  CXX compiler  : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA compiler : ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA version  : ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA arch     : ${PYSIGLIB_CUDA_ARCH}")
message(STATUS "  Install dest  : ${CUSIG_INSTALL_DESTINATION}/")
message(STATUS "  -------------------------------------")
message(STATUS "")
