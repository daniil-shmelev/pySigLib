# ---------------------------------------------------------------------------
# CodSpeed / Google Benchmark harness for cpsig
# ---------------------------------------------------------------------------
include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/CodSpeedHQ/codspeed-cpp
    SOURCE_SUBDIR  google_benchmark
    GIT_TAG        c856e49567da7fff740518cb34ce93c3f9d4f82c
)
FetchContent_MakeAvailable(google_benchmark)

add_executable(bench_cpsig bench_cpsig.cpp)
target_link_libraries(bench_cpsig PRIVATE benchmark::benchmark cpsig)
target_include_directories(bench_cpsig PRIVATE
    "${CMAKE_SOURCE_DIR}/siglib/cpsig"
    "${CMAKE_SOURCE_DIR}/siglib/shared"
)
target_compile_features(bench_cpsig PRIVATE cxx_std_20)

if(WIN32)
    add_custom_command(TARGET bench_cpsig POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:tbb>
            $<TARGET_FILE_DIR:bench_cpsig>
    )
endif()
