add_library(pysiglib_preparation STATIC
    log_sig/lyndon_words.cpp
    log_sig/log_sig_cache.cpp
    log_sig/bch_cache.cpp
    branched_sig/branched_log_sig_cache.cpp
    branched_sig/branched_sig_cache_io.cpp
    branched_sig/branched_sig_cache.cpp
    branched_sig/branched_sig_coef_cache.cpp
)

# Harald Hofstatter's BCH implementation has no CMake project of its own.
# Build only the library sources needed for the ordinary BCH series; in
# particular, this deliberately excludes its parser and command-line program.
# Upstream uses C variable-length arrays and __int128, so MSVC keeps the
# existing project-local fallback.
if(NOT MSVC)
    include(FetchContent)
    FetchContent_Declare(
        hofstaetter_bch
        GIT_REPOSITORY https://github.com/HaraldHofstaetter/BCH.git
        GIT_TAG d84ed5d3b77ce14c637f30963972ee6719ee2857
    )
    FetchContent_MakeAvailable(hofstaetter_bch)

    add_library(pysiglib_bch STATIC
        "${hofstaetter_bch_SOURCE_DIR}/phi.c"
        "${hofstaetter_bch_SOURCE_DIR}/expr.c"
        "${hofstaetter_bch_SOURCE_DIR}/lie_series.c"
        "${hofstaetter_bch_SOURCE_DIR}/lyndon.c"
        "${hofstaetter_bch_SOURCE_DIR}/rightnormed.c"
        "${hofstaetter_bch_SOURCE_DIR}/goldberg.c"
        "${hofstaetter_bch_SOURCE_DIR}/convert_lyndon.c"
        "${hofstaetter_bch_SOURCE_DIR}/convert_rightnormed.c"
        "${hofstaetter_bch_SOURCE_DIR}/convert_hall.c"
    )
    set_property(TARGET pysiglib_bch PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_include_directories(pysiglib_bch PUBLIC "${hofstaetter_bch_SOURCE_DIR}")
    if(APPLE)
        target_include_directories(pysiglib_bch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/apple_compat")
    endif()
    if(_enable_vec)
        if(APPLE)
            target_compile_options(pysiglib_bch PRIVATE -ftree-vectorize)
        elseif(DEFINED ENV{CIBUILDWHEEL})
            target_compile_options(pysiglib_bch PRIVATE -march=x86-64-v3 -ftree-vectorize)
        else()
            # Its fixed-width integer kernel is slower with AVX-512 code generation.
            target_compile_options(pysiglib_bch PRIVATE -march=native -mno-avx512f -ftree-vectorize)
        endif()
    endif()

    target_sources(pysiglib_preparation PRIVATE log_sig/bch_formula.cpp)
    target_compile_definitions(pysiglib_preparation PRIVATE PYSIGLIB_USE_UPSTREAM_BCH=1)
    target_link_libraries(pysiglib_preparation PRIVATE pysiglib_bch)
endif()

set_target_properties(pysiglib_preparation PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE ON
)
target_precompile_headers(pysiglib_preparation PRIVATE preparation_pch.h)

target_include_directories(pysiglib_preparation PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
