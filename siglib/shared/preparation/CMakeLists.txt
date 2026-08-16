add_library(pysiglib_preparation STATIC
    log_sig/lyndon_words.cpp
    log_sig/log_sig_cache.cpp
    log_sig/bch_cache.cpp
    branched_sig/branched_log_sig_cache.cpp
    branched_sig/branched_sig_cache_io.cpp
    branched_sig/branched_sig_cache.cpp
    branched_sig/branched_sig_cache_common.cpp
    branched_sig/branched_sig_cache_bck.cpp
    branched_sig/branched_sig_cache_mkw.cpp
    branched_sig/branched_sig_coef_cache.cpp
)

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
