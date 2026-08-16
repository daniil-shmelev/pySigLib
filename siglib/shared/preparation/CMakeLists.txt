add_library(pysiglib_preparation STATIC
    lyndon_words.cpp
    log_sig_cache.cpp
    bch_cache.cpp
    branched_log_sig_cache.cpp
    branched_sig_cache_io.cpp
    branched_sig_cache.cpp
    branched_sig_coef_cache.cpp
)

set_target_properties(pysiglib_preparation PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(pysiglib_preparation PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
