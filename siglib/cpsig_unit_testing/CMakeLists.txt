if(NOT TARGET GTest::gtest_main)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.16.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

set(_pysiglib_cpsig_test_sources
    cp_test_misc.cpp
    cp_test_signature.cpp
    cp_test_sig_coef.cpp
    cp_test_log_sig.cpp
    cp_test_sig_kernel.cpp
    cp_test_polynomial_sig_kernel.cpp
    cp_test_preparation.cpp
    cp_test_branched.cpp
    cp_test_branched_sig_coef.cpp
)
if(NOT PYSIGLIB_CPSIG_PREBUILT_LIBRARY)
    list(APPEND _pysiglib_cpsig_test_sources ../cpsig/cp_utils.cpp)
endif()
add_executable(cpsig_tests ${_pysiglib_cpsig_test_sources})

target_include_directories(cpsig_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpsig"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shared"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shared/preparation"
)

target_compile_features(cpsig_tests PRIVATE cxx_std_20)
if(NOT PYSIGLIB_CPSIG_PREBUILT_LIBRARY)
    target_compile_definitions(cpsig_tests PRIVATE CPSIG_EXPORTS)
else()
    set(TBB_TEST OFF CACHE BOOL "" FORCE)
    set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(TBB_STRICT OFF CACHE BOOL "" FORCE)
    set(TBBMALLOC_BUILD OFF CACHE BOOL "" FORCE)
    set(TBBMALLOC_PROXY_BUILD OFF CACHE BOOL "" FORCE)
    set(TBB_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        pysiglib_test_onetbb
        GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
        GIT_TAG 2fea40755cccad7eb8b0ee0c07ed0c4e75bf562c
    )
    FetchContent_MakeAvailable(pysiglib_test_onetbb)
    target_include_directories(cpsig_tests PRIVATE
        "${pysiglib_test_onetbb_SOURCE_DIR}/include"
    )
endif()
target_link_libraries(cpsig_tests PRIVATE cpsig pysiglib_preparation Threads::Threads GTest::gtest_main)

if(_enable_vec)
    target_compile_definitions(cpsig_tests PRIVATE VEC)
    if(MSVC)
        if(_has_avx2)
            target_compile_options(cpsig_tests PRIVATE /arch:AVX2)
        elseif(_has_avx)
            target_compile_options(cpsig_tests PRIVATE /arch:AVX)
        endif()
    elseif(NOT APPLE)
        target_compile_options(cpsig_tests PRIVATE -march=native -ftree-vectorize)
    else()
        target_compile_options(cpsig_tests PRIVATE -ftree-vectorize)
    endif()
endif()

if(WIN32)
    add_custom_command(TARGET cpsig_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:cpsig>
            $<TARGET_FILE_DIR:cpsig_tests>
    )
    if(PYSIGLIB_CPSIG_PREBUILT_LIBRARY)
        add_custom_command(TARGET cpsig_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${PYSIGLIB_CPSIG_PREBUILT_TBB_LIBRARY}"
                $<TARGET_FILE_DIR:cpsig_tests>
        )
    else()
        add_custom_command(TARGET cpsig_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:tbb>
                $<TARGET_FILE_DIR:cpsig_tests>
        )
    endif()
endif()

set(_pysiglib_test_cache_root "${CMAKE_CURRENT_BINARY_DIR}/pysiglib_test_cache")
file(MAKE_DIRECTORY "${_pysiglib_test_cache_root}")
set(_pysiglib_test_environment
    "LOCALAPPDATA=${_pysiglib_test_cache_root}"
    "HOME=${_pysiglib_test_cache_root}"
)

include(GoogleTest)
gtest_discover_tests(cpsig_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        ENVIRONMENT "${_pysiglib_test_environment}"
)
