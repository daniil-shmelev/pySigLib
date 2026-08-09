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

add_executable(cpsig_tests
    cp_test_misc.cpp
    cp_test_signature.cpp
    cp_test_sig_coef.cpp
    cp_test_log_sig.cpp
    cp_test_sig_kernel.cpp
    cp_test_polynomial_sig_kernel.cpp
    cp_test_branched.cpp
    cp_test_branched_sig_coef.cpp
    ../cpsig/words.cpp
    ../cpsig/cp_utils.cpp
)

target_include_directories(cpsig_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpsig"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shared"
)

target_compile_features(cpsig_tests PRIVATE cxx_std_20)
target_compile_definitions(cpsig_tests PRIVATE CPSIG_EXPORTS)
target_link_libraries(cpsig_tests PRIVATE cpsig Threads::Threads GTest::gtest_main)

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

add_custom_command(TARGET cpsig_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:cpsig>
        $<TARGET_FILE_DIR:cpsig_tests>
)
if(WIN32)
    add_custom_command(TARGET cpsig_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:tbb>
            $<TARGET_FILE_DIR:cpsig_tests>
    )
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
