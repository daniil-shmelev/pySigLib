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

add_executable(cusig_tests
    cu_test_batch_grid.cpp
    cu_test_signature.cpp
    cu_test_sig_backprop.cpp
    cu_test_sig_coef.cpp
    cu_test_log_sig.cpp
    cu_test_sig_kernel.cpp
    cu_test_branched.cpp
)

target_include_directories(cusig_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../cusig"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shared"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shared/preparation"
)

target_compile_features(cusig_tests PRIVATE cxx_std_20)
target_link_libraries(cusig_tests PRIVATE
    cusig
    CUDA::cudart
    Threads::Threads
    GTest::gtest_main
)

add_custom_command(TARGET cusig_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:cusig>
        $<TARGET_FILE_DIR:cusig_tests>
)

set(_pysiglib_test_cache_root "${CMAKE_CURRENT_BINARY_DIR}/pysiglib_test_cache")
file(MAKE_DIRECTORY "${_pysiglib_test_cache_root}")
set(_pysiglib_test_environment
    "LOCALAPPDATA=${_pysiglib_test_cache_root}"
    "HOME=${_pysiglib_test_cache_root}"
)

include(GoogleTest)
gtest_discover_tests(cusig_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        LABELS cuda
        ENVIRONMENT "${_pysiglib_test_environment}"
)
