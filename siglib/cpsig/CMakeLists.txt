if(NOT TARGET pysiglib_preparation)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../shared/preparation"
        "${CMAKE_CURRENT_BINARY_DIR}/preparation"
    )
endif()

add_library(cpsig SHARED
    multithreading.cpp
    cp_branched_polynomial_sig_kernel.cpp
    cp_bch.cpp
    cache_lifecycle/cp_branched_cache.cpp
    cache_lifecycle/cp_branched_log_cache.cpp
    cp_branched_log_signature.cpp
    cp_branched_sig_coef.cpp
    cache_lifecycle/cp_branched_sig_coef_cache.cpp
    cp_branched_signature.cpp
    cp_branched_sig_kernel.cpp
    cp_log_signatures.cpp
    cp_path_transforms.cpp
    cp_polynomial_sig_kernel.cpp
    cp_sig_coef.cpp
    cp_sig_lasso.cpp
    cp_sig_kernel.cpp
    cp_signature.cpp
    cp_utils.cpp
    cp_sig_combine.cpp
    cp_sig_join.cpp
    cpsig.cpp
    cache_lifecycle/cp_bch_cache.cpp
    cache_lifecycle/cp_poly_cache.cpp
    cache_lifecycle/log_sig_cache.cpp
)

include(FetchContent)

set(TBB_TEST OFF CACHE BOOL "" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TBB_STRICT OFF CACHE BOOL "" FORCE)
set(TBBMALLOC_BUILD OFF CACHE BOOL "" FORCE)
set(TBBMALLOC_PROXY_BUILD OFF CACHE BOOL "" FORCE)
set(TBB_INSTALL OFF CACHE BOOL "" FORCE)
set(_pysiglib_build_shared_libs "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS ON)
FetchContent_Declare(
    oneTBB
    GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
    GIT_TAG 2fea40755cccad7eb8b0ee0c07ed0c4e75bf562c
)
FetchContent_MakeAvailable(oneTBB)
set(BUILD_SHARED_LIBS "${_pysiglib_build_shared_libs}")
unset(_pysiglib_build_shared_libs)

set_target_properties(cpsig PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_precompile_headers(cpsig PRIVATE cppch.h)
if(APPLE)
    set_target_properties(cpsig PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_target_properties(cpsig PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
target_include_directories(cpsig PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shared"
)
target_compile_definitions(cpsig PRIVATE CPSIG_EXPORTS)

target_link_libraries(cpsig
    PRIVATE Threads::Threads
    PRIVATE pysiglib_preparation
    PUBLIC TBB::tbb
)

if(WIN32)
    add_custom_command(TARGET cpsig POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:tbb>
            $<TARGET_FILE_DIR:cpsig>
    )
endif()

if(MSVC)
    target_compile_options(cpsig PRIVATE /Zi)
    target_link_options(cpsig PRIVATE /DEBUG /OPT:REF /OPT:ICF)
endif()

# ---- Highway ---------------------------------------------------------------
option(PYSIGLIB_AVX "Enable runtime CPU SIMD dispatch (SSE2/AVX2/NEON)" ON)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/Highway.cmake")
target_link_libraries(cpsig PRIVATE hwy)
if(PYSIGLIB_TESTS)
    target_sources(cpsig PRIVATE cp_dispatch_test.cpp)
    if(MSVC)
        target_link_options(cpsig PRIVATE "/MAP:${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/cpsig.map")
    endif()
endif()

set(_cpsig_kernels
    cp_bch_impl.cpp
    cp_branched_log_signature_impl.cpp
    cp_branched_polynomial_sig_kernel_impl.cpp
    cp_branched_sig_coef_impl.cpp
    cp_branched_sig_kernel_impl.cpp
    cp_branched_signature_impl.cpp
    cp_log_signatures_impl.cpp
    cp_path_transforms_impl.cpp
    cp_polynomial_sig_kernel_impl.cpp
    cp_sig_coef_impl.cpp
    cp_sig_lasso_impl.cpp
    cp_sig_combine_impl.cpp
    cp_sig_join_impl.cpp
    cp_sig_kernel_impl.cpp
    cp_signature_impl.cpp
)
if(NOT PYSIGLIB_AVX)
    target_compile_definitions(cpsig PRIVATE CPSIG_SCALAR_ONLY)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(_cpsig_targets baseline)
    if(PYSIGLIB_AVX AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
        list(APPEND _cpsig_targets avx2)
    endif()
    foreach(_target IN LISTS _cpsig_targets)
        add_library(cpsig_${_target} OBJECT ${_cpsig_kernels})
        target_compile_definitions(cpsig_${_target} PRIVATE CPSIG_STATIC_TARGET CPSIG_EXPORTS)
        if(NOT PYSIGLIB_AVX)
            target_compile_definitions(cpsig_${_target} PRIVATE CPSIG_SCALAR_ONLY)
        endif()
        target_include_directories(cpsig_${_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../shared")
        target_link_libraries(cpsig_${_target} PRIVATE hwy TBB::tbb pysiglib_preparation)
        target_compile_options(cpsig_${_target} PRIVATE /Zi)
        target_precompile_headers(cpsig_${_target} PRIVATE cppch.h)
        if(_target STREQUAL "avx2")
            target_compile_options(cpsig_${_target} PRIVATE /arch:AVX2)
        endif()
        target_sources(cpsig PRIVATE $<TARGET_OBJECTS:cpsig_${_target}>)
    endforeach()
else()
    target_sources(cpsig PRIVATE ${_cpsig_kernels})
endif()

# ---- Install ----------------------------------------------------------------
# CPSIG_SKIP_INSTALL lets dependents (e.g. the pysiglib-cuda plugin, which
# needs cpsig at link time but must not ship it in its own wheel) add cpsig
# via add_subdirectory() without pulling the library into their install set.
if(NOT CPSIG_SKIP_INSTALL)
    install(TARGETS cpsig tbb
        LIBRARY DESTINATION pysiglib
        RUNTIME DESTINATION pysiglib   # .dll on Windows
    )
    install(FILES "${highway_SOURCE_DIR}/LICENSE"
        DESTINATION pysiglib/licenses
        RENAME Highway-LICENSE.txt
    )
    install(FILES "${onetbb_SOURCE_DIR}/LICENSE.txt"
        DESTINATION pysiglib/licenses
        RENAME oneTBB-LICENSE.txt
    )
    install(FILES "${onetbb_SOURCE_DIR}/third-party-programs.txt"
        DESTINATION pysiglib/licenses
    )
endif()
