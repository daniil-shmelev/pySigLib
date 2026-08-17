if(NOT TARGET pysiglib_preparation)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../shared/preparation"
        "${CMAKE_CURRENT_BINARY_DIR}/preparation"
    )
endif()

add_library(cpsig SHARED
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
    cp_sig_kernel.cpp
    cp_signature.cpp
    cp_utils.cpp
    cp_sig_combine.cpp
    cp_sig_join.cpp
    cppch.cpp
    cpsig.cpp
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

# ---- Vectorization ----------------------------------------------------------
# Under cibuildwheel, force a portable AVX2 baseline so wheels don't embed
# AVX-512 (built on a CPU that has it) and then SIGILL on end-user CPUs that
# don't. Local source installs get -march=native for max perf.
if(_enable_vec)
    target_compile_definitions(cpsig PRIVATE VEC)

    if(MSVC)
        if(DEFINED ENV{CIBUILDWHEEL})
            target_compile_options(cpsig PRIVATE /arch:AVX2)
        elseif(_has_avx512f)
            target_compile_options(cpsig PRIVATE /arch:AVX512)
        elseif(_has_avx2)
            target_compile_options(cpsig PRIVATE /arch:AVX2)
        elseif(_has_avx)
            target_compile_options(cpsig PRIVATE /arch:AVX)
        endif()
        target_compile_options(cpsig PRIVATE /Qvec-report:2)
    elseif(APPLE)
        # No -march=native: some clang versions don't recognise the newest
        # apple-mN CPU names and fail with "unknown target CPU".
        target_compile_options(cpsig PRIVATE -ftree-vectorize)
    else()
        if(DEFINED ENV{CIBUILDWHEEL})
            target_compile_options(cpsig PRIVATE -march=x86-64-v3 -ftree-vectorize)
        else()
            target_compile_options(cpsig PRIVATE -march=native -ftree-vectorize)
        endif()
    endif()
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
    install(FILES "${onetbb_SOURCE_DIR}/LICENSE.txt"
        DESTINATION pysiglib/licenses
        RENAME oneTBB-LICENSE.txt
    )
    install(FILES "${onetbb_SOURCE_DIR}/third-party-programs.txt"
        DESTINATION pysiglib/licenses
    )
endif()
