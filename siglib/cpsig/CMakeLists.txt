add_library(cpsig SHARED
    cp_bch.cpp
    cp_branched_cache.cpp
    cp_branched_log_signature.cpp
    cp_branched_signature.cpp
    cp_branched_sig_kernel.cpp
    cp_log_signatures.cpp
    cp_path_transforms.cpp
    cp_sig_coef.cpp
    cp_sig_kernel.cpp
    cp_signature.cpp
    cp_utils.cpp
    cp_sig_combine.cpp
    cp_sig_join.cpp
    cppch.cpp
    cpsig.cpp
    log_sig_cache.cpp
    words.cpp
)

set_target_properties(cpsig PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_include_directories(cpsig PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../shared")
target_compile_definitions(cpsig PRIVATE CPSIG_EXPORTS)

target_link_libraries(cpsig PRIVATE Threads::Threads)

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
    install(TARGETS cpsig
        LIBRARY DESTINATION pysiglib
        RUNTIME DESTINATION pysiglib   # .dll on Windows
    )
endif()
