add_library(cpsig SHARED
    cp_bch.cpp
    cp_branched_cache.cpp
    cp_branched_signature.cpp
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

target_compile_features(cpsig PRIVATE cxx_std_20)
target_include_directories(cpsig PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../shared")
target_compile_definitions(cpsig PRIVATE CPSIG_EXPORTS)

# B2 uses <threading>multi — on Linux this adds -pthread
target_link_libraries(cpsig PRIVATE Threads::Threads)

# ---- Vectorization ----------------------------------------------------------
if(_enable_vec)
    target_compile_definitions(cpsig PRIVATE VEC)

    if(MSVC)
        if(_has_avx512f)
            target_compile_options(cpsig PRIVATE /arch:AVX512)
        elseif(_has_avx2)
            target_compile_options(cpsig PRIVATE /arch:AVX2)
        elseif(_has_avx)
            target_compile_options(cpsig PRIVATE /arch:AVX)
        endif()
        # Match B2: <toolset>msvc:<cxxflags>"/Qvec-report:2"
        target_compile_options(cpsig PRIVATE /Qvec-report:2)
    elseif(APPLE)
        # Don't use -march=native on Apple: clang resolves "native" against the
        # host CPU (apple-mN), and not every clang version in the Xcode toolchain
        # recognises the newest Apple CPU names (e.g. apple-m3), which can break
        # the build with "unknown target CPU" errors. NEON is baseline on arm64
        # Apple Silicon, so -ftree-vectorize alone is sufficient.
        target_compile_options(cpsig PRIVATE -ftree-vectorize)
    else()
        # Under cibuildwheel (CI wheel builds), target the portable AVX2
        # baseline (x86-64-v3 = Haswell 2013+). -march=native would emit
        # AVX-512 whenever the build runner happens to have it, then crash
        # end users with SIGILL on CPUs that don't. For source installs,
        # -march=native still applies so local developers get maximum perf
        # on their own hardware.
        if(DEFINED ENV{CIBUILDWHEEL})
            target_compile_options(cpsig PRIVATE -march=x86-64-v3 -ftree-vectorize)
        else()
            target_compile_options(cpsig PRIVATE -march=native -ftree-vectorize)
        endif()
        # -fopt-info-vec-missed is GCC-only; Clang uses -Rpass-missed=loop-vectorize
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(cpsig PRIVATE -fopt-info-vec-missed)
        endif()
    endif()
endif()

# ---- Install ----------------------------------------------------------------
install(TARGETS cpsig
    LIBRARY DESTINATION pysiglib
    RUNTIME DESTINATION pysiglib   # .dll on Windows
)
