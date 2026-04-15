# Install dest overridden by pysiglib-cuda plugin's CMakeLists.txt.
if(NOT DEFINED CUSIG_INSTALL_DESTINATION)
    set(CUSIG_INSTALL_DESTINATION "pysiglib")
endif()

add_library(cusig SHARED
    cu_branched_signature.cu
    cu_exp_host.cpp
    cu_exp_signature.cu
    cu_log_sig_cache.cu
    cu_log_sig_combine.cu
    cu_log_signature.cu
    cu_path_transforms.cu
    cu_sig_coef.cu
    cu_sig_kernel.cu
    cu_signature.cu
    cu_sig_combine.cu
    cusig_lifecycle.cu
    cupch.cpp
)

set_target_properties(cusig PROPERTIES
    CUDA_STANDARD 17
    CXX_STANDARD  20
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES         "${PYSIGLIB_CUDA_ARCH}"
)

target_include_directories(cusig PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../shared")
target_compile_definitions(cusig PRIVATE CUSIG_EXPORTS NDEBUG)

# ---------------------------------------------------------------------------
# Match old build_cusig.bat nvcc flags:
#   -lineinfo   (debug line info in device code)
#   --dopt=on   (device code optimizer)
# ---------------------------------------------------------------------------
target_compile_options(cusig PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--dopt=on>
)

# ---------------------------------------------------------------------------
# Match old build_cusig.bat cl.exe flags for cupch.cpp:
#   /Oi  — enable intrinsic functions
#   /GL  — whole-program optimization (requires /LTCG at link)
#   /Gy  — function-level linking (enables /OPT:REF)
#   /sdl — additional security development checks
#   /Zi  — generate PDB debug info
#   /permissive- — strict C++ conformance
# ---------------------------------------------------------------------------
if(MSVC)
    set_source_files_properties(cupch.cpp PROPERTIES
        COMPILE_FLAGS "/Oi /GL /Gy /sdl /Zi /permissive-"
    )

    # Old build defines (Unicode mode, not MBCS)
    target_compile_definitions(cusig PRIVATE _USRDLL _UNICODE UNICODE)
endif()

# ---------------------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------------------
target_link_libraries(cusig PRIVATE CUDA::cudart)

# cudadevrt is needed for separable compilation (device linking).
if(TARGET CUDA::cudadevrt)
    target_link_libraries(cusig PRIVATE CUDA::cudadevrt)
else()
    find_library(_cudadevrt cudadevrt PATHS "${CUDAToolkit_LIBRARY_DIR}")
    if(_cudadevrt)
        target_link_libraries(cusig PRIVATE "${_cudadevrt}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Match old build_cusig.bat link.exe flags:
#   /LTCG:incremental — link-time code generation (pairs with /GL)
#   /OPT:REF          — eliminate unreferenced functions/data
#   /OPT:ICF          — fold identical COMDATs
#   /DEBUG             — generate PDB
# ---------------------------------------------------------------------------
if(MSVC)
    target_link_options(cusig PRIVATE
        /LTCG:incremental
        /OPT:REF
        /OPT:ICF
        /DEBUG
    )
endif()

# ---- Install ----------------------------------------------------------------
install(TARGETS cusig
    LIBRARY DESTINATION ${CUSIG_INSTALL_DESTINATION}
    RUNTIME DESTINATION ${CUSIG_INSTALL_DESTINATION}   # .dll on Windows
)
