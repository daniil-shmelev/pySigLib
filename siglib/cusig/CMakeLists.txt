# Install dest overridden by pysiglib-cuda plugin's CMakeLists.txt.
if(NOT DEFINED CUSIG_INSTALL_DESTINATION)
    set(CUSIG_INSTALL_DESTINATION "pysiglib")
endif()

option(PYSIGLIB_CUDA_LINEINFO "Generate CUDA device line information" ON)
option(PYSIGLIB_CUDA_SEPARABLE_COMPILATION
    "Enable CUDA separable compilation" ON)
set(PYSIGLIB_CUDA_COMPRESS_MODE "default" CACHE STRING
    "CUDA fatbin compression mode")
set_property(CACHE PYSIGLIB_CUDA_COMPRESS_MODE PROPERTY STRINGS
    default size speed balance none)
set(_pysiglib_cuda_compress_modes default size speed balance none)
if(NOT PYSIGLIB_CUDA_COMPRESS_MODE IN_LIST _pysiglib_cuda_compress_modes)
    message(FATAL_ERROR
        "Invalid PYSIGLIB_CUDA_COMPRESS_MODE: ${PYSIGLIB_CUDA_COMPRESS_MODE}")
endif()

if(NOT TARGET pysiglib_preparation)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../shared/preparation"
        "${CMAKE_CURRENT_BINARY_DIR}/preparation"
    )
endif()

set(_pysiglib_broken_cccl_header "")
if(WIN32)
    set(_pysiglib_cuda_include_dirs
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}
    )
    list(REMOVE_DUPLICATES _pysiglib_cuda_include_dirs)
    foreach(_pysiglib_cuda_include_dir IN LISTS _pysiglib_cuda_include_dirs)
        set(_pysiglib_clusterlaunchcontrol_header
            "${_pysiglib_cuda_include_dir}/cuda/__ptx/instructions/generated/clusterlaunchcontrol.h")
        if(EXISTS "${_pysiglib_clusterlaunchcontrol_header}")
            file(STRINGS "${_pysiglib_clusterlaunchcontrol_header}"
                _pysiglib_broken_cccl_lines
                REGEX "reinterpret_cast<long2")
            if(_pysiglib_broken_cccl_lines)
                set(_pysiglib_broken_cccl_header
                    "${_pysiglib_clusterlaunchcontrol_header}")
                break()
            endif()
        endif()
    endforeach()
endif()

set(_pysiglib_cccl_include_dirs "")
if(_pysiglib_broken_cccl_header)
    include(FetchContent)
    FetchContent_Declare(
        pysiglib_cccl
        URL https://github.com/NVIDIA/cccl/archive/refs/tags/v3.4.0.tar.gz
        URL_HASH SHA256=F566AC36E2E4E9F8B85122BAA131E445113B6C1B4D395C94944FE9037B2A6352
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR pysiglib_headers_only
    )
    FetchContent_MakeAvailable(pysiglib_cccl)
    list(APPEND _pysiglib_cccl_include_dirs
        "${pysiglib_cccl_SOURCE_DIR}/thrust"
        "${pysiglib_cccl_SOURCE_DIR}/libcudacxx/include"
        "${pysiglib_cccl_SOURCE_DIR}/cub"
    )
endif()

add_library(cusig SHARED
	cu_branched_log_bch.cu
    cu_branched_log_signature.cu
    cu_branched_signature.cu
    cu_branched_sig_kernel.cu
    cu_exp_host.cpp
    cu_exp_signature.cu
    cache_lifecycle/cu_log_sig_cache.cu
    cu_log_sig_combine.cu
    cu_log_signature.cu
    cu_path_transforms.cu
    cu_polynomial_sig_kernel.cu
    cu_sig_coef.cu
    cu_sig_kernel.cu
    cu_signature.cu
    cu_sig_combine.cu
    cache_lifecycle/cusig_lifecycle.cu
)

set_target_properties(cusig PROPERTIES
    CUDA_STANDARD          20
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD           20
    CUDA_SEPARABLE_COMPILATION ${PYSIGLIB_CUDA_SEPARABLE_COMPILATION}
    CUDA_ARCHITECTURES         "${PYSIGLIB_CUDA_ARCH}"
)

if(_pysiglib_cccl_include_dirs)
    # Affected CCCL headers use 32-bit long for 64-bit PTX operands on Windows.
    target_include_directories(cusig BEFORE PRIVATE ${_pysiglib_cccl_include_dirs})
    if(MSVC)
        target_compile_options(cusig PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        )
    endif()
endif()

target_include_directories(cusig PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shared"
)
target_compile_definitions(cusig PRIVATE CUSIG_EXPORTS NDEBUG)

# nvcc flags: optional device line info and the device optimizer.
target_compile_options(cusig PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${PYSIGLIB_CUDA_LINEINFO}>>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--dopt=on>
)

if(NOT PYSIGLIB_CUDA_COMPRESS_MODE STREQUAL "default")
    target_compile_options(cusig PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=${PYSIGLIB_CUDA_COMPRESS_MODE}>
    )
    if(PYSIGLIB_CUDA_SEPARABLE_COMPILATION)
        target_link_options(cusig PRIVATE
            "$<DEVICE_LINK:--compress-mode=${PYSIGLIB_CUDA_COMPRESS_MODE}>"
        )
    endif()
endif()

if(MSVC)
    target_compile_definitions(cusig PRIVATE _USRDLL _UNICODE UNICODE)
endif()

target_link_libraries(cusig PRIVATE CUDA::cudart pysiglib_preparation)

# cudadevrt is needed for separable compilation (device linking).
if(PYSIGLIB_CUDA_SEPARABLE_COMPILATION)
    if(TARGET CUDA::cudadevrt)
        target_link_libraries(cusig PRIVATE CUDA::cudadevrt)
    else()
        find_library(_cudadevrt cudadevrt PATHS "${CUDAToolkit_LIBRARY_DIR}")
        if(_cudadevrt)
            target_link_libraries(cusig PRIVATE "${_cudadevrt}")
        endif()
    endif()
endif()

if(MSVC)
    # link.exe flags: /OPT:REF drops unused code, /OPT:ICF folds identical
    # COMDATs, and /DEBUG emits a PDB.
    target_link_options(cusig PRIVATE
        "$<HOST_LINK:/OPT:REF>"
        "$<HOST_LINK:/OPT:ICF>"
        "$<HOST_LINK:/DEBUG>"
    )
endif()

# ---- Install ----------------------------------------------------------------
install(TARGETS cusig
    LIBRARY DESTINATION ${CUSIG_INSTALL_DESTINATION}
    RUNTIME DESTINATION ${CUSIG_INSTALL_DESTINATION}   # .dll on Windows
)
