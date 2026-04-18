# CPU-only JAX FFI shared library. Built from the shared source
# pysiglib_jax_ffi.cc with PYSIGLIB_JAX_WITH_CUDA undefined so the existing
# #ifdef guards exclude all CUDA handlers. The CUDA counterpart is built by
# plugins/cuda/CMakeLists.txt from the same source file with the macro set.
add_library(pysiglib_jax_ffi_cpu SHARED
    pysiglib_jax_ffi.cc
)

set_target_properties(pysiglib_jax_ffi_cpu PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    # Resolve libcpsig next to this DLL. Needed by auditwheel on Linux and
    # for the wheel to be importable without setting LD_LIBRARY_PATH.
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
    MACOSX_RPATH TRUE
)

target_include_directories(pysiglib_jax_ffi_cpu PRIVATE
    "${JAXLIB_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpsig"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shared"
)

target_link_libraries(pysiglib_jax_ffi_cpu PRIVATE cpsig)

install(TARGETS pysiglib_jax_ffi_cpu
    LIBRARY DESTINATION pysiglib
    RUNTIME DESTINATION pysiglib
    ARCHIVE DESTINATION pysiglib
)
