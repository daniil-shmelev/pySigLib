add_library(pysiglib_jax_ffi SHARED
    pysiglib_jax_ffi.cc
)

set_target_properties(pysiglib_jax_ffi PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_include_directories(pysiglib_jax_ffi PRIVATE
    "${_jax_include_dir}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpsig"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shared"
)

target_link_libraries(pysiglib_jax_ffi PRIVATE cpsig)

if(_cuda_enabled)
    target_compile_definitions(pysiglib_jax_ffi PRIVATE PYSIGLIB_JAX_WITH_CUDA)
    target_include_directories(pysiglib_jax_ffi PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../cusig")
    target_link_libraries(pysiglib_jax_ffi PRIVATE cusig CUDA::cudart)
endif()

install(TARGETS pysiglib_jax_ffi
    LIBRARY DESTINATION pysiglib
    RUNTIME DESTINATION pysiglib
    ARCHIVE DESTINATION pysiglib
)
