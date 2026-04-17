add_library(pysiglib_jax_ffi SHARED
    pysiglib_jax_ffi.cc
)

set_target_properties(pysiglib_jax_ffi PROPERTIES
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

target_include_directories(pysiglib_jax_ffi PRIVATE
    "${_jax_include_dir}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpsig"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shared"
)

target_link_libraries(pysiglib_jax_ffi PRIVATE cpsig)

install(TARGETS pysiglib_jax_ffi
    LIBRARY DESTINATION pysiglib
    RUNTIME DESTINATION pysiglib
    ARCHIVE DESTINATION pysiglib
)
