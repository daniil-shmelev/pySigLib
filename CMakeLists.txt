cmake_minimum_required(VERSION 3.24)
project(pysiglib LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global C++ toolchain requirements
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Error for old GNU toolchains.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
    message(FATAL_ERROR
        "\n"
        "pysiglib requires GCC/G++ >= 10 for source builds.\n"
        "Detected: ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_VERSION})\n"
        "\n"
        "Please use a newer compiler, for example:\n"
        "  CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 uv pip install -e . --no-build-isolation --reinstall -v\n"
    )
endif()

include(CheckLanguage)
find_package(Python COMPONENTS Interpreter REQUIRED)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(PYSIGLIB_AVX  "Enable AVX/NEON vectorization"      ON)
option(PYSIGLIB_JAX_FFI "Build JAX FFI support when headers are available" ON)
option(PYSIGLIB_BENCHMARKS "Build CodSpeed / Google Benchmark harness" OFF)

# Static CRT (/MT) on MSVC, to match the original B2 build.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# AVX / vectorization detection
# ---------------------------------------------------------------------------
set(_enable_vec FALSE)
set(_has_avx 0)
set(_has_avx2 0)
set(_has_avx512f 0)
set(_jax_ffi_enabled FALSE)
set(_jax_include_dir "")

if(PYSIGLIB_AVX)
    if(APPLE)
        # ARM NEON is always available on Apple Silicon; on Intel Macs
        # -march=native enables whatever the CPU supports.
        set(_enable_vec TRUE)
    else()
        # Compile and run the existing avx_info utility to probe the CPU.
        try_run(_avx_run_result _avx_compile_result
            "${CMAKE_CURRENT_BINARY_DIR}/_avx_detect"
            "${CMAKE_CURRENT_SOURCE_DIR}/avx_info/avx_info.cpp"
        )
        if(_avx_compile_result AND NOT _avx_run_result STREQUAL "FAILED_TO_RUN")
            math(EXPR _has_avx     "${_avx_run_result} & 1")
            math(EXPR _has_avx2    "${_avx_run_result} & 2")
            math(EXPR _has_avx512f "${_avx_run_result} & 4")
            if(_has_avx2)
                set(_enable_vec TRUE)
            else()
                set(PYSIGLIB_AVX OFF)
            endif()
        else()
            message(WARNING "AVX detection failed — disabling vectorization")
            set(PYSIGLIB_AVX OFF)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# JAX FFI detection (optional). Probe the Python env for jaxlib's XLA FFI
# headers; enable the JAX FFI subproject if found.
# ---------------------------------------------------------------------------
if(PYSIGLIB_JAX_FFI)
    set(_jax_probe_script [=[
import importlib.util
import importlib.metadata
import pathlib
import site
import sys

log = []
MIN_JAXLIB = (0, 9, 1)

# Check jaxlib version first - older versions have incompatible XLA FFI headers
def get_jaxlib_version():
    """Return jaxlib version tuple, or None if not found."""
    for finder in (
        lambda: importlib.metadata.version("jaxlib"),
        lambda: _version_from_site_packages(),
    ):
        try:
            ver = finder()
            if ver:
                parts = tuple(int(x) for x in ver.split(".")[:3])
                return parts
        except Exception:
            pass
    return None

def _version_from_site_packages():
    """Try to read jaxlib version from dist-info in site-packages."""
    try:
        sp = site.getsitepackages()
    except AttributeError:
        sp = []
    try:
        sp.append(site.getusersitepackages())
    except AttributeError:
        pass
    # Also check base Python's site-packages
    base = pathlib.Path(sys.base_prefix) / "Lib" / "site-packages"
    if base.is_dir() and str(base) not in sp:
        sp.append(str(base))
    base_posix = pathlib.Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if base_posix.is_dir() and str(base_posix) not in sp:
        sp.append(str(base_posix))
    for d in sp:
        for dist in pathlib.Path(d).glob("jaxlib-*.dist-info"):
            ver = dist.name.split("-")[1]
            return ver
    return None

ver = get_jaxlib_version()
if ver:
    log.append(f"  jaxlib version: {'.'.join(str(x) for x in ver)}")
    if ver < MIN_JAXLIB:
        log.append(f"  jaxlib too old (need >= {'.'.join(str(x) for x in MIN_JAXLIB)}), skipping JAX FFI")
        sys.stderr.write("\n".join(log))
        raise SystemExit(1)
else:
    log.append("  jaxlib version: not found")

def search_roots():
    """Yield candidate jaxlib/jax root directories."""
    # 1. Current interpreter (works with --no-build-isolation)
    for mod in ("jaxlib", "jax"):
        spec = importlib.util.find_spec(mod)
        locs = getattr(spec, "submodule_search_locations", None) if spec else None
        if locs:
            for loc in locs:
                log.append(f"  importlib found {mod}: {loc}")
                yield loc
        else:
            log.append(f"  importlib: {mod} not found")

    # 2. System/user site-packages (finds jaxlib in host env during isolated builds)
    try:
        sp = site.getsitepackages()
    except AttributeError:
        sp = []
    try:
        sp.append(site.getusersitepackages())
    except AttributeError:
        pass
    # Also check the base Python's site-packages (outside the venv)
    base = pathlib.Path(sys.base_prefix) / "Lib" / "site-packages"
    if base.is_dir() and str(base) not in sp:
        sp.append(str(base))
    base_posix = pathlib.Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if base_posix.is_dir() and str(base_posix) not in sp:
        sp.append(str(base_posix))
    log.append(f"  site-packages dirs: {sp}")
    for d in sp:
        for mod in ("jaxlib", "jax"):
            p = pathlib.Path(d) / mod
            if p.is_dir():
                log.append(f"  site-packages found {mod}: {p}")
                yield str(p)

for root in search_roots():
    root_path = pathlib.Path(root)
    for candidate in (root_path / "include", root_path):
        ffi_h = candidate / "xla" / "ffi" / "api" / "ffi.h"
        c_api_h = candidate / "xla" / "ffi" / "api" / "c_api.h"
        if ffi_h.exists() and c_api_h.exists():
            log.append(f"  XLA FFI headers found: {candidate}")
            sys.stderr.write("\n".join(log))
            sys.stdout.write(str(candidate))
            raise SystemExit(0)
        else:
            log.append(f"  checked {candidate}: ffi.h={ffi_h.exists()} c_api.h={c_api_h.exists()}")

log.append("  no XLA FFI headers found")
sys.stderr.write("\n".join(log))
raise SystemExit(1)
]=])

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "${_jax_probe_script}"
        RESULT_VARIABLE _jax_include_result
        OUTPUT_VARIABLE _jax_include_dir
        ERROR_VARIABLE _jax_include_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if(_jax_include_result EQUAL 0)
        set(_jax_ffi_enabled TRUE)
    endif()

    # Fallback: accept explicit JAX_INCLUDE_DIR
    if(NOT _jax_ffi_enabled AND DEFINED JAX_INCLUDE_DIR)
        if(EXISTS "${JAX_INCLUDE_DIR}/xla/ffi/api/ffi.h")
            set(_jax_include_dir "${JAX_INCLUDE_DIR}")
            set(_jax_ffi_enabled TRUE)
        endif()
    endif()
endif()

message(STATUS "")
message(STATUS "  ---- pysiglib build summary ----")
message(STATUS "  System        : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Generator     : ${CMAKE_GENERATOR}")
message(STATUS "  CXX compiler  : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "  Build configs : ${CMAKE_CONFIGURATION_TYPES}")
else()
    message(STATUS "  Build type    : ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "  CUDA          : extracted to pysiglib-cuda plugin (install via pip install pysiglib[cuda])")
message(STATUS "  JAX FFI opt   : ${PYSIGLIB_JAX_FFI}")
message(STATUS "  JAX FFI       : ${_jax_ffi_enabled}")
if(_jax_ffi_enabled)
    message(STATUS "  JAX include   : ${_jax_include_dir}")
endif()
if(PYSIGLIB_JAX_FFI AND _jax_include_error)
    message(STATUS "  JAX probe log :")
    # Print each line of the probe log as a separate STATUS message
    string(REPLACE "\n" ";" _jax_log_lines "${_jax_include_error}")
    foreach(_line IN LISTS _jax_log_lines)
        message(STATUS "  ${_line}")
    endforeach()
endif()
message(STATUS "  AVX enabled   : ${_enable_vec}")
if(_enable_vec AND NOT APPLE)
    if(_has_avx2)
        set(_avx2_str "yes")
    else()
        set(_avx2_str "no")
    endif()
    if(_has_avx512f)
        set(_avx512_str "yes")
    else()
        set(_avx512_str "no")
    endif()
    message(STATUS "  AVX2          : ${_avx2_str}")
    message(STATUS "  AVX-512F      : ${_avx512_str}")
endif()
message(STATUS "  --------------------------------")
message(STATUS "")

# ---------------------------------------------------------------------------
# Generate _config.py  (consumed by pysiglib.load_siglib at runtime)
# ---------------------------------------------------------------------------
if(_enable_vec)
    set(_vec_py "True")
else()
    set(_vec_py "False")
endif()
if(_jax_ffi_enabled)
    set(_jax_py "True")
else()
    set(_jax_py "False")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_config.py"
"# This file is automatically generated by CMake and should not be edited.\n\
# It contains information about how the package was built.\n\
SYSTEM = '${CMAKE_SYSTEM_NAME}'\n\
BUILT_WITH_JAX_FFI = ${_jax_py}\n\
BUILT_WITH_AVX = ${_vec_py}\n\
CXX_COMPILER = '${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}'\n\
CMAKE_GENERATOR = '${CMAKE_GENERATOR}'\n")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_config.py" DESTINATION pysiglib)

# ---------------------------------------------------------------------------
# Sub-projects
# ---------------------------------------------------------------------------
add_subdirectory(siglib/cpsig)

if(_jax_ffi_enabled)
    add_subdirectory(siglib/jax_ffi)
endif()

if(PYSIGLIB_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
