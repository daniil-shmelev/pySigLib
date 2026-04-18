# Probe the active Python environment for jaxlib's XLA FFI headers.
#
# Sets:
#   JAXLIB_FOUND        - TRUE if the headers are available
#   JAXLIB_INCLUDE_DIR  - directory containing xla/ffi/api/ffi.h and c_api.h
#   JAXLIB_PROBE_LOG    - stderr log from the probe (for diagnostic STATUS prints)
#
# Inputs:
#   Python_EXECUTABLE  - set by find_package(Python COMPONENTS Interpreter)
#   JAX_INCLUDE_DIR    - optional explicit override (short-circuits the probe)
#
# The probe rejects jaxlib < 0.9.1 because older versions expose XLA_FFI_API_MINOR
# less than 3, which is incompatible with the handlers compiled from this tree.

if(NOT Python_EXECUTABLE)
    message(FATAL_ERROR "FindJaxlib.cmake requires Python_EXECUTABLE to be set (call find_package(Python COMPONENTS Interpreter) first)")
endif()

set(JAXLIB_FOUND FALSE)
set(JAXLIB_INCLUDE_DIR "")
set(JAXLIB_PROBE_LOG "")

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
    RESULT_VARIABLE _jax_probe_result
    OUTPUT_VARIABLE _jax_probe_stdout
    ERROR_VARIABLE _jax_probe_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

set(JAXLIB_PROBE_LOG "${_jax_probe_stderr}")

if(_jax_probe_result EQUAL 0)
    set(JAXLIB_FOUND TRUE)
    set(JAXLIB_INCLUDE_DIR "${_jax_probe_stdout}")
endif()

# Explicit override via JAX_INCLUDE_DIR
if(NOT JAXLIB_FOUND AND DEFINED JAX_INCLUDE_DIR)
    if(EXISTS "${JAX_INCLUDE_DIR}/xla/ffi/api/ffi.h")
        set(JAXLIB_INCLUDE_DIR "${JAX_INCLUDE_DIR}")
        set(JAXLIB_FOUND TRUE)
    endif()
endif()
