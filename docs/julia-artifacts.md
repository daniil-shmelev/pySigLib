# Using the native library from Julia

The **Julia release assets** workflow repackages the tested CPU wheels as three native-library archives (Linux x86_64/glibc, Windows x86_64, macOS Apple Silicon) and an `Artifacts.toml`. It reuses the existing binaries; Julia users need neither Python nor a compiler. CUDA is not included in this initial workflow.

Run the existing **Release** workflow successfully, then publish a GitHub Release whose tag points to that exact commit and matches the wheel version (`v4.0.0` or `4.0.0`, for example). Publishing triggers packaging, Julia smoke tests on all three platforms, and attachment of the archives and manifest. The source workflow's wheel artifacts must still be available. If the release was published before the build completed, rerun **Julia release assets** manually with the existing tag. This workflow does not create releases or publish to PyPI.

Copy the release's `Artifacts.toml` into your Julia package root (merge its `pysiglib_cpu` entries if you already have one). Add the `Artifacts` and `Libdl` standard libraries to your package dependencies. Julia selects and downloads the matching archive automatically, using its download checksum and content hash. See the [Julia artifact documentation](https://pkgdocs.julialang.org/v1/artifacts/).

For example, inside your package module:

```julia
using Artifacts, Libdl

const libcpsig = Ref{String}()

function __init__()
    directory = joinpath(artifact"pysiglib_cpu", "pysiglib")
    if Sys.iswindows()
        Libdl.dlopen(joinpath(directory, "tbb12.dll"))
    end
    filename = Sys.iswindows() ? "cpsig.dll" : "libcpsig.$(Libdl.dlext)"
    libcpsig[] = joinpath(directory, filename)
end

sig_length(dimension, degree) = ccall(
    (:sig_length, libcpsig[]), UInt64, (UInt64, UInt64), dimension, degree)
```

The archives retain the wheel's native-library paths, bundled dependencies and licenses so relative library references keep working. Python modules and JAX FFI libraries are omitted. Windows preloads the bundled TBB DLL just as the Python loader does. These binaries retain the wheel requirements: Linux glibc 2.28+, an x86-64-v3/AVX2-capable CPU on Linux x86_64, AVX2 on Windows, and macOS 11+ on Apple Silicon. The artifact platform selector does not check CPU instruction support or the glibc version. There are no Intel macOS, Linux ARM or musl archives.

The `x86_64-w64-mingw32` filename is Julia's Windows platform identifier; the library inside is the existing MSVC-built DLL and is accessed through its C ABI. A future CUDA artifact will also need explicit runtime/driver requirements and GPU tests.
