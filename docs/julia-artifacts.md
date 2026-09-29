# Using the native library from Julia

The **Julia release assets** workflow repackages the tested wheels as CPU archives (Linux x86_64/glibc, Windows x86_64, macOS Apple Silicon), optional CUDA 12 archives (Linux and Windows x86_64), and an `Artifacts.toml`. It reuses the existing binaries; Julia users need neither Python nor a compiler. CUDA archives include the CUDA runtime, so GPU users need a compatible NVIDIA driver but no locally installed toolkit.

Run the existing **Release** workflow successfully, then publish a GitHub Release whose tag points to that exact commit and matches the wheel version (`v4.0.0` or `4.0.0`, for example). Publishing triggers packaging, Julia smoke tests on all three platforms, and attachment of the archives and manifest. CUDA archives are included only if both source GPU test jobs passed; a CPU-only release does not publish untested CUDA wheels. The source workflow's wheel artifacts must still be available. If the release was published before the build completed, rerun **Julia release assets** manually with the existing tag. This workflow does not create releases or publish to PyPI.

Copy the release's `Artifacts.toml` into your Julia package root (merge its `pysiglib_cpu` and `pysiglib_cuda12` entries if you already have one). Add the `Artifacts` and `Libdl` standard libraries to your package dependencies. Julia selects and downloads the matching archive automatically, using its download checksum and content hash. CUDA entries are lazy, so CPU users do not download them. Use `Pkg.Artifacts` when accessing lazy artifacts, and add `Pkg` to the GPU wrapper's dependencies. See the [Julia artifact documentation](https://pkgdocs.julialang.org/v1/artifacts/).

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

The `x86_64-w64-mingw32` filename is Julia's Windows platform identifier; the library inside is the existing MSVC-built DLL and is accessed through its C ABI.

## CUDA through `ccall`

[cuda_ccall.jl](../examples/julia/cuda_ccall.jl) is a runnable example using only Julia standard libraries. It loads `cusig` and the bundled runtime, allocates device memory, calls `linear_sig_cuda_d`, synchronizes, copies the result to Julia, and frees the allocations even if a call fails. Copy the example into your wrapper, then use:

```julia
using Pkg.Artifacts
include("cuda_ccall.jl")

libraries = SigLibCUDAExample.load_libraries(artifact"pysiglib_cuda12")
signature = SigLibCUDAExample.linear_signature(libraries, [1.0, 2.0], 2)
# [1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0]
```

Load the handles during your package's runtime initialization, not during precompilation. A wrapper using CUDA.jl can instead pass device pointers from `CuArray`s to the same C ABI without copying through host memory. Keep the arrays alive across the call, use the same device/context, and synchronize before and after the call: cusig does not accept CUDA.jl's task-local stream. General batched inputs must also respect the C API's row-major layout.

These artifacts use CUDA 12.9.79, matching the CUDA 12.9.1 toolkit in the wheel workflow. Linux retains auditwheel's renamed runtime and relative RPATH. Windows adds `cudart64_12.dll` from NVIDIA's checksum-pinned redistributable and preloads it before `cusig.dll`; both archives include the runtime license. The packager's runtime pin must be updated alongside the toolkit in `release.yml`. Users need an NVIDIA GPU supported by the wheel's `CUDA_ARCH` list and a driver compatible with CUDA 12.9. There are no macOS CUDA artifacts.

Hosted CI validates artifact hashes, CPU computation, and CUDA library/runtime loading; it does not have a GPU. Publication also requires the original CUDA wheels to have passed both upstream GPU test jobs. To additionally run the Julia example against a GPU after generating the artifacts, set `PYSIGLIB_JULIA_TEST_GPU=1` and run `julia scripts/test-julia-artifact.jl /path/to/julia-artifacts`. This checks numerical CUDA results and fails if the CUDA artifact or a working GPU is missing.
