# Julia native artifacts

After the existing **Release** workflow succeeds, publish a GitHub Release at that exact commit with a matching version tag (`v4.0.0` or `4.0.0`). **Julia release assets** attaches native archives and `Artifacts.toml`. Manual dispatch retries an existing tag while the source wheel artifacts remain available. CPU supports Linux/Windows x86_64 and macOS arm64; CUDA 12 supports Linux/Windows x86_64 and is included only when both source GPU test jobs passed.

Copy the release's `Artifacts.toml` into your Julia package and add `Pkg` and `Libdl` to its dependencies. Load libraries at runtime, for example in `__init__()`:

```julia
using Pkg.Artifacts, Libdl
cpu = joinpath(artifact"pysiglib_cpu", "pysiglib")
Sys.iswindows() && Libdl.dlopen(joinpath(cpu, "tbb12.dll"))
cpsig = Libdl.dlopen(joinpath(cpu, Sys.iswindows() ? "cpsig.dll" : "libcpsig.$(Libdl.dlext)"))
ccall(Libdl.dlsym(cpsig, :sig_length), UInt64, (UInt64, UInt64), 2, 2) # 7

# Only when CUDA is requested (the CUDA artifact is lazy):
gpu = joinpath(artifact"pysiglib_cuda12", "pysiglib_cuda")
Sys.iswindows() && Libdl.dlopen(joinpath(gpu, "cudart64_12.dll"))
cusig = Libdl.dlopen(joinpath(gpu, Sys.iswindows() ? "cusig.dll" : "libcusig.$(Libdl.dlext)"))
```

Call `cusig` functions with device pointers, not ordinary Julia arrays. CUDA.jl `CuArray`s can supply those pointers: preserve their lifetime, use the same device/context, and synchronize before and after calls because cusig does not accept CUDA.jl's task-local stream. Batched buffers must follow the C API's row-major layout. Signatures and return codes are declared in `siglib/cpsig/cpsig.h` and `siglib/cusig/cusig.h`.

Archives preserve bundled libraries and licenses, omit Python/JAX code, and need no Python or compiler. CUDA includes runtime 12.9.79 and requires a compatible NVIDIA driver/GPU; update the pinned runtime alongside `release.yml` when upgrading CUDA. CPU requirements remain those of the wheels: glibc 2.28+ and x86-64-v3 on Linux, AVX2 on Windows, macOS 11+ on Apple Silicon. Julia's platform selector does not check those CPU/glibc requirements. There are no musl, Linux ARM, Intel Mac or macOS CUDA artifacts.
