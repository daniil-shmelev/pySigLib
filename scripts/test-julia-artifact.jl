# Exercise artifact selection, checksum verification, unpacking and the C ABI
# without Python installed. Use local archives before publishing release URLs.
using Pkg.Artifacts
using Base.BinaryPlatforms
using Libdl
using Test

output = abspath(only(ARGS))
manifest = joinpath(output, "Artifacts.toml")
function install_local_artifact(name)
    metadata = artifact_meta(name, manifest)
    @test metadata !== nothing
    download = only(metadata["download"])
    archive = joinpath(output, basename(download["url"]))
    hash = Base.SHA1(metadata["git-tree-sha1"])
    local_url = "file://" * (Sys.iswindows() ? "/" : "") * replace(archive, '\\' => '/')
    @test download_artifact(hash, local_url, download["sha256"]) === true
    @test verify_artifact(hash)
    return artifact_path(hash)
end

directory = joinpath(install_local_artifact("pysiglib_cpu"), "pysiglib")
# Match the wheel loader: preload the bundled TBB DLL on Windows.
if Sys.iswindows()
    Libdl.dlopen(joinpath(directory, "tbb12.dll"))
end
filename = Sys.iswindows() ? "cpsig.dll" : "libcpsig.$(Libdl.dlext)"
library = Libdl.dlopen(joinpath(directory, filename))
@test ccall(Libdl.dlsym(library, :sig_length), UInt64, (UInt64, UInt64), 2, 2) == 7
displacement = [1.0, 2.0]
signature = zeros(7)
@test ccall(Libdl.dlsym(library, :linear_sig_d), Cint,
    (Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, Bool, Cint),
    displacement, signature, 1, 2, 2, true, 1) == 0
@test signature ≈ [1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0]
ccall(Libdl.dlsym(library, :cpsig_shutdown), Cvoid, ())
println("Julia artifact and C ABI verified on $(triplet(HostPlatform()))")

gpu_test = get(ENV, "PYSIGLIB_JULIA_TEST_GPU", "0") == "1"
cuda_metadata = artifact_meta("pysiglib_cuda12", manifest)
gpu_test && cuda_metadata === nothing && error("GPU test requested but no CUDA artifact exists")
if cuda_metadata !== nothing
    include(joinpath(@__DIR__, "..", "examples", "julia", "cuda_ccall.jl"))
    libraries = SigLibCUDAExample.load_libraries(install_local_artifact("pysiglib_cuda12"))
    # Hosted runners have no GPU: still check the actual runtime and C ABI load.
    @test Libdl.dlsym(libraries.cusig, :linear_sig_cuda_d) != C_NULL
    @test ccall(Libdl.dlsym(libraries.cusig, :cusig_last_error_message), Cstring, ()) != C_NULL
    println("CUDA library and bundled runtime loaded")
    if gpu_test
        try
            @test SigLibCUDAExample.linear_signature(libraries, [1.0, 2.0], 2) ≈ signature
            @test SigLibCUDAExample.linear_signature(libraries, [3.0], 3) ≈ [1.0, 3.0, 4.5, 4.5]
            println("Julia CUDA computations verified on GPU")
        finally
            ccall(Libdl.dlsym(libraries.cusig, :cusig_shutdown), Cvoid, ())
        end
    end
end
