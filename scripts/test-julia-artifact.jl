# Exercise artifact selection, checksum verification, unpacking and the C ABI
# without Python installed. Use local archives before publishing release URLs.
using Pkg.Artifacts
using Base.BinaryPlatforms
using Libdl
using Test

output = abspath(only(ARGS))
manifest = joinpath(output, "Artifacts.toml")
metadata = artifact_meta("pysiglib_cpu", manifest)
@test metadata !== nothing
download = only(metadata["download"])
archive = joinpath(output, basename(download["url"]))
hash = Base.SHA1(metadata["git-tree-sha1"])
local_url = "file://" * (Sys.iswindows() ? "/" : "") * replace(archive, '\\' => '/')
@test download_artifact(hash, local_url, download["sha256"]) === true
@test verify_artifact(hash)

directory = joinpath(artifact_path(hash), "pysiglib")
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
