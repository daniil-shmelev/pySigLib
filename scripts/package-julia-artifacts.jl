# Reuse repaired wheels; keep native dependency paths intact for RPATH/DLL loading.
using Pkg.Artifacts, Base.BinaryPlatforms, Downloads, SHA

wheels, output, repository, tag = ARGS
version = startswith(tag, "v") ? tag[2:end] : tag
occursin(r"^[0-9][A-Za-z0-9.!+_-]*$", version) || error("Invalid version: $version")
include_cuda = get(ENV, "INCLUDE_CUDA", "false") == "true"
mkpath(output)
manifest = joinpath(output, "Artifacts.toml")
isfile(manifest) && rm(manifest)
platforms = (("x86_64-linux-gnu", "manylinux", "lib%s.so"),
    ("x86_64-w64-mingw32", "win_amd64", "%s.dll"),
    ("aarch64-apple-darwin", "macosx", "lib%s.dylib"))

mktempdir() do runtime
    if include_cuda
        # Match release.yml's CUDA 12.9.1 toolkit. SHA256 from NVIDIA's redistrib_12.9.1.json.
        bundle = "cuda_cudart-windows-x86_64-12.9.79-archive"
        url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/$bundle.zip"
        archive = Downloads.download(url, joinpath(runtime, "cudart.zip"))
        bytes2hex(open(sha256, archive)) == "179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8" || error("CUDA runtime checksum mismatch")
        run(`unzip -q -j $archive $bundle/bin/cudart64_12.dll $bundle/LICENSE -d $runtime`)
    end
    for cuda in (false, true), (triplet, wheel_tag, library_pattern) in platforms
        cuda && (!include_cuda || startswith(triplet, "aarch64")) && continue
        package = cuda ? "pysiglib_cuda" : "pysiglib"
        arch = startswith(triplet, "aarch64") ? "arm64" : "x86_64"
        wheel = only(filter(readdir(wheels; join=true)) do path
            name = basename(path)
            startswith(name, "$package-$version-") && endswith(name, ".whl") &&
                occursin(wheel_tag, name) && (wheel_tag == "win_amd64" || occursin(arch, name))
        end)
        hash = create_artifact() do root
            run(`unzip -q $wheel -d $root`)
            library = replace(library_pattern, "%s" => cuda ? "cusig" : "cpsig")
            isfile(joinpath(root, package, library)) || error("Missing $library in $wheel")
            for (directory, _, files) in walkdir(root), file in files
                native = occursin(r"\.(so(\..*)?|dylib|dll)$", file) && !occursin("jax_ffi", file)
                native || "licenses" in splitpath(directory) || rm(joinpath(directory, file))
            end
            if cuda
                licenses = joinpath(root, package, "licenses")
                mkpath(licenses)
                cp(joinpath(@__DIR__, "..", "LICENSE"), joinpath(licenses, "LICENSE"); force=true)
                cp(joinpath(runtime, "LICENSE"), joinpath(licenses, "NVIDIA-CUDA-LICENSE.txt"); force=true)
                if wheel_tag == "win_amd64"
                    cp(joinpath(runtime, "cudart64_12.dll"), joinpath(root, package, "cudart64_12.dll"); force=true)
                else
                    # auditwheel already bundled and renamed the Linux runtime.
                    any(endswith(".so.12.9.79"), readdir(joinpath(root, "pysiglib_cuda.libs"))) || error("Update the CUDA runtime pin with release.yml")
                end
            end
        end
        prefix, name = cuda ? ("pysiglib-cuda12", "pysiglib_cuda12") : ("pysiglib", "pysiglib_cpu")
        filename = "$prefix-$version-$triplet.tar.gz"
        checksum = archive_artifact(hash, joinpath(output, filename))
        url = "https://github.com/$repository/releases/download/$tag/$filename"
        bind_artifact!(manifest, name, hash; platform=parse(Platform, triplet),
            download_info=[(url, checksum)], lazy=cuda, force=true)
    end
end
