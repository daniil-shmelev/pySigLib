# Generate a manifest for the release assets using Julia's own artifact hashing.
using Pkg.Artifacts
using Base.BinaryPlatforms
using SHA

output, repository, tag = ARGS
version = startswith(tag, "v") ? tag[2:end] : tag
manifest = joinpath(output, "Artifacts.toml")
isfile(manifest) && rm(manifest)

for platform in ("x86_64-linux-gnu", "x86_64-w64-mingw32", "aarch64-apple-darwin")
    filename = "pysiglib-$version-$platform.tar.gz"
    archive = joinpath(output, filename)
    # gzip is available on the Linux packaging runner; no Julia packages needed.
    hash = create_artifact() do directory
        run(`tar -xzf $archive -C $directory`)
    end
    checksum = bytes2hex(open(sha256, archive))
    url = "https://github.com/$repository/releases/download/$tag/$filename"
    bind_artifact!(manifest, "pysiglib_cpu", hash;
        platform=parse(Platform, platform), download_info=[(url, checksum)], force=true)
end
println(read(manifest, String))
