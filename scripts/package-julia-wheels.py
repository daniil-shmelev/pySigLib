"""Repack repaired wheels, preserving native dependency paths and licenses."""

import argparse
import gzip
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import tarfile
import urllib.request
import zipfile


PLATFORMS = {
    "x86_64-linux-gnu": ("manylinux", "x86_64", "libcpsig.so"),
    "x86_64-w64-mingw32": ("win_amd64", "win_amd64", "cpsig.dll"),
    "aarch64-apple-darwin": ("macosx", "arm64", "libcpsig.dylib"),
}


# Match CUDA 12.9.1 in release.yml. Hash from NVIDIA's redistrib_12.9.1.json.
CUDART_ARCHIVE = "cuda_cudart-windows-x86_64-12.9.79-archive"
CUDART_SHA256 = "179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8"
CUDART_URL = (
    "https://developer.download.nvidia.com/compute/cuda/redist/"
    f"cuda_cudart/windows-x86_64/{CUDART_ARCHIVE}.zip"
)


def cuda_runtime_files():
    with urllib.request.urlopen(CUDART_URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != CUDART_SHA256:
        raise ValueError("NVIDIA CUDA runtime archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {
            "pysiglib_cuda/cudart64_12.dll": archive.read(f"{CUDART_ARCHIVE}/bin/cudart64_12.dll"),
            "pysiglib_cuda/licenses/NVIDIA-CUDA-LICENSE.txt": archive.read(f"{CUDART_ARCHIVE}/LICENSE"),
            "pysiglib_cuda/licenses/LICENSE": (Path(__file__).resolve().parents[1] / "LICENSE").read_bytes(),
        }


def package(wheel_dir, output_dir, version, include_cuda=False):
    if not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version):
        raise ValueError(f"Invalid version: {version!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [("pysiglib", platform, spec) for platform, spec in PLATFORMS.items()]
    runtime = cuda_runtime_files() if include_cuda else {}
    if include_cuda:
        targets += [
            ("pysiglib_cuda", platform, (spec[0], spec[1], spec[2].replace("cpsig", "cusig")))
            for platform, spec in PLATFORMS.items() if "apple" not in platform
        ]
    for package_name, platform, (os_tag, arch_tag, library) in targets:
        wheels = list(wheel_dir.rglob(f"{package_name}-{version}-*.whl"))
        candidates = [
            p for p in wheels
            if os_tag in p.name.rsplit("-", 1)[1]
            and arch_tag in p.name.rsplit("-", 1)[1]
        ]
        if len(candidates) != 1:
            raise ValueError(f"{platform}: expected one {version} {package_name} wheel, got {candidates}")
        prefix = "pysiglib-cuda12" if package_name == "pysiglib_cuda" else "pysiglib"
        target = output_dir / f"{prefix}-{version}-{platform}.tar.gz"
        with zipfile.ZipFile(candidates[0]) as wheel:
            if f"{package_name}/{library}" not in wheel.namelist():
                raise ValueError(f"{candidates[0]} is missing {library}")
            extra_files = {}
            if package_name == "pysiglib_cuda":
                extra_files = {n: d for n, d in runtime.items() if "licenses/" in n or "mingw32" in platform}
                if "linux" in platform and not any(
                    re.fullmatch(r"pysiglib_cuda\.libs/libcudart-[^/]+\.so\.12\.9\.79", n)
                    for n in wheel.namelist()
                ):
                    raise ValueError("Expected the CUDA 12.9.79 runtime bundled by auditwheel; update the runtime pin with release.yml")
            # Keep auditwheel/delocate dependency directories in their original
            # locations: moving the libraries would break their relative RPATHs.
            with target.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
                with tarfile.open(fileobj=gz, mode="w") as archive:
                    for name in sorted(wheel.namelist()):
                        path = PurePosixPath(name)
                        if path.is_absolute() or ".." in path.parts:
                            raise ValueError(f"Unsafe wheel path: {name}")
                        native = re.search(r"(?:\.so(?:\..*)?|\.dylib|\.dll)$", name)
                        license_file = "licenses" in path.parts
                        if name.endswith("/") or not (native or license_file):
                            continue
                        if "pysiglib_jax_ffi" in path.name:
                            continue
                        data = wheel.read(name)
                        info = tarfile.TarInfo(name)
                        info.size = len(data)
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(data))
                    for name, data in sorted(extra_files.items()):
                        if name in wheel.namelist():
                            if wheel.read(name) != data:
                                raise ValueError(f"Conflicting bundled runtime file: {name}")
                            continue
                        info = tarfile.TarInfo(name)
                        info.size = len(data)
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(data))
        print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("version")
    parser.add_argument("--cuda", action="store_true", help="Also package Linux/Windows CUDA 12 wheels and runtime")
    args = parser.parse_args()
    package(args.wheel_dir, args.output_dir, args.version, args.cuda)
