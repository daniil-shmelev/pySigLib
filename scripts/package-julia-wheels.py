"""Repack repaired CPU wheels, preserving native dependency paths and licenses."""

import argparse
import gzip
import io
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile


PLATFORMS = {
    "x86_64-linux-gnu": ("manylinux", "x86_64", "libcpsig.so"),
    "x86_64-w64-mingw32": ("win_amd64", "win_amd64", "cpsig.dll"),
    "aarch64-apple-darwin": ("macosx", "arm64", "libcpsig.dylib"),
}


def package(wheel_dir, output_dir, version):
    if not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version):
        raise ValueError(f"Invalid version: {version!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    wheels = list(wheel_dir.rglob(f"pysiglib-{version}-*.whl"))
    for platform, (os_tag, arch_tag, library) in PLATFORMS.items():
        candidates = [
            p for p in wheels
            if os_tag in p.name.rsplit("-", 1)[1]
            and arch_tag in p.name.rsplit("-", 1)[1]
        ]
        if len(candidates) != 1:
            raise ValueError(f"{platform}: expected one {version} CPU wheel, got {candidates}")
        target = output_dir / f"pysiglib-{version}-{platform}.tar.gz"
        with zipfile.ZipFile(candidates[0]) as wheel:
            if f"pysiglib/{library}" not in wheel.namelist():
                raise ValueError(f"{candidates[0]} is missing {library}")
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
        print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("version")
    args = parser.parse_args()
    package(args.wheel_dir, args.output_dir, args.version)
