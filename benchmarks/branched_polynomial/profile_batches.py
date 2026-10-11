"""Measure public PyTorch forward/backward latency for branched polynomial kernels.

Run before and after a native rebuild with different --label values. Inputs,
outputs and both gradients are saved so performance changes can be checked for
numerical equivalence. CPU uses one core; CUDA timings include synchronization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import pysiglib.torch_api as ps
from pysiglib.load_siglib import CPSIG, CUSIG

ROOT = Path(__file__).resolve().parents[2]
CASES = [("pairs", 64, n, -1) for n in (32, 128, 512)] + [
    ("mmd", 8, 32, 8), ("mmd", 32, 128, 8), ("mmd", 8, 512, 4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if "cuda" in args.devices and not torch.cuda.is_available():
        raise RuntimeError("The CPU/CUDA profile requires an available CUDA device")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    core = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {core})
    rows, arrays = [], {}
    # Record binaries: local builds may lag source edits.
    libraries = [CPSIG] + ([CUSIG] if "cuda" in args.devices else [])
    hashes = {str(Path(lib._name).resolve()): hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()
              for lib in libraries}
    for dtype in (torch.float32, torch.float64):
        for workload, batch, segments, max_batch in CASES:
            generator = torch.Generator().manual_seed(715)
            inputs = [torch.cat((torch.zeros(batch, 1, 3, dtype=dtype),
                      (.25 / (3 * segments)**.5 * torch.randn(batch, segments, 3,
                       dtype=dtype, generator=generator)).cumsum(1)), dim=1) for _ in range(2)]
            for device in args.devices:
                x, y = [p.to(device).requires_grad_() for p in inputs]
                options = dict(depth=3, method="polynomial", order=3, n_jobs=1)

                def run():
                    if workload == "pairs":
                        value = ps.branched_sig_kernel(x, y, **options)
                    else:
                        value = ps.branched_sig_mmd(x, y, max_batch=max_batch, **options)
                    gradients = torch.autograd.grad(value.sum(), (x, y))
                    if device == "cuda":
                        torch.cuda.synchronize()
                    return value, gradients

                for _ in range(2):
                    run()
                times = []
                for _ in range(args.samples):
                    start = time.perf_counter()
                    value, gradients = run()
                    times.append(1000 * (time.perf_counter() - start))
                key = f"{str(dtype).split('.')[-1]}_{workload}_b{batch}_n{segments}_{device}"
                arrays[key + "_value"] = value.detach().cpu().numpy()
                arrays[key + "_dx"] = gradients[0].cpu().numpy()
                arrays[key + "_dy"] = gradients[1].cpu().numpy()
                rows.append(dict(case=key, device=device, dtype=str(dtype), workload=workload,
                                 batch=batch, segments=segments, max_batch=max_batch,
                                 median_ms=float(np.median(times)), min_ms=min(times), max_ms=max(times)))
                print(key, f"{np.median(times):.3f} ms", flush=True)
    report = dict(label=args.label, depth=3, order=3, samples=args.samples,
                  cpu_core=core, gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  torch=torch.__version__,
                  binary_sha256=hashes,
                  scope="Public PyTorch API forward + gradients of both inputs. Includes static Gram construction, path-gradient chain, allocation, launch and synchronization; excludes initial device transfers. Warmed calls. MMD includes all three Gram matrices and their gradients. CPU n_jobs=1, pinned to one core.",
                  measurements=rows)
    (args.output / f"batches_{args.label}.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / f"batches_{args.label}.npz", **arrays)


if __name__ == "__main__":
    main()
