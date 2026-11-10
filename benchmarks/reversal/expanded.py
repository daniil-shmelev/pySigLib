# Copyright 2026 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Screen and confirm prefix-suffix replacement costs on CPU and CUDA."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import torch

from run import Native, ROOT, measure


PAIRS = [(2, 2), (2, 4), (2, 6), (2, 8), (2, 10), (2, 12), (2, 14),
         (3, 2), (3, 4), (3, 6), (3, 8), (3, 10),
         (4, 2), (4, 4), (4, 6), (4, 8),
         (8, 2), (8, 3), (8, 4), (8, 5), (16, 2), (16, 3), (16, 4)]


def save(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii", newline="\r\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["screen", "confirm"], default="screen")
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--max-call-seconds", type=float, default=2)
    args = parser.parse_args()
    args.warmup = 1 if args.stage == "screen" else 3
    args.samples = 3 if args.stage == "screen" else 9
    args.sample_ms = 10 if args.stage == "screen" else 20
    if args.stage == "confirm":
        args.max_call_seconds = max(args.max_call_seconds, 5)
    native = Native(args.device, ROOT / "out/build/reversal/Release")
    digest = hashlib.sha256(native.path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        report = json.loads(args.output.read_text(encoding="ascii"))
        assert report["metadata"]["device"] == args.device and report["metadata"]["library_sha256"] == digest
    else:
        report = dict(metadata=dict(device=args.device, platform=platform.platform(), python=sys.version,
                                    torch=torch.__version__, gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                                    library_sha256=digest, seed=args.seed, threads=args.threads,
                                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    reference="Retained old CPU float64 prefix-suffix on identical rounded paths and cotangents",
                                    workspace="Native numeric temporary allocations, excluding inputs and output; CUDA includes 4 MiB cuBLAS workspace"),
                      screen=[], confirm=[])
    report["metadata"][args.stage + "_settings"] = dict(samples=args.samples, warmup=args.warmup,
        sample_ms=args.sample_ms, max_call_seconds=args.max_call_seconds, blocks=[32, 128, 512],
        small_path_blocks=[512])
    if args.stage == "confirm":
        report["metadata"].setdefault("screen_settings", dict(samples=3, warmup=1, sample_ms=10,
            max_call_seconds=2, blocks=[32, 128, 512], small_path_blocks=[512]))
    if args.stage == "screen":
        shapes = [(d, m, n, batch, dtype) for (d, m), n, batch, dtype in itertools.product(
            PAIRS, [16, 256, 4096], [1, 32], [torch.float32, torch.float64])]
        shapes += [(d, m, n, 8, dtype) for (d, m), n, dtype in itertools.product(
            PAIRS, [64, 1024], [torch.float32, torch.float64])]
    else:
        ranked = []
        for case in report["screen"]:
            if "methods" not in case or case["methods"][0]["status"] != "ok" or case["methods"][-1]["status"] != "ok":
                continue
            ratio = case["methods"][-1]["median_ms"] / case["methods"][0]["median_ms"]
            ranked.append((ratio, case))
        choices = []
        for ordered in (sorted(ranked, key=lambda x: -x[0]), sorted(ranked, key=lambda x: abs(x[0] - 1.1)),
                        sorted(ranked, key=lambda x: x[0])):
            added = 0
            for _, case in ordered:
                if case not in choices:
                    choices.append(case)
                    added += 1
                if added == 4:
                    break
        for case in report["screen"]:
            if "methods" in case and any(m["status"] == "excluded" and m.get("first_call_ms", 0) < 5000 for m in case["methods"]):
                if case not in choices:
                    choices.append(case)
        shapes = [(c["dimension"], c["degree"], c["segments"], c["batch"], getattr(torch, c["dtype"])) for c in choices]
    completed = {(c["dimension"], c["degree"], c["segments"], c["batch"], c["dtype"]) for c in report[args.stage]}
    for d, m, n, batch, dtype in shapes:
        key = (d, m, n, batch, str(dtype).split(".")[-1])
        if key in completed:
            continue
        args.blocks = [0, 512] if n <= 16 else [0, 32, 128, 512]
        try:
            case = measure(native, d, m, n, batch, dtype, args)
        except Exception as exc:
            case = dict(device=args.device, dimension=d, degree=m, segments=n, batch=batch,
                        dtype=key[-1], status="failed", reason=str(exc))
        report[args.stage].append(case)
        save(args.output, report)
        print(f"{args.device} {args.stage} {len(report[args.stage])}/{len(shapes)} d={d} m={m} n={n} batch={batch} {dtype}", flush=True)
    report["metadata"][args.stage + "_finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(args.output, report)


if __name__ == "__main__":
    main()
