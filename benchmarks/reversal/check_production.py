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

"""Check production reversal against the retained benchmark baseline."""

import argparse
import hashlib
import json

import numpy as np
import torch

from run import Native, ROOT, pysiglib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", default="benchmarks/reversal/results/expanded_cpu.json")
    parser.add_argument("--output", default="tmp/reversible_backward_20261110/production_accuracy.json")
    args = parser.parse_args()
    screen = json.loads((ROOT / args.screen).read_text())
    native = Native("cpu", ROOT / "out/build/reversal/Release")
    selected = [case for case in screen["screen"] if any(
        method.get("gradient_pass") is False for method in case.get("methods", []))]
    records = []
    for case in selected:
        d, m, n, batch = (case[key] for key in ("dimension", "degree", "segments", "batch"))
        dtype = getattr(torch, case["dtype"])
        rng = np.random.default_rng(screen["metadata"]["seed"] + d * 1000 + m * 100 + n + batch)
        path = torch.tensor(rng.normal(scale=.03, size=(batch, n + 1, d)).cumsum(1), dtype=dtype)
        final = pysiglib.sig(path, m, method="ps", scalar_term=True, n_jobs=20)
        seeds = torch.tensor(rng.normal(scale=.05, size=final.shape), dtype=dtype)
        final64 = pysiglib.sig(path.double(), m, method="ps", scalar_term=True, n_jobs=20)
        reference = torch.empty_like(path, dtype=torch.float64)
        native.call(path.double(), final64, seeds.double(), reference, m, 0, 0, True, 20)
        baseline, prototype = torch.empty_like(path), torch.empty_like(path)
        native.call(path, final, seeds, baseline, m, 0, 0, True, 20)
        native.call(path, final, seeds, prototype, m, 0, 128, True, 20)
        saved_final, saved_seeds = final.clone(), seeds.clone()
        production = pysiglib.sig_backprop(path, final, seeds, m, method="ps", n_jobs=20)
        torch.testing.assert_close(final, saved_final, rtol=0, atol=0)
        torch.testing.assert_close(seeds, saved_seeds, rtol=0, atol=0)
        tolerance = 3e-5 if dtype == torch.float32 else 1e-10
        torch.testing.assert_close(production, prototype, rtol=tolerance, atol=tolerance)
        record = {key: case[key] for key in ("dimension", "degree", "segments", "batch", "dtype")}
        for name, value in (("baseline", baseline), ("production", production)):
            error = value.double() - reference
            record[name] = dict(max_absolute=float(error.abs().max()),
                relative_l2=float(error.norm() / reference.norm().clamp_min(1e-300)),
                gradient_pass=bool(torch.all(error.abs() <= tolerance * (1 + reference.abs()))))
        records.append(record)
        print(record, flush=True)
    report = dict(reference="Retained old CPU implementation in float64 on identical rounded inputs",
        selection="Every CPU screen case with an accuracy failure in any candidate",
        production_sha256=hashlib.sha256((ROOT / "pysiglib/cpsig.dll").read_bytes()).hexdigest(),
        benchmark_sha256=hashlib.sha256(native.path.read_bytes()).hexdigest(), cases=records)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii", newline="\r\n")


if __name__ == "__main__":
    main()
