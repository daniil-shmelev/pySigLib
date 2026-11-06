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

"""Plot endpoint-GEMM speedups and paired timing intervals."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def comparison(case, baseline):
    methods = (baseline, "endpoint")
    keys = sorted({(v["seed"], v["repeat"]) for v in case["samples"]})
    times = {(v["seed"], v["repeat"], v["method"]): v["ms"] for v in case["samples"]}
    paired = np.array([[times[(*key, method)] for method in methods] for key in keys])
    rng = np.random.default_rng(470)
    indices = rng.integers(len(keys), size=(5000, len(keys)))
    medians = np.median(paired[indices], axis=1)
    interval = np.quantile(medians[:, 0] / medians[:, 1], [0.025, 0.975])
    median = np.median(paired, axis=0)
    return {"speedup": float(median[0] / median[1]), "low": float(interval[0]), "high": float(interval[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = []
    for dtype in ("float32", "float64"):
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
        labels = []
        for ax, tag, title in zip(axes, ("cpu1", "cpu20", "cuda1"), ("CPU: 1 thread", "CPU: 20 threads", "CUDA")):
            path = args.directory / f"{tag}_{dtype}.json"
            if not path.exists():
                ax.set_visible(False)
                continue
            data = json.loads(path.read_text())
            for i, case in enumerate(data["cases"]):
                batch, n, d, m = case["shape"]
                if len(labels) <= i: labels.append(f"d={d}, m={m}\nb={batch}, n={n}")
                medians = {method: float(np.median([v["ms"] for v in case["samples"] if v["method"] == method]))
                           for method in ("horner", "prefix_suffix", "endpoint")}
                row = {"backend": tag, "dtype": dtype, "shape": case["shape"], "block": case["block"], "median_ms": medians}
                for j, (baseline, color, label) in enumerate((("horner", "#2364aa", "Relative to Horner"),
                                                            ("prefix_suffix", "#e57a25", "Relative to prefix-suffix"))):
                    result = comparison(case, baseline)
                    row[baseline] = result
                    x = i + (j - 0.5) * 0.28
                    y = result["speedup"]
                    ax.errorbar(x, y, yerr=[[max(0, y - result["low"])] , [max(0, result["high"] - y)]],
                                fmt="o", capsize=3, color=color, label=label if i == 0 else None)
                summary.append(row)
            ax.set_yscale("log", base=2)
            ax.axhline(1, color="#666666", linewidth=1, linestyle="--")
            ax.set_title(title, loc="left", fontsize=12)
            ax.set_ylabel("Speedup")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(loc="upper left", ncol=2, fontsize=9)
        axes[-1].set_xticks(range(len(labels)), labels, fontsize=8)
        fig.suptitle(f"Endpoint GEMM, {dtype.upper()} throughout\nAbove 1 is faster; bars show 95% paired bootstrap intervals", fontsize=15)
        for extension in ("png", "pdf"):
            fig.savefig(args.directory / f"endpoint_speedup_{dtype}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii", newline="\r\n")
    sources = [p for p in Path(__file__).parent.iterdir() if p.is_file()]
    measurements = [args.directory / f"{tag}_{dtype}.json" for tag in ("cpu1", "cpu20", "cuda1")
                    for dtype in ("float32", "float64")]
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(),
                "sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sources + measurements if p.exists()}}
    (args.directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii", newline="\r\n")
    for row in summary:
        print(row["backend"], row["dtype"], row["shape"], "block", row["block"],
              "ms", {k: round(v, 4) for k, v in row["median_ms"].items()},
              "speedup", round(row["horner"]["speedup"], 2), round(row["prefix_suffix"]["speedup"], 2))


if __name__ == "__main__":
    main()
