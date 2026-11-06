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

"""Show the three batch-32 length grids in the paper heatmap style."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "tmp" / "prefix_suffix_dispatch" / "heatmaps_simple"
LENGTHS = (16, 256, 4096)
DIMENSIONS = (2, 3, 4, 6, 8, 10, 12, 16, 24, 32)
DEGREES = tuple(range(2, 17))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-forward-update", type=Path, help="Overlay a completed CPU forward benchmark report.")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    updates = {}
    if args.cpu_forward_update:
        if args.output_dir.resolve() == ROOT.resolve():
            parser.error("use a separate output directory for a partial update")
        update = json.loads(args.cpu_forward_update.read_text())
        settings = update["metadata"]["settings"]
        assert update["complete"] and not update["skipped"] and update["cases"]
        assert settings["device"] == "cpu" and settings["repeats"] == 9 and settings["sample_seconds"] == .02
        for case in update["cases"]:
            assert (case["batch"], case["dtype"], case["n_jobs"]) == (32, "float32", 20)
            key = (case["segments"], case["dimension"], case["degree"])
            assert key not in updates
            updates[key] = case
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#E7EBF0")
    summary = []
    hashes = []
    for kind in (("forward",) if args.cpu_forward_update else ("forward", "backward")):
        limit = 4
        norm = Normalize(-limit, limit, clip=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
        for row, device in enumerate(("cpu", "cuda")):
            for col, segments in enumerate(LENGTHS):
                source = ROOT / f"segments{segments}" / kind / f"{device}.json"
                report = json.loads(source.read_text())
                assert report["complete"]
                cases = report["cases"] + report["skipped"]
                assert len(cases) == 150
                assert {(c["batch"], c["segments"], c["dimension"], c["degree"])
                        for c in cases} == {(32, segments, d, m) for d in DIMENSIONS for m in DEGREES}
                hashes.append((device, report["metadata"]["native_sha256"][device]))
                values = np.full((len(DEGREES), len(DIMENSIONS)), np.nan)
                for case in report["cases"]:
                    if device == "cpu" and updates:
                        case = updates.pop((segments, case["dimension"], case["degree"]), case)
                    if kind == "forward":
                        assert case["methods"]["gemm"]["passes_strict_tolerance"]
                        value = case["methods"]["gemm"]["speedup"]
                    else:
                        value = case["public_speedup"]
                    values[DEGREES.index(case["degree"]), DIMENSIONS.index(case["dimension"])] = value
                ax = axes[row, col]
                picture = ax.imshow(np.ma.masked_invalid(np.log2(values)), origin="lower",
                                    aspect="auto", norm=norm, cmap=cmap)
                ax.set_title(f"{device.upper()} | n={segments:,}",
                             fontsize=12, weight="bold", pad=10)
                ax.set_xticks(range(len(DIMENSIONS)), DIMENSIONS)
                ax.set_yticks(range(len(DEGREES)), DEGREES)
                ax.set_xticks(np.arange(-.5, len(DIMENSIONS), 1), minor=True)
                ax.set_yticks(np.arange(-.5, len(DEGREES), 1), minor=True)
                ax.grid(which="minor", color="white", linewidth=1)
                ax.tick_params(which="minor", bottom=False, left=False)
                if row == 1:
                    ax.set_xlabel("Dimension")
                if col == 0:
                    ax.set_ylabel("Signature depth")
                for i, j in np.ndindex(values.shape):
                    value = values[i, j]
                    if np.isfinite(value):
                        fraction = norm(np.log2(value))
                        color = "white" if fraction < .18 or fraction > .82 else "#142436"
                        label = f"{value:.2f}" if value < 10 else f"{value:.1f}"
                        ax.text(j, i, label, ha="center", va="center", fontsize=6.5, color=color)
                skipped = report["skipped"]
                timeouts = sum("120-second" in c.get("reason", "") for c in skipped)
                summary.append(f"{segments} {device} {kind}: {len(report['cases'])} measured, "
                               f"{len(skipped)} excluded ({timeouts} timed out), "
                               f"{int(np.sum(values > 1))} faster than Horner")
        fig.suptitle(f"Prefix-suffix {kind} speedup over Horner", fontsize=20, weight="bold", y=.97)
        fig.subplots_adjust(left=.055, right=.91, top=.90, bottom=.055, wspace=.15, hspace=.20)
        cax = fig.add_axes((.935, .16, .015, .68))
        colorbar = fig.colorbar(picture, cax=cax, extend="both")
        ticks = range(-limit, limit + 1)
        colorbar.set_ticks(ticks, labels=tuple(f"1/{2 ** -tick}" if tick < 0 else str(2 ** tick)
                                                for tick in ticks))
        colorbar.set_label("Horner time / prefix-suffix time (log scale)")
        for extension in ("png", "pdf"):
            fig.savefig(args.output_dir / f"{kind}_length_comparison.{extension}", dpi=180,
                        facecolor="white", bbox_inches="tight", pad_inches=.08)
        plt.close(fig)
    assert all(len({value for backend, value in hashes if backend == device}) == 1 for device in ("cpu", "cuda"))
    assert not updates, "update contains cells outside the measured grid"
    (args.output_dir / "coverage.txt").write_text("\n".join(summary) + "\n", encoding="ascii", newline="\r\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
