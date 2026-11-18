# Audio experiments for the revised manuscript

The three classification tables and the Spoken Arabic Digit dictionary comparison were rerun on 18 November 2026 using solver commit `4b5a257d3fdf1b9fe6dfdbc8989dccefaaa17e0d`. The machine was an eight-core Apple M3 with 8 GiB RAM. The runner verifies checksums for the prepared paths, split identifiers and frozen penalties before fitting. The Speech Commands degree-four penalty vector is the training-only reconstruction described in the [protocol](../protocol.json).

Each of the 12 classification rows has one fresh-process tree fit. The nine rows whose dense training matrix fits within 4 GiB also have a fresh-process fit using the full design, cyclic coordinate descent and Gap Safe screening. Both methods use the same per-class penalties. The reported memory is the increase in process high-water RSS during the fit. The dictionary comparison uses one warm-up and three timed fresh-process fits per dictionary, with alternating order. Its memory statistic is the median tracked native peak, and its time IQR is the difference between the linearly interpolated 75th and 25th percentiles. All fits use eight feature workers and one BLAS thread.

The runner independently evaluates selected training and held-out columns, objectives and stationarity. It streams every coordinate for the nine feasible signature classification rows and both Arabic dictionaries. The paired classification checks also compare the held-out labels with the full-design predictions. The three largest classification dictionaries use the tree certificate for inactive words. [Measurements and audits](measurements.json)

## Reproduction

From the solver repository root, prepare the original recordings and splits following the [dataset instructions](../README.md#reproduction), then run the benchmark in a fresh output directory.

```sh
python benchmarks/benchmark_prefix_suffix.py --data /tmp/audio-data prepare \
  --experiments /path/to/sig_branch_and_bound
python -m benchmarks.benchmark_paper_audio --data /tmp/audio-data --output /tmp/paper-audio-rerun
```

The output includes the per-fit JSON records and model arrays, the independent audits, and the combined measurements used in the manuscript. Dataset preparation, calibration, test prediction and audits are outside the timed calls.
