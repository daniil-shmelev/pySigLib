# Reversal experiment records

These files are unchanged copies of selected records from `tmp/reversible_backward_20261110/`. They support the reports in the parent directory. Raw JSON retains timing samples, accuracy failures, settings, and native-library hashes. Historical CUDA Pascal data precedes removal of that implementation; the binaries and the separate Overleaf paper repository are not included.

- `expanded_cpu.json` and `expanded_cuda.json`: broad screen and confirmation samples.
- `prefix_suffix_cpu.json` and `prefix_suffix_cuda.json`: original block-size experiment samples.
- `prefix_suffix_checks_cpu.json` and `prefix_suffix_checks_cuda.json`: correctness and workspace checks.
- `production_accuracy.json`: final CPU implementation checked against the retained old implementation in float64.
- `pascal_accuracy.json`, `pascal_ab_*`, `pascal_heatmap_*`, and `pascal_backward_*`: Pascal accuracy and timing records.
- `replacement_*.log`: final Python and native validation summaries. Their suites overlap, so their pass counts must not be added as distinct tests.

To regenerate the prefix-suffix summaries from the archived data, pass `--results-dir benchmarks/reversal/results` to `summarize.py` or `summarize_expanded.py`, with a new output report path. The experiment DLLs are Windows-specific and must be built locally before running new measurements. `check_production.py` reads the archived CPU screen by default and writes new results under `tmp/reversible_backward_20261110/`.
