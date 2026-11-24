# Working-set search experiments

The reference is pySigLib commit `eb26853f145582ed23bd5967bcaf413dbfca5e2e`. The manuscript protocol and reference objectives come from commit `02da5ef180616028bcf1533d859a4f5310848f6b`. The clean top-32 implementation is `07a48aa6b` on `study/top32-search`. The three pilots and the primary timing comparison used the experimental library at `bb9739838`. The degree-two extremes and the separate batch-width comparison used the following revision on this branch, which adds the opt-in all-violations rule. Use the scripts from the branch tip with either library.

Build the `cpsig` target in Release mode in all three checkouts using the same compiler and SIMD settings. Build the CPU library with `PYSIGLIB_CUDA=OFF` and `PYSIGLIB_JAX_FFI=OFF`. Place this branch's built library, its generated `_config.py` and the oneTBB runtime in its `pysiglib` directory. Supply the paths to the other two libraries with `--control` and `--plain`. On Windows their filename is `cpsig.dll`.

The data directory is the local `sig_branch_and_bound_experiments` archive containing the three `results/normalized/integral_comparison/prepared.npz` files and `results/optimized_degree4_20261028/protocol.json`. The driver checks those file hashes and reconstructs the manuscript's training, validation and test sample-ID hashes. It uses the manuscript's exact per-response penalties, including its degree-four Speech Commands reconstruction and the two separate Arabic dictionary calibrations.

The commands below use PowerShell with Python 3.12, NumPy, PyTorch, psutil and threadpoolctl. Substitute the local data and library paths. Use new output directories on a repeat run.

```powershell
$data = '../../sig_branch_and_bound_experiments'
$control = 'out/control/cpsig.dll'
$plain = '../lasso_search_plain/out/study/siglib/cpsig/Release/cpsig.dll'
py -3.12 benchmarks/benchmark_search_policy.py --data $data --control $control --out out/pilot --policies current,top16,top32,top64,bins16,bins32,first16,half32,grow64,bins32_cache32 --cases arabic4,insect4,speech4,dict_log,arabic2,insect3,speech3 --reps 2
py -3.12 benchmarks/benchmark_search_policy.py --data $data --control $control --out out/pool_pilot --policies current,legacy,top32,bins32_pool,raw32_feedback,grow_pool,best32_256,best32_512,group32_256,group32_512,quarter_pool --cases insect4,arabic4,speech3,speech4,dict_log --reps 2 --seed 8821
py -3.12 benchmarks/benchmark_search_policy.py --data $data --control $control --out out/peer_pilot --policies current,legacytrace,top32,bins32_pool,peer32,peer128,peergroups --cases insect4,arabic4,speech3,speech4,dict_log --reps 2 --seed 3161
py -3.12 benchmarks/benchmark_search_policy.py --data $data --control $control --plain $plain --out out/final_search --policies current,plain32,bins32_pool,legacytrace --cases arabic1,insect2,speech3,dict_log,arabic4,insect1,speech2,dict_sig,arabic3,insect4,speech1,arabic2,insect3,speech4 --reps 7 --seed 91462 --fresh --save
py -3.12 benchmarks/audit_search_policy.py --data $data --models out/final_search --policies current,plain32,bins32_pool,legacytrace
py -3.12 benchmarks/audit_search_policy.py --models out/final_search --ties
py -3.12 benchmarks/audit_search_policy.py --models out/final_search --losses
py -3.12 benchmarks/benchmark_search_policy.py --data $data --control $control --out out/extremes --policies current,first1,all_violators,top32 --cases arabic2,insect2,speech2 --reps 3 --seed 1248
py -3.12 benchmarks/benchmark_search_policy.py --data $data --out out/widths --policies legacy,top16,top32,top64 --cases insect2,dict_sig,speech1,arabic4,insect4,speech3,arabic2,dict_log,speech4,insect1,arabic3,speech2,arabic1,insect3 --reps 5 --seed 29711 --fresh
py -3.12 benchmarks/summarize_search_policy.py out/final_search
```

Each row fits all ten responses through the public CPU API. The driver rotates method order, sets eight feature workers and limits BLAS to one thread. It checks the numerical status after each timed call. With `--fresh` each method and case starts in its own process. The first fit measures the increase in process high-water memory, saves the model if requested and warms up the worker. The remaining repetitions measure fit time. The native allocator's own peak is recorded per fit. Data loading, test prediction and audits are outside fit timing.

`current` loads the reference library and `plain32` loads the clean top-32 library. All other names select policies in the same experimental library. `legacy` uses the reference rule and `legacytrace` additionally counts candidate sources and score ties. `top16`, `top32` and `top64` admit the strongest indices available at the current residual, up to the indicated limit. `bins32_pool` admits one representative per score bin and retains the usual larger reusable candidate pool. `first1` returns one violation as soon as it is available. `all_violators` disables candidate reuse and lower-degree proposals, searches the entire dictionary at the current residual and admits every violation. Limit this last comparison to tractable dictionaries.

The audit computes training and held-out features independently using signature coefficients and the tensor logarithm series. It checks the selected and intercept stationarity conditions, the objectives against the manuscript archive, and all coordinates for dictionaries below the paper's 4 GiB cutoff. The largest three classification rows use the native tree certificate for the remaining inactive coordinates. The tie diagnostic uses five repeated columns and five independent Walsh columns. The small loss diagnostic checks every coordinate for both dictionaries under all four supported losses.
