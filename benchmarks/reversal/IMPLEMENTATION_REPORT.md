# CPU replacement and CUDA Pascal removal

Archived measurements and validation logs are included in [results](results/README.md). The `tmp/reversible_backward_20261110/` paths below identify the original runs and the default location for new outputs.

The standard CPU prefix-suffix backward implementation now uses two directional reversal sweeps. Each sweep reconstructs states in blocks of up to 128 segments. The existing factor and workspace budgets can reduce the block size. There is no selection between the old and new backward algorithms, and there is no full-path checkpoint fallback. The old code is retained only in `cpu_baseline.h` for reproducible benchmarks.

The forward algorithm, Horner, corrected-path prefix-suffix algorithm, and CUDA prefix-suffix algorithm are unchanged. No paper source or figure was changed in this implementation step. Nothing was committed or pushed to Overleaf.

## Interfaces and storage

- CPU backward uses the final signature supplied through the existing interface. Torch already saves this output; JAX now saves it as well. No `_return_state` flag was added.
- The C function signatures are unchanged. Standard CPU prefix-suffix backward now requires the final signature pointer. Passing null returns an error instead of rebuilding the forward trajectory. The native header documents this change.
- Numeric auxiliary storage is bounded in path length, for fixed dimension, depth, split, and thread count. This excludes inputs, the final signature, output gradients, and opaque BLAS/TBB allocations. Tests use a fixed 16 KiB workspace budget and 4 KiB factor budget for 16, 257, and 4096 segments, both dtypes, three split settings, and both scalar conventions.
- CUDA coefficient Pascal kernels and their build entry were removed. Explicit `method="pascal"` requests raise an unsupported-method error in the base and Torch APIs, including empty batches. The C ABI and JAX CUDA FFI reject this method too. CUDA `auto` and `horner` use Horner.

## Runtime evidence

The prior CPU screen measured block 128 on 276 configurations with more than 16 segments. Median runtime overhead was 7%; the 90th percentile was 33%. Median numeric temporary-memory reduction was 82.9%. These are prototype measurements from `EXPANDED_REPORT.md`, not new production timings or a uniform runtime guarantee. The new production code uses the same recurrence and block size, with the existing path transformations and workspace limits.

## Validation

- Main Python, Torch, JAX, and API-parity suite: 1085 passed, 27 skipped.
- Additional signature-backward and JAX integration suite: 854 passed. This suite overlaps the main suite; the counts must not be added as distinct tests.
- Final coefficient suite after adding JAX device checks: 154 passed, 2 skipped. JAX CUDA runtime tests are skipped because this Windows JAX installation has no GPU backend.
- Native CPU prefix-suffix and Pascal tests: 16 passed.
- Native CUDA prefix-suffix and coefficient tests: 5 passed.
- The CPU library, CUDA library without Pascal, isolated CPU benchmark, and CPU/CUDA JAX FFI compiled successfully. The CUDA FFI target was compiled against existing import libraries; it was not executed through JAX on GPU.
- Workspace checks also verify that repeated backward calls preserve final signatures and cotangents. Existing tests cover transformations, views, explicit splits, partial blocks, empty inputs, threading, and CUDA streams.

## Numerical limits

`check_production.py` repeats every CPU screen case in which any candidate failed the original accuracy check. The reference is the retained old implementation in float64 on identical rounded paths and cotangents. Production agrees with the block-128 prototype at the existing tolerance, and preserves its inputs.

All 15 selected cases fail the fixed pointwise float32 tolerance of `3e-5 * (1 + abs(reference))`: 10 pass with the old implementation and 5 fail with both. Fourteen cases have 4096 segments; one has 1024 segments. The largest absolute error is about 0.00596 for dimension 3, depth 8, batch 32, and 4096 segments; its relative L2 error is about 2.85e-5. No tolerance was relaxed. The passing unit tests do not remove this known long-path rounding limitation. These failures are separate from the passing test counts above.

## Reproduction and records

From the repository root:

```powershell
$env:PYTHONPATH="$PWD\plugins\cuda;$PWD"
cmake --build out/build/windows-cpu --config Release --target cpsig cpsig_tests -j 2
cmake --build out/build/windows-cuda --config Release --target cusig cusig_tests -j 2
cmake --build out/build/reversal --config Release --target reversal_cpu -j 2
Copy-Item out/build/windows-cpu/siglib/cpsig/Release/cpsig.dll pysiglib/cpsig.dll -Force
Copy-Item out/build/windows-cuda/cusig_build/Release/cusig.dll plugins/cuda/pysiglib_cuda/cusig.dll -Force
python -m pytest tests/test_prefix_suffix_backprop.py tests/test_prefix_suffix.py tests/test_pascal_coefficients.py tests/test_sig_coef.py tests/test_sig_coef_backprop.py tests/test_apis_match.py -q
python -m pytest tests/test_prefix_suffix_backprop.py tests/test_jax_api.py tests/test_sig_backprop.py -q
python benchmarks/reversal/check_production.py
```

Raw accuracy results, native-library hashes, and validation logs are in `tmp/reversible_backward_20261110/`: `production_accuracy.json`, `replacement_tests.log`, `replacement_integration_tests.log`, `replacement_pascal_final_tests.log`, `replacement_native_cpu.log`, and `replacement_native_cuda.log`. The original screen and timing samples remain in `expanded_cpu.json`. The benchmark scripts now use the retained old implementation for the reference so that a production replacement does not change their baseline.
