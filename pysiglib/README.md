# Python API structure

`pysiglib` is the NumPy API. `pysiglib.torch_api` and `pysiglib.jax_api` provide the corresponding framework APIs and import their optional dependencies only when selected. The public functions enforce their array contracts and provide concrete array annotations; `py.typed` makes those annotations available to downstream type checkers.

`_core` contains shared native calls, mathematical utilities, static kernel formulas, and generic streaming algorithms. Common array operations use `array-api-compat`; `_array.py` defines concrete array type variables and the few operations that need special handling, such as preserving Torch gradients when copying. `_storage.py` handles NumPy/Torch memory access for ctypes. Optional framework types are imported only during type checking, and there is no backend registry. Torch autograd wrappers call `_core` directly. JAX retains its FFI and custom differentiation implementation, with NumPy callbacks where needed.

The C++ and CUDA numerical implementations are shared and unchanged by this separation. Add backend-specific behavior at the array or autodiff boundary; keep common algorithms in `_core`.

Tests parametrized over NumPy and Torch storage use `tests/native_api.py` to exercise the shared native layer. `tests/test_backend_apis.py` checks the public contracts and execution with other frameworks unavailable. `tests/typing/backend_arrays.py` checks both inferred return types and rejection of foreign arrays; run `basedpyright --project tests/typing/pyrightconfig.json --pythonpath /path/to/python` in an environment with both frameworks installed.
