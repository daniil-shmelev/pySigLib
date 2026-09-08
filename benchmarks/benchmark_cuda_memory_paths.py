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

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time


SEED = 20260529


MATRIX_CASES = {}


def add_case_pair(name, **spec):
    for direction in ("forward", "backprop"):
        MATRIX_CASES[f"{name}_{direction}"] = {
            **spec, "direction": direction,
        }


add_case_pair(
    "signature_f32_main", kind="signature", dtype="float32",
    batch=256, length=1024, dimension=8, degree=3)
add_case_pair(
    "signature_f64_main", kind="signature", dtype="float64",
    batch=64, length=256, dimension=8, degree=3)
add_case_pair(
    "signature_f32_naive", kind="signature", dtype="float32",
    batch=256, length=1024, dimension=8, degree=3, horner=False)
add_case_pair(
    "signature_f32_correction", kind="signature", dtype="float32",
    batch=256, length=1024, dimension=8, degree=3, correction=True)
add_case_pair(
    "signature_f32_lead_lag", kind="signature", dtype="float32",
    batch=256, length=1024, dimension=4, degree=3, lead_lag=True)
add_case_pair(
    "signature_f32_near_limit", kind="signature", dtype="float32",
    batch=8, length=2, dimension=12000, degree=1)
add_case_pair(
    "signature_f64_near_limit", kind="signature", dtype="float64",
    batch=8, length=2, dimension=6000, degree=1)
add_case_pair(
    "signature_f32_fallback", kind="signature", dtype="float32",
    batch=16, length=2, dimension=30000, degree=1, fallback=True)
add_case_pair(
    "signature_f64_fallback", kind="signature", dtype="float64",
    batch=8, length=2, dimension=15000, degree=1, fallback=True)
add_case_pair(
    "signature_f32_naive_fallback", kind="signature", dtype="float32",
    batch=16, length=2, dimension=30000, degree=1, horner=False,
    fallback=True)
add_case_pair(
    "signature_f32_lead_lag_fallback", kind="signature", dtype="float32",
    batch=16, length=2, dimension=15000, degree=1, lead_lag=True,
    fallback=True)

add_case_pair(
    "method3_f32_main", kind="method3", dtype="float32",
    batch=64, length=128, dimension=16, degree=3)
add_case_pair(
    "method3_f32_near_limit", kind="method3", dtype="float32",
    batch=8, length=16, dimension=33, degree=3)
add_case_pair(
    "method3_f64_near_limit", kind="method3", dtype="float64",
    batch=8, length=16, dimension=26, degree=3)
add_case_pair(
    "method3_f32_torch", kind="method3", dtype="float32",
    batch=8, length=16, dimension=16, degree=3, torch_api=True)
add_case_pair(
    "method3_f32_fallback", kind="method3", dtype="float32",
    batch=8, length=16, dimension=34, degree=3, fallback=True)
add_case_pair(
    "method3_f64_fallback", kind="method3", dtype="float64",
    batch=8, length=16, dimension=27, degree=3, fallback=True)

add_case_pair(
    "branched_dense_nonplanar", kind="branched_dense", dtype="float32",
    batch=32, length=128, dimension=3, degree=4, planar=False)
add_case_pair(
    "branched_dense_planar", kind="branched_dense", dtype="float32",
    batch=32, length=64, dimension=1, degree=6, planar=True)
add_case_pair(
    "branched_dense_planar_correction", kind="branched_dense",
    dtype="float32", batch=32, length=64, dimension=1, degree=6,
    planar=True, correction=True)
add_case_pair(
    "branched_dense_planar_f32_fallback", kind="branched_dense",
    dtype="float32", batch=16, length=64, dimension=1, degree=7,
    planar=True, fallback=True)
add_case_pair(
    "branched_dense_planar_f64_fallback", kind="branched_dense",
    dtype="float64", batch=16, length=64, dimension=1, degree=7,
    planar=True, fallback=True)
add_case_pair(
    "branched_dense_nonplanar_f64_fallback", kind="branched_dense",
    dtype="float64", batch=16, length=64, dimension=4, degree=4,
    planar=False, fallback=True)
add_case_pair(
    "branched_dense_planar_correction_fallback", kind="branched_dense",
    dtype="float32", batch=16, length=64, dimension=1, degree=7,
    planar=True, correction=True, fallback=True)
add_case_pair(
    "branched_dense_nonplanar_1024_fallback", kind="branched_dense",
    dtype="float32", batch=8, length=16, dimension=5, degree=4,
    planar=False, fallback=True)
add_case_pair(
    "branched_dense_planar_1024_fallback", kind="branched_dense",
    dtype="float32", batch=4, length=16, dimension=1, degree=8,
    planar=True, fallback=True)

add_case_pair(
    "branched_combine", kind="branched_combine", dtype="float32",
    batch=128, dimension=1, degree=6, planar=True)
add_case_pair(
    "branched_combine_f32_fallback", kind="branched_combine",
    dtype="float32", batch=64, dimension=1, degree=7, planar=True,
    fallback=True)
add_case_pair(
    "branched_combine_f64_fallback", kind="branched_combine",
    dtype="float64", batch=64, dimension=1, degree=7, planar=True,
    fallback=True)

add_case_pair(
    "branched_coef", kind="branched_coef", dtype="float32",
    batch=32, length=64, dimension=1, degree=6, planar=True)
add_case_pair(
    "branched_coef_f32_fallback", kind="branched_coef", dtype="float32",
    batch=16, length=64, dimension=1, degree=7, planar=True,
    fallback=True)
add_case_pair(
    "branched_coef_f64_fallback", kind="branched_coef", dtype="float64",
    batch=16, length=64, dimension=1, degree=7, planar=True,
    fallback=True)
add_case_pair(
    "branched_coef_large_batch_fallback", kind="branched_coef",
    dtype="float32", batch=64, length=64, dimension=1, degree=7,
    planar=True, fallback=True)
add_case_pair(
    "branched_coef_correction_fallback", kind="branched_coef",
    dtype="float32", batch=16, length=64, dimension=1, degree=7,
    planar=True, correction=True, fallback=True)

for method in (0, 1, 2):
    add_case_pair(
        f"branched_conversion_method{method}", kind="branched_conversion",
        dtype="float32", batch=32, dimension=8, degree=3, method=method,
        planar=method != 0)
    add_case_pair(
        f"branched_conversion_method{method}_fallback",
        kind="branched_conversion", dtype="float32", batch=16,
        dimension=14, degree=3, method=method, planar=method != 0,
        fallback=True)
add_case_pair(
    "branched_conversion_f32_near_limit", kind="branched_conversion",
    dtype="float32", batch=8, dimension=13, degree=3, method=0,
    planar=False)
add_case_pair(
    "branched_conversion_f64_near_limit", kind="branched_conversion",
    dtype="float64", batch=8, dimension=10, degree=3, method=0,
    planar=False)
add_case_pair(
    "branched_conversion_f64_fallback", kind="branched_conversion",
    dtype="float64", batch=8, dimension=11, degree=3, method=0,
    planar=False, fallback=True)
add_case_pair(
    "branched_conversion_f64_extreme", kind="branched_conversion",
    dtype="float64", batch=1, dimension=20, degree=3, method=0,
    planar=False, fallback=True)

add_case_pair(
    "branched_log", kind="branched_log", dtype="float32",
    batch=16, length=16, dimension=8, degree=3, method=0,
    planar=False)
add_case_pair(
    "branched_log_fallback", kind="branched_log", dtype="float32",
    batch=8, length=8, dimension=14, degree=3, method=0,
    planar=False, fallback=True)


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def first_tensor(value):
    if isinstance(value, (tuple, list)):
        return first_tensor(value[0])
    return value


def make_path(torch, batch, length, dimension, dtype):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED)
    increments = torch.randn(
        (batch, length - 1, dimension), generator=generator,
        device="cuda", dtype=dtype)
    zeros = torch.zeros((batch, 1, dimension), device="cuda", dtype=dtype)
    return torch.cat((zeros, increments.cumsum(dim=1)), dim=1)


def make_matrix_case(name, spec):
    import pysiglib
    import pysiglib.torch_api
    from pysiglib._core.branched_log_sig import branched_log_sig as _native_branched_log_sig
    from pysiglib._core.branched_sig import branched_sig as _native_branched_sig
    from pysiglib._core.branched_sig_backprop import branched_sig_backprop as _native_branched_sig_backprop
    from pysiglib._core.branched_sig_coef import branched_sig_coef as _native_branched_sig_coef
    from pysiglib._core.branched_sig_coef_backprop import branched_sig_coef_backprop as _native_branched_sig_coef_backprop
    from pysiglib._core.branched_sig import branched_sig_combine as _native_branched_sig_combine
    from pysiglib._core.branched_sig_backprop import branched_sig_combine_backprop as _native_branched_sig_combine_backprop
    from pysiglib._core.branched_sig import branched_sig_length as _native_branched_sig_length
    from pysiglib._core.branched_log_sig import branched_sig_to_log_sig as _native_branched_sig_to_log_sig
    from pysiglib._core.branched_log_sig_backprop import branched_sig_to_log_sig_backprop as _native_branched_sig_to_log_sig_backprop
    from pysiglib._core.log_sig import log_sig as _native_log_sig
    from pysiglib._core.branched_log_sig import prepare_branched_log_sig as _native_prepare_branched_log_sig
    from pysiglib._core.branched_sig import prepare_branched_sig as _native_prepare_branched_sig
    from pysiglib._core.branched_sig_coef import prepare_branched_sig_coef as _native_prepare_branched_sig_coef
    from pysiglib._core.log_sig import prepare_log_sig as _native_prepare_log_sig
    from pysiglib._core.log_sig import set_cache_dir as _native_set_cache_dir
    from pysiglib._core.sig import sig as _native_sig
    from pysiglib._core.sig_backprop import sig_backprop as _native_sig_backprop
    import torch

    dtype = getattr(torch, spec["dtype"])
    batch = spec["batch"]
    length = spec.get("length", 1)
    dimension = spec["dimension"]
    degree = spec["degree"]
    direction = spec["direction"]
    path_steps = batch * max(length - 1, 0)
    kind = spec["kind"]

    if kind == "signature":
        path = make_path(torch, batch, length, dimension, dtype)
        options = {
            "horner": spec.get("horner", True),
            "lead_lag": spec.get("lead_lag", False),
        }
        correction = None
        if spec.get("correction"):
            generator = torch.Generator(device="cuda")
            generator.manual_seed(SEED + 1)
            correction = 0.001 * torch.randn(
                (batch, length - 1, dimension * dimension),
                generator=generator, device="cuda", dtype=dtype)
            options["correction"] = correction
        if direction == "backprop":
            signature = _native_sig(path, degree, **options)
            derivs = torch.ones_like(signature)
            backprop_options = {
                key: value for key, value in options.items()
                if key not in ("horner",)
            }
            call = lambda: _native_sig_backprop(
                path, signature, derivs, degree, **backprop_options)
        else:
            call = lambda: _native_sig(path, degree, **options)
        return call, batch, path_steps

    if kind == "method3":
        _native_prepare_log_sig(dimension, degree, 3, device="cuda")
        path = make_path(torch, batch, length, dimension, dtype)
        if spec.get("torch_api"):
            if direction == "backprop":
                cotangent = torch.ones_like(
                    _native_log_sig(path, degree, method=3))

                def call():
                    value = path.detach().requires_grad_(True)
                    result = pysiglib.torch_api.log_sig(
                        value, degree, method=3)
                    return torch.autograd.grad(result, value, cotangent)
            else:
                call = lambda: pysiglib.torch_api.log_sig(
                    path, degree, method=3)
        elif direction == "backprop":
            from pysiglib._core.log_sig_backprop import _log_sig_from_path_backprop
            derivs = torch.ones_like(
                _native_log_sig(path, degree, method=3))
            call = lambda: _log_sig_from_path_backprop(
                derivs, path, degree)
        else:
            call = lambda: _native_log_sig(path, degree, method=3)
        return call, batch, path_steps

    if kind == "branched_dense":
        planar = spec["planar"]
        _native_prepare_branched_sig(
            dimension, degree, planar=planar, device="cuda")
        path = make_path(torch, batch, length, dimension, dtype)
        options = {"planar": planar}
        if spec.get("correction"):
            generator = torch.Generator(device="cuda")
            generator.manual_seed(SEED + 2)
            correction_width = sum(
                dimension ** level for level in range(2, degree + 1))
            options["correction"] = 0.001 * torch.randn(
                (batch, length - 1, correction_width),
                generator=generator, device="cuda", dtype=dtype)
        if direction == "backprop":
            signature = _native_branched_sig(path, degree, **options)
            derivs = torch.ones_like(signature)
            call = lambda: _native_branched_sig_backprop(
                path, signature, derivs, degree, **options)
        else:
            call = lambda: _native_branched_sig(path, degree, **options)
        return call, batch, path_steps

    if kind == "branched_combine":
        planar = spec["planar"]
        _native_prepare_branched_sig(
            dimension, degree, planar=planar, device="cuda")
        left = _native_branched_sig(
            make_path(torch, batch, 3, dimension, dtype), degree,
            planar=planar)
        right = _native_branched_sig(
            make_path(torch, batch, 4, dimension, dtype), degree,
            planar=planar)
        if direction == "backprop":
            derivs = torch.ones_like(left)
            call = lambda: _native_branched_sig_combine_backprop(
                derivs, left, right, dimension, degree, planar=planar)
        else:
            call = lambda: _native_branched_sig_combine(
                left, right, dimension, degree, planar=planar)
        return call, batch, batch

    if kind == "branched_coef":
        planar = spec["planar"]
        trees = list(pysiglib.trees(dimension, degree, planar=planar)[1:])
        _native_prepare_branched_sig_coef(
            dimension, trees, planar=planar, device="cuda")
        path = make_path(torch, batch, length, dimension, dtype)
        options = {"planar": planar}
        if spec.get("correction"):
            generator = torch.Generator(device="cuda")
            generator.manual_seed(SEED + 3)
            correction_width = sum(
                dimension ** level for level in range(2, degree + 1))
            options["correction"] = 0.001 * torch.randn(
                (batch, length - 1, correction_width),
                generator=generator, device="cuda", dtype=dtype)
        if direction == "backprop":
            coefs = _native_branched_sig_coef(path, trees, **options)
            derivs = torch.ones_like(coefs)
            call = lambda: _native_branched_sig_coef_backprop(
                path, trees, coefs, derivs, **options)
        else:
            call = lambda: _native_branched_sig_coef(
                path, trees, **options)
        return call, batch, path_steps

    if kind == "branched_conversion":
        method = spec["method"]
        planar = spec["planar"]
        _native_prepare_branched_log_sig(
            dimension, degree, method, planar=planar, device="cuda")
        generator = torch.Generator(device="cuda")
        generator.manual_seed(SEED)
        signature = torch.randn(
            (batch, _native_branched_sig_length(
                dimension, degree, planar=planar)),
            generator=generator, device="cuda", dtype=dtype)
        if direction == "backprop":
            log_signature = _native_branched_sig_to_log_sig(
                signature, dimension, degree, planar=planar, method=method)
            derivs = torch.ones_like(log_signature)
            call = lambda: _native_branched_sig_to_log_sig_backprop(
                signature, derivs, dimension, degree,
                planar=planar, method=method)
        else:
            call = lambda: _native_branched_sig_to_log_sig(
                signature, dimension, degree, planar=planar, method=method)
        return call, batch, batch

    if kind == "branched_log":
        method = spec["method"]
        planar = spec["planar"]
        _native_prepare_branched_log_sig(
            dimension, degree, method, planar=planar, device="cuda")
        path = make_path(torch, batch, length, dimension, dtype)
        if direction == "backprop":
            cotangent = torch.ones_like(_native_branched_log_sig(
                path, degree, planar=planar, method=method))

            def call():
                value = path.detach().requires_grad_(True)
                result = pysiglib.torch_api.branched_log_sig(
                    value, degree, planar=planar, method=method)
                return torch.autograd.grad(result, value, cotangent)
        else:
            call = lambda: _native_branched_log_sig(
                path, degree, planar=planar, method=method)
        return call, batch, path_steps

    raise ValueError(f"Unknown benchmark kind: {kind}")


def make_case(name, cache_dir):
    import pysiglib
    import pysiglib.torch_api
    from pysiglib._core.branched_log_sig import branched_log_sig as _native_branched_log_sig
    from pysiglib._core.branched_sig import branched_sig as _native_branched_sig
    from pysiglib._core.branched_sig_backprop import branched_sig_backprop as _native_branched_sig_backprop
    from pysiglib._core.branched_sig_coef import branched_sig_coef as _native_branched_sig_coef
    from pysiglib._core.branched_sig_coef_backprop import branched_sig_coef_backprop as _native_branched_sig_coef_backprop
    from pysiglib._core.branched_sig import branched_sig_combine as _native_branched_sig_combine
    from pysiglib._core.branched_sig_backprop import branched_sig_combine_backprop as _native_branched_sig_combine_backprop
    from pysiglib._core.branched_sig import branched_sig_length as _native_branched_sig_length
    from pysiglib._core.branched_log_sig import branched_sig_to_log_sig as _native_branched_sig_to_log_sig
    from pysiglib._core.branched_log_sig_backprop import branched_sig_to_log_sig_backprop as _native_branched_sig_to_log_sig_backprop
    from pysiglib._core.log_sig import log_sig as _native_log_sig
    from pysiglib._core.branched_log_sig import prepare_branched_log_sig as _native_prepare_branched_log_sig
    from pysiglib._core.branched_sig import prepare_branched_sig as _native_prepare_branched_sig
    from pysiglib._core.branched_sig_coef import prepare_branched_sig_coef as _native_prepare_branched_sig_coef
    from pysiglib._core.log_sig import prepare_log_sig as _native_prepare_log_sig
    from pysiglib._core.log_sig import set_cache_dir as _native_set_cache_dir
    from pysiglib._core.sig import sig as _native_sig
    from pysiglib._core.sig_backprop import sig_backprop as _native_sig_backprop
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _native_set_cache_dir(str(cache_dir))
    float32 = torch.float32

    if name in MATRIX_CASES:
        return make_matrix_case(name, MATRIX_CASES[name])

    if name.startswith("sig_fast"):
        path = make_path(torch, 64, 256, 8, float32)
        degree = 3
        out = _native_sig(path, degree)
        deriv = torch.ones_like(out)
        if name.endswith("backprop"):
            call = lambda: _native_sig_backprop(path, out, deriv, degree)
        else:
            call = lambda: _native_sig(path, degree)
        return call, 64, 64 * 255

    if name.startswith("sig_high_dim_fast"):
        path = make_path(torch, 8, 2, 12000, float32)
        degree = 1
        out = _native_sig(path, degree)
        deriv = torch.ones_like(out)
        if name.endswith("backprop"):
            call = lambda: _native_sig_backprop(path, out, deriv, degree)
        else:
            call = lambda: _native_sig(path, degree)
        return call, 8, 8

    if name.startswith("log_method3_fast"):
        path = make_path(torch, 8, 16, 16, float32)
        degree = 3
        _native_prepare_log_sig(16, degree, 3, device="cuda")
        if name.endswith("backprop"):
            cotangent = torch.ones_like(
                _native_log_sig(path, degree, method=3))

            def call():
                value = path.detach().requires_grad_(True)
                result = pysiglib.torch_api.log_sig(value, degree, method=3)
                return torch.autograd.grad(result, value, cotangent)
        else:
            call = lambda: _native_log_sig(path, degree, method=3)
        return call, 8, 8 * 15

    if name.startswith("branched_dense_nonplanar_fast"):
        batch, length, dimension, degree = 16, 64, 3, 4
        _native_prepare_branched_sig(
            dimension, degree, planar=False, device="cuda")
        path = make_path(torch, batch, length, dimension, float32)
        out = _native_branched_sig(path, degree, planar=False)
        deriv = torch.ones_like(out)
        if name.endswith("backprop"):
            call = lambda: _native_branched_sig_backprop(
                path, out, deriv, degree, planar=False)
        else:
            call = lambda: _native_branched_sig(
                path, degree, planar=False)
        return call, batch, batch * (length - 1)

    if name.startswith("branched_dense_planar_fast"):
        batch, length, dimension, degree = 16, 64, 1, 6
        _native_prepare_branched_sig(
            dimension, degree, planar=True, device="cuda")
        path = make_path(torch, batch, length, dimension, float32)
        out = _native_branched_sig(path, degree, planar=True)
        deriv = torch.ones_like(out)
        if name.endswith("backprop"):
            call = lambda: _native_branched_sig_backprop(
                path, out, deriv, degree, planar=True)
        else:
            call = lambda: _native_branched_sig(
                path, degree, planar=True)
        return call, batch, batch * (length - 1)

    if name.startswith("branched_combine_fast"):
        batch, dimension, degree = 64, 1, 6
        _native_prepare_branched_sig(
            dimension, degree, planar=True, device="cuda")
        path1 = make_path(torch, batch, 3, dimension, float32)
        path2 = make_path(torch, batch, 4, dimension, float32)
        sig1 = _native_branched_sig(path1, degree, planar=True)
        sig2 = _native_branched_sig(path2, degree, planar=True)
        combined = _native_branched_sig_combine(
            sig1, sig2, dimension, degree, planar=True)
        deriv = torch.ones_like(combined)
        if name.endswith("backprop"):
            call = lambda: _native_branched_sig_combine_backprop(
                deriv, sig1, sig2, dimension, degree, planar=True)
        else:
            call = lambda: _native_branched_sig_combine(
                sig1, sig2, dimension, degree, planar=True)
        return call, batch, batch

    if name.startswith("branched_coef_fast"):
        batch, length, dimension, degree = 16, 32, 1, 6
        trees = list(pysiglib.trees(dimension, degree, planar=True))
        _native_prepare_branched_sig_coef(
            dimension, trees, planar=True, device="cuda")
        path = make_path(torch, batch, length, dimension, float32)
        coefs = _native_branched_sig_coef(path, trees, planar=True)
        deriv = torch.ones_like(coefs)
        if name.endswith("backprop"):
            call = lambda: _native_branched_sig_coef_backprop(
                path, trees, coefs, deriv, planar=True)
        else:
            call = lambda: _native_branched_sig_coef(
                path, trees, planar=True)
        return call, batch, batch * (length - 1)

    if name.startswith("branched_conversion_fast"):
        batch, dimension, degree = 8, 8, 3
        _native_prepare_branched_log_sig(
            dimension, degree, 0, planar=False, device="cuda")
        generator = torch.Generator(device="cuda")
        generator.manual_seed(SEED)
        bsig = torch.randn(
            (batch, _native_branched_sig_length(
                dimension, degree, planar=False)),
            generator=generator, device="cuda", dtype=float32)
        blogsig = _native_branched_sig_to_log_sig(
            bsig, dimension, degree, planar=False, method=0)
        deriv = torch.ones_like(blogsig)
        if name.endswith("backprop"):
            call = lambda: _native_branched_sig_to_log_sig_backprop(
                bsig, deriv, dimension, degree, planar=False, method=0)
        else:
            call = lambda: _native_branched_sig_to_log_sig(
                bsig, dimension, degree, planar=False, method=0)
        return call, batch, batch

    if name.startswith("branched_conversion_near_fast"):
        batch, dimension, degree = 4, 13, 3
        _native_prepare_branched_log_sig(
            dimension, degree, 0, planar=False, device="cuda")
        generator = torch.Generator(device="cuda")
        generator.manual_seed(SEED)
        bsig = torch.randn(
            (batch, _native_branched_sig_length(
                dimension, degree, planar=False)),
            generator=generator, device="cuda", dtype=float32)
        blogsig = _native_branched_sig_to_log_sig(
            bsig, dimension, degree, planar=False, method=0)
        deriv = torch.ones_like(blogsig)
        if name.endswith("backprop"):
            call = lambda: _native_branched_sig_to_log_sig_backprop(
                bsig, deriv, dimension, degree, planar=False, method=0)
        else:
            call = lambda: _native_branched_sig_to_log_sig(
                bsig, dimension, degree, planar=False, method=0)
        return call, batch, batch

    if name.startswith("branched_log_fast"):
        batch, length, dimension, degree = 8, 8, 8, 3
        _native_prepare_branched_log_sig(
            dimension, degree, 0, planar=False, device="cuda")
        path = make_path(torch, batch, length, dimension, float32)
        if name.endswith("backprop"):
            cotangent = torch.ones_like(_native_branched_log_sig(
                path, degree, planar=False, method=0))

            def call():
                value = path.detach().requires_grad_(True)
                result = pysiglib.torch_api.branched_log_sig(
                    value, degree, planar=False, method=0)
                return torch.autograd.grad(result, value, cotangent)
        else:
            call = lambda: _native_branched_log_sig(
                path, degree, planar=False, method=0)
        return call, batch, batch * (length - 1)

    raise ValueError(f"Unknown benchmark case: {name}")


FAST_CASES = (
    "sig_fast_forward",
    "sig_fast_backprop",
    "sig_high_dim_fast_forward",
    "sig_high_dim_fast_backprop",
    "log_method3_fast_forward",
    "log_method3_fast_backprop",
    "branched_dense_nonplanar_fast_forward",
    "branched_dense_nonplanar_fast_backprop",
    "branched_dense_planar_fast_forward",
    "branched_dense_planar_fast_backprop",
    "branched_combine_fast_forward",
    "branched_combine_fast_backprop",
    "branched_coef_fast_forward",
    "branched_coef_fast_backprop",
    "branched_conversion_fast_forward",
    "branched_conversion_fast_backprop",
    "branched_conversion_near_fast_forward",
    "branched_conversion_near_fast_backprop",
    "branched_log_fast_forward",
    "branched_log_fast_backprop",
)

ALL_CASES = FAST_CASES + tuple(MATRIX_CASES)
FALLBACK_CASES = tuple(
    name for name, spec in MATRIX_CASES.items() if spec.get("fallback"))


def command_output(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL,
            timeout=10).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def gpu_snapshot():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {}
    fields = (
        "uuid", "clocks.sm", "temperature.gpu", "power.draw",
        "power.limit", "driver_version",
    )
    output = command_output([
        executable, "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits", "--id=0",
    ])
    if output == "unknown":
        return {}
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    if len(values) != len(fields):
        return {}
    result = dict(zip(fields, values))
    for key in ("clocks.sm", "temperature.gpu", "power.draw", "power.limit"):
        try:
            result[key] = float(result[key])
        except ValueError:
            pass
    return result


def quantiles(values):
    import numpy as np

    p25, median, p75, p95 = np.quantile(values, (0.25, 0.5, 0.75, 0.95))
    return {
        "minimum": min(values),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
        "p95": float(p95),
        "mad": statistics.median(abs(value - median) for value in values),
    }


def capture(args):
    import pysiglib
    import pysiglib.torch_api
    from pysiglib._core.branched_log_sig import branched_log_sig as _native_branched_log_sig
    from pysiglib._core.branched_sig import branched_sig as _native_branched_sig
    from pysiglib._core.branched_sig_backprop import branched_sig_backprop as _native_branched_sig_backprop
    from pysiglib._core.branched_sig_coef import branched_sig_coef as _native_branched_sig_coef
    from pysiglib._core.branched_sig_coef_backprop import branched_sig_coef_backprop as _native_branched_sig_coef_backprop
    from pysiglib._core.branched_sig import branched_sig_combine as _native_branched_sig_combine
    from pysiglib._core.branched_sig_backprop import branched_sig_combine_backprop as _native_branched_sig_combine_backprop
    from pysiglib._core.branched_sig import branched_sig_length as _native_branched_sig_length
    from pysiglib._core.branched_log_sig import branched_sig_to_log_sig as _native_branched_sig_to_log_sig
    from pysiglib._core.branched_log_sig_backprop import branched_sig_to_log_sig_backprop as _native_branched_sig_to_log_sig_backprop
    from pysiglib._core.log_sig import log_sig as _native_log_sig
    from pysiglib._core.branched_log_sig import prepare_branched_log_sig as _native_prepare_branched_log_sig
    from pysiglib._core.branched_sig import prepare_branched_sig as _native_prepare_branched_sig
    from pysiglib._core.branched_sig_coef import prepare_branched_sig_coef as _native_prepare_branched_sig_coef
    from pysiglib._core.log_sig import prepare_log_sig as _native_prepare_log_sig
    from pysiglib._core.log_sig import set_cache_dir as _native_set_cache_dir
    from pysiglib._core.sig import sig as _native_sig
    from pysiglib._core.sig_backprop import sig_backprop as _native_sig_backprop
    import torch

    call, batch, path_steps = make_case(args.case, args.cache_dir)
    gpu_before = gpu_snapshot()
    torch.cuda.synchronize()
    first_start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    first_seconds = time.perf_counter() - first_start
    tensor = first_tensor(result)
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError("Benchmark result is not finite")

    for _ in range(args.warmups):
        result = call()
    torch.cuda.synchronize()

    repetitions = 1
    while True:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repetitions):
            result = call()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if elapsed >= args.min_sample_seconds:
            break
        repetitions = max(
            repetitions + 1,
            math.ceil(repetitions * args.min_sample_seconds * 1.1
                      / max(elapsed, 1e-9)),
        )

    wall_samples = []
    event_samples = []
    gpu_samples = []
    clock_input = None
    clock_result = None
    if args.stabilize_clock:
        clock_input = torch.randn(
            (2048, 2048), device="cuda", dtype=torch.float32)
        clock_result = torch.empty_like(clock_input)
    for _ in range(args.samples):
        if clock_input is not None:
            for _ in range(args.stabilize_clock):
                torch.mm(clock_input, clock_input, out=clock_result)
            torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        start_event.record()
        for _ in range(repetitions):
            result = call()
        end_event.record()
        torch.cuda.synchronize()
        wall_samples.append((time.perf_counter() - start) / repetitions)
        event_samples.append(
            start_event.elapsed_time(end_event) * 1e-3 / repetitions)
        gpu_samples.append(gpu_snapshot())

    gpu_after = gpu_snapshot()
    gpu_clocks = [
        sample["clocks.sm"] for sample in gpu_samples
        if isinstance(sample.get("clocks.sm"), (int, float))
    ]
    gpu_temperatures = [
        sample["temperature.gpu"] for sample in gpu_samples
        if isinstance(sample.get("temperature.gpu"), (int, float))
    ]
    properties = torch.cuda.get_device_properties(0)
    row = {
        "case": args.case,
        "round": args.round,
        "label": args.label,
        "revision": args.revision or git_revision(),
        "dirty": bool(command_output(["git", "status", "--porcelain"])
                      not in ("", "unknown")),
        "first_seconds": first_seconds,
        "repetitions": repetitions,
        "samples": args.samples,
        "batch": batch,
        "path_steps": path_steps,
        "wall": quantiles(wall_samples),
        "event": quantiles(event_samples),
        "wall_samples": wall_samples,
        "event_samples": event_samples,
        "batch_per_second": batch / statistics.median(wall_samples),
        "path_steps_per_second": path_steps / statistics.median(wall_samples),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch": torch.__version__,
        "pysiglib": getattr(pysiglib, "__version__", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_compiler": platform.python_compiler(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvcc": command_output(["nvcc", "--version"]),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "gpu_samples": gpu_samples,
        "gpu_clock_median": (
            statistics.median(gpu_clocks) if gpu_clocks else "unknown"),
        "gpu_temperature_median": (
            statistics.median(gpu_temperatures)
            if gpu_temperatures else "unknown"),
        "gpu_uuid": gpu_after.get("uuid", gpu_before.get("uuid", "unknown")),
        "driver": gpu_after.get(
            "driver_version", gpu_before.get("driver_version", "unknown")),
        "shared_memory_per_block": properties.shared_memory_per_block,
        "shared_memory_per_block_optin": getattr(
            properties, "shared_memory_per_block_optin", "unknown"),
        "shared_memory_per_multiprocessor": (
            properties.shared_memory_per_multiprocessor),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=("sample", "wall_seconds", "event_seconds"))
        writer.writeheader()
        for index, (wall, event) in enumerate(
                zip(wall_samples, event_samples), start=1):
            writer.writerow({
                "sample": index,
                "wall_seconds": wall,
                "event_seconds": event,
            })
    print(
        f"{args.case}: wall={1e3 * row['wall']['median']:.3f} ms, "
        f"event={1e3 * row['event']['median']:.3f} ms, "
        f"repetitions={repetitions}",
        flush=True,
    )


def compare(args):
    input_paths = []
    for path in args.input:
        if path.is_dir():
            input_paths.extend(sorted(path.rglob("*.json")))
        else:
            input_paths.append(path)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    grouped = {}
    for row in rows:
        grouped.setdefault((row["case"], row["label"]), []).append(row)
    cases = sorted({case for case, _ in grouped})
    output_rows = []
    failed = False
    for case in cases:
        baseline = sorted(grouped.get((case, "baseline"), []), key=lambda x: x["round"])
        candidate = sorted(grouped.get((case, "candidate"), []), key=lambda x: x["round"])
        if not baseline and candidate:
            output_rows.append({
                "case": case,
                "wall_ratio": "",
                "event_ratio": "",
                "wall_failed": False,
                "event_failed": False,
                "first_call_ratio": "",
                "environment_valid": True,
                "candidate_wall_seconds": statistics.median(
                    row["wall"]["median"] for row in candidate),
                "candidate_event_seconds": statistics.median(
                    row["event"]["median"] for row in candidate),
            })
            continue
        if not baseline or not candidate or len(baseline) != len(candidate):
            continue
        wall_ratios = [
            new["wall"]["median"] / old["wall"]["median"]
            for old, new in zip(baseline, candidate)
        ]
        event_ratios = [
            new["event"]["median"] / old["event"]["median"]
            for old, new in zip(baseline, candidate)
        ]
        wall_delta = statistics.median(
            row["wall"]["median"] for row in candidate) - statistics.median(
                row["wall"]["median"] for row in baseline)
        event_delta = statistics.median(
            row["event"]["median"] for row in candidate) - statistics.median(
                row["event"]["median"] for row in baseline)
        wall_failed = (
            statistics.median(wall_ratios) > 1.05
            and sum(ratio > 1.05 for ratio in wall_ratios) >= 2
            and wall_delta > 10e-6
        )
        event_failed = (
            statistics.median(event_ratios) > 1.05
            and sum(ratio > 1.05 for ratio in event_ratios) >= 2
            and event_delta > 2e-6
        )
        environment_valid = True
        for old, new in zip(baseline, candidate):
            old_gpu = old.get("gpu_after", {})
            new_gpu = new.get("gpu_after", {})
            old_clock = old.get("gpu_clock_median", old_gpu.get("clocks.sm"))
            new_clock = new.get("gpu_clock_median", new_gpu.get("clocks.sm"))
            old_temp = old.get(
                "gpu_temperature_median", old_gpu.get("temperature.gpu"))
            new_temp = new.get(
                "gpu_temperature_median", new_gpu.get("temperature.gpu"))
            if isinstance(old_clock, (int, float)) and isinstance(
                    new_clock, (int, float)) and old_clock:
                environment_valid &= abs(new_clock / old_clock - 1) <= 0.02
            if isinstance(old_temp, (int, float)) and isinstance(
                    new_temp, (int, float)):
                environment_valid &= abs(new_temp - old_temp) <= 5
        first_call_ratio = statistics.median(
            row["first_seconds"] for row in candidate) / statistics.median(
                row["first_seconds"] for row in baseline)
        failed = failed or wall_failed or event_failed or not environment_valid
        output_rows.append({
            "case": case,
            "wall_ratio": statistics.median(wall_ratios),
            "event_ratio": statistics.median(event_ratios),
            "wall_failed": wall_failed,
            "event_failed": event_failed,
            "first_call_ratio": first_call_ratio,
            "environment_valid": environment_valid,
            "candidate_wall_seconds": statistics.median(
                row["wall"]["median"] for row in candidate),
            "candidate_event_seconds": statistics.median(
                row["event"]["median"] for row in candidate),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        with args.output.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)
    for row in output_rows:
        if row["wall_ratio"] == "":
            print(
                f"{row['case']}: candidate wall="
                f"{1e3 * row['candidate_wall_seconds']:.3f} ms, event="
                f"{1e3 * row['candidate_event_seconds']:.3f} ms")
        else:
            first_note = " first-call>20%" if row["first_call_ratio"] > 1.2 else ""
            environment_note = " environment-invalid" if not row[
                "environment_valid"] else ""
            print(
                f"{row['case']}: wall={row['wall_ratio']:.4f}, "
                f"event={row['event_ratio']:.4f}"
                f"{first_note}{environment_note}")
    if failed:
        raise SystemExit(1)


def profile(args):
    import torch

    call, _, _ = make_case(args.case, args.cache_dir)
    for _ in range(args.warmups):
        call()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(args.case)
    try:
        for _ in range(args.repetitions):
            call()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--case", choices=ALL_CASES, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--cache-dir", type=Path, required=True)
    capture_parser.add_argument("--label", choices=("baseline", "candidate"), required=True)
    capture_parser.add_argument("--round", type=int, required=True)
    capture_parser.add_argument("--revision")
    capture_parser.add_argument("--warmups", type=int, default=5)
    capture_parser.add_argument("--samples", type=int, default=15)
    capture_parser.add_argument("--min-sample-seconds", type=float, default=0.2)
    capture_parser.add_argument(
        "--stabilize-clock", type=int, default=0, metavar="MATMULS")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--input", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--case", choices=ALL_CASES, required=True)
    profile_parser.add_argument("--cache-dir", type=Path, required=True)
    profile_parser.add_argument("--warmups", type=int, default=5)
    profile_parser.add_argument("--repetitions", type=int, default=1)
    list_parser = subparsers.add_parser("list")
    list_parser.set_defaults(list_cases=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "capture":
        capture(arguments)
    elif arguments.command == "compare":
        compare(arguments)
    elif arguments.command == "profile":
        profile(arguments)
    else:
        print("\n".join(ALL_CASES))
