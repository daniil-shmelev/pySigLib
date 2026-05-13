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

import json
from pathlib import Path

import kauri
import numpy as np
import pytest
import torch

import pysiglib
import pysiglib.torch_api as torch_api
from conftest import check_close
from test_branched_sig import (
    compute_kauri_to_pysiglib_permutation,
    enumerate_decorated_trees,
    linear_branched_sig_ref,
    reorder_kauri_to_pysiglib,
)


FIXTURE = Path(__file__).parent / "fixtures" / "branched_log_sig_stochastax_degree2.json"


def _branched_sig_map_reference(path, d, N):
    trees = enumerate_decorated_trees(d, N)
    X = kauri.Map(lambda t: 1.0 if t.nodes() == 0 else 0.0)

    for n in range(len(path) - 1):
        z = path[n + 1] - path[n]
        coeffs = linear_branched_sig_ref(z, trees)

        def make_char(c):
            def char_func(t):
                if t.nodes() == 0:
                    return 1.0
                return c.get(t.sorted_list_repr(), 0.0)
            return char_func

        X = X * kauri.Map(make_char(coeffs))
    return X


def _kauri_log_reference(path, d, N):
    trees = enumerate_decorated_trees(d, N)
    sig_map = _branched_sig_map_reference(path, d, N)
    h = kauri.Map(lambda t: 0.0 if t.nodes() == 0 else sig_map(t))
    acc = h
    power = h

    for k in range(2, N + 1):
        power = power * h
        coeff = -1.0 / k if k % 2 == 0 else 1.0 / k
        prev = acc
        acc = kauri.Map(lambda t, prev=prev, power=power, coeff=coeff: prev(t) + coeff * power(t))

    return np.array([acc(t) for t in trees])


def test_branched_sig_to_log_sig_stochastax_degree2_fixture():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    d = fixture["dimension"]
    N = fixture["degree"]
    pysiglib.prepare_branched_sig(d, N)

    bsig = np.array(fixture["signature_scalar"], dtype=np.float64)
    expected = np.array(fixture["log_signature_scalar"], dtype=np.float64)

    out = pysiglib.branched_sig_to_log_sig(bsig, d, N)
    np.testing.assert_allclose(out, expected, atol=1e-14)

    path = np.array(fixture["path"], dtype=np.float64)
    direct = pysiglib.branched_log_sig(path, N, scalar_term=True)
    np.testing.assert_allclose(direct, expected, atol=1e-14)


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_branched_log_sig_vs_kauri_hopf_log(d, N):
    pysiglib.prepare_branched_sig(d, N)
    perm = compute_kauri_to_pysiglib_permutation(d, N)

    rng = np.random.default_rng(1234)
    path = np.cumsum(rng.normal(scale=0.1, size=(6, d)), axis=0)

    out = pysiglib.branched_log_sig(path, N)
    ref = _kauri_log_reference(path, d, N)
    ref_reordered = reorder_kauri_to_pysiglib(ref, perm)

    np.testing.assert_allclose(out, ref_reordered, atol=1e-10)


def test_branched_sig_to_log_sig_scalar_term_roundtrip():
    d, N = 2, 3
    pysiglib.prepare_branched_sig(d, N)
    path = np.array([[0.0, 0.0], [0.2, 0.5], [0.8, -0.1]], dtype=np.float64)
    bsig_scalar = pysiglib.branched_sig(path, N, scalar_term=True)
    bsig_tail = bsig_scalar[1:]

    logsig_scalar = pysiglib.branched_sig_to_log_sig(bsig_scalar, d, N)
    logsig_tail = pysiglib.branched_sig_to_log_sig(bsig_tail, d, N)

    assert logsig_scalar[0] == 0.0
    np.testing.assert_allclose(logsig_scalar[1:], logsig_tail, atol=1e-14)


@pytest.mark.parametrize("time_aug,lead_lag", [(True, False), (False, True), (True, True)])
def test_branched_sig_to_log_sig_uses_path_dimension_with_augmentation(time_aug, lead_lag):
    d, N = 2, 2
    pysiglib.prepare_branched_sig(d, N, time_aug=time_aug, lead_lag=lead_lag)

    rng = np.random.default_rng(9876)
    path = np.cumsum(rng.normal(scale=0.1, size=(5, d)), axis=0)

    bsig = pysiglib.branched_sig(
        path, N, time_aug=time_aug, lead_lag=lead_lag, scalar_term=True)
    direct = pysiglib.branched_log_sig(
        path, N, time_aug=time_aug, lead_lag=lead_lag, scalar_term=True)
    converted = pysiglib.branched_sig_to_log_sig(
        bsig, d, N, time_aug=time_aug, lead_lag=lead_lag)

    np.testing.assert_allclose(converted, direct, atol=1e-12)


def test_branched_sig_to_log_sig_backprop_finite_difference():
    d, N = 2, 3
    pysiglib.prepare_branched_sig(d, N)
    rng = np.random.default_rng(4321)
    path = np.cumsum(rng.normal(scale=0.2, size=(5, d)), axis=0)
    bsig = pysiglib.branched_sig(path, N, scalar_term=True)
    derivs = rng.normal(size=bsig.shape)

    grad = pysiglib.branched_sig_to_log_sig_backprop(bsig, derivs, d, N)
    eps = 1e-6
    fd = np.zeros_like(bsig)
    for i in range(bsig.shape[0]):
        plus = bsig.copy()
        minus = bsig.copy()
        plus[i] += eps
        minus[i] -= eps
        f_plus = np.dot(pysiglib.branched_sig_to_log_sig(plus, d, N), derivs)
        f_minus = np.dot(pysiglib.branched_sig_to_log_sig(minus, d, N), derivs)
        fd[i] = (f_plus - f_minus) / (2 * eps)

    np.testing.assert_allclose(grad, fd, atol=2e-6, rtol=2e-6)


def test_torch_branched_sig_to_log_sig_backward_matches_explicit_backprop():
    d, N = 2, 3
    pysiglib.prepare_branched_sig(d, N)
    path = np.array([[0.0, 0.0], [0.2, -0.3], [0.7, 0.4]], dtype=np.float64)
    bsig_np = pysiglib.branched_sig(path, N, scalar_term=True)
    weights_np = np.linspace(-0.4, 0.7, bsig_np.shape[0])

    bsig = torch.tensor(bsig_np, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor(weights_np, dtype=torch.float64)
    out = torch_api.branched_sig_to_log_sig(bsig, d, N)
    (out * weights).sum().backward()

    expected = pysiglib.branched_sig_to_log_sig_backprop(bsig_np, weights_np, d, N)
    check_close(bsig.grad, torch.tensor(expected, dtype=torch.float64), atol=1e-10)


def test_torch_branched_log_sig_backward_runs():
    d, N = 2, 3
    path = torch.tensor(
        [[0.0, 0.0], [0.2, -0.3], [0.7, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )

    out = torch_api.branched_log_sig(path, N, scalar_term=True)
    out.sum().backward()

    assert path.grad is not None
    assert torch.all(torch.isfinite(path.grad))


def test_torch_branched_sig_to_log_sig_uses_path_dimension_with_augmentation():
    d, N = 2, 2
    path = torch.tensor(
        [[0.0, 0.0], [0.2, -0.3], [0.7, 0.4]],
        dtype=torch.float64,
    )
    bsig = torch_api.branched_sig(path, N, time_aug=True, scalar_term=True)
    bsig = bsig.detach().clone().requires_grad_(True)

    converted = torch_api.branched_sig_to_log_sig(bsig, d, N, time_aug=True)
    direct = torch_api.branched_log_sig(path, N, time_aug=True, scalar_term=True)

    check_close(converted, direct, atol=1e-12)
    converted.sum().backward()
    assert bsig.grad is not None
    assert torch.all(torch.isfinite(bsig.grad))


def test_jax_branched_sig_to_log_sig_value_and_grad():
    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import pysiglib.jax_api as jax_api

    d, N = 2, 2
    pysiglib.prepare_branched_sig(d, N)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    bsig = jnp.asarray(fixture["signature_scalar"], dtype=jnp.float64)
    expected = np.array(fixture["log_signature_scalar"], dtype=np.float64)

    out = jax_api.branched_sig_to_log_sig(bsig, d, N)
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-10)

    grad = jax.grad(lambda x: jnp.sum(jax_api.branched_sig_to_log_sig(x, d, N)))(bsig)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_jax_branched_sig_to_log_sig_uses_path_dimension_with_augmentation():
    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import pysiglib.jax_api as jax_api

    d, N = 2, 2
    pysiglib.prepare_branched_sig(d, N, lead_lag=True)

    path = jnp.asarray(
        [[0.0, 0.0], [0.2, -0.3], [0.7, 0.4]],
        dtype=jnp.float64,
    )
    bsig = jax_api.branched_sig(path, N, lead_lag=True, scalar_term=True)

    converted = jax_api.branched_sig_to_log_sig(bsig, d, N, lead_lag=True)
    direct = jax_api.branched_log_sig(path, N, lead_lag=True, scalar_term=True)

    np.testing.assert_allclose(np.asarray(converted), np.asarray(direct), atol=1e-12)
    grad = jax.grad(lambda x: jnp.sum(jax_api.branched_sig_to_log_sig(x, d, N, lead_lag=True)))(bsig)
    assert np.all(np.isfinite(np.asarray(grad)))
