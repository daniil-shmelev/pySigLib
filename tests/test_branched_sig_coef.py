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

import numpy as np
import pytest
import torch

import pysiglib


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_branched_sig_coef_matches_full(planar, dtype):
    dimension, degree = 3, 3
    pysiglib.prepare_branched_sig(dimension, degree, planar=planar)
    basis = pysiglib.trees(dimension, degree, planar=planar)
    requested = [basis[0], basis[1], basis[5], basis[-1], basis[5]]
    path = np.random.default_rng(101).normal(size=(2, 3, 7, dimension)).astype(dtype)

    actual = pysiglib.branched_sig_coef(path, requested, planar=planar)
    full = pysiglib.branched_sig(path, degree, planar=planar, scalar_term=True)
    indices = [
        pysiglib.tree_to_idx(
            tree, dimension, degree, planar=planar, scalar_term=True)
        for tree in requested
    ]

    tolerance = 2e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(actual, full[..., indices], rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "time_aug,lead_lag",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_branched_sig_coef_augmentation_matches_full(time_aug, lead_lag):
    dimension, degree = 2, 3
    augmented_dimension = (2 * dimension if lead_lag else dimension) + int(time_aug)
    pysiglib.prepare_branched_sig(
        dimension, degree, time_aug=time_aug, lead_lag=lead_lag)
    basis = pysiglib.trees(augmented_dimension, degree)
    requested = [basis[0], basis[augmented_dimension], basis[-1]]
    path = np.random.default_rng(102).normal(size=(3, 6, dimension))

    actual = pysiglib.branched_sig_coef(
        path, requested, time_aug=time_aug, lead_lag=lead_lag, end_time=2.0)
    full = pysiglib.branched_sig(
        path, degree, time_aug=time_aug, lead_lag=lead_lag,
        end_time=2.0, scalar_term=True)
    indices = [
        pysiglib.tree_to_idx(tree, augmented_dimension, degree, scalar_term=True)
        for tree in requested
    ]

    np.testing.assert_allclose(actual, full[..., indices], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("planar", [False, True])
def test_branched_sig_coef_correction_and_backprop_match_full(planar):
    dimension, degree, length = 2, 3, 6
    pysiglib.prepare_branched_sig(dimension, degree, planar=planar)
    basis = pysiglib.trees(dimension, degree, planar=planar)
    requested = [basis[0], basis[1], basis[4], basis[-2], basis[4]]
    rng = np.random.default_rng(103)
    path = rng.normal(size=(2, length, dimension))
    correction_len = dimension ** 2 + dimension ** 3
    correction = rng.normal(scale=0.01, size=(2, length - 1, correction_len))
    derivs = rng.normal(size=(2, len(requested)))

    coefs = pysiglib.branched_sig_coef(
        path, requested, planar=planar, correction=correction)
    full = pysiglib.branched_sig(
        path, degree, planar=planar, correction=correction, scalar_term=True)
    indices = [
        pysiglib.tree_to_idx(
            tree, dimension, degree, planar=planar, scalar_term=True)
        for tree in requested
    ]
    np.testing.assert_allclose(coefs, full[..., indices], rtol=1e-12, atol=1e-12)

    actual_grad = pysiglib.branched_sig_coef_backprop(
        path, requested, coefs, derivs, planar=planar, correction=correction)
    full_derivs = np.zeros_like(full)
    for i, index in enumerate(indices):
        full_derivs[..., index] += derivs[..., i]
    expected_grad = pysiglib.branched_sig_backprop(
        path, full, full_derivs, degree, planar=planar, correction=correction)

    np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("time_aug,lead_lag", [(True, False), (False, True), (True, True)])
def test_branched_sig_coef_backprop_augmentation_matches_full(time_aug, lead_lag):
    dimension, degree = 2, 3
    augmented_dimension = (2 * dimension if lead_lag else dimension) + int(time_aug)
    pysiglib.prepare_branched_sig(
        dimension, degree, time_aug=time_aug, lead_lag=lead_lag)
    basis = pysiglib.trees(augmented_dimension, degree)
    requested = [basis[1], basis[augmented_dimension + 2], basis[-1]]
    rng = np.random.default_rng(104)
    path = rng.normal(size=(2, 5, dimension))
    derivs = rng.normal(size=(2, len(requested)))
    options = dict(time_aug=time_aug, lead_lag=lead_lag, end_time=1.7)

    coefs = pysiglib.branched_sig_coef(path, requested, **options)
    actual_grad = pysiglib.branched_sig_coef_backprop(
        path, requested, coefs, derivs, **options)
    full = pysiglib.branched_sig(path, degree, scalar_term=True, **options)
    indices = [
        pysiglib.tree_to_idx(tree, augmented_dimension, degree, scalar_term=True)
        for tree in requested
    ]
    full_derivs = np.zeros_like(full)
    for i, index in enumerate(indices):
        full_derivs[..., index] += derivs[..., i]
    expected_grad = pysiglib.branched_sig_backprop(
        path, full, full_derivs, degree, **options)

    np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-11, atol=1e-11)


def test_branched_sig_coef_parallel_matches_serial():
    dimension, degree = 3, 3
    pysiglib.prepare_branched_sig(dimension, degree)
    basis = pysiglib.trees(dimension, degree)
    requested = [basis[1], basis[4], basis[12], basis[-1]]
    path = np.random.default_rng(105).normal(size=(12, 8, dimension))

    serial = pysiglib.branched_sig_coef(path, requested, n_jobs=1)
    parallel = pysiglib.branched_sig_coef(path, requested, n_jobs=2)
    np.testing.assert_array_equal(parallel, serial)

    derivs = np.ones_like(serial)
    serial_grad = pysiglib.branched_sig_coef_backprop(
        path, requested, serial, derivs, n_jobs=1)
    parallel_grad = pysiglib.branched_sig_coef_backprop(
        path, requested, parallel, derivs, n_jobs=2)
    np.testing.assert_array_equal(parallel_grad, serial_grad)


@pytest.mark.parametrize("planar", [False, True])
def test_torch_branched_sig_coef_value_and_gradient_match_full(planar):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_sig(dimension, degree, planar=planar)
    basis = pysiglib.trees(dimension, degree, planar=planar)
    requested = [basis[1], basis[4], basis[-1], basis[4]]
    path_coef = torch.randn(2, 6, dimension, dtype=torch.float64, requires_grad=True)
    path_full = path_coef.detach().clone().requires_grad_(True)
    weights = torch.randn(2, len(requested), dtype=torch.float64)

    coefs = pysiglib.torch_api.branched_sig_coef(
        path_coef, requested, planar=planar)
    coefs.backward(weights)

    full = pysiglib.torch_api.branched_sig(
        path_full, degree, planar=planar, scalar_term=True)
    indices = [
        pysiglib.tree_to_idx(
            tree, dimension, degree, planar=planar, scalar_term=True)
        for tree in requested
    ]
    full[..., indices].backward(weights)

    torch.testing.assert_close(coefs, full[..., indices], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(path_coef.grad, path_full.grad, rtol=1e-11, atol=1e-11)


def test_jax_branched_sig_coef_value_and_gradient_match_cpu():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    import pysiglib.jax_api as jax_api

    jax.config.update("jax_enable_x64", True)

    dimension, degree = 2, 3
    pysiglib.prepare_branched_sig(dimension, degree)
    basis = pysiglib.trees(dimension, degree)
    requested = [basis[1], basis[4], basis[-1], basis[4]]
    path = np.random.default_rng(106).normal(size=(2, 6, dimension))
    weights = np.random.default_rng(107).normal(size=(2, len(requested)))

    jax_path = jnp.asarray(path)
    jax_weights = jnp.asarray(weights)
    coefs = jax_api.branched_sig_coef(jax_path, requested)
    jax_grad = jax.grad(
        lambda value: jnp.sum(
            jax_api.branched_sig_coef(value, requested) * jax_weights)
    )(jax_path)

    cpu_coefs = pysiglib.branched_sig_coef(path, requested)
    cpu_grad = pysiglib.branched_sig_coef_backprop(
        path, requested, cpu_coefs, weights)
    np.testing.assert_allclose(np.asarray(coefs), cpu_coefs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax_grad), cpu_grad, rtol=1e-11, atol=1e-11)


def test_branched_sig_coef_validation():
    path = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="non-empty"):
        pysiglib.branched_sig_coef(path, [])
    with pytest.raises(ValueError):
        pysiglib.branched_sig_coef(path, (3,))
    pysiglib.prepare_branched_sig(2, 1)
    expected = pysiglib.branched_sig_coef(path, (0,))
    actual = pysiglib.torch_api.branched_sig_coef(path, (0,))
    np.testing.assert_array_equal(actual, expected)


def test_branched_sig_coef_rejects_empty_lead_lag_path():
    path = np.empty((0, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="at least one point"):
        pysiglib.branched_sig_coef(path, None, lead_lag=True)

    torch_path = torch.empty((0, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="at least one point"):
        pysiglib.torch_api.branched_sig_coef(torch_path, None, lead_lag=True)


def test_jax_branched_sig_coef_rejects_empty_lead_lag_path():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    import pysiglib.jax_api as jax_api

    with pytest.raises(ValueError, match="at least one point"):
        jax_api.branched_sig_coef(
            jnp.empty((0, 2), dtype=jnp.float32), None, lead_lag=True)


def test_branched_sig_coef_backprop_validates_path_metadata():
    dimension, degree = 2, 2
    pysiglib.prepare_branched_sig(dimension, degree)
    basis = pysiglib.trees(dimension, degree)
    requested = [basis[1], basis[-1]]
    path = np.arange(16, dtype=np.float64).reshape(2, 4, dimension) * 0.01
    coefs = pysiglib.branched_sig_coef(path, requested)

    single_coefs = coefs[0].copy()
    with pytest.raises(ValueError, match="batch shape"):
        pysiglib.branched_sig_coef_backprop(
            path, requested, single_coefs, np.ones_like(single_coefs))

    coefs_float = coefs.astype(np.float32)
    with pytest.raises(ValueError, match="same dtype"):
        pysiglib.branched_sig_coef_backprop(
            path, requested, coefs_float, np.ones_like(coefs_float))

    torch_coefs = torch.from_numpy(coefs)
    with pytest.raises(ValueError, match="same array type"):
        pysiglib.branched_sig_coef_backprop(
            path, requested, torch_coefs, torch.ones_like(torch_coefs))
