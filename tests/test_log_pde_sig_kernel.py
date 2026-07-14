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


def _random_paths(seed, batch, length, dimension=2):
    rng = np.random.default_rng(seed)
    increments = rng.normal(scale=0.04, size=(batch, length, dimension))
    return np.ascontiguousarray(np.cumsum(increments, axis=-2))


def test_log_pde_matches_tensordev_reference_grid():
    path1 = np.array(
        [[0.0, 0.0], [0.1, 0.2], [0.3, 0.1], [0.2, -0.1], [0.4, 0.0]],
        dtype=np.float64,
    )
    path2 = np.array(
        [
            [0.0, 0.0],
            [-0.1, 0.2],
            [0.1, 0.3],
            [0.2, 0.1],
            [0.0, -0.1],
            [0.3, 0.0],
            [0.2, 0.2],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0365700172524006, 1.040772345560078],
            [1.0, 1.0737800099166548, 1.0823609296996786],
            [1.0, 1.0792209359400757, 1.0824348112577176],
            [1.0, 1.0846752878260841, 1.0825079628081127],
        ],
        dtype=np.float64,
    )
    actual = pysiglib.sig_kernel(
        path1,
        path2,
        (1, 0),
        method="log_pde",
        log_degree=(3, 2),
        log_steps=(2, 3),
        return_grid=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_log_pde_degree_one_matches_tensordev_reference_grid():
    path1 = np.array(
        [[0.0, 0.0], [0.5, -0.2], [0.1, 0.7], [0.8, 0.4], [1.0, -0.1]],
        dtype=np.float64,
    )
    path2 = np.array(
        [
            [0.0, 0.0],
            [-0.3, 0.4],
            [0.6, 0.2],
            [0.2, -0.5],
            [0.9, 0.1],
            [0.4, 0.8],
            [1.1, 0.3],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.8418062500000001, 1.1661135810546877],
            [1.0, 0.6963045823046876, 1.3470576735436974],
            [1.0, 0.9611653480198525, 1.7118268936701422],
            [1.0, 1.2662871572967702, 2.145890324584627],
        ],
        dtype=np.float64,
    )
    actual = pysiglib.sig_kernel(
        path1,
        path2,
        (1, 0),
        method="log_pde",
        log_degree=1,
        log_steps=(2, 3),
        return_grid=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize(
    "dtype,batch_shape,lengths,dimension,degrees,log_steps,dyadic,return_grid,n_jobs",
    [
        (np.float32, (), (3, 5), 1, (1, 1), (1, 2), (0, 0), False, 1),
        (np.float64, (1,), (7, 9), 2, (2, 3), (3, 2), (1, 0), True, 1),
        (np.float32, (3,), (9, 13), 3, (3, 2), (2, 3), (0, 1), False, 2),
        (np.float64, (2, 2), (9, 7), 2, (4, 3), (4, 2), (1, 1), False, 4),
        (np.float64, (2,), (5, 9), 4, (2, 4), (2, 4), (0, 0), True, 2),
    ],
)
def test_log_pde_parameter_matrix_is_thread_consistent(
        dtype, batch_shape, lengths, dimension, degrees, log_steps,
        dyadic, return_grid, n_jobs):
    rng = np.random.default_rng(123)
    path1 = np.cumsum(
        rng.normal(scale=0.03, size=batch_shape + (lengths[0], dimension)),
        axis=-2,
        dtype=dtype,
    )
    path2 = np.cumsum(
        rng.normal(scale=0.03, size=batch_shape + (lengths[1], dimension)),
        axis=-2,
        dtype=dtype,
    )
    expected = pysiglib.sig_kernel(
        path1, path2, dyadic, method="log_pde", log_degree=degrees,
        log_steps=log_steps, return_grid=return_grid, n_jobs=1,
    )
    actual = pysiglib.sig_kernel(
        path1, path2, dyadic, method="log_pde", log_degree=degrees,
        log_steps=log_steps, return_grid=return_grid, n_jobs=n_jobs,
    )
    tolerance = 2e-5 if dtype == np.float32 else 2e-13
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_log_pde_standard_backprop_directional_derivative():
    path1 = _random_paths(1, 1, 7)
    path2 = _random_paths(2, 1, 9)
    direction1 = _random_paths(3, 1, 7)
    direction2 = _random_paths(4, 1, 9)
    kwargs = dict(
        method="log_pde",
        log_degree=(3, 2),
        log_steps=(3, 2),
    )
    grad1, grad2 = pysiglib.sig_kernel_backprop(
        np.ones(1), path1, path2, (1, 0), right_deriv=True, **kwargs
    )
    predicted = np.sum(grad1 * direction1) + np.sum(grad2 * direction2)
    eps = 1e-6
    plus = pysiglib.sig_kernel(
        path1 + eps * direction1,
        path2 + eps * direction2,
        (1, 0),
        **kwargs,
    )
    minus = pysiglib.sig_kernel(
        path1 - eps * direction1,
        path2 - eps * direction2,
        (1, 0),
        **kwargs,
    )
    observed = np.sum(plus - minus) / (2.0 * eps)
    np.testing.assert_allclose(predicted, observed, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize(
    "dtype,batch,lengths,dimension,degrees,log_steps,dyadic,return_grid,n_jobs,time_aug,lead_lag",
    [
        (np.float64, 2, (5, 7), 1, (1, 1), (2, 3), (0, 0), False, 1, False, False),
        (np.float64, 3, (7, 9), 2, (3, 2), (3, 2), (1, 0), True, 3, False, False),
        (np.float32, 2, (9, 13), 3, (2, 3), (4, 3), (0, 1), False, 2, False, False),
        (np.float64, 1, (5, 7), 2, (2, 2), (4, 3), (0, 0), True, 2, True, True),
    ],
)
def test_log_pde_backprop_parameter_matrix(
        dtype, batch, lengths, dimension, degrees, log_steps, dyadic,
        return_grid, n_jobs, time_aug, lead_lag):
    path1 = _random_paths(20, batch, lengths[0], dimension).astype(dtype)
    path2 = _random_paths(21, batch, lengths[1], dimension).astype(dtype)
    direction1 = _random_paths(22, batch, lengths[0], dimension).astype(dtype)
    direction2 = _random_paths(23, batch, lengths[1], dimension).astype(dtype)
    kwargs = dict(
        method="log_pde", log_degree=degrees, log_steps=log_steps,
        time_aug=time_aug, lead_lag=lead_lag, n_jobs=n_jobs,
        return_grid=return_grid,
    )
    value = pysiglib.sig_kernel(path1, path2, dyadic, **kwargs)
    derivs = np.random.default_rng(24).normal(size=value.shape).astype(dtype)
    grad1, grad2 = pysiglib.sig_kernel_backprop(
        derivs, path1, path2, dyadic, right_deriv=True, **kwargs
    )
    predicted = np.sum(grad1 * direction1) + np.sum(grad2 * direction2)
    eps = 2e-3 if dtype == np.float32 else 1e-6
    plus = pysiglib.sig_kernel(
        path1 + eps * direction1, path2 + eps * direction2, dyadic, **kwargs
    )
    minus = pysiglib.sig_kernel(
        path1 - eps * direction1, path2 - eps * direction2, dyadic, **kwargs
    )
    observed = np.sum((plus - minus) * derivs) / (2 * eps)
    tolerance = 2e-2 if dtype == np.float32 else 3e-7
    np.testing.assert_allclose(predicted, observed, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "dtype,batch_shape,lengths,dimension,degrees,log_steps,dyadic,return_grid,n_jobs,time_aug,lead_lag,use_torch",
    [
        (np.float64, (2,), (5, 7), 1, (1, 1), (2, 3), (0, 0), False, 1, False, False, False),
        (np.float64, (3,), (7, 9), 2, (3, 2), (3, 2), (1, 0), True, 2, False, False, False),
        (np.float32, (2,), (9, 13), 3, (2, 3), (4, 3), (0, 1), False, 2, False, False, True),
        (np.float64, (4,), (5, 7), 2, (2, 2), (4, 3), (0, 0), True, 3, True, True, False),
    ],
)
def test_log_pde_backprop_k_grid_matches_recomputation(
        dtype, batch_shape, lengths, dimension, degrees, log_steps, dyadic,
        return_grid, n_jobs, time_aug, lead_lag, use_torch):
    rng = np.random.default_rng(40)
    path1 = np.ascontiguousarray(np.cumsum(
        rng.normal(scale=0.03, size=batch_shape + (lengths[0], dimension)), axis=-2
    ).astype(dtype))
    path2 = np.ascontiguousarray(np.cumsum(
        rng.normal(scale=0.03, size=batch_shape + (lengths[1], dimension)), axis=-2
    ).astype(dtype))
    if use_torch:
        path1 = torch.from_numpy(path1)
        path2 = torch.from_numpy(path2)
    kwargs = dict(
        method="log_pde", log_degree=degrees, log_steps=log_steps,
        time_aug=time_aug, lead_lag=lead_lag, n_jobs=n_jobs,
    )
    k_grid = pysiglib.sig_kernel(
        path1, path2, dyadic, return_grid=True, **kwargs
    )
    value = k_grid if return_grid else k_grid[..., -1, -1]
    derivs = rng.normal(size=value.shape).astype(dtype)
    if use_torch:
        derivs = torch.from_numpy(derivs)
    expected = pysiglib.sig_kernel_backprop(
        derivs, path1, path2, dyadic, right_deriv=True,
        return_grid=return_grid, **kwargs
    )
    actual = pysiglib.sig_kernel_backprop(
        derivs, path1, path2, dyadic, right_deriv=True,
        k_grid=k_grid, return_grid=return_grid, **kwargs
    )
    np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(expected[0]))
    np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(expected[1]))


def test_log_pde_torch_and_jax_autodiff_match_standard_api():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy")
    jax_api = pytest.importorskip("pysiglib.jax_api")
    torch_api = pytest.importorskip("pysiglib.torch_api")

    path1 = _random_paths(5, 2, 7)
    path2 = _random_paths(6, 2, 9)
    kwargs = dict(
        method="log_pde",
        log_degree=(3, 2),
        log_steps=(3, 2),
    )
    expected = pysiglib.sig_kernel(path1, path2, (1, 0), **kwargs)
    expected_grad1, expected_grad2 = pysiglib.sig_kernel_backprop(
        np.ones(2), path1, path2, (1, 0), right_deriv=True, **kwargs
    )

    torch_path1 = torch.tensor(path1, requires_grad=True)
    torch_path2 = torch.tensor(path2, requires_grad=True)
    torch_value = torch_api.sig_kernel(
        torch_path1, torch_path2, (1, 0), **kwargs
    )
    torch_value.sum().backward()

    def objective(x, y):
        return jnp.sum(jax_api.sig_kernel(x, y, (1, 0), **kwargs))

    jax_path1 = jnp.asarray(path1)
    jax_path2 = jnp.asarray(path2)
    jax_value = jax.jit(
        lambda x, y: jax_api.sig_kernel(x, y, (1, 0), **kwargs)
    )(jax_path1, jax_path2)
    jax_grad1, jax_grad2 = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        jax_path1, jax_path2
    )

    np.testing.assert_allclose(torch_value.detach().numpy(), expected, atol=1e-14)
    np.testing.assert_allclose(torch_path1.grad.numpy(), expected_grad1, atol=1e-14)
    np.testing.assert_allclose(torch_path2.grad.numpy(), expected_grad2, atol=1e-14)
    np.testing.assert_allclose(np.asarray(jax_value), expected, atol=1e-14)
    np.testing.assert_allclose(np.asarray(jax_grad1), expected_grad1, atol=1e-14)
    np.testing.assert_allclose(np.asarray(jax_grad2), expected_grad2, atol=1e-14)


def test_log_pde_gram_values_and_gradients_match_across_apis():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy")
    jax_api = pytest.importorskip("pysiglib.jax_api")
    torch_api = pytest.importorskip("pysiglib.torch_api")

    path1 = _random_paths(7, 3, 7)
    path2 = _random_paths(8, 2, 9)
    derivs = np.random.default_rng(9).normal(size=(3, 2))
    kwargs = dict(
        method="log_pde",
        log_degree=(3, 2),
        log_steps=(3, 2),
    )
    expected = pysiglib.sig_kernel_gram(path1, path2, (1, 0), **kwargs)
    expected_grad1, expected_grad2 = pysiglib.sig_kernel_gram_backprop(
        derivs,
        path1,
        path2,
        (1, 0),
        right_deriv=True,
        max_batch=2,
        **kwargs,
    )

    torch_path1 = torch.tensor(path1, requires_grad=True)
    torch_path2 = torch.tensor(path2, requires_grad=True)
    torch_value = torch_api.sig_kernel_gram(
        torch_path1, torch_path2, (1, 0), max_batch=2, **kwargs
    )
    (torch_value * torch.tensor(derivs)).sum().backward()

    def objective(x, y):
        value = jax_api.sig_kernel_gram(x, y, (1, 0), max_batch=2, **kwargs)
        return jnp.sum(value * jnp.asarray(derivs))

    jax_path1 = jnp.asarray(path1)
    jax_path2 = jnp.asarray(path2)
    jax_value = jax.jit(
        lambda x, y: jax_api.sig_kernel_gram(x, y, (1, 0), max_batch=2, **kwargs)
    )(jax_path1, jax_path2)
    jax_grad1, jax_grad2 = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        jax_path1, jax_path2
    )

    np.testing.assert_allclose(torch_value.detach().numpy(), expected, atol=1e-14)
    np.testing.assert_allclose(torch_path1.grad.numpy(), expected_grad1, atol=1e-14)
    np.testing.assert_allclose(torch_path2.grad.numpy(), expected_grad2, atol=1e-14)
    np.testing.assert_allclose(np.asarray(jax_value), expected, atol=1e-14)
    np.testing.assert_allclose(np.asarray(jax_grad1), expected_grad1, atol=1e-14)
    np.testing.assert_allclose(np.asarray(jax_grad2), expected_grad2, atol=1e-14)


@pytest.mark.parametrize("symmetric,return_grid", [(False, False), (True, True)])
def test_log_pde_gram_backprop_k_grid_matches_recomputation(symmetric, return_grid):
    path1 = _random_paths(50, 3, 7)
    path2 = path1 if symmetric else _random_paths(51, 2, 9)
    degrees = 3 if symmetric else (3, 2)
    log_steps = 3 if symmetric else (3, 2)
    dyadic = (1, 1) if symmetric else (1, 0)
    kwargs = dict(
        method="log_pde", log_degree=degrees, log_steps=log_steps,
        n_jobs=2, max_batch=2,
    )
    k_grid = pysiglib.sig_kernel_gram(
        path1, path2, dyadic, return_grid=True, **kwargs
    )
    value = k_grid if return_grid else k_grid[..., -1, -1]
    derivs = np.random.default_rng(52).normal(size=value.shape)
    expected = pysiglib.sig_kernel_gram_backprop(
        derivs, path1, path2, dyadic, right_deriv=True,
        return_grid=return_grid, **kwargs
    )
    actual = pysiglib.sig_kernel_gram_backprop(
        derivs, path1, path2, dyadic, right_deriv=True,
        k_grid=k_grid, return_grid=return_grid, **kwargs
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_log_pde_metrics_match_manual_gram_formulas_across_apis():
    jnp = pytest.importorskip("jax.numpy")
    jax_api = pytest.importorskip("pysiglib.jax_api")
    torch_api = pytest.importorskip("pysiglib.torch_api")

    sample1 = _random_paths(30, 3, 7)
    sample2 = _random_paths(31, 2, 9)
    lam = 0.4
    kwargs = dict(
        method="log_pde",
        log_degree=(3, 2),
        log_steps=(3, 4),
        lead_lag=True,
        n_jobs=2,
        max_batch=2,
    )
    xx = pysiglib.sig_kernel_gram(
        sample1, sample1, (1, 1), method="log_pde",
        log_degree=3, log_steps=3, lead_lag=True, n_jobs=2, max_batch=2,
    )
    xy = pysiglib.sig_kernel_gram(sample1, sample2, (1, 0), **kwargs)
    yy = pysiglib.sig_kernel_gram(
        sample2, sample2, (0, 0), method="log_pde",
        log_degree=2, log_steps=4, lead_lag=True, n_jobs=2, max_batch=2,
    )
    xx_mean = (np.sum(xx) - np.trace(xx)) / 6
    xy_mean = 2.0 * np.sum(xy, axis=0) / 3
    yy_mean = (np.sum(yy) - np.trace(yy)) / 2
    expected_score = lam * xx_mean - xy_mean
    expected_expected_score = expected_score.mean().reshape(1)
    expected_mmd = xx_mean - 2.0 * np.mean(xy) + yy_mean

    cases = [
        (pysiglib, sample1, sample2, np.asarray),
        (
            torch_api,
            torch.as_tensor(sample1),
            torch.as_tensor(sample2),
            lambda value: value.detach().numpy(),
        ),
        (jax_api, jnp.asarray(sample1), jnp.asarray(sample2), np.asarray),
    ]
    for api, x, y, to_numpy in cases:
        score = api.sig_score(x, y, (1, 0), lam=lam, **kwargs)
        expected_score_value = api.expected_sig_score(
            x, y, (1, 0), lam=lam, **kwargs
        )
        mmd = api.sig_mmd(x, y, (1, 0), **kwargs)
        np.testing.assert_allclose(
            to_numpy(score), expected_score, rtol=2e-13, atol=2e-13
        )
        np.testing.assert_allclose(
            to_numpy(expected_score_value), expected_expected_score,
            rtol=2e-13, atol=2e-13,
        )
        np.testing.assert_allclose(
            to_numpy(mmd), expected_mmd, rtol=2e-13, atol=2e-13
        )


def test_log_pde_gram_backprop_with_path_augmentation():
    torch_api = pytest.importorskip("pysiglib.torch_api")
    path1 = _random_paths(10, 1, 5)
    path2 = _random_paths(11, 1, 5)
    kwargs = dict(
        method="log_pde",
        log_degree=2,
        log_steps=2,
        time_aug=True,
        lead_lag=True,
    )
    expected_grad1, expected_grad2 = pysiglib.sig_kernel_gram_backprop(
        np.ones((1, 1)), path1, path2, 0,
        right_deriv=True, **kwargs
    )
    torch_path1 = torch.tensor(path1, requires_grad=True)
    torch_path2 = torch.tensor(path2, requires_grad=True)
    torch_api.sig_kernel_gram(torch_path1, torch_path2, 0, **kwargs).sum().backward()
    np.testing.assert_allclose(torch_path1.grad.numpy(), expected_grad1, atol=1e-14)
    np.testing.assert_allclose(torch_path2.grad.numpy(), expected_grad2, atol=1e-14)


def test_log_pde_lead_lag_log_steps_count_original_intervals():
    jnp = pytest.importorskip("jax.numpy")
    jax_api = pytest.importorskip("pysiglib.jax_api")
    torch_api = pytest.importorskip("pysiglib.torch_api")
    path1 = _random_paths(15, 2, 5)
    path2 = _random_paths(16, 1, 7)
    transformed1 = pysiglib.transform_path(path1, lead_lag=True)
    transformed2 = pysiglib.transform_path(path2, lead_lag=True)
    kwargs = dict(
        method="log_pde",
        log_degree=(2, 2),
        log_steps=(2, 3),
        return_grid=True,
    )
    expected = pysiglib.sig_kernel_gram(
        transformed1,
        transformed2,
        (1, 0),
        method="log_pde",
        log_degree=(2, 2),
        log_steps=(4, 6),
        return_grid=True,
    )
    actual = pysiglib.sig_kernel_gram(
        path1, path2, (1, 0), lead_lag=True, **kwargs
    )
    torch_actual = torch_api.sig_kernel_gram(
        torch.as_tensor(path1), torch.as_tensor(path2),
        (1, 0), lead_lag=True, **kwargs
    )
    jax_actual = jax_api.sig_kernel_gram(
        jnp.asarray(path1), jnp.asarray(path2),
        (1, 0), lead_lag=True, **kwargs
    )
    assert actual.shape == (2, 1, 5, 3)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(torch_actual.detach().numpy(), expected, atol=2e-15)
    np.testing.assert_allclose(np.asarray(jax_actual), expected, atol=2e-15)


def test_log_pde_normalization_uses_side_specific_refinement():
    path1 = _random_paths(12, 1, 7)
    path2 = _random_paths(13, 1, 9)
    kwargs = dict(
        method="log_pde",
        log_degree=(3, 2),
        log_steps=(3, 2),
    )
    raw = pysiglib.sig_kernel(path1, path2, (1, 0), **kwargs)
    self1 = pysiglib.sig_kernel(
        path1, path1, (1, 1), method="log_pde", log_degree=3, log_steps=3
    )
    self2 = pysiglib.sig_kernel(
        path2, path2, (0, 0), method="log_pde", log_degree=2, log_steps=2
    )
    normalized = pysiglib.sig_kernel(
        path1, path2, (1, 0), normalize=True, **kwargs
    )
    np.testing.assert_allclose(normalized, raw / np.sqrt(self1 * self2), atol=1e-14)


def test_log_pde_parameter_validation():
    path = _random_paths(14, 1, 5)
    np.testing.assert_allclose(
        pysiglib.sig_kernel(path, path, 0),
        pysiglib.sig_kernel(path, path, 0, method="pde"),
    )
    with pytest.raises(ValueError, match="method must be 'pde' or 'log_pde'"):
        pysiglib.sig_kernel(path, path, 0, method="finite_difference")
    with pytest.raises(ValueError, match="requires log_degree and log_steps"):
        pysiglib.sig_kernel(path, path, 0, method="log_pde")
    with pytest.raises(ValueError, match="must divide"):
        pysiglib.sig_kernel(
            path, path, 0, method="log_pde", log_degree=2, log_steps=3
        )
    with pytest.raises(ValueError, match="linear static kernel"):
        pysiglib.sig_kernel(
            path,
            path,
            0,
            method="log_pde",
            log_degree=2,
            log_steps=2,
            static_kernel=pysiglib.RBFKernel(sigma=1.0),
        )


def test_log_pde_k_grid_validation():
    path = _random_paths(60, 2, 5)
    kwargs = dict(method="log_pde", log_degree=2, log_steps=2)
    derivs = np.ones(2)
    with pytest.raises(ValueError, match="k_grid has shape"):
        pysiglib.sig_kernel_backprop(
            derivs, path, path, 0, k_grid=np.ones((2, 2, 2)), **kwargs
        )
    with pytest.raises(ValueError, match="same array type, device and dtype"):
        pysiglib.sig_kernel_backprop(
            derivs, path, path, 0, k_grid=np.ones((2, 3, 3), dtype=np.float32),
            **kwargs,
        )


_CUDA_AVAILABLE = pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
@pytest.mark.parametrize(
    "dtype,batch_shape,lengths,dimension,degrees,log_steps,dyadic,return_grid,checkpoint,time_aug,lead_lag",
    [
        (np.float32, (), (3, 5), 1, (1, 1), (1, 2), (0, 0), False, False, False, False),
        (np.float64, (1,), (7, 9), 2, (2, 3), (3, 2), (1, 0), True, True, False, False),
        (np.float32, (3,), (9, 13), 3, (3, 2), (4, 3), (0, 1), False, True, False, False),
        (np.float64, (2, 2), (9, 7), 2, (4, 3), (4, 2), (1, 1), False, False, False, False),
        (np.float64, (2,), (5, 7), 2, (2, 2), (2, 3), (0, 0), True, True, True, True),
    ],
)
def test_log_pde_cuda_standard_matches_cpu(
        dtype, batch_shape, lengths, dimension, degrees, log_steps, dyadic,
        return_grid, checkpoint, time_aug, lead_lag):
    rng = np.random.default_rng(70)
    path1 = np.ascontiguousarray(np.cumsum(
        rng.normal(scale=0.03, size=batch_shape + (lengths[0], dimension)), axis=-2
    ).astype(dtype))
    path2 = np.ascontiguousarray(np.cumsum(
        rng.normal(scale=0.03, size=batch_shape + (lengths[1], dimension)), axis=-2
    ).astype(dtype))
    kwargs = dict(
        method="log_pde", log_degree=degrees, log_steps=log_steps,
        time_aug=time_aug, lead_lag=lead_lag,
    )

    expected = pysiglib.sig_kernel(
        path1, path2, dyadic, return_grid=return_grid, **kwargs
    )
    path1_cuda = torch.as_tensor(path1, device="cuda")
    path2_cuda = torch.as_tensor(path2, device="cuda")
    actual = pysiglib.sig_kernel(
        path1_cuda, path2_cuda, dyadic, return_grid=return_grid, **kwargs
    )

    derivs = np.asarray(rng.normal(size=expected.shape), dtype=dtype)
    cpu_grid = None
    cuda_grid = None
    if checkpoint:
        cpu_grid = pysiglib.sig_kernel(
            path1, path2, dyadic, return_grid=True, **kwargs
        )
        cuda_grid = actual if return_grid else pysiglib.sig_kernel(
            path1_cuda, path2_cuda, dyadic, return_grid=True, **kwargs
        )
    expected_grad = pysiglib.sig_kernel_backprop(
        derivs, path1, path2, dyadic, right_deriv=True,
        k_grid=cpu_grid, return_grid=return_grid, **kwargs,
    )
    actual_grad = pysiglib.sig_kernel_backprop(
        torch.as_tensor(derivs, device="cuda"), path1_cuda, path2_cuda,
        dyadic, right_deriv=True, k_grid=cuda_grid,
        return_grid=return_grid, **kwargs,
    )

    tolerance = 5e-4 if dtype == np.float32 else 2e-10
    np.testing.assert_allclose(
        actual.cpu().numpy(), expected, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        actual_grad[0].cpu().numpy(), expected_grad[0],
        rtol=tolerance, atol=tolerance,
    )
    np.testing.assert_allclose(
        actual_grad[1].cpu().numpy(), expected_grad[1],
        rtol=tolerance, atol=tolerance,
    )


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
def test_log_pde_cuda_empty_batch():
    path1 = torch.empty((0, 5, 2), dtype=torch.float64, device="cuda")
    path2 = torch.empty((0, 7, 2), dtype=torch.float64, device="cuda")
    kwargs = dict(method="log_pde", log_degree=(2, 3), log_steps=(2, 3))
    value = pysiglib.sig_kernel(path1, path2, 0, **kwargs)
    grad1, grad2 = pysiglib.sig_kernel_backprop(
        torch.empty(0, dtype=torch.float64, device="cuda"),
        path1, path2, 0, right_deriv=True, **kwargs,
    )
    assert value.shape == (0,)
    assert grad1.shape == path1.shape
    assert grad2.shape == path2.shape


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
def test_log_pde_torch_cuda_autograd_and_gram_match_cpu():
    torch_api = pytest.importorskip("pysiglib.torch_api")
    path1 = _random_paths(80, 3, 7)
    path2 = _random_paths(81, 2, 9)
    kwargs = dict(
        method="log_pde", log_degree=(3, 2), log_steps=(3, 4),
    )

    expected_gram = pysiglib.sig_kernel_gram(
        path1, path2, (1, 0), max_batch=2, **kwargs
    )
    weights = np.random.default_rng(82).normal(size=expected_gram.shape)
    expected_grad = pysiglib.sig_kernel_gram_backprop(
        weights, path1, path2, (1, 0), right_deriv=True,
        max_batch=2, **kwargs,
    )

    path1_cuda = torch.as_tensor(path1, device="cuda").requires_grad_()
    path2_cuda = torch.as_tensor(path2, device="cuda").requires_grad_()
    actual_gram = torch_api.sig_kernel_gram(
        path1_cuda, path2_cuda, (1, 0), max_batch=2, **kwargs
    )
    (actual_gram * torch.as_tensor(weights, device="cuda")).sum().backward()

    np.testing.assert_allclose(actual_gram.detach().cpu().numpy(), expected_gram, atol=2e-10)
    np.testing.assert_allclose(path1_cuda.grad.cpu().numpy(), expected_grad[0], atol=2e-10)
    np.testing.assert_allclose(path2_cuda.grad.cpu().numpy(), expected_grad[1], atol=2e-10)
    expected_mmd = pysiglib.sig_mmd(
        path1, path2, (1, 0), max_batch=2, **kwargs
    )
    actual_mmd = torch_api.sig_mmd(
        path1_cuda.detach(), path2_cuda.detach(), (1, 0), max_batch=2, **kwargs
    )
    np.testing.assert_allclose(actual_mmd.cpu().numpy(), expected_mmd, atol=2e-10)


def test_log_pde_torch_saved_grid_detects_inplace_mutation():
    torch_api = pytest.importorskip("pysiglib.torch_api")
    path1 = torch.as_tensor(_random_paths(83, 1, 7)).requires_grad_()
    path2 = torch.as_tensor(_random_paths(84, 1, 7)).requires_grad_()
    value = torch_api.sig_kernel(
        path1, path2, 0, method="log_pde", log_degree=3,
        log_steps=3, return_grid=True,
    )
    value.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        value.sum().backward()


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
def test_log_pde_jax_cuda_jit_and_grad_match_cpu():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax_api = pytest.importorskip("pysiglib.jax_api")
    devices = [device for device in jax.devices() if device.platform in {"gpu", "cuda"}]
    if not devices:
        pytest.skip("JAX has no CUDA device")
    jax.config.update("jax_enable_x64", True)

    path1 = _random_paths(85, 2, 7)
    path2 = _random_paths(86, 2, 9)
    kwargs = dict(
        method="log_pde", log_degree=(3, 2), log_steps=(3, 2),
        return_grid=True,
    )
    expected = pysiglib.sig_kernel(path1, path2, (1, 0), **kwargs)
    weights = np.random.default_rng(87).normal(size=expected.shape)
    expected_grad = pysiglib.sig_kernel_backprop(
        weights, path1, path2, (1, 0), right_deriv=True, **kwargs
    )

    device = devices[0]
    path1_cuda = jax.device_put(path1, device=device)
    path2_cuda = jax.device_put(path2, device=device)
    weights_cuda = jax.device_put(weights, device=device)
    kernel = jax.jit(lambda x, y: jax_api.sig_kernel(x, y, (1, 0), **kwargs))
    objective = lambda x, y: jnp.sum(kernel(x, y) * weights_cuda)
    actual = kernel(path1_cuda, path2_cuda)
    actual_grad = jax.jit(jax.grad(objective, argnums=(0, 1)))(path1_cuda, path2_cuda)

    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-10)
    np.testing.assert_allclose(np.asarray(actual_grad[0]), expected_grad[0], atol=2e-10)
    np.testing.assert_allclose(np.asarray(actual_grad[1]), expected_grad[1], atol=2e-10)
