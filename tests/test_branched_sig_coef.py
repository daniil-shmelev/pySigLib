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
from conftest import DEVICES, assert_device, skip_no_cuda


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("scalar_term", [False, True])
def test_extract_branched_sig_coef_all(device, planar, scalar_term):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_sig(
        dimension, degree, planar=planar, device=device)
    path = torch.rand((2, 5, dimension), dtype=torch.float64, device=device)
    bsig = pysiglib.branched_sig(
        path, degree, planar=planar, scalar_term=scalar_term)
    basis = pysiglib.trees(dimension, degree, planar=planar)
    requested = list(basis if scalar_term else basis[1:])

    actual = pysiglib.extract_branched_sig_coef(
        bsig,
        requested,
        dimension,
        planar=planar,
        scalar_term=scalar_term,
    )

    assert_device(actual, device)
    torch.testing.assert_close(actual, bsig, rtol=0, atol=0)


def test_extract_branched_sig_coef_from_higher_degree():
    dimension, degree = 2, 4
    pysiglib.prepare_branched_sig(dimension, degree, device="cpu")
    path = np.random.default_rng(100).normal(size=(3, 6, dimension))
    bsig = pysiglib.branched_sig(path, degree)
    basis = pysiglib.trees(dimension, degree)
    requested = [basis[1], basis[4], basis[1]]
    indices = [
        pysiglib.tree_to_idx(tree, dimension, degree)
        for tree in requested
    ]

    actual = pysiglib.extract_branched_sig_coef(
        bsig, requested, dimension)

    np.testing.assert_array_equal(actual, bsig[..., indices])


def test_extract_branched_sig_coef_augmentation():
    dimension, degree = 2, 3
    augmented_dimension = 2 * dimension + 1
    pysiglib.prepare_branched_sig(
        dimension, degree, time_aug=True, lead_lag=True, device="cpu")
    path = np.random.default_rng(101).normal(size=(2, 5, dimension))
    bsig = pysiglib.branched_sig(
        path,
        degree,
        time_aug=True,
        lead_lag=True,
        scalar_term=True,
    )
    basis = pysiglib.trees(augmented_dimension, degree)
    requested = [None, basis[1], basis[augmented_dimension + 2], basis[-1]]
    indices = [
        pysiglib.tree_to_idx(
            tree, augmented_dimension, degree, scalar_term=True)
        for tree in requested
    ]

    actual = pysiglib.extract_branched_sig_coef(
        bsig,
        requested,
        dimension,
        time_aug=True,
        lead_lag=True,
        scalar_term=True,
    )

    np.testing.assert_array_equal(actual, bsig[..., indices])


def test_extract_branched_sig_coef_validation():
    bsig = np.ones(10)
    with pytest.raises(ValueError, match="non-empty list"):
        pysiglib.extract_branched_sig_coef(bsig, [], 2)
    with pytest.raises(ValueError, match="invalid decorated tree"):
        pysiglib.extract_branched_sig_coef(bsig, (2,), 2)
    with pytest.raises(ValueError, match="empty tree has no index"):
        pysiglib.extract_branched_sig_coef(bsig, None, 2)


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_branched_sig_coef_matches_full(planar, dtype):
    dimension, degree = 3, 3
    basis = pysiglib.trees(dimension, degree, planar=planar)
    requested = [basis[0], basis[1], basis[5], basis[-1], basis[5]]
    pysiglib.prepare_branched_sig_coef(dimension, requested, planar=planar)
    path = np.random.default_rng(101).normal(size=(2, 3, 7, dimension)).astype(dtype)

    actual = pysiglib.branched_sig_coef(path, requested, planar=planar)
    pysiglib.prepare_branched_sig(dimension, degree, planar=planar)
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
    basis = pysiglib.trees(augmented_dimension, degree)
    requested = [basis[0], basis[augmented_dimension], basis[-1]]
    pysiglib.prepare_branched_sig_coef(
        dimension, requested, time_aug=time_aug, lead_lag=lead_lag)
    path = np.random.default_rng(102).normal(size=(3, 6, dimension))

    actual = pysiglib.branched_sig_coef(
        path, requested, time_aug=time_aug, lead_lag=lead_lag, end_time=2.0)
    pysiglib.prepare_branched_sig(
        dimension, degree, time_aug=time_aug, lead_lag=lead_lag)
    full = pysiglib.branched_sig(
        path, degree, time_aug=time_aug, lead_lag=lead_lag,
        end_time=2.0, scalar_term=True)
    indices = [
        pysiglib.tree_to_idx(tree, augmented_dimension, degree, scalar_term=True)
        for tree in requested
    ]

    np.testing.assert_allclose(actual, full[..., indices], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("correction_degree", [2, 3])
def test_branched_sig_coef_correction_and_backprop_match_full(planar, correction_degree):
    dimension, degree, length = 2, 3, 6
    basis = pysiglib.trees(dimension, degree, planar=planar)
    requested = [basis[0], basis[1], basis[4], basis[-2], basis[4]]
    pysiglib.prepare_branched_sig_coef(dimension, requested, planar=planar)
    rng = np.random.default_rng(103)
    path = rng.normal(size=(2, length, dimension))
    correction_len = sum(
        dimension ** level for level in range(2, correction_degree + 1))
    correction = rng.normal(scale=0.01, size=(2, length - 1, correction_len))
    derivs = rng.normal(size=(2, len(requested)))

    coefs = pysiglib.branched_sig_coef(
        path, requested, planar=planar, correction=correction)
    pysiglib.prepare_branched_sig(dimension, degree, planar=planar)
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
    basis = pysiglib.trees(augmented_dimension, degree)
    requested = [basis[1], basis[augmented_dimension + 2], basis[-1]]
    pysiglib.prepare_branched_sig_coef(
        dimension, requested, time_aug=time_aug, lead_lag=lead_lag)
    rng = np.random.default_rng(104)
    path = rng.normal(size=(2, 5, dimension))
    derivs = rng.normal(size=(2, len(requested)))
    options = dict(time_aug=time_aug, lead_lag=lead_lag, end_time=1.7)

    coefs = pysiglib.branched_sig_coef(path, requested, **options)
    actual_grad = pysiglib.branched_sig_coef_backprop(
        path, requested, coefs, derivs, **options)
    pysiglib.prepare_branched_sig(
        dimension, degree, time_aug=time_aug, lead_lag=lead_lag)
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
    basis = pysiglib.trees(dimension, degree)
    requested = [basis[1], basis[4], basis[12], basis[-1]]
    pysiglib.prepare_branched_sig_coef(dimension, requested)
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
    basis = pysiglib.trees(dimension, degree, planar=planar)
    requested = [basis[1], basis[4], basis[-1], basis[4]]
    pysiglib.torch_api.prepare_branched_sig_coef(
        dimension, requested, planar=planar)
    path_coef = torch.randn(2, 6, dimension, dtype=torch.float64, requires_grad=True)
    path_full = path_coef.detach().clone().requires_grad_(True)
    weights = torch.randn(2, len(requested), dtype=torch.float64)

    coefs = pysiglib.torch_api.branched_sig_coef(
        path_coef, requested, planar=planar)
    coefs.backward(weights)

    pysiglib.torch_api.prepare_branched_sig(
        dimension, degree, planar=planar)
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


@skip_no_cuda
@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_branched_sig_coef_correction_and_autograd_match_cpu(planar, dtype):
    dimension, degree, length = 2, 3, 6
    basis = pysiglib.trees(dimension, degree, planar=planar)
    requested = [basis[0], basis[3], basis[-2], basis[3]]
    pysiglib.prepare_branched_sig_coef(dimension, requested, planar=planar)
    rng = np.random.default_rng(108)
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    path = rng.normal(size=(2, length, dimension)).astype(np_dtype)
    correction_len = dimension ** 2 + dimension ** 3
    correction = rng.normal(
        scale=0.01, size=(2, length - 1, correction_len)).astype(np_dtype)
    weights = rng.normal(size=(2, len(requested))).astype(np_dtype)

    cpu_coefs = pysiglib.branched_sig_coef(
        path, requested, planar=planar, correction=correction)
    cpu_grad = pysiglib.branched_sig_coef_backprop(
        path, requested, cpu_coefs, weights, planar=planar,
        correction=correction)

    cuda_path = torch.tensor(
        path, dtype=dtype, device="cuda", requires_grad=True)
    cuda_correction = torch.tensor(correction, dtype=dtype, device="cuda")
    cuda_weights = torch.tensor(weights, dtype=dtype, device="cuda")
    cuda_coefs = pysiglib.torch_api.branched_sig_coef(
        cuda_path, requested, planar=planar, correction=cuda_correction)
    cuda_coefs.backward(cuda_weights)

    tolerance = 3e-5 if dtype == torch.float32 else 2e-11
    np.testing.assert_allclose(
        cuda_coefs.detach().cpu().numpy(), cpu_coefs,
        rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        cuda_path.grad.detach().cpu().numpy(), cpu_grad,
        rtol=tolerance, atol=tolerance)


@skip_no_cuda
@pytest.mark.parametrize(
    "time_aug,lead_lag",
    [(True, False), (False, True), (True, True)],
)
def test_cuda_branched_sig_coef_augmentation_matches_cpu(time_aug, lead_lag):
    dimension, degree = 2, 3
    augmented_dimension = (2 * dimension if lead_lag else dimension) + int(time_aug)
    basis = pysiglib.trees(augmented_dimension, degree)
    requested = [basis[1], basis[augmented_dimension + 1], basis[-1]]
    pysiglib.prepare_branched_sig_coef(
        dimension, requested, time_aug=time_aug, lead_lag=lead_lag)
    path = np.random.default_rng(109).normal(size=(2, 5, dimension))
    options = dict(
        time_aug=time_aug, lead_lag=lead_lag, end_time=1.6)

    expected = pysiglib.branched_sig_coef(path, requested, **options)
    actual = pysiglib.branched_sig_coef(
        torch.tensor(path, dtype=torch.float64, device="cuda"),
        requested, **options)
    np.testing.assert_allclose(
        actual.cpu().numpy(), expected, rtol=2e-11, atol=2e-11)


def test_jax_branched_sig_coef_value_and_gradient_match_cpu():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    import pysiglib.jax_api as jax_api

    jax.config.update("jax_enable_x64", True)

    dimension, degree = 2, 3
    basis = pysiglib.trees(dimension, degree)
    requested = [basis[1], basis[4], basis[-1], basis[4]]
    jax_api.prepare_branched_sig_coef(dimension, requested)
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


@skip_no_cuda
def test_jax_cuda_branched_sig_coef_value_and_gradient_match_cpu():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    import pysiglib.jax_api as jax_api

    try:
        gpu = jax.devices("gpu")[0]
    except (IndexError, RuntimeError):
        pytest.skip("JAX CUDA device is unavailable")
    jax.config.update("jax_enable_x64", True)

    dimension, degree = 2, 3
    basis = pysiglib.trees(dimension, degree)
    requested = [basis[1], basis[4], basis[-1], basis[4]]
    jax_api.prepare_branched_sig_coef(dimension, requested)
    path = np.random.default_rng(110).normal(size=(2, 6, dimension))
    weights = np.random.default_rng(111).normal(size=(2, len(requested)))
    jax_path = jax.device_put(jnp.asarray(path), gpu)
    jax_weights = jax.device_put(jnp.asarray(weights), gpu)

    coefs = jax_api.branched_sig_coef(jax_path, requested)
    grad = jax.grad(lambda value: jnp.sum(
        jax_api.branched_sig_coef(value, requested) * jax_weights))(jax_path)

    expected = pysiglib.branched_sig_coef(path, requested)
    expected_grad = pysiglib.branched_sig_coef_backprop(
        path, requested, expected, weights)
    np.testing.assert_allclose(
        np.asarray(coefs), expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        np.asarray(grad), expected_grad, rtol=2e-10, atol=2e-10)


def test_branched_sig_coef_validation():
    path = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="non-empty"):
        pysiglib.branched_sig_coef(path, [])
    with pytest.raises(ValueError):
        pysiglib.branched_sig_coef(path, (3,))
    pysiglib.prepare_branched_sig_coef(2, (0,))
    expected = pysiglib.branched_sig_coef(path, (0,))
    actual = pysiglib.torch_api.branched_sig_coef(path, (0,))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "prepare,args",
    [
        (pysiglib.prepare_branched_sig, (2, 2)),
        (pysiglib.prepare_branched_sig_coef, (2, [(0,)])),
    ],
)
def test_branched_sig_prepare_rejects_invalid_device(prepare, args):
    with pytest.raises(ValueError, match="device must"):
        prepare(*args, device="tpu")


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_branched_sig_dense_cuda_cache_is_per_device():
    pysiglib.clear_cache()
    dimension, degree = 2, 3
    path = np.random.default_rng(102).normal(size=(5, dimension))
    pysiglib.prepare_branched_sig(dimension, degree, device="cpu")
    expected = pysiglib.branched_sig(path, degree)

    for device_index in range(2):
        with torch.cuda.device(device_index):
            pysiglib.prepare_branched_sig(
                dimension, degree, device="cuda")
            cuda_path = torch.tensor(
                path, dtype=torch.float64, device=f"cuda:{device_index}")
            actual = pysiglib.branched_sig(cuda_path, degree)
            np.testing.assert_allclose(
                actual.cpu().numpy(), expected, rtol=1e-12, atol=1e-12)


@skip_no_cuda
def test_branched_sig_dense_cuda_disk_cache(tmp_path):
    pysiglib.set_cache_dir(str(tmp_path))
    dimension, degree = 2, 3
    cache_file = tmp_path / "pysiglib_cache" / "branched_2_3_v3.bin"
    pysiglib.clear_cache(use_disk=True)
    try:
        pysiglib.prepare_branched_sig(
            dimension, degree, device="cuda", use_disk=True)
        assert cache_file.is_file()

        pysiglib.clear_cache(device="cuda")
        cache_file.write_bytes(b"invalid")
        with pytest.raises(Exception, match="invalid cache file"):
            pysiglib.prepare_branched_sig(
                dimension, degree, device="cuda", use_disk=True)
    finally:
        pysiglib.clear_cache(use_disk=True)


@skip_no_cuda
def test_branched_sig_coef_cuda_disk_cache(tmp_path):
    pysiglib.set_cache_dir(str(tmp_path))
    dimension, degree = 2, 3
    trees = [pysiglib.trees(dimension, degree)[-1]]
    pysiglib.clear_cache(use_disk=True)
    try:
        pysiglib.prepare_branched_sig_coef(
            dimension, trees, device="cuda", use_disk=True)
        cache_files = list((tmp_path / "pysiglib_cache").glob(
            "branched_coef_*.bin"))
        assert len(cache_files) == 1

        pysiglib.clear_cache(device="cuda")
        cache_files[0].write_bytes(b"invalid")
        with pytest.raises(Exception, match="invalid cache file"):
            pysiglib.prepare_branched_sig_coef(
                dimension, trees, device="cuda", use_disk=True)
    finally:
        pysiglib.clear_cache(use_disk=True)


def test_branched_sig_coef_cpu_disk_cache(tmp_path):
    pysiglib.set_cache_dir(str(tmp_path))
    dimension, degree = 2, 3
    trees = [pysiglib.trees(dimension, degree)[-1]]
    pysiglib.clear_cache(use_disk=True)
    try:
        pysiglib.prepare_branched_sig_coef(
            dimension, trees, device="cpu", use_disk=True)
        cache_files = list((tmp_path / "pysiglib_cache").glob(
            "branched_coef_*.bin"))
        assert len(cache_files) == 1

        pysiglib.clear_cache(device="cpu")
        cache_files[0].write_bytes(b"invalid")
        with pytest.raises(Exception, match="invalid cache file"):
            pysiglib.prepare_branched_sig_coef(
                dimension, trees, device="cpu", use_disk=True)
    finally:
        pysiglib.clear_cache(use_disk=True)


def test_branched_sig_coef_requires_preparation():
    pysiglib.clear_cache()
    dimension, degree = 2, 2
    basis = pysiglib.trees(dimension, degree)
    requested = [basis[-1]]
    path = np.zeros((3, dimension), dtype=np.float64)

    with pytest.raises(Exception, match="Could not find prepared cache"):
        pysiglib.branched_sig_coef(path, requested)

    pysiglib.prepare_branched_sig_coef(dimension, requested)
    pysiglib.branched_sig_coef(path, requested)
    with pytest.raises(Exception, match="Could not find prepared cache"):
        pysiglib.branched_sig(path, degree)


@skip_no_cuda
def test_cuda_branched_sig_coef_requires_preparation():
    pysiglib.clear_cache()
    dimension, degree = 2, 2
    basis = pysiglib.trees(dimension, degree)
    requested = [basis[-1]]
    path = torch.zeros((3, dimension), dtype=torch.float64, device="cuda")

    with pytest.raises(Exception, match="Could not find prepared cache"):
        pysiglib.branched_sig_coef(path, requested)

    pysiglib.prepare_branched_sig_coef(dimension, requested)
    pysiglib.branched_sig_coef(path, requested)


@skip_no_cuda
def test_branched_sig_coef_prepare_device_selection():
    pysiglib.clear_cache()
    dimension, degree = 2, 2
    basis = pysiglib.trees(dimension, degree)
    requested = [basis[-1]]
    cpu_path = np.zeros((3, dimension), dtype=np.float64)
    cuda_path = torch.zeros(
        (3, dimension), dtype=torch.float64, device="cuda")

    pysiglib.prepare_branched_sig_coef(
        dimension, requested, device="cpu")
    pysiglib.branched_sig_coef(cpu_path, requested)
    with pytest.raises(Exception, match="Could not find prepared cache"):
        pysiglib.branched_sig_coef(cuda_path, requested)

    pysiglib.clear_cache()
    pysiglib.prepare_branched_sig_coef(
        dimension, requested, device="cuda")
    pysiglib.branched_sig_coef(cuda_path, requested)
    with pytest.raises(Exception, match="Could not find prepared cache"):
        pysiglib.branched_sig_coef(cpu_path, requested)


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
    basis = pysiglib.trees(dimension, degree)
    requested = [basis[1], basis[-1]]
    pysiglib.prepare_branched_sig_coef(dimension, requested)
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
