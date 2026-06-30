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
from conftest import load_fixtures

try:
    import scipy  # noqa: F401
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

requires_scipy = pytest.mark.skipif(
    not SCIPY_AVAILABLE, reason="scipy is required for the fractional kernel BL2 fit")


FIXTURES = load_fixtures("volterra_signature_tensordev.npz")


def _identity_A(dimension, dtype=np.float64):
    return np.eye(dimension, dtype=dtype).reshape(1, dimension, dimension)


def test_volterra_sig_constant_kernel_matches_signature():
    rng = np.random.default_rng(123)
    path = rng.standard_normal((2, 6, 3)).astype(np.float64)
    degree = 4
    kernel = pysiglib.VolterraFSSK(
        Lambda=np.array([0.], dtype=np.float64),
        A=_identity_A(path.shape[-1]),
        b=np.array([1.], dtype=np.float64),
        quad_order=48,
    )

    kernel.prepare(degree, dt=0.2, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(
            path,
            degree,
            kernel,
            dt=0.2,
            scalar_term=True,
        )
    finally:
        kernel.clear_cache()
    expected = pysiglib.sig(path, degree, scalar_term=True)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def test_volterra_sig_scalar_term_and_one_point_path():
    path = np.array([[1., -2.]], dtype=np.float64)
    kernel = pysiglib.VolterraFSSK(Lambda=[0.5], A=_identity_A(2), b=[1.2])

    kernel.prepare(3, dt=1., dtype=path.dtype)
    try:
        full = pysiglib.volterra_sig(path, 3, kernel, dt=1., scalar_term=True)
        stripped = pysiglib.volterra_sig(path, 3, kernel, dt=1., scalar_term=False)
    finally:
        kernel.clear_cache()

    np.testing.assert_allclose(full[0], 1.)
    np.testing.assert_allclose(full[1:], 0.)
    np.testing.assert_allclose(stripped, 0.)


def test_volterra_sig_degree_zero():
    path = np.array([[[0., 1.], [0.5, -0.2], [1., 0.3]]], dtype=np.float64)
    kernel = pysiglib.VolterraFSSK(Lambda=[0.2, 1.0], A=_identity_A(2), b=[0.7, 0.1])

    kernel.prepare(0, dt=0.5, dtype=path.dtype)
    try:
        full = pysiglib.volterra_sig(path, 0, kernel, dt=0.5, scalar_term=True)
        stripped = pysiglib.volterra_sig(path, 0, kernel, dt=0.5, scalar_term=False)
    finally:
        kernel.clear_cache()

    np.testing.assert_allclose(full, np.ones((1, 1)))
    assert stripped.shape == (1, 0)


def test_volterra_sig_torch_cpu_output():
    path = torch.tensor([[0., 0.], [0.2, -0.1], [0.4, 0.3]], dtype=torch.float64)
    kernel = pysiglib.VolterraFSSK(Lambda=[0.3], A=_identity_A(2), b=[0.8])

    kernel.prepare(2, dt=0.2, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(path, 2, kernel, dt=0.2, scalar_term=True)
    finally:
        kernel.clear_cache()

    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == "cpu"
    assert actual.dtype == path.dtype


def test_prepare_volterra_sig_matches_direct_call():
    path = np.array(FIXTURES["matrix_path"], copy=True)
    degree = int(FIXTURES["matrix_degree"])
    dt = float(FIXTURES["matrix_dt"])
    readout_lag = float(FIXTURES["matrix_tau_dt"])
    quad_order = int(FIXTURES["matrix_quad_order"])
    kernel = pysiglib.VolterraFSSK(
        Lambda=np.diag(np.array(FIXTURES["matrix_lambdas"], copy=True)),
        A=np.array(FIXTURES["matrix_A"], copy=True),
        b=np.array(FIXTURES["matrix_b"], copy=True),
        quad_order=quad_order,
    )

    expected = FIXTURES["matrix_expected"]
    assert kernel.prepare(degree, dt=dt, readout_lag=readout_lag) is None
    try:
        actual = pysiglib.volterra_sig(
            path,
            degree,
            kernel,
            dt=dt,
            readout_lag=readout_lag,
            scalar_term=True,
        )
    finally:
        kernel.clear_cache()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_prepare_volterra_sig_torch_cpu_output():
    path = torch.tensor([[0., 0.], [0.2, -0.1], [0.4, 0.3]], dtype=torch.float64)
    kernel = pysiglib.VolterraFSSK(Lambda=[0.3], A=_identity_A(2), b=[0.8])
    kernel.prepare(2, dt=0.2, dtype=torch.float64)
    try:
        actual = pysiglib.volterra_sig(path, 2, kernel, dt=0.2, scalar_term=True)
        expected = pysiglib.volterra_sig(path, 2, kernel, dt=0.2, scalar_term=True)
    finally:
        kernel.clear_cache()

    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == "cpu"
    assert actual.dtype == path.dtype
    torch.testing.assert_close(actual, expected)


def test_prepare_volterra_sig_caches_readout_lag():
    path = np.array([[0.], [1.]], dtype=np.float64)
    kernel = pysiglib.VolterraFSSK(
        Lambda=[1.0],
        A=_identity_A(1),
        b=[1.0],
    )
    degree = 1
    dt = 1.
    readout_lag = 0.5

    kernel.prepare(degree, dt=dt, readout_lag=0., dtype=path.dtype)
    kernel.prepare(degree, dt=dt, readout_lag=readout_lag, dtype=path.dtype)
    try:
        base = pysiglib.volterra_sig(
            path, degree, kernel, dt=dt, readout_lag=0., scalar_term=True)
        lagged = pysiglib.volterra_sig(
            path, degree, kernel, dt=dt, readout_lag=readout_lag, scalar_term=True)
    finally:
        kernel.clear_cache()

    np.testing.assert_allclose(lagged[..., 1:], base[..., 1:] * np.exp(-readout_lag))


def test_prepare_volterra_sig_rejects_dtype_mismatch():
    path = np.array([[0., 0.], [1., 1.]], dtype=np.float32)
    kernel = pysiglib.VolterraFSSK(Lambda=[0.1], A=_identity_A(2), b=[1.0])
    pysiglib.prepare_volterra_sig(1, kernel, dt=1., dtype=np.float64)
    try:
        with pytest.raises(ValueError, match="dtype"):
            pysiglib.volterra_sig(path, 1, kernel, dt=1.)
    finally:
        kernel.clear_cache()


def test_volterra_sig_matches_tensordev_scalar_kernel():
    path = np.array(FIXTURES["scalar_path"], copy=True)
    degree = int(FIXTURES["scalar_degree"])
    dt = float(FIXTURES["scalar_dt"])
    quad_order = int(FIXTURES["scalar_quad_order"])
    Lambda = np.array(FIXTURES["scalar_lambdas"], copy=True)
    A = np.array(FIXTURES["scalar_A"], copy=True)
    b = np.array(FIXTURES["scalar_b"], copy=True)
    expected = FIXTURES["scalar_expected"]
    kernel = pysiglib.VolterraFSSK(
        Lambda=Lambda,
        A=A,
        b=np.ascontiguousarray(b[0]),
        quad_order=quad_order,
    )

    kernel.prepare(degree, dt=dt, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(
            path,
            degree,
            kernel,
            dt=dt,
            scalar_term=True,
        )
        stripped = pysiglib.volterra_sig(
            path,
            degree,
            kernel,
            dt=dt,
            scalar_term=False,
        )
    finally:
        kernel.clear_cache()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(stripped, expected[..., 1:], rtol=1e-10, atol=1e-11)


def test_volterra_sig_matches_tensordev_matrix_kernel():
    path = np.array(FIXTURES["matrix_path"], copy=True)
    degree = int(FIXTURES["matrix_degree"])
    dt = float(FIXTURES["matrix_dt"])
    readout_lag = float(FIXTURES["matrix_tau_dt"])
    quad_order = int(FIXTURES["matrix_quad_order"])
    Lambda = np.diag(np.array(FIXTURES["matrix_lambdas"], copy=True))
    b = np.array(FIXTURES["matrix_b"], copy=True)
    A = np.array(FIXTURES["matrix_A"], copy=True)
    expected = FIXTURES["matrix_expected"]
    kernel = pysiglib.VolterraFSSK(Lambda=Lambda, A=A, b=b, quad_order=quad_order)

    kernel.prepare(degree, dt=dt, readout_lag=readout_lag, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(
            path,
            degree,
            kernel,
            dt=dt,
            readout_lag=readout_lag,
            scalar_term=True,
        )
    finally:
        kernel.clear_cache()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


@requires_scipy
def test_volterra_fractional_kernel_bl2_matches_tensordev_nodes():
    from pysiglib._rough_kernel import bl2_quadrature_rule

    beta = float(FIXTURES["fractional_beta"])
    R = int(FIXTURES["fractional_R"])
    T = float(FIXTURES["fractional_T"])
    nodes, weights = bl2_quadrature_rule(beta, R, T)

    # The BL2 rule is an iterative scipy.optimize result, so it is not bit-exact
    # across platforms/BLAS; a tight relative tolerance confirms the port without
    # demanding non-portable bit equality (the end-to-end signature test below
    # guards the downstream accuracy to 1e-10).
    np.testing.assert_allclose(nodes, FIXTURES["fractional_nodes"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(weights, FIXTURES["fractional_weights"], rtol=1e-6, atol=1e-9)


@requires_scipy
def test_volterra_sig_matches_tensordev_fractional_kernel():
    path = np.array(FIXTURES["fractional_path"], copy=True)
    degree = int(FIXTURES["fractional_degree"])
    dt = float(FIXTURES["fractional_dt"])
    readout_lag = float(FIXTURES["fractional_tau_dt"])
    beta = float(FIXTURES["fractional_beta"])
    R = int(FIXTURES["fractional_R"])
    T = float(FIXTURES["fractional_T"])
    quad_order = int(FIXTURES["fractional_quad_order"])
    A = np.array(FIXTURES["fractional_A"], copy=True)
    expected = FIXTURES["fractional_expected"]

    kernel = pysiglib.VolterraFractionalKernel(
        A, beta=beta, R=R, T=T, quad_order=quad_order)
    kernel.prepare(degree, dt=dt, readout_lag=readout_lag, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(
            path, degree, kernel, dt=dt, readout_lag=readout_lag, scalar_term=True)
        stripped = pysiglib.volterra_sig(
            path, degree, kernel, dt=dt, readout_lag=readout_lag, scalar_term=False)
    finally:
        kernel.clear_cache()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(stripped, expected[..., 1:], rtol=1e-10, atol=1e-11)


def test_volterra_fractional_kernel_rejects_bad_parameters():
    A = _identity_A(2)
    with pytest.raises(ValueError, match="beta"):
        pysiglib.VolterraFractionalKernel(A, beta=0.4, R=4)
    with pytest.raises(ValueError, match="beta"):
        pysiglib.VolterraFractionalKernel(A, beta=1.0, R=4)
    with pytest.raises(ValueError, match="R"):
        pysiglib.VolterraFractionalKernel(A, beta=0.7, R=0)
    with pytest.raises(ValueError, match="T"):
        pysiglib.VolterraFractionalKernel(A, beta=0.7, R=4, T=0.)


def _prony_fixture_kwargs():
    return dict(
        A=np.array(FIXTURES["prony_A"], copy=True),
        real_rates=np.array(FIXTURES["prony_real_rates"], copy=True),
        real_sizes=np.array(FIXTURES["prony_real_sizes"], copy=True),
        osc_decays=np.array(FIXTURES["prony_osc_decays"], copy=True),
        osc_freqs=np.array(FIXTURES["prony_osc_freqs"], copy=True),
        osc_sizes=np.array(FIXTURES["prony_osc_sizes"], copy=True),
        quad_order=int(FIXTURES["prony_quad_order"]),
    )


def test_volterra_sig_matches_tensordev_from_prony():
    degree = int(FIXTURES["prony_degree"])
    dt = float(FIXTURES["prony_dt"])
    lag = float(FIXTURES["prony_tau_dt"])
    path = np.array(FIXTURES["prony_path"], copy=True)
    expected = FIXTURES["prony_expected"]
    kernel = pysiglib.VolterraFSSK.from_prony(
        alpha=np.array(FIXTURES["prony_alpha"], copy=True),
        beta=np.array(FIXTURES["prony_beta"], copy=True),
        delta=np.array(FIXTURES["prony_delta"], copy=True),
        **_prony_fixture_kwargs(),
    )
    kernel.prepare(degree, dt=dt, readout_lag=lag, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(
            path, degree, kernel, dt=dt, readout_lag=lag, scalar_term=True)
    finally:
        kernel.clear_cache()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_volterra_sig_matches_tensordev_from_jordan():
    degree = int(FIXTURES["prony_degree"])
    dt = float(FIXTURES["prony_dt"])
    lag = float(FIXTURES["prony_tau_dt"])
    path = np.array(FIXTURES["prony_path"], copy=True)
    expected = FIXTURES["prony_expected"]
    kwargs = _prony_fixture_kwargs()
    A = kwargs.pop("A")
    kernel = pysiglib.VolterraFSSK.from_jordan(
        A=A, b=np.array(FIXTURES["prony_b"], copy=True), **kwargs)
    kernel.prepare(degree, dt=dt, readout_lag=lag, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(
            path, degree, kernel, dt=dt, readout_lag=lag, scalar_term=True)
    finally:
        kernel.clear_cache()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_volterra_from_jordan_real_only_uses_real_path():
    # Real poles only (no oscillatory) -> a real-spectrum realization.
    kernel = pysiglib.VolterraFSSK.from_jordan(
        A=_identity_A(2), b=np.array([[1.0, 0.5]]),
        real_rates=[0.3, 1.2], real_sizes=[1, 1])
    path = np.array([[0., 0.], [0.2, -0.1], [0.4, 0.3]], dtype=np.float64)
    kernel.prepare(2, dt=0.2, dtype=path.dtype)
    try:
        out = np.asarray(pysiglib.volterra_sig(path, 2, kernel, dt=0.2, scalar_term=True))
    finally:
        kernel.clear_cache()
    assert np.isfinite(out).all()


def test_volterra_sig_matches_tensordev_jordan_block():
    # A genuine size-2 Jordan block (repeated pole, defective Lambda): handled by
    # the general matrix recursion.
    degree = int(FIXTURES["jordan_degree"])
    dt = float(FIXTURES["jordan_dt"])
    lag = float(FIXTURES["jordan_tau_dt"])
    path = np.array(FIXTURES["jordan_path"], copy=True)
    expected = FIXTURES["jordan_expected"]
    kernel = pysiglib.VolterraFSSK.from_jordan(
        A=np.array(FIXTURES["jordan_A"], copy=True),
        b=np.array(FIXTURES["jordan_b"], copy=True),
        real_rates=np.array(FIXTURES["jordan_real_rates"], copy=True),
        real_sizes=np.array(FIXTURES["jordan_real_sizes"], copy=True),
        quad_order=int(FIXTURES["jordan_quad_order"]),
    )
    kernel.prepare(degree, dt=dt, readout_lag=lag, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(
            path, degree, kernel, dt=dt, readout_lag=lag, scalar_term=True)
        stripped = pysiglib.volterra_sig(
            path, degree, kernel, dt=dt, readout_lag=lag, scalar_term=False)
    finally:
        kernel.clear_cache()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(stripped, expected[..., 1:], rtol=1e-10, atol=1e-11)


def test_volterra_sig_jordan_block_batch_matches_per_path():
    kernel = pysiglib.VolterraFSSK.from_jordan(
        A=_identity_A(3), b=np.array([[0.7, 0.2, -0.1]]),
        real_rates=[0.6], real_sizes=[3])
    path = np.ascontiguousarray(
        np.random.default_rng(5).standard_normal((9, 7, 3)) * 0.01)
    kernel.prepare(3, dt=0.1, dtype=path.dtype)
    try:
        batched = np.asarray(pysiglib.volterra_sig(path, 3, kernel, dt=0.1, scalar_term=True))
        ref = np.stack([
            np.asarray(pysiglib.volterra_sig(
                np.ascontiguousarray(path[i:i + 1]), 3, kernel, dt=0.1, scalar_term=True))[0]
            for i in range(path.shape[0])])
        multi = np.asarray(pysiglib.volterra_sig(path, 3, kernel, dt=0.1, scalar_term=True, n_jobs=-1))
    finally:
        kernel.clear_cache()
    assert np.isfinite(batched).all()
    np.testing.assert_allclose(batched, ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(multi, batched)


def test_volterra_sig_rejects_unknown_scheme():
    path = np.array([[0., 0.], [1., 1.]], dtype=np.float64)
    kernel = pysiglib.VolterraFSSK(Lambda=[0.1], A=_identity_A(2), b=[1.0])

    with pytest.raises(ValueError, match="scheme"):
        pysiglib.volterra_sig(path, 1, kernel, dt=1., scheme="fft")


def test_volterra_sig_requires_prepared_kernel():
    path = np.array([[0., 0.], [1., 1.]], dtype=np.float64)
    kernel = pysiglib.VolterraFSSK(Lambda=[0.1], A=_identity_A(2), b=[1.0])

    with pytest.raises(RuntimeError, match="kernel.prepare"):
        pysiglib.volterra_sig(path, 1, kernel, dt=1.)


def test_volterra_sig_exact_dense_real_Lambda_matches_diagonalization():
    # A dense Lambda with a real spectrum reduces to an equivalent diagonal
    # realization; the signature must match that built directly from the
    # eigenvalue diagonal and the transformed b.
    rng = np.random.default_rng(7)
    R, m, d, degree = 3, 2, 3, 3
    S = rng.standard_normal((R, R))
    mu = np.array([0.3, 0.8, 1.5])
    Lambda = S @ np.diag(mu) @ np.linalg.inv(S)
    A = rng.standard_normal((2, m, d)) * 0.1
    b = rng.standard_normal((2, R)) * 0.2
    path = np.ascontiguousarray(rng.standard_normal((6, 10, d)) * 0.01)

    dense = pysiglib.VolterraFSSK(Lambda=Lambda, A=A, b=b, quad_order=32)
    dense.prepare(degree, dt=0.1, dtype=path.dtype)

    eigvals, V = np.linalg.eig(Lambda)
    w = np.ones(R) @ V
    b_eff = (w[None, :] * np.linalg.solve(V, b.T).T).real
    diag = pysiglib.VolterraFSSK(
        Lambda=eigvals.real, A=A, b=np.ascontiguousarray(b_eff), quad_order=32)
    diag.prepare(degree, dt=0.1, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(path, degree, dense, dt=0.1, scalar_term=True)
        expected = pysiglib.volterra_sig(path, degree, diag, dt=0.1, scalar_term=True)
    finally:
        dense.clear_cache()
        diag.clear_cache()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_volterra_sig_matches_tensordev_oscillatory_kernel():
    # Dense Lambda with a complex (oscillatory) spectrum: eigenvalues a +- i w.
    path = np.array(FIXTURES["osc_path"], copy=True)
    degree = int(FIXTURES["osc_degree"])
    dt = float(FIXTURES["osc_dt"])
    readout_lag = float(FIXTURES["osc_tau_dt"])
    quad_order = int(FIXTURES["osc_quad_order"])
    Lambda = np.array(FIXTURES["osc_Lambda"], copy=True)
    b = np.array(FIXTURES["osc_b"], copy=True)
    A = np.array(FIXTURES["osc_A"], copy=True)
    expected = FIXTURES["osc_expected"]
    kernel = pysiglib.VolterraFSSK(Lambda=Lambda, A=A, b=b, quad_order=quad_order)

    kernel.prepare(degree, dt=dt, readout_lag=readout_lag, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(
            path, degree, kernel, dt=dt, readout_lag=readout_lag, scalar_term=True)
        stripped = pysiglib.volterra_sig(
            path, degree, kernel, dt=dt, readout_lag=readout_lag, scalar_term=False)
    finally:
        kernel.clear_cache()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(stripped, expected[..., 1:], rtol=1e-10, atol=1e-11)


def test_volterra_sig_oscillatory_batch_matches_per_path():
    rng = np.random.default_rng(3)
    Lambda = np.array([[0.7, 1.6], [-1.6, 0.7]], dtype=np.float64)
    A = rng.standard_normal((2, 2, 3)) * 0.1
    b = rng.standard_normal((2, 2)) * 0.2
    path = np.ascontiguousarray(rng.standard_normal((10, 8, 3)) * 0.01)
    kernel = pysiglib.VolterraFSSK(Lambda=Lambda, A=A, b=b, quad_order=32)
    kernel.prepare(3, dt=0.1, dtype=path.dtype)
    try:
        batched = np.asarray(pysiglib.volterra_sig(path, 3, kernel, dt=0.1, scalar_term=True))
        ref = np.stack([
            np.asarray(pysiglib.volterra_sig(
                np.ascontiguousarray(path[i:i + 1]), 3, kernel, dt=0.1, scalar_term=True))[0]
            for i in range(path.shape[0])])
        multi = np.asarray(pysiglib.volterra_sig(path, 3, kernel, dt=0.1, scalar_term=True, n_jobs=-1))
    finally:
        kernel.clear_cache()
    assert np.isfinite(batched).all()
    np.testing.assert_allclose(batched, ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(multi, batched)


def test_volterra_sig_requires_dt():
    path = np.array([[0., 0.], [1., 1.]], dtype=np.float64)
    kernel = pysiglib.VolterraFSSK(Lambda=[0.1], A=_identity_A(2), b=[1.0])

    with pytest.raises(TypeError):
        pysiglib.volterra_sig(path, 1, kernel)


@requires_scipy
def test_volterra_conv_fractional_matches_tensordev():
    path = np.array(FIXTURES["conv_path"], copy=True)
    degree = int(FIXTURES["conv_degree"])
    dt = float(FIXTURES["conv_dt"])
    A = np.array(FIXTURES["conv_A"], copy=True)
    beta = float(FIXTURES["conv_frac_beta"])
    expected = FIXTURES["conv_frac_expected"]
    kernel = pysiglib.VolterraConvFractionalKernel(A, beta=beta)

    kernel.prepare(degree, dt=dt, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(path, degree, kernel, dt=dt, scalar_term=True)
        stripped = pysiglib.volterra_sig(path, degree, kernel, dt=dt, scalar_term=False)
    finally:
        kernel.clear_cache()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(stripped, expected[..., 1:], rtol=1e-10, atol=1e-11)


@requires_scipy
def test_volterra_conv_fractional_multivariate_matches_tensordev():
    path = np.array(FIXTURES["conv_path"], copy=True)
    degree = int(FIXTURES["conv_degree"])
    dt = float(FIXTURES["conv_dt"])
    A = np.array(FIXTURES["conv_frac2_A"], copy=True)
    beta = np.array(FIXTURES["conv_frac2_beta"], copy=True)
    expected = FIXTURES["conv_frac2_expected"]
    kernel = pysiglib.VolterraConvFractionalKernel(A, beta=beta)

    kernel.prepare(degree, dt=dt, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(path, degree, kernel, dt=dt, scalar_term=True)
        stripped = pysiglib.volterra_sig(path, degree, kernel, dt=dt, scalar_term=False)
    finally:
        kernel.clear_cache()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(stripped, expected[..., 1:], rtol=1e-10, atol=1e-11)


@requires_scipy
def test_volterra_conv_multivariate_batch_matches_per_path():
    base = np.array(FIXTURES["conv_path"], copy=True)
    degree = int(FIXTURES["conv_degree"])
    dt = float(FIXTURES["conv_dt"])
    A = np.array(FIXTURES["conv_frac2_A"], copy=True)
    beta = np.array(FIXTURES["conv_frac2_beta"], copy=True)
    path = np.stack([base, base * 0.5, base + 0.2])
    kernel = pysiglib.VolterraConvFractionalKernel(A, beta=beta)

    kernel.prepare(degree, dt=dt, dtype=path.dtype)
    try:
        batched = np.asarray(pysiglib.volterra_sig(path, degree, kernel, dt=dt, scalar_term=True))
        ref = np.stack([
            np.asarray(pysiglib.volterra_sig(
                np.ascontiguousarray(path[i]), degree, kernel, dt=dt, scalar_term=True))
            for i in range(path.shape[0])])
        multi = np.asarray(pysiglib.volterra_sig(path, degree, kernel, dt=dt, scalar_term=True, n_jobs=-1))
    finally:
        kernel.clear_cache()

    assert np.isfinite(batched).all()
    np.testing.assert_allclose(batched, ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(multi, batched)


@requires_scipy
def test_volterra_conv_gamma_matches_tensordev():
    path = np.array(FIXTURES["conv_path"], copy=True)
    degree = int(FIXTURES["conv_degree"])
    dt = float(FIXTURES["conv_dt"])
    A = np.array(FIXTURES["conv_A"], copy=True)
    expected = FIXTURES["conv_gamma_expected"]
    kernel = pysiglib.VolterraConvGammaKernel(
        A,
        beta=float(FIXTURES["conv_gamma_beta"]),
        scale=float(FIXTURES["conv_gamma_scale"]),
        rate=float(FIXTURES["conv_gamma_rate"]),
        quad_order=int(FIXTURES["conv_gamma_quad_order"]),
    )

    kernel.prepare(degree, dt=dt, dtype=path.dtype)
    try:
        actual = pysiglib.volterra_sig(path, degree, kernel, dt=dt, scalar_term=True)
    finally:
        kernel.clear_cache()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


@requires_scipy
def test_volterra_conv_batch_matches_per_path():
    base = np.array(FIXTURES["conv_path"], copy=True)
    degree = int(FIXTURES["conv_degree"])
    dt = float(FIXTURES["conv_dt"])
    A = np.array(FIXTURES["conv_A"], copy=True)
    beta = float(FIXTURES["conv_frac_beta"])
    path = np.stack([base, base * 0.5, base + 0.2])
    kernel = pysiglib.VolterraConvFractionalKernel(A, beta=beta)

    kernel.prepare(degree, dt=dt, dtype=path.dtype)
    try:
        batched = np.asarray(pysiglib.volterra_sig(path, degree, kernel, dt=dt, scalar_term=True))
        ref = np.stack([
            np.asarray(pysiglib.volterra_sig(
                np.ascontiguousarray(path[i]), degree, kernel, dt=dt, scalar_term=True))
            for i in range(path.shape[0])])
        multi = np.asarray(pysiglib.volterra_sig(path, degree, kernel, dt=dt, scalar_term=True, n_jobs=-1))
    finally:
        kernel.clear_cache()

    assert np.isfinite(batched).all()
    np.testing.assert_allclose(batched, ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(multi, batched)


@requires_scipy
def test_volterra_conv_torch_cpu_output():
    path = torch.tensor(FIXTURES["conv_path"], dtype=torch.float64)
    degree = int(FIXTURES["conv_degree"])
    dt = float(FIXTURES["conv_dt"])
    A = np.array(FIXTURES["conv_A"], copy=True)
    beta = float(FIXTURES["conv_frac_beta"])
    expected = FIXTURES["conv_frac_expected"]
    kernel = pysiglib.VolterraConvFractionalKernel(A, beta=beta)

    kernel.prepare(degree, dt=dt, dtype=torch.float64)
    try:
        actual = pysiglib.volterra_sig(path, degree, kernel, dt=dt, scalar_term=True)
    finally:
        kernel.clear_cache()

    assert isinstance(actual, torch.Tensor)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-10, atol=1e-11)


def test_volterra_conv_gamma_rejects_multivariate():
    A = np.zeros((2, 2, 2), dtype=np.float64)
    kernel = pysiglib.VolterraConvGammaKernel(A, beta=0.8)
    with pytest.raises(ValueError):
        kernel.prepare(2, dt=0.1)


def test_volterra_conv_fractional_rejects_beta_length_mismatch():
    A = np.zeros((2, 2, 2), dtype=np.float64)        # q = 2
    kernel = pysiglib.VolterraConvFractionalKernel(A, beta=[0.6, 0.7, 0.8])  # 3 != 2
    with pytest.raises(ValueError):
        kernel.prepare(2, dt=0.1)


def test_volterra_conv_rejects_readout_lag():
    A = _identity_A(2)
    kernel = pysiglib.VolterraConvFractionalKernel(A, beta=0.7)
    with pytest.raises(ValueError):
        kernel.prepare(2, dt=0.1, readout_lag=0.05)
