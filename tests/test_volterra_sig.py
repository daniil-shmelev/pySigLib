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


def test_volterra_sig_exact_rejects_non_diagonal_Lambda():
    path = np.array([[0., 0.], [1., 1.]], dtype=np.float64)
    Lambda = np.array([[0.1, 0.2], [0., 0.3]], dtype=np.float64)
    kernel = pysiglib.VolterraFSSK(
        Lambda=Lambda,
        A=_identity_A(2),
        b=[1.0, 0.5],
    )

    with pytest.raises(ValueError, match="diagonal kernel.Lambda"):
        kernel.prepare(1, dt=1., dtype=path.dtype)


def test_volterra_sig_requires_dt():
    path = np.array([[0., 0.], [1., 1.]], dtype=np.float64)
    kernel = pysiglib.VolterraFSSK(Lambda=[0.1], A=_identity_A(2), b=[1.0])

    with pytest.raises(TypeError):
        pysiglib.volterra_sig(path, 1, kernel)
