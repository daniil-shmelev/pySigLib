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


METHOD = "polynomial"


class DoubleLinearKernel(pysiglib.StaticKernel):
    def __call__(self, ctx, x, y):
        return 2 * torch.bmm(torch.diff(x, dim=1), torch.diff(y, dim=1).transpose(1, 2))

    def grad_x(self, ctx, derivs):
        raise NotImplementedError

    def grad_y(self, ctx, derivs):
        raise NotImplementedError


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polynomial_sig_kernel_golden(dtype):
    x = np.array([[0., 0.], [.3, -.2], [.2, .2], [.45, .35]], dtype=dtype)
    y = np.array([[0., 0.], [.2, .1], [-.15, .15]], dtype=dtype)
    expected = 0.9868046051301949

    result = pysiglib.sig_kernel(x, y, method=METHOD, order=7)

    assert isinstance(result, np.ndarray)
    assert result.dtype == dtype
    assert np.allclose(result, expected, rtol=2e-6, atol=2e-6)


def test_polynomial_sig_kernel_order_is_highest_degree():
    x = torch.tensor([[0.], [1.]], dtype=torch.float64)
    y = torch.tensor([[0.], [2.]], dtype=torch.float64)

    result = pysiglib.sig_kernel(x, y, method=METHOD, order=2)

    assert torch.equal(result, torch.tensor(4., dtype=torch.float64))


def test_polynomial_sig_kernel_custom_static_kernel():
    x = torch.tensor([[0.], [1.]], dtype=torch.float64)
    y = torch.tensor([[0.], [1.]], dtype=torch.float64)

    result = pysiglib.sig_kernel(
        x, y, method=METHOD, order=2, static_kernel=DoubleLinearKernel())

    assert torch.equal(result, torch.tensor(4., dtype=torch.float64))


def test_polynomial_sig_kernel_normalize():
    x = torch.tensor([[0., 0.], [.2, .1], [.3, -.1]], dtype=torch.float64)
    y = torch.tensor([[0., 0.], [-.1, .2], [.15, .25]], dtype=torch.float64)
    kxy = pysiglib.sig_kernel(x, y, method=METHOD, order=7)
    kxx = pysiglib.sig_kernel(x, x, method=METHOD, order=7)
    kyy = pysiglib.sig_kernel(y, y, method=METHOD, order=7)

    normalized = pysiglib.sig_kernel(x, y, method=METHOD, order=7, normalize=True)

    assert torch.allclose(normalized, kxy / torch.sqrt(kxx * kyy), rtol=1e-13, atol=1e-13)


def test_polynomial_sig_kernel_gram_and_batching():
    x = torch.tensor([
        [[0., 0.], [.2, .1], [.3, -.1]],
        [[0., 0.], [-.1, .3], [.2, .2]],
    ], dtype=torch.float64)
    y = torch.tensor([
        [[0., 0.], [.1, -.2], [.2, .1]],
        [[0., 0.], [.3, .1], [.1, .2]],
        [[0., 0.], [-.2, .1], [.2, -.1]],
    ], dtype=torch.float64)

    result = pysiglib.sig_kernel_gram(x, y, method=METHOD, order=7, max_batch=1, n_jobs=2)
    expected = torch.stack([
        torch.stack([pysiglib.sig_kernel(xi, yj, method=METHOD, order=7) for yj in y])
        for xi in x
    ])

    assert torch.allclose(result, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("transform_kwargs", [
    {"time_aug": True},
    {"lead_lag": True},
    {"time_aug": True, "lead_lag": True},
])
def test_polynomial_sig_kernel_path_transforms(transform_kwargs):
    x = torch.tensor([[0., 0.], [.2, .1], [.3, -.1]], dtype=torch.float64)
    y = torch.tensor([[0., 0.], [-.1, .2], [.15, .25]], dtype=torch.float64)
    tx = pysiglib.transform_path(x, **transform_kwargs)
    ty = pysiglib.transform_path(y, **transform_kwargs)

    transformed = pysiglib.sig_kernel(x, y, method=METHOD, order=7, **transform_kwargs)
    explicit = pysiglib.sig_kernel(tx, ty, method=METHOD, order=7)

    assert torch.allclose(transformed, explicit, rtol=1e-13, atol=1e-13)


def test_polynomial_sig_kernel_gram_normalize():
    x = torch.tensor([
        [[0., 0.], [.2, .1], [.3, -.1]],
        [[0., 0.], [-.1, .3], [.2, .2]],
    ], dtype=torch.float64)

    result = pysiglib.sig_kernel_gram(x, x, method=METHOD, order=7, normalize=True)

    assert torch.allclose(torch.diagonal(result), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(result, result.T, rtol=1e-13, atol=1e-13)


def test_polynomial_sig_kernel_rejects_grid():
    x = np.zeros((2, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="return_grid"):
        pysiglib.sig_kernel(x, x, method=METHOD, order=5, return_grid=True)


def test_polynomial_sig_kernel_rejects_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    x = torch.zeros((2, 1), dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="only supports CPU"):
        pysiglib.sig_kernel(x, x, method=METHOD, order=5)


def test_polynomial_sig_kernel_converges_to_direct_signature():
    rng = np.random.default_rng(25022025)
    increments = rng.normal(scale=0.04, size=(7, 2))
    x = np.vstack([np.zeros(2), np.cumsum(increments, axis=0)])
    y = np.vstack([np.zeros(2), np.cumsum(increments[::-1], axis=0)])
    sig_x = pysiglib.sig(x, 12, scalar_term=True)
    sig_y = pysiglib.sig(y, 12, scalar_term=True)
    reference = float(np.dot(sig_x, sig_y))

    errors = [
        abs(float(pysiglib.sig_kernel(x, y, method=METHOD, order=order)) - reference)
        for order in (2, 5, 8, 12)
    ]

    assert errors[-1] < errors[0]
    assert all(next_error <= error + 1e-14 for error, next_error in zip(errors, errors[1:]))


@pytest.mark.parametrize("kwargs", [
    {},
    {"method": "invalid", "dyadic_order": 0},
    {"method": "finite_difference", "dyadic_order": 0, "order": 2},
    {"method": "polynomial", "order": 1},
    {"method": "polynomial", "order": 65},
])
def test_sig_kernel_method_validation(kwargs):
    x = np.zeros((2, 1), dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        pysiglib.sig_kernel(x, x, **kwargs)


@pytest.mark.parametrize("api_name", ["torch_api", "jax_api"])
def test_reverse_apis_reject_polynomial_methods(api_name):
    api = pytest.importorskip("pysiglib." + api_name)
    x = np.zeros((2, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="finite_difference"):
        api.sig_kernel(x, x, method=METHOD, order=5)
