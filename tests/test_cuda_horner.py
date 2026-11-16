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
from conftest import skip_no_cuda


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scalar_term", [False, True])
@pytest.mark.parametrize("batch,dimension,degree,time_aug,lead_lag", [
    (8, 2, 14, False, False),
    (4, 3, 9, False, False),
    (4, 3, 11, False, False),
    (3, 4, 8, False, False),
    (3, 6, 6, False, False),
    (2, 6, 7, False, False),
    (1, 6, 8, False, False),
    (1, 6, 9, False, False),
    (1, 5, 9, True, False),
    (4, 8, 5, False, False),
    (32, 8, 5, False, False),
    (3, 10, 5, False, False),
    (6, 12, 4, False, False),
    (3, 16, 4, False, False),
    (16, 2, 14, False, False),
    (16, 4, 8, False, False),
    (16, 8, 5, False, False),
    (1, 24, 4, False, False),
    (1, 32, 4, False, False),
    (8, 1, 14, False, True),
    (4, 2, 11, True, False),
    (4, 1, 11, True, True),
])
def test_cuda_horner_matches_cpu(dtype, scalar_term, batch, dimension, degree, time_aug, lead_lag):
    generator = torch.Generator().manual_seed(829)
    direction = torch.linspace(2., 4., dimension, dtype=dtype)
    path = torch.linspace(0., 1., 9, dtype=dtype)[None, :, None] * direction
    path = path + .01 * torch.randn(batch, 9, dimension, generator=generator, dtype=dtype)
    options = dict(method="horner", scalar_term=scalar_term, time_aug=time_aug,
                   lead_lag=lead_lag, end_time=1.7)
    expected = pysiglib.sig(path, degree, **options)
    device_path = path.cuda()
    actual = pysiglib.sig(device_path, degree, **options)
    tolerance = 3e-4 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    seeds = .01 * torch.randn(expected.shape, generator=generator, dtype=dtype)
    options.pop("scalar_term")
    expected_grad = pysiglib.sig_backprop(path, expected, seeds, degree, **options)
    actual_grad = pysiglib.sig_backprop(device_path, actual, seeds.cuda(), degree, **options)
    torch.testing.assert_close(actual_grad.cpu(), expected_grad, atol=tolerance, rtol=tolerance)


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("degree", [13, 14, 15, 16])
def test_cuda_horner_high_degree_linear_path(dtype, degree):
    direction = torch.tensor([3., 4.], dtype=dtype)
    path = (torch.linspace(0., 1., 7, dtype=dtype)[:, None] * direction).repeat(8, 1, 1)
    actual = pysiglib.sig(path.cuda(), degree, method="horner", scalar_term=True).cpu()
    levels = [torch.ones(1, dtype=dtype)]
    for level in range(1, degree + 1):
        levels.append(torch.kron(levels[-1], direction) / level)
    expected = torch.cat(levels).expand_as(actual)
    tolerance = 3e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@skip_no_cuda
@pytest.mark.parametrize("dimension,degree", [(1, 16), (1, 65), (37, 1)])
@pytest.mark.parametrize("scalar_term", [False, True])
def test_cuda_horner_endpoint_gradients(dimension, degree, scalar_term):
    generator = torch.Generator().manual_seed(121)
    path = torch.randn(3, 2, 17, dimension, generator=generator, dtype=torch.float64, requires_grad=True)
    actual = pysiglib.torch_api.signature(path.cuda(), degree, method="horner", scalar_term=scalar_term)
    expected = pysiglib.torch_api.signature(path, degree, method="horner", scalar_term=scalar_term)
    weights = torch.randn(expected.shape, generator=generator, dtype=torch.float64)
    actual_grad = torch.autograd.grad(actual, path, weights.cuda())[0]
    expected_grad = torch.autograd.grad(expected, path, weights)[0]
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(actual_grad[..., 1:-1, :], torch.zeros_like(actual_grad[..., 1:-1, :]), atol=0, rtol=0)


@skip_no_cuda
@pytest.mark.parametrize("dimension,degree", [(2, 14), (3, 10), (4, 8)])
def test_cuda_horner_directional_derivative(dimension, degree):
    generator = torch.Generator().manual_seed(657)
    path = torch.randn(2, 9, dimension, generator=generator, dtype=torch.float64) * .07
    path += torch.linspace(0., 3., 9)[None, :, None]
    path = path.cuda().requires_grad_()
    value = pysiglib.torch_api.signature(path, degree, method="horner", scalar_term=True)
    seeds = .1 * torch.randn(value.shape, generator=generator, dtype=torch.float64).cuda()
    gradient = torch.autograd.grad(value, path, seeds)[0]
    direction = torch.randn(path.shape, generator=generator, dtype=torch.float64).cuda()
    epsilon = 1e-5
    plus = pysiglib.sig(path.detach() + epsilon * direction, degree, method="horner", scalar_term=True)
    minus = pysiglib.sig(path.detach() - epsilon * direction, degree, method="horner", scalar_term=True)
    expected = ((plus - minus) * seeds).sum() / (2 * epsilon)
    torch.testing.assert_close((gradient * direction).sum(), expected, atol=2e-7, rtol=2e-7)


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scalar_term", [False, True])
def test_cuda_horner_long_noncontiguous_batch(dtype, scalar_term):
    generator = torch.Generator().manual_seed(98)
    data = .025 * torch.randn(2, 2, 131, 6, generator=generator, dtype=dtype).cumsum(dim=-2)
    path = data[..., ::2].requires_grad_()
    expected = pysiglib.torch_api.signature(path, 10, method="horner", scalar_term=scalar_term)
    seeds = .03 * torch.randn(expected.shape, generator=generator, dtype=dtype)
    expected_grad = torch.autograd.grad(expected, path, seeds)[0]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        device_path = data.cuda()[..., ::2].requires_grad_()
        actual = pysiglib.torch_api.signature(device_path, 10, method="horner", scalar_term=scalar_term)
        actual_grad = torch.autograd.grad(actual, device_path, seeds.cuda())[0]
    stream.synchronize()
    tolerance = 3e-5 if dtype == torch.float32 else 3e-10
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_grad.cpu(), expected_grad, atol=tolerance, rtol=tolerance)


@skip_no_cuda
def test_cuda_horner_long_reconstruction():
    rng = np.random.default_rng(19513)
    path = np.zeros((16, 4097, 3), dtype=np.float32)
    path[:, 1:] = (rng.normal(size=(16, 4096, 3)) * .03).astype(np.float32).cumsum(axis=1)
    device_path = torch.from_numpy(path).cuda()
    signature = pysiglib.sig(device_path, 8, method="horner")
    seeds = rng.normal(0, .05, signature.shape).astype(np.float32)
    reference_path = path.astype(np.float64)
    reference = pysiglib.sig(reference_path, 8, method="horner", n_jobs=2)
    expected = pysiglib.sig_backprop(reference_path, reference, seeds.astype(np.float64), 8,
                                    method="horner", n_jobs=2)
    actual = pysiglib.sig_backprop(device_path, signature, torch.from_numpy(seeds).cuda(), 8, method="horner")
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=2e-4, rtol=1e-3)


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scalar_term", [False, True])
@pytest.mark.parametrize("dimension,segments", [(2, 64), (3, 1024)])
def test_cuda_horner_backward_updated_buffers(dtype, scalar_term, dimension, segments):
    generator = torch.Generator().manual_seed(731)
    path = .02 * torch.randn(4, segments + 1, dimension, generator=generator, dtype=dtype).cumsum(dim=1)
    signature = pysiglib.sig(path, 4, method="horner", scalar_term=scalar_term)
    seeds = .05 * torch.randn(signature.shape, generator=generator, dtype=dtype)
    device_path, device_signature, device_seeds = path.cuda(), signature.cuda(), seeds.cuda()
    tolerance = 3e-5 if dtype == torch.float32 else 2e-10
    for iteration in range(2):
        if iteration:
            path.copy_(.02 * torch.randn(path.shape, generator=generator, dtype=dtype).cumsum(dim=1))
            signature.copy_(pysiglib.sig(path, 4, method="horner", scalar_term=scalar_term))
            seeds.copy_(.05 * torch.randn(seeds.shape, generator=generator, dtype=dtype))
            device_path.copy_(path)
            device_signature.copy_(signature)
            device_seeds.copy_(seeds)
        expected = pysiglib.sig_backprop(path, signature, seeds, 4, method="horner")
        actual = pysiglib.sig_backprop(device_path, device_signature, device_seeds, 4, method="horner")
        torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


@skip_no_cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("batch,segments,dimension", [(4, 64, 2), (4, 1024, 3), (16, 8, 8)])
def test_cuda_horner_backward_device_changes(batch, segments, dimension):
    generator = torch.Generator().manual_seed(92)
    path = .02 * torch.randn(batch, segments + 1, dimension, generator=generator, dtype=torch.float64).cumsum(dim=1)
    signature = pysiglib.sig(path, 4, method="horner")
    seeds = .05 * torch.randn(signature.shape, generator=generator, dtype=torch.float64)
    expected = pysiglib.sig_backprop(path, signature, seeds, 4, method="horner")
    for index in (0, 1, 0):
        device = torch.device("cuda", index)
        actual = pysiglib.sig_backprop(path.to(device), signature.to(device), seeds.to(device), 4, method="horner")
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-10, rtol=2e-10)
