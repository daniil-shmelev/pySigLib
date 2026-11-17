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
import pysiglib.torch_api as torch_api


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scalar_term", [False, True])
@pytest.mark.parametrize("dimension,degree,time_aug,lead_lag", [
    (1, 1, False, False), (1, 8, False, False), (2, 8, False, False),
    (3, 4, False, False), (3, 6, False, False), (4, 5, False, False),
    (6, 4, False, False), (8, 6, False, False), (24, 2, False, False), (2, 5, True, False),
    (2, 4, False, True), (2, 4, True, True), (1, 5, True, True),
])
def test_cpu_horner_reverse_matches_join_autograd(dtype, scalar_term, dimension, degree, time_aug, lead_lag):
    generator = torch.Generator().manual_seed(721)
    storage = torch.randn((1, 3, 9, 2 * dimension), generator=generator, dtype=dtype) * .03
    path = storage[..., ::2].requires_grad_()
    transformed = path.reshape(3, 9, dimension)
    if lead_lag:
        lag = torch.cat((transformed[:, :-1].repeat_interleave(2, dim=1), transformed[:, -1:]), dim=1)
        lead = torch.cat((transformed[:, :1], transformed[:, 1:].repeat_interleave(2, dim=1)), dim=1)
        transformed = torch.cat((lag, lead), dim=-1)
    if time_aug:
        time = torch.linspace(0, .7, transformed.shape[1], dtype=dtype).reshape(1, -1, 1)
        transformed = torch.cat((transformed, time.expand(3, -1, -1)), dim=-1)
    channels = transformed.shape[-1]
    width = pysiglib.sig_length(channels, degree, scalar_term=scalar_term)
    reference = torch.zeros((3, width), dtype=dtype)
    if scalar_term:
        reference[:, 0] = 1
    for increment in transformed.diff(dim=1).unbind(dim=1):
        reference = torch_api.sig_join(reference, increment, channels, degree)
    seeds = torch.randn((1, 3, 2 * width), generator=generator, dtype=dtype)[..., ::2]
    expected, = torch.autograd.grad(reference, path, seeds.reshape(3, width))
    options = dict(time_aug=time_aug, lead_lag=lead_lag, end_time=.7)
    signature = pysiglib.sig(path.detach(), degree, scalar_term=scalar_term, **options)
    saved = [value.detach().clone() for value in (path, signature, seeds)]
    tolerance = 2e-5 if dtype == torch.float32 else 2e-10
    for jobs in [1, 4]:
        actual = pysiglib.sig_backprop(path.detach(), signature, seeds, degree, n_jobs=jobs, **options)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        numpy_result = pysiglib.sig_backprop(path.detach().numpy(), signature.numpy(), seeds.numpy(),
                                            degree, n_jobs=jobs, **options)
        np.testing.assert_allclose(numpy_result, expected.numpy(), atol=tolerance, rtol=tolerance)
    result = torch_api.sig(path, degree, scalar_term=scalar_term, n_jobs=4, **options)
    gradient, = torch.autograd.grad(result, path, seeds)
    torch.testing.assert_close(gradient, expected, atol=tolerance, rtol=tolerance)
    for value, original in zip((path, signature, seeds), saved):
        torch.testing.assert_close(value, original, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scalar_term", [False, True])
@pytest.mark.parametrize("batch,dimension,degree", [
    (8, 3, 4), (9, 3, 4), (8, 3, 5), (9, 3, 5), (8, 4, 6), (16, 4, 6), (17, 4, 6),
])
def test_cpu_horner_reverse_simd_boundary(dtype, scalar_term, batch, dimension, degree):
    generator = torch.Generator().manual_seed(731)
    path = torch.randn((batch, 17, dimension), generator=generator, dtype=dtype) * .03
    signature = pysiglib.sig(path, degree, scalar_term=scalar_term)
    seeds = torch.randn(signature.shape, generator=generator, dtype=dtype)
    expected = torch.stack([pysiglib.sig_backprop(x, s, g, degree)
                            for x, s, g in zip(path, signature, seeds)])
    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
    for jobs in [1, 4, 8]:
        parallel_signature = pysiglib.sig(path, degree, scalar_term=scalar_term, n_jobs=jobs)
        torch.testing.assert_close(parallel_signature, signature, atol=tolerance, rtol=tolerance)
        actual = pysiglib.sig_backprop(path, signature, seeds, degree, n_jobs=jobs)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


