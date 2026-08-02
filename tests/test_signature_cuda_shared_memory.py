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

import pytest
import torch

import pysiglib
from conftest import skip_no_cuda


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signature_cuda_adapts_forward_shared_memory(dtype):
    torch.manual_seed(29)
    path = torch.randn(16, 127, 98, dtype=dtype)

    expected = pysiglib.sig(path, 1)
    actual = pysiglib.sig(path.cuda(), 1)

    torch.testing.assert_close(actual.cpu(), expected)


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signature_cuda_adapts_backprop_shared_memory(dtype):
    torch.manual_seed(29)
    path_cpu = torch.randn(1, 2, 350, dtype=dtype, requires_grad=True)
    path_cuda = path_cpu.detach().cuda().requires_grad_()

    sig_cpu = pysiglib.torch_api.signature(path_cpu, 2)
    sig_cuda = pysiglib.torch_api.signature(path_cuda, 2)
    grad_cpu = torch.autograd.grad(sig_cpu.sum(), path_cpu)[0]
    grad_cuda = torch.autograd.grad(sig_cuda.sum(), path_cuda)[0]

    torch.testing.assert_close(sig_cuda.cpu(), sig_cpu)
    torch.testing.assert_close(
        grad_cuda.cpu(), grad_cpu, rtol=2e-4, atol=2e-4)


@skip_no_cuda
def test_signature_cuda_caps_shared_memory_chunks():
    torch.manual_seed(29)
    path_cpu = torch.randn(
        1, 127, 1000, dtype=torch.float32, requires_grad=True)
    path_cuda = path_cpu.detach().cuda().requires_grad_()

    sig_cpu = pysiglib.torch_api.signature(path_cpu, 1)
    sig_cuda = pysiglib.torch_api.signature(path_cuda, 1)
    grad_cpu = torch.autograd.grad(sig_cpu.sum(), path_cpu)[0]
    grad_cuda = torch.autograd.grad(sig_cuda.sum(), path_cuda)[0]

    torch.testing.assert_close(sig_cuda.cpu(), sig_cpu)
    torch.testing.assert_close(grad_cuda.cpu(), grad_cpu)
