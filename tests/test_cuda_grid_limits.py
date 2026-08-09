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

import torch

import pysiglib
from conftest import skip_no_cuda


BATCH_SIZE = 65536


@skip_no_cuda
def test_standard_cuda_batch_above_grid_y_limit():
    path = torch.zeros((BATCH_SIZE, 2, 1), dtype=torch.float32, device="cuda")
    path[:, 1, 0] = 1

    sig = pysiglib.sig(path, 1)
    torch.testing.assert_close(sig, torch.ones_like(sig))

    sig_grad = pysiglib.sig_backprop(path, sig, torch.ones_like(sig), 1)
    torch.testing.assert_close(sig_grad[:, 0, :], -torch.ones_like(sig_grad[:, 0, :]))
    torch.testing.assert_close(sig_grad[:, 1, :], torch.ones_like(sig_grad[:, 1, :]))

    d_sig1, d_sig2 = pysiglib.sig_combine_backprop(
        torch.ones_like(sig), sig, sig, 1, 1)
    torch.testing.assert_close(d_sig1, torch.ones_like(d_sig1))
    torch.testing.assert_close(d_sig2, torch.ones_like(d_sig2))

    pysiglib.prepare_log_sig(1, 1, method=1, device="cuda", use_disk=False)
    log_sig = pysiglib.log_sig(path, 1, method=1)
    torch.testing.assert_close(log_sig, torch.ones_like(log_sig))

    projected_grad = pysiglib.sig_to_log_sig_backprop(
        sig, torch.ones_like(log_sig), 1, 1, method=1)
    torch.testing.assert_close(projected_grad, torch.ones_like(projected_grad))

    recovered = pysiglib.logsig_to_sig(log_sig, 1, 1, method=1)
    torch.testing.assert_close(recovered, sig)

    exp_grad = pysiglib.logsig_to_sig_backprop(
        log_sig, torch.ones_like(recovered), 1, 1, method=1)
    torch.testing.assert_close(exp_grad, torch.ones_like(exp_grad))


@skip_no_cuda
def test_branched_cuda_batch_above_grid_y_limit():
    path = torch.zeros((BATCH_SIZE, 2, 1), dtype=torch.float32, device="cuda")
    path[:, 1, 0] = 1

    pysiglib.prepare_branched_sig(1, 1)
    bsig = pysiglib.branched_sig(path, 1)
    torch.testing.assert_close(bsig, torch.ones_like(bsig))

    path_grad = pysiglib.branched_sig_backprop(
        path, bsig, torch.ones_like(bsig), 1)
    torch.testing.assert_close(path_grad[:, 0, :], -torch.ones_like(path_grad[:, 0, :]))
    torch.testing.assert_close(path_grad[:, 1, :], torch.ones_like(path_grad[:, 1, :]))

    combined = pysiglib.branched_sig_combine(bsig, bsig, 1, 1)
    torch.testing.assert_close(combined, 2 * torch.ones_like(combined))

    d_bsig1, d_bsig2 = pysiglib.branched_sig_combine_backprop(
        torch.ones_like(combined), bsig, bsig, 1, 1)
    torch.testing.assert_close(d_bsig1, torch.ones_like(d_bsig1))
    torch.testing.assert_close(d_bsig2, torch.ones_like(d_bsig2))

    pysiglib.prepare_branched_log_sig(1, 1, device="cuda")
    blog_sig = pysiglib.branched_sig_to_log_sig(bsig, 1, 1)
    torch.testing.assert_close(blog_sig, torch.ones_like(blog_sig))

    blog_grad = pysiglib.branched_sig_to_log_sig_backprop(
        bsig, torch.ones_like(blog_sig), 1, 1)
    torch.testing.assert_close(blog_grad, torch.ones_like(blog_grad))

    requested = [(0,)]
    pysiglib.prepare_branched_sig_coef(
        1, requested, device="cuda", use_disk=False)
    coefs = pysiglib.branched_sig_coef(path, requested)
    torch.testing.assert_close(coefs, torch.ones_like(coefs))

    coef_grad = pysiglib.branched_sig_coef_backprop(
        path, requested, coefs, torch.ones_like(coefs))
    torch.testing.assert_close(coef_grad[:, 0, :], -torch.ones_like(coef_grad[:, 0, :]))
    torch.testing.assert_close(coef_grad[:, 1, :], torch.ones_like(coef_grad[:, 1, :]))


@skip_no_cuda
def test_cuda_kernel_batches_above_grid_y_limit():
    path = torch.zeros((BATCH_SIZE, 2, 1), dtype=torch.float32, device="cuda")
    path[:, 1, 0] = 1
    weights = torch.ones(BATCH_SIZE, dtype=torch.float32, device="cuda")

    sig_kernel = pysiglib.sig_kernel(path, path, 0)
    torch.testing.assert_close(sig_kernel, sig_kernel[0].expand_as(sig_kernel))
    sig_left, sig_right = pysiglib.sig_kernel_backprop(
        weights, path, path, 0, left_deriv=True, right_deriv=True)
    torch.testing.assert_close(sig_left, sig_left[0].expand_as(sig_left))
    torch.testing.assert_close(sig_right, sig_right[0].expand_as(sig_right))

    branched_kernel = pysiglib.branched_sig_kernel(path, path, 1, 0)
    torch.testing.assert_close(
        branched_kernel, branched_kernel[0].expand_as(branched_kernel))
    branched_left, branched_right = pysiglib.branched_sig_kernel_backprop(
        weights, path, path, 1, 0, left_deriv=True, right_deriv=True)
    torch.testing.assert_close(
        branched_left, branched_left[0].expand_as(branched_left))
    torch.testing.assert_close(
        branched_right, branched_right[0].expand_as(branched_right))
