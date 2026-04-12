# Copyright 2025 Daniil Shmelev
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

from typing import Union, Optional
import numpy as np
import torch
from ..sig import sig as sig_forward
from ..sig import sig_combine as sig_combine_forward
from ..sig_coef import sig_coef as sig_coef_forward
from ..sig_coef_backprop import sig_coef_backprop
from ..sig_backprop import sig_backprop, sig_combine_backprop
from ..log_sig import sig_to_log_sig as sig_to_log_sig_forward
from ..log_sig import log_sig as log_sig_forward
from ..log_sig_backprop import sig_to_log_sig_backprop, _log_sig_from_path_backprop
from ..static_kernels import StaticKernel
from ..log_sig_combine import log_sig_combine as log_sig_combine_forward
from ..log_sig_combine import log_sig_combine_backprop
from ..logsig_to_sig import logsig_to_sig as logsig_to_sig_forward
from ..logsig_to_sig_backprop import logsig_to_sig_backprop
from ..sig_kernel import sig_kernel as sig_kernel_forward
from ..sig_kernel_backprop import sig_kernel_backprop
from ..sig_kernel import sig_kernel_gram as sig_kernel_gram_forward
from ..sig_kernel_backprop import sig_kernel_gram_backprop
from ..sig_metrics import sig_score as sig_score_forward
from ..sig_metrics import expected_sig_score as expected_sig_score_forward
from ..sig_metrics import sig_mmd as sig_mmd_forward
from ..transform_path import transform_path as transform_path_forward
from ..transform_path_backprop import transform_path_backprop

from ..linear_sig import linear_sig as linear_sig_forward
from ..sig_join import sig_join as sig_join_forward
from ..sig_join_backprop import sig_join_backprop
from ..log_sig_join import log_sig_join as log_sig_join_forward
from ..log_sig_join_backprop import log_sig_join_backprop

from ..param_checks import check_type, check_word_or_word_list

class Sig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, degree, time_aug, lead_lag, end_time, horner, n_jobs):
        sig_ = sig_forward(path, degree, time_aug, lead_lag, end_time, horner, n_jobs)

        ctx.save_for_backward(path, sig_)
        ctx.degree = degree
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.horner = horner
        ctx.n_jobs = n_jobs

        return sig_

    @staticmethod
    def backward(ctx, grad_output):
        path, sig_ = ctx.saved_tensors
        grad = sig_backprop(path, sig_, grad_output, ctx.degree, ctx.time_aug, ctx.lead_lag, ctx.end_time, ctx.n_jobs)
        return grad, None, None, None, None, None, None

def sig(
        path : Union[np.ndarray, torch.tensor],
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        horner: bool = True,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return Sig.apply(path, degree, time_aug, lead_lag, end_time, horner, n_jobs)

sig.__doc__ = sig_forward.__doc__

class SigCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sig1, sig2, dimension, degree, time_aug, lead_lag, n_jobs):
        sig_combined = sig_combine_forward(sig1, sig2, dimension, degree, time_aug, lead_lag, n_jobs)

        ctx.save_for_backward(sig1, sig2)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.n_jobs = n_jobs

        return sig_combined

    @staticmethod
    def backward(ctx, grad_output):
        sig1, sig2 = ctx.saved_tensors
        sig1_grad, sig2_grad = sig_combine_backprop(grad_output, sig1, sig2, ctx.dimension, ctx.degree, ctx.time_aug, ctx.lead_lag, ctx.n_jobs)
        return sig1_grad, sig2_grad, None, None, None, None, None

def sig_combine(
        sig1 : Union[np.ndarray, torch.tensor],
        sig2 : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        time_aug: bool = False,
        lead_lag: bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return SigCombine.apply(sig1, sig2, dimension, degree, time_aug, lead_lag, n_jobs)


sig_combine.__doc__ = sig_combine_forward.__doc__

def _get_coef_from_prefixes(word, coef, prefixes):
    if prefixes:
        return coef

    if isinstance(word, tuple):
        word = [word]
    degrees = [max(len(w), 1) for w in word]

    _coef_idx = [degrees[0] - 1]
    for i in range(1, len(word)):
        _coef_idx.append(_coef_idx[-1] + degrees[i])

    return coef[..., _coef_idx].contiguous()

class SigCoef(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, word, time_aug, lead_lag, end_time, n_jobs):
        coefs_ = sig_coef_forward(path, word, time_aug, lead_lag, end_time, True, n_jobs)

        ctx.save_for_backward(coefs_, path)
        ctx.word = word
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.n_jobs = n_jobs

        return coefs_


    @staticmethod
    def backward(ctx, grad_output):
        coefs_, path = ctx.saved_tensors
        grad = sig_coef_backprop(path, ctx.word, coefs_, grad_output, ctx.time_aug, ctx.lead_lag, ctx.end_time, ctx.n_jobs)
        return grad, None, None, None, None, None

def sig_coef(
        path: Union[np.ndarray, torch.tensor],
        words: Union[tuple[int, ...], list[tuple[int, ...]]],
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.,
        prefixes: bool = False,
        n_jobs: int = 1
) -> Union[np.ndarray, torch.tensor]:
    coefs_ = SigCoef.apply(path, words, time_aug, lead_lag, end_time, n_jobs)
    return _get_coef_from_prefixes(words, coefs_, prefixes)

sig_coef.__doc__ = sig_coef_forward.__doc__

class TransformPath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, time_aug, lead_lag, end_time, n_jobs):
        new_path = transform_path_forward(path, time_aug, lead_lag, end_time, n_jobs)

        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.n_jobs = n_jobs

        return new_path

    @staticmethod
    def backward(ctx, grad_output):
        new_derivs = transform_path_backprop(grad_output, ctx.time_aug, ctx.lead_lag, ctx.end_time, ctx.n_jobs)
        return new_derivs, None, None, None, None

def transform_path(
    path : Union[np.ndarray, torch.tensor],
    time_aug : bool = False,
    lead_lag : bool = False,
    end_time : float = 1.,
    n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return TransformPath.apply(path, time_aug, lead_lag, end_time, n_jobs)

transform_path.__doc__ = transform_path_forward.__doc__

class SigToLogSig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sig, dimension, degree, time_aug, lead_lag, method, n_jobs):
        log_sig_ = sig_to_log_sig_forward(sig, dimension, degree, time_aug, lead_lag, method, n_jobs)

        ctx.save_for_backward(sig)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.method = method
        ctx.n_jobs = n_jobs

        return log_sig_

    @staticmethod
    def backward(ctx, grad_output):
        sig_ = ctx.saved_tensors[0]
        grad = sig_to_log_sig_backprop(sig_, grad_output, ctx.dimension, ctx.degree, ctx.time_aug, ctx.lead_lag, ctx.method, ctx.n_jobs)
        return grad, None, None, None, None, None, None

def sig_to_log_sig(
        sig : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        method : int = 1,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return SigToLogSig.apply(sig, dimension, degree, time_aug, lead_lag, method, n_jobs)


sig_to_log_sig.__doc__ = sig_to_log_sig_forward.__doc__

def log_sig(
        path : Union[np.ndarray, torch.tensor],
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        method : int = 1,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    if method == 3:
        return _log_sig_via_combine_torch(path, degree, time_aug, lead_lag, end_time, n_jobs)

    sig_ = sig(path, degree, time_aug, lead_lag, end_time, True, n_jobs)
    dimension = path.shape[-1]
    log_sig_ = sig_to_log_sig(sig_, dimension, degree, time_aug, lead_lag, method, n_jobs)
    return log_sig_

log_sig.__doc__ = log_sig_forward.__doc__


class LogSigFromPath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, degree, time_aug, lead_lag, end_time, n_jobs):
        # Apply transforms if needed
        if time_aug or lead_lag:
            transformed = transform_path(path, time_aug, lead_lag, end_time, n_jobs)
        else:
            transformed = path

        result = log_sig_forward(transformed, degree, False, False, 1.0, 3, n_jobs)

        ctx.save_for_backward(transformed)
        ctx.degree = degree
        ctx.n_jobs = n_jobs
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time

        return result

    @staticmethod
    def backward(ctx, grad_output):
        transformed, = ctx.saved_tensors
        d_path = _log_sig_from_path_backprop(grad_output, transformed, ctx.degree, ctx.n_jobs)

        # If transforms were applied, backprop through them
        if ctx.time_aug or ctx.lead_lag:
            d_path = transform_path_backprop(d_path, ctx.time_aug, ctx.lead_lag, ctx.end_time, ctx.n_jobs)

        return d_path, None, None, None, None, None

def _log_sig_via_combine_torch(path, degree, time_aug, lead_lag, end_time, n_jobs):
    """Compute log-signature via sequential BCH combination with C++ forward and backward."""
    if isinstance(path, torch.Tensor) and path.requires_grad:
        return LogSigFromPath.apply(path, degree, time_aug, lead_lag, end_time, n_jobs)
    else:
        return log_sig_forward(path, degree, time_aug, lead_lag, end_time, 3, n_jobs)

class SigKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path1, path2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, return_grid):
        k_grid = sig_kernel_forward(path1, path2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, True)

        ctx.save_for_backward(k_grid, path1, path2)
        ctx.dyadic_order = dyadic_order
        ctx.static_kernel = static_kernel
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.return_grid = return_grid
        ctx.n_jobs = n_jobs

        if return_grid:
            return k_grid
        return k_grid[..., -1, -1]

    @staticmethod
    def backward(ctx, grad_output):
        left_deriv = ctx.needs_input_grad[0]
        right_deriv = ctx.needs_input_grad[1]

        k_grid, path1, path2 = ctx.saved_tensors
        new_derivs = sig_kernel_backprop(grad_output, path1, path2, ctx.dyadic_order, ctx.static_kernel,
                                         ctx.time_aug, ctx.lead_lag, ctx.end_time,
                                         left_deriv, right_deriv, k_grid, ctx.n_jobs, ctx.return_grid)

        return new_derivs[0], new_derivs[1], None, None, None, None, None, None, None

def sig_kernel(
        path1 : Union[np.ndarray, torch.tensor],
        path2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        return_grid: bool = False,
        normalize : bool = False
) -> Union[np.ndarray, torch.tensor]:
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")
    k = SigKernel.apply(path1, path2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, return_grid)
    if normalize:
        k1 = SigKernel.apply(path1, path1, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, False)
        k2 = SigKernel.apply(path2, path2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, False)
        k = k / torch.sqrt(torch.clamp(k1 * k2, min=1e-30))
    return k

sig_kernel.__doc__ = sig_kernel_forward.__doc__

class SigKernelGram(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path1, path2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, return_grid):
        k_grid = sig_kernel_gram_forward(path1, path2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, return_grid = True)

        ctx.save_for_backward(k_grid, path1, path2)
        ctx.symmetric = path1 is path2

        ctx.dyadic_order = dyadic_order
        ctx.static_kernel = static_kernel
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.n_jobs = n_jobs
        ctx.max_batch = max_batch
        ctx.return_grid = return_grid

        if return_grid:
            return k_grid
        return k_grid[..., -1, -1]

    @staticmethod
    def backward(ctx, grad_output):
        left_deriv = ctx.needs_input_grad[0]
        right_deriv = ctx.needs_input_grad[1]

        k_grid, path1, path2 = ctx.saved_tensors
        if ctx.symmetric:
            path2 = path1

        new_derivs = sig_kernel_gram_backprop(grad_output, path1, path2, ctx.dyadic_order, ctx.static_kernel,
                                         ctx.time_aug, ctx.lead_lag, ctx.end_time,
                                         left_deriv, right_deriv, k_grid, ctx.n_jobs, ctx.return_grid, ctx.max_batch)

        return new_derivs[0], new_derivs[1], None, None, None, None, None, None, None, None

def sig_kernel_gram(
        path1: Union[np.ndarray, torch.tensor],
        path2: Union[np.ndarray, torch.tensor],
        dyadic_order: Union[int, tuple],
        static_kernel : Optional[StaticKernel] = None,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.,
        n_jobs: int = 1,
        max_batch: int = -1,
        return_grid: bool = False,
        normalize : bool = False
) -> Union[np.ndarray, torch.tensor]:
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")
    gram = SigKernelGram.apply(path1, path2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, return_grid)
    if normalize:
        d1 = SigKernel.apply(path1, path1, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, False)
        d2 = d1 if path1 is path2 else SigKernel.apply(path2, path2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, False)
        gram = gram / torch.sqrt(torch.clamp(d1.unsqueeze(1) * d2.unsqueeze(0), min=1e-30))
    return gram

sig_kernel_gram.__doc__ = sig_kernel_gram_forward.__doc__

def sig_score(
        sample : Union[np.ndarray, torch.tensor],
        y : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        lam : float = 1.,
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.tensor]:
    check_type(sample, "sample", torch.Tensor)
    check_type(y, "y", torch.Tensor)

    if len(y.shape) == 2:
        y = y.unsqueeze(0).contiguous().clone()

    B = sample.shape[0]
    if B < 2:
        raise ValueError("sig_score requires at least 2 sample paths (got {}).".format(B))

    xx = sig_kernel_gram(sample, sample, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    xy = sig_kernel_gram(sample, y, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, False)

    xx_sum = (torch.sum(xx) - torch.trace(xx)) / (B * (B - 1.))
    xy_sum = torch.sum(xy, dim=0) * (2. / B)

    return lam * xx_sum - xy_sum

sig_score.__doc__ = sig_score_forward.__doc__

def expected_sig_score(
        sample1 : Union[np.ndarray, torch.tensor],
        sample2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        lam : float = 1.,
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.tensor]:
    res = sig_score(sample1, sample2, dyadic_order, lam, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch)
    res = torch.mean(res, 0, True)
    return res

expected_sig_score.__doc__ = expected_sig_score_forward.__doc__

def sig_mmd(
        sample1 : Union[np.ndarray, torch.tensor],
        sample2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.tensor]:
    m = sample1.shape[0]
    n = sample2.shape[0]
    if m < 2:
        raise ValueError("sig_mmd requires at least 2 paths in sample1 (got {}).".format(m))
    if n < 2:
        raise ValueError("sig_mmd requires at least 2 paths in sample2 (got {}).".format(n))

    xx = sig_kernel_gram(sample1, sample1, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    xy = sig_kernel_gram(sample1, sample2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    yy = sig_kernel_gram(sample2, sample2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, False)

    xx_sum = (torch.sum(xx) - torch.trace(xx)) / (m * (m - 1))
    xy_sum = 2. * torch.mean(xy)
    yy_sum = (torch.sum(yy) - torch.trace(yy)) / (n * (n - 1))

    return xx_sum - xy_sum + yy_sum

sig_mmd.__doc__ = sig_mmd_forward.__doc__

class LogSigCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ls1, ls2, dimension, degree, time_aug, lead_lag, n_jobs):
        combined = log_sig_combine_forward(ls1, ls2, dimension, degree, time_aug, lead_lag, n_jobs)

        ctx.save_for_backward(ls1, ls2)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.n_jobs = n_jobs

        return combined

    @staticmethod
    def backward(ctx, grad_output):
        ls1, ls2 = ctx.saved_tensors
        ls1_grad, ls2_grad = log_sig_combine_backprop(grad_output, ls1, ls2, ctx.dimension, ctx.degree, ctx.time_aug, ctx.lead_lag, ctx.n_jobs)
        return ls1_grad, ls2_grad, None, None, None, None, None

def log_sig_combine(
        log_sig1 : Union[np.ndarray, torch.tensor],
        log_sig2 : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        time_aug: bool = False,
        lead_lag: bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return LogSigCombine.apply(log_sig1, log_sig2, dimension, degree, time_aug, lead_lag, n_jobs)

log_sig_combine.__doc__ = log_sig_combine_forward.__doc__


# =========================================================================
# Branched Signature (with autograd)
# =========================================================================

from ..branched_sig import branched_sig as branched_sig_forward, prepare_branched_sig, branched_sig_length, branched_sig_combine as branched_sig_combine_forward
from ..branched_sig_backprop import branched_sig_backprop, branched_sig_combine_backprop

class BranchedSig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, degree, time_aug, lead_lag, end_time, n_jobs):
        bsig = branched_sig_forward(path, degree, n_jobs=n_jobs,
                                     time_aug=time_aug, lead_lag=lead_lag, end_time=end_time)

        ctx.save_for_backward(path, bsig)
        ctx.degree = degree
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.n_jobs = n_jobs

        return bsig

    @staticmethod
    def backward(ctx, grad_output):
        path, bsig = ctx.saved_tensors
        grad = branched_sig_backprop(path, bsig, grad_output, ctx.degree,
                                     ctx.time_aug, ctx.lead_lag, ctx.end_time, ctx.n_jobs)
        return grad, None, None, None, None, None

def branched_sig(
        path: Union[np.ndarray, torch.Tensor],
        degree: int,
        n_jobs: int = 1,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.0
) -> Union[np.ndarray, torch.Tensor]:
    return BranchedSig.apply(path, degree, time_aug, lead_lag, end_time, n_jobs)

branched_sig.__doc__ = branched_sig_forward.__doc__


class BranchedSigCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bsig1, bsig2, dimension, degree, n_jobs):
        combined = branched_sig_combine_forward(bsig1, bsig2, dimension, degree, n_jobs)
        ctx.save_for_backward(bsig1, bsig2)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.n_jobs = n_jobs
        return combined

    @staticmethod
    def backward(ctx, grad_output):
        bsig1, bsig2 = ctx.saved_tensors
        d1, d2 = branched_sig_combine_backprop(
            grad_output, bsig1, bsig2, ctx.dimension, ctx.degree, ctx.n_jobs)
        return d1, d2, None, None, None

def branched_sig_combine(
        bsig1: Union[np.ndarray, torch.Tensor],
        bsig2: Union[np.ndarray, torch.Tensor],
        dimension: int,
        degree: int,
        n_jobs: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    return BranchedSigCombine.apply(bsig1, bsig2, dimension, degree, n_jobs)

branched_sig_combine.__doc__ = branched_sig_combine_forward.__doc__


class LogSigToSig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_sig, dimension, degree, time_aug, lead_lag, method, n_jobs):
        sig_ = logsig_to_sig_forward(log_sig, dimension, degree, time_aug, lead_lag, method, n_jobs)

        ctx.save_for_backward(log_sig)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.method = method
        ctx.n_jobs = n_jobs

        return sig_

    @staticmethod
    def backward(ctx, grad_output):
        log_sig_ = ctx.saved_tensors[0]
        grad = logsig_to_sig_backprop(log_sig_, grad_output, ctx.dimension, ctx.degree, ctx.time_aug, ctx.lead_lag, ctx.method, ctx.n_jobs)
        return grad, None, None, None, None, None, None

def logsig_to_sig(
        log_sig : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        method : int = 1,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return LogSigToSig.apply(log_sig, dimension, degree, time_aug, lead_lag, method, n_jobs)

logsig_to_sig.__doc__ = logsig_to_sig_forward.__doc__


def linear_sig(
        displacement : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    # Delegate to sig() on a 2-point path [0, v] to reuse Sig's autograd.
    if isinstance(displacement, torch.Tensor):
        if displacement.shape[-1] != dimension:
            raise ValueError(
                f"displacement last-dim ({displacement.shape[-1]}) does not match dimension ({dimension})")
        zeros = torch.zeros_like(displacement)
        path = torch.stack([zeros, displacement], dim=-2)
    else:
        displacement = np.asarray(displacement)
        if displacement.shape[-1] != dimension:
            raise ValueError(
                f"displacement last-dim ({displacement.shape[-1]}) does not match dimension ({dimension})")
        zeros = np.zeros_like(displacement)
        path = np.stack([zeros, displacement], axis=-2)
    return sig(path, degree, n_jobs=n_jobs)

linear_sig.__doc__ = linear_sig_forward.__doc__

class SigJoin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sig, displacement, dimension, degree, prepend, n_jobs):
        result = sig_join_forward(sig, displacement, dimension, degree, prepend=prepend, n_jobs=n_jobs)

        ctx.save_for_backward(sig, displacement)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.prepend = prepend
        ctx.n_jobs = n_jobs

        return result

    @staticmethod
    def backward(ctx, grad_output):
        sig, displacement = ctx.saved_tensors
        d_sig, d_displacement = sig_join_backprop(grad_output, sig, displacement, ctx.dimension, ctx.degree, prepend=ctx.prepend, n_jobs=ctx.n_jobs)
        return d_sig, d_displacement, None, None, None, None

def sig_join(
        sig : Union[np.ndarray, torch.tensor],
        displacement : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        prepend : bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return SigJoin.apply(sig, displacement, dimension, degree, int(prepend), n_jobs)

sig_join.__doc__ = sig_join_forward.__doc__

class LogSigJoin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_sig, displacement, dimension, degree, n_jobs):
        result = log_sig_join_forward(log_sig, displacement, dimension, degree, n_jobs)

        ctx.save_for_backward(log_sig, displacement)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.n_jobs = n_jobs

        return result

    @staticmethod
    def backward(ctx, grad_output):
        log_sig, displacement = ctx.saved_tensors
        d_logsig, d_displacement = log_sig_join_backprop(grad_output, log_sig, displacement, ctx.dimension, ctx.degree, ctx.n_jobs)
        return d_logsig, d_displacement, None, None, None

def log_sig_join(
        log_sig : Union[np.ndarray, torch.tensor],
        displacement : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return LogSigJoin.apply(log_sig, displacement, dimension, degree, n_jobs)

log_sig_join.__doc__ = log_sig_join_forward.__doc__
