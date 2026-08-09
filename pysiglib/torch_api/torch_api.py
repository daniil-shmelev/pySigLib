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
from ..sig_kernel import sig_kernel as sig_kernel_forward, _ensure_3d
from ..sig_kernel_backprop import sig_kernel_backprop
from ..sig_kernel import sig_kernel_gram as sig_kernel_gram_forward
from ..sig_kernel_backprop import sig_kernel_gram_backprop
from ..branched_sig_kernel import branched_sig_kernel as branched_sig_kernel_forward
from ..branched_sig_kernel import branched_sig_kernel_gram as branched_sig_kernel_gram_forward
from ..branched_sig_kernel_backprop import branched_sig_kernel_backprop
from ..branched_sig_kernel_backprop import branched_sig_kernel_gram_backprop
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
from ..sig_kernel import _safe_normalize

def _has_non_empty_correction(correction):
    if correction is None:
        return False
    if isinstance(correction, torch.Tensor):
        return correction.numel() != 0
    return np.asarray(correction).size != 0


class Sig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, correction, degree, time_aug, lead_lag, end_time, horner, scalar_term, n_jobs):
        sig_ = sig_forward(path, degree, scalar_term=scalar_term, time_aug=time_aug, lead_lag=lead_lag,
                           end_time=end_time, horner=horner, correction=correction, n_jobs=n_jobs)

        saved_correction = path.new_empty((0,)) if correction is None else correction.detach().clone()
        ctx.save_for_backward(path, sig_, saved_correction)
        ctx.degree = degree
        ctx.scalar_term = scalar_term
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.horner = horner
        ctx.n_jobs = n_jobs

        return sig_

    @staticmethod
    def backward(ctx, grad_output):
        path, sig_, saved_correction = ctx.saved_tensors
        correction = saved_correction if saved_correction.numel() != 0 else None
        grad = sig_backprop(path, sig_, grad_output, ctx.degree, time_aug=ctx.time_aug, lead_lag=ctx.lead_lag,
                            end_time=ctx.end_time, correction=correction, n_jobs=ctx.n_jobs)
        return grad, None, None, None, None, None, None, None, None

def sig(
        path : Union[np.ndarray, torch.Tensor],
        degree : int,
        *,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        horner: bool = True,
        scalar_term : bool = False,
        correction = None,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    if not isinstance(path, torch.Tensor):
        return sig_forward(path, degree, scalar_term=scalar_term, time_aug=time_aug, lead_lag=lead_lag,
                           end_time=end_time, horner=horner, correction=correction, n_jobs=n_jobs)
    return Sig.apply(path, correction, degree, time_aug, lead_lag, end_time, horner, scalar_term, n_jobs)

sig.__doc__ = sig_forward.__doc__

class SigCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sig1, sig2, dimension, degree, time_aug, lead_lag, n_jobs):
        sig_combined = sig_combine_forward(sig1, sig2, dimension, degree, time_aug=time_aug, lead_lag=lead_lag, n_jobs=n_jobs)

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
        sig1_grad, sig2_grad = sig_combine_backprop(grad_output, sig1, sig2, ctx.dimension, ctx.degree, time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, n_jobs=ctx.n_jobs)
        return sig1_grad, sig2_grad, None, None, None, None, None

def sig_combine(
        sig1 : Union[np.ndarray, torch.Tensor],
        sig2 : Union[np.ndarray, torch.Tensor],
        dimension : int,
        degree : int,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
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
        coefs_ = sig_coef_forward(path, word, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, prefixes=True, n_jobs=n_jobs)

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
        grad = sig_coef_backprop(path, ctx.word, coefs_, grad_output, time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, end_time=ctx.end_time, n_jobs=ctx.n_jobs)
        return grad, None, None, None, None, None

def sig_coef(
        path: Union[np.ndarray, torch.Tensor],
        words: Union[tuple[int, ...], list[tuple[int, ...]]],
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.,
        prefixes: bool = False,
        n_jobs: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    coefs_ = SigCoef.apply(path, words, time_aug, lead_lag, end_time, n_jobs)
    return _get_coef_from_prefixes(words, coefs_, prefixes)

sig_coef.__doc__ = sig_coef_forward.__doc__

class TransformPath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, time_aug, lead_lag, end_time, n_jobs):
        new_path = transform_path_forward(path, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)

        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.n_jobs = n_jobs

        return new_path

    @staticmethod
    def backward(ctx, grad_output):
        new_derivs = transform_path_backprop(grad_output, time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, end_time=ctx.end_time, n_jobs=ctx.n_jobs)
        return new_derivs, None, None, None, None

def transform_path(
    path : Union[np.ndarray, torch.Tensor],
    *,
    time_aug : bool = False,
    lead_lag : bool = False,
    end_time : float = 1.,
    n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    return TransformPath.apply(path, time_aug, lead_lag, end_time, n_jobs)

transform_path.__doc__ = transform_path_forward.__doc__

class SigToLogSig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sig, dimension, degree, time_aug, lead_lag, method, n_jobs):
        log_sig_ = sig_to_log_sig_forward(sig, dimension, degree, time_aug=time_aug, lead_lag=lead_lag, method=method, n_jobs=n_jobs)

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
        grad = sig_to_log_sig_backprop(sig_, grad_output, ctx.dimension, ctx.degree, time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, method=ctx.method, n_jobs=ctx.n_jobs)
        return grad, None, None, None, None, None, None

def sig_to_log_sig(
        sig : Union[np.ndarray, torch.Tensor],
        dimension : int,
        degree : int,
        *,
        time_aug : bool = False,
        lead_lag : bool = False,
        method : int = 1,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    return SigToLogSig.apply(sig, dimension, degree, time_aug, lead_lag, method, n_jobs)


sig_to_log_sig.__doc__ = sig_to_log_sig_forward.__doc__

def log_sig(
        path : Union[np.ndarray, torch.Tensor],
        degree : int,
        *,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        method : int = 1,
        scalar_term : bool = False,
        correction = None,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    if method == 3:
        if not isinstance(path, torch.Tensor) or _has_non_empty_correction(correction):
            return log_sig_forward(
                path, degree, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
                method=method, scalar_term=scalar_term, correction=correction, n_jobs=n_jobs)
        return _log_sig_via_combine_torch(path, degree, time_aug, lead_lag, end_time, n_jobs)

    sig_ = sig(path, degree, scalar_term=scalar_term, time_aug=time_aug, lead_lag=lead_lag,
               end_time=end_time, horner=True, correction=correction, n_jobs=n_jobs)
    dimension = path.shape[-1]
    return sig_to_log_sig(sig_, dimension, degree, time_aug=time_aug, lead_lag=lead_lag, method=method, n_jobs=n_jobs)

log_sig.__doc__ = log_sig_forward.__doc__


class LogSigFromPath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, degree, time_aug, lead_lag, end_time, n_jobs):
        # Apply transforms if needed
        if time_aug or lead_lag:
            transformed = transform_path(path, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)
        else:
            transformed = path

        result = log_sig_forward(transformed, degree, scalar_term=True, time_aug=False, lead_lag=False, end_time=1.0, method=3, n_jobs=n_jobs)

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
        d_path = _log_sig_from_path_backprop(grad_output, transformed, ctx.degree, n_jobs=ctx.n_jobs)

        # If transforms were applied, backprop through them
        if ctx.time_aug or ctx.lead_lag:
            d_path = transform_path_backprop(d_path, time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, end_time=ctx.end_time, n_jobs=ctx.n_jobs)

        return d_path, None, None, None, None, None

def _log_sig_via_combine_torch(path, degree, time_aug, lead_lag, end_time, n_jobs):
    """Compute log-signature via sequential BCH combination."""
    if isinstance(path, torch.Tensor) and path.requires_grad:
        return LogSigFromPath.apply(path, degree, time_aug, lead_lag, end_time, n_jobs)
    else:
        return log_sig_forward(path, degree, scalar_term=True, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, method=3, n_jobs=n_jobs)

class SigKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path1, path2, dyadic_order, method, order, static_kernel, time_aug, lead_lag, end_time, n_jobs, return_grid):
        if method == "polynomial":
            result, solver_state = sig_kernel_forward(
                path1, path2, dyadic_order, method=method, order=order,
                static_kernel=static_kernel, time_aug=time_aug,
                lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs,
                return_grid=return_grid, _return_state=True)
        else:
            solver_state = sig_kernel_forward(
                path1, path2, dyadic_order, method=method, order=order,
                static_kernel=static_kernel, time_aug=time_aug,
                lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs,
                return_grid=True)
            result = solver_state if return_grid else solver_state[..., -1, -1]

        ctx.save_for_backward(solver_state, path1, path2)
        ctx.batch_shape = path1.shape[:-2]
        ctx.dyadic_order = dyadic_order
        ctx.method = method
        ctx.order = order
        ctx.static_kernel = static_kernel
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.return_grid = return_grid
        ctx.n_jobs = n_jobs

        return result

    @staticmethod
    def backward(ctx, grad_output):
        left_deriv = ctx.needs_input_grad[0]
        right_deriv = ctx.needs_input_grad[1]

        solver_state, path1, path2 = ctx.saved_tensors
        batch_shape = ctx.batch_shape

        # Flatten multi-dim batch to 3D for sig_kernel_backprop
        if len(batch_shape) > 1:
            flat_path1 = path1.reshape(-1, path1.shape[-2], path1.shape[-1])
            flat_path2 = path2.reshape(-1, path2.shape[-2], path2.shape[-1])
            if ctx.method == "polynomial":
                flat_k_grid = None
                flat_state = solver_state.reshape(-1, *solver_state.shape[-4:])
            else:
                flat_k_grid = solver_state.reshape(-1, solver_state.shape[-2], solver_state.shape[-1])
                flat_state = None
            if ctx.return_grid:
                flat_grad = grad_output.reshape(-1, grad_output.shape[-2], grad_output.shape[-1])
            else:
                flat_grad = grad_output.reshape(-1)
        else:
            flat_path1, flat_path2, flat_grad = path1, path2, grad_output
            flat_k_grid = None if ctx.method == "polynomial" else solver_state
            flat_state = solver_state if ctx.method == "polynomial" else None

        new_derivs = sig_kernel_backprop(flat_grad, flat_path1, flat_path2, ctx.dyadic_order,
                                         method=ctx.method, order=ctx.order,
                                         static_kernel=ctx.static_kernel,
                                         time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, end_time=ctx.end_time,
                                         left_deriv=left_deriv, right_deriv=right_deriv, k_grid=flat_k_grid,
                                         n_jobs=ctx.n_jobs, return_grid=ctx.return_grid,
                                         _state=flat_state)

        # Reshape gradients back to original batch shape
        d0 = new_derivs[0].reshape(path1.shape) if new_derivs[0] is not None else None
        d1 = new_derivs[1].reshape(path2.shape) if new_derivs[1] is not None else None

        return d0, d1, None, None, None, None, None, None, None, None, None

def sig_kernel(
        path1 : Union[np.ndarray, torch.Tensor],
        path2 : Union[np.ndarray, torch.Tensor],
        dyadic_order : Optional[Union[int, tuple]] = None,
        *,
        method : str = "finite_difference",
        order : Optional[int] = None,
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        return_grid: bool = False,
        normalize : bool = False
) -> Union[np.ndarray, torch.Tensor]:
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")
    k = SigKernel.apply(path1, path2, dyadic_order, method, order, static_kernel,
                        time_aug, lead_lag, end_time, n_jobs, return_grid)
    if normalize:
        k1 = SigKernel.apply(path1, path1, dyadic_order, method, order, static_kernel,
                             time_aug, lead_lag, end_time, n_jobs, False)
        k2 = SigKernel.apply(path2, path2, dyadic_order, method, order, static_kernel,
                             time_aug, lead_lag, end_time, n_jobs, False)
        k = _safe_normalize(k, k1, k2, "sig_kernel(normalize=True)")
    return k

sig_kernel.__doc__ = sig_kernel_forward.__doc__

class SigKernelGram(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path1, path2, dyadic_order, method, order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, return_grid):
        compute_grid = return_grid or method == "finite_difference"
        result = sig_kernel_gram_forward(
            path1, path2, dyadic_order, method=method, order=order,
            static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
            end_time=end_time, n_jobs=n_jobs, max_batch=max_batch,
            return_grid=compute_grid)

        solver_grid = result if compute_grid else path1.new_empty((0,))
        ctx.save_for_backward(solver_grid, path1, path2)
        ctx.symmetric = path1 is path2

        ctx.dyadic_order = dyadic_order
        ctx.method = method
        ctx.order = order
        ctx.static_kernel = static_kernel
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.n_jobs = n_jobs
        ctx.max_batch = max_batch
        ctx.return_grid = return_grid

        if return_grid or method == "polynomial":
            return result
        return result[..., -1, -1]

    @staticmethod
    def backward(ctx, grad_output):
        left_deriv = ctx.needs_input_grad[0]
        right_deriv = ctx.needs_input_grad[1]

        solver_grid, path1, path2 = ctx.saved_tensors
        if ctx.symmetric:
            path2 = path1

        new_derivs = sig_kernel_gram_backprop(grad_output, path1, path2, ctx.dyadic_order,
                                         method=ctx.method, order=ctx.order,
                                         static_kernel=ctx.static_kernel,
                                         time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, end_time=ctx.end_time,
                                         left_deriv=left_deriv, right_deriv=right_deriv,
                                         k_grid=None if ctx.method == "polynomial" else solver_grid,
                                         n_jobs=ctx.n_jobs, return_grid=ctx.return_grid, max_batch=ctx.max_batch)

        return new_derivs[0], new_derivs[1], None, None, None, None, None, None, None, None, None, None

def sig_kernel_gram(
        path1: Union[np.ndarray, torch.Tensor],
        path2: Union[np.ndarray, torch.Tensor],
        dyadic_order: Optional[Union[int, tuple]] = None,
        *,
        method: str = "finite_difference",
        order: Optional[int] = None,
        static_kernel : Optional[StaticKernel] = None,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.,
        n_jobs: int = 1,
        max_batch: int = -1,
        return_grid: bool = False,
        normalize : bool = False
) -> Union[np.ndarray, torch.Tensor]:
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")
    gram = SigKernelGram.apply(path1, path2, dyadic_order, method, order,
                               static_kernel, time_aug, lead_lag, end_time,
                               n_jobs, max_batch, return_grid)
    if normalize:
        d1 = SigKernel.apply(path1, path1, dyadic_order, method, order, static_kernel,
                             time_aug, lead_lag, end_time, n_jobs, False)
        d2 = d1 if path1 is path2 else SigKernel.apply(
            path2, path2, dyadic_order, method, order, static_kernel,
            time_aug, lead_lag, end_time, n_jobs, False)
        gram = _safe_normalize(gram, d1.unsqueeze(1), d2.unsqueeze(0), "sig_kernel_gram(normalize=True)")
    return gram

sig_kernel_gram.__doc__ = sig_kernel_gram_forward.__doc__

class BranchedSigKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path1, path2, depth, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, return_grid):
        result = branched_sig_kernel_forward(
            path1, path2, depth, dyadic_order,
            static_kernel=static_kernel, time_aug=time_aug,
            lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs,
            return_grid=return_grid)

        ctx.save_for_backward(path1, path2)
        ctx.depth = depth
        ctx.dyadic_order = dyadic_order
        ctx.static_kernel = static_kernel
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.return_grid = return_grid
        ctx.n_jobs = n_jobs

        return result

    @staticmethod
    def backward(ctx, grad_output):
        left_deriv = ctx.needs_input_grad[0]
        right_deriv = ctx.needs_input_grad[1]

        path1, path2 = ctx.saved_tensors
        new_derivs = branched_sig_kernel_backprop(
            grad_output, path1, path2, ctx.depth, ctx.dyadic_order,
            static_kernel=ctx.static_kernel,
            time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, end_time=ctx.end_time,
            left_deriv=left_deriv, right_deriv=right_deriv,
            k_stack=None, n_jobs=ctx.n_jobs, return_grid=ctx.return_grid)

        return new_derivs[0], new_derivs[1], None, None, None, None, None, None, None, None


def branched_sig_kernel(
        path1: Union[np.ndarray, torch.Tensor],
        path2: Union[np.ndarray, torch.Tensor],
        depth: int,
        dyadic_order: Union[int, tuple],
        *,
        static_kernel: Optional[StaticKernel] = None,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.,
        n_jobs: int = 1,
        return_grid: bool = False,
        normalize: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")
    k = BranchedSigKernel.apply(path1, path2, depth, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, return_grid)
    if normalize:
        k1 = BranchedSigKernel.apply(path1, path1, depth, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, False)
        k2 = BranchedSigKernel.apply(path2, path2, depth, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, False)
        k = _safe_normalize(k, k1, k2, "branched_sig_kernel(normalize=True)")
    return k


branched_sig_kernel.__doc__ = branched_sig_kernel_forward.__doc__


class BranchedSigKernelGram(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path1, path2, depth, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, return_grid):
        result = branched_sig_kernel_gram_forward(
            path1, path2, depth, dyadic_order,
            static_kernel=static_kernel, time_aug=time_aug,
            lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs,
            max_batch=max_batch, return_grid=return_grid)

        ctx.save_for_backward(path1, path2)
        ctx.symmetric = path1 is path2
        ctx.depth = depth
        ctx.dyadic_order = dyadic_order
        ctx.static_kernel = static_kernel
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.n_jobs = n_jobs
        ctx.max_batch = max_batch
        ctx.return_grid = return_grid

        return result

    @staticmethod
    def backward(ctx, grad_output):
        left_deriv = ctx.needs_input_grad[0]
        right_deriv = ctx.needs_input_grad[1]

        path1, path2 = ctx.saved_tensors
        if ctx.symmetric:
            path2 = path1

        new_derivs = branched_sig_kernel_gram_backprop(
            grad_output, path1, path2, ctx.depth, ctx.dyadic_order,
            static_kernel=ctx.static_kernel,
            time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, end_time=ctx.end_time,
            left_deriv=left_deriv, right_deriv=right_deriv,
            k_stack=None, n_jobs=ctx.n_jobs, return_grid=ctx.return_grid,
            max_batch=ctx.max_batch)

        return new_derivs[0], new_derivs[1], None, None, None, None, None, None, None, None, None


def branched_sig_kernel_gram(
        path1: Union[np.ndarray, torch.Tensor],
        path2: Union[np.ndarray, torch.Tensor],
        depth: int,
        dyadic_order: Union[int, tuple],
        *,
        static_kernel: Optional[StaticKernel] = None,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.,
        n_jobs: int = 1,
        max_batch: int = -1,
        return_grid: bool = False,
        normalize: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")
    gram = BranchedSigKernelGram.apply(
        path1, path2, depth, dyadic_order, static_kernel, time_aug,
        lead_lag, end_time, n_jobs, max_batch, return_grid)
    if normalize:
        d1 = BranchedSigKernel.apply(path1, path1, depth, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, False)
        d2 = d1 if path1 is path2 else BranchedSigKernel.apply(path2, path2, depth, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, False)
        gram = _safe_normalize(gram, d1.unsqueeze(1), d2.unsqueeze(0), "branched_sig_kernel_gram(normalize=True)")
    return gram


branched_sig_kernel_gram.__doc__ = branched_sig_kernel_gram_forward.__doc__

def sig_score(
        sample : Union[np.ndarray, torch.Tensor],
        y : Union[np.ndarray, torch.Tensor],
        dyadic_order : Union[int, tuple],
        *,
        lam : float = 1.,
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.Tensor]:
    check_type(sample, "sample", torch.Tensor)
    check_type(y, "y", torch.Tensor)

    batch_shape_y = tuple(y.shape[:-2])
    sample = _ensure_3d(sample)
    y = _ensure_3d(y)

    B = sample.shape[0]
    if B < 2:
        raise ValueError("sig_score requires at least 2 sample paths (got {}).".format(B))

    xx = sig_kernel_gram(sample, sample, dyadic_order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False)
    xy = sig_kernel_gram(sample, y, dyadic_order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False)

    xx_sum = (torch.sum(xx) - torch.trace(xx)) / (B * (B - 1.))
    xy_sum = torch.sum(xy, dim=0) * (2. / B)

    res = lam * xx_sum - xy_sum
    if batch_shape_y:
        res = res.reshape(*batch_shape_y)
    return res

sig_score.__doc__ = sig_score_forward.__doc__

def expected_sig_score(
        sample1 : Union[np.ndarray, torch.Tensor],
        sample2 : Union[np.ndarray, torch.Tensor],
        dyadic_order : Union[int, tuple],
        *,
        lam : float = 1.,
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.Tensor]:
    res = sig_score(sample1, sample2, dyadic_order, lam=lam, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch)
    return res.mean().reshape(1)

expected_sig_score.__doc__ = expected_sig_score_forward.__doc__

def sig_mmd(
        sample1 : Union[np.ndarray, torch.Tensor],
        sample2 : Union[np.ndarray, torch.Tensor],
        dyadic_order : Union[int, tuple],
        *,
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.Tensor]:
    sample1 = _ensure_3d(sample1)
    sample2 = _ensure_3d(sample2)

    m = sample1.shape[0]
    n = sample2.shape[0]
    if m < 2:
        raise ValueError("sig_mmd requires at least 2 paths in sample1 (got {}).".format(m))
    if n < 2:
        raise ValueError("sig_mmd requires at least 2 paths in sample2 (got {}).".format(n))

    xx = sig_kernel_gram(sample1, sample1, dyadic_order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False)
    xy = sig_kernel_gram(sample1, sample2, dyadic_order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False)
    yy = sig_kernel_gram(sample2, sample2, dyadic_order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False)

    xx_sum = (torch.sum(xx) - torch.trace(xx)) / (m * (m - 1))
    xy_sum = 2. * torch.mean(xy)
    yy_sum = (torch.sum(yy) - torch.trace(yy)) / (n * (n - 1))

    return xx_sum - xy_sum + yy_sum

sig_mmd.__doc__ = sig_mmd_forward.__doc__

class LogSigCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ls1, ls2, dimension, degree, time_aug, lead_lag, n_jobs):
        combined = log_sig_combine_forward(ls1, ls2, dimension, degree, time_aug=time_aug, lead_lag=lead_lag, n_jobs=n_jobs)

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
        ls1_grad, ls2_grad = log_sig_combine_backprop(grad_output, ls1, ls2, ctx.dimension, ctx.degree, time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, n_jobs=ctx.n_jobs)
        return ls1_grad, ls2_grad, None, None, None, None, None

def log_sig_combine(
        log_sig1 : Union[np.ndarray, torch.Tensor],
        log_sig2 : Union[np.ndarray, torch.Tensor],
        dimension : int,
        degree : int,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    return LogSigCombine.apply(log_sig1, log_sig2, dimension, degree, time_aug, lead_lag, n_jobs)

log_sig_combine.__doc__ = log_sig_combine_forward.__doc__


# =========================================================================
# Branched Signature (with autograd)
# =========================================================================

from ..branched_sig import branched_sig as branched_sig_forward, prepare_branched_sig, branched_sig_length, branched_sig_combine as branched_sig_combine_forward
from ..branched_sig_backprop import branched_sig_backprop, branched_sig_combine_backprop
from ..branched_sig_coef import branched_sig_coef as branched_sig_coef_forward, prepare_branched_sig_coef
from ..branched_sig_coef_backprop import branched_sig_coef_backprop
from ..branched_log_sig import prepare_branched_log_sig, branched_sig_to_log_sig as branched_sig_to_log_sig_forward
from ..branched_log_sig import branched_log_sig as branched_log_sig_forward
from ..branched_log_sig_backprop import branched_sig_to_log_sig_backprop

class BranchedSigCoef(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, correction, basis_elements, time_aug, lead_lag, end_time, planar, n_jobs):
        coefs = branched_sig_coef_forward(
            path, basis_elements, time_aug=time_aug, lead_lag=lead_lag,
            end_time=end_time, planar=planar, correction=correction, n_jobs=n_jobs)
        saved_correction = correction.detach().clone() if correction is not None else path.new_empty((0,))
        ctx.save_for_backward(path, coefs, saved_correction)
        ctx.basis_elements = basis_elements
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.planar = planar
        ctx.n_jobs = n_jobs
        return coefs

    @staticmethod
    def backward(ctx, grad_output):
        path, coefs, correction = ctx.saved_tensors
        grad = branched_sig_coef_backprop(
            path, ctx.basis_elements, coefs, grad_output,
            time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, end_time=ctx.end_time,
            planar=ctx.planar, correction=correction, n_jobs=ctx.n_jobs)
        return grad, None, None, None, None, None, None, None

def branched_sig_coef(
        path: Union[np.ndarray, torch.Tensor],
        trees,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.0,
        planar: bool = False,
        correction=None,
        n_jobs: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(path, np.ndarray):
        return branched_sig_coef_forward(
            path, trees, time_aug=time_aug, lead_lag=lead_lag,
            end_time=end_time, planar=planar, correction=correction, n_jobs=n_jobs)
    check_type(path, "path", torch.Tensor)
    return BranchedSigCoef.apply(
        path, correction, trees, time_aug, lead_lag, end_time, planar, n_jobs)

branched_sig_coef.__doc__ = branched_sig_coef_forward.__doc__

class BranchedSig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, correction, degree, time_aug, lead_lag, end_time, planar, scalar_term, n_jobs):
        bsig = branched_sig_forward(path, degree, scalar_term=scalar_term, n_jobs=n_jobs,
                                     time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, planar=planar,
                                     correction=correction)

        saved_correction = correction.detach().clone() if correction is not None else path.new_empty((0,))
        ctx.save_for_backward(path, bsig, saved_correction)
        ctx.degree = degree
        ctx.scalar_term = scalar_term
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.n_jobs = n_jobs
        ctx.planar = planar

        return bsig

    @staticmethod
    def backward(ctx, grad_output):
        path, bsig, saved_correction = ctx.saved_tensors
        grad = branched_sig_backprop(path, bsig, grad_output, ctx.degree,
                                     time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, end_time=ctx.end_time,
                                     planar=ctx.planar, correction=saved_correction, n_jobs=ctx.n_jobs)
        return grad, None, None, None, None, None, None, None, None

def branched_sig(
        path: Union[np.ndarray, torch.Tensor],
        degree: int,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.0,
        planar: bool = False,
        scalar_term : bool = False,
        correction = None,
        n_jobs: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    return BranchedSig.apply(path, correction, degree, time_aug, lead_lag, end_time, planar, scalar_term, n_jobs)

branched_sig.__doc__ = branched_sig_forward.__doc__


class BranchedSigCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bsig1, bsig2, dimension, degree, planar, n_jobs):
        combined = branched_sig_combine_forward(bsig1, bsig2, dimension, degree,
                                                 planar=planar, n_jobs=n_jobs)
        ctx.save_for_backward(bsig1, bsig2)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.n_jobs = n_jobs
        ctx.planar = planar

        return combined

    @staticmethod
    def backward(ctx, grad_output):
        bsig1, bsig2 = ctx.saved_tensors
        d1, d2 = branched_sig_combine_backprop(
            grad_output, bsig1, bsig2, ctx.dimension, ctx.degree,
            planar=ctx.planar, n_jobs=ctx.n_jobs)
        return d1, d2, None, None, None, None

def branched_sig_combine(
        bsig1: Union[np.ndarray, torch.Tensor],
        bsig2: Union[np.ndarray, torch.Tensor],
        dimension: int,
        degree: int,
        *,
        planar: bool = False,
        n_jobs: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    return BranchedSigCombine.apply(bsig1, bsig2, dimension, degree, planar, n_jobs)

branched_sig_combine.__doc__ = branched_sig_combine_forward.__doc__


class BranchedSigToLogSig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bsig, dimension, degree, time_aug, lead_lag, planar, n_jobs):
        blogsig = branched_sig_to_log_sig_forward(
            bsig, dimension, degree, time_aug=time_aug, lead_lag=lead_lag,
            planar=planar, n_jobs=n_jobs)
        ctx.save_for_backward(bsig)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.planar = planar
        ctx.n_jobs = n_jobs
        return blogsig

    @staticmethod
    def backward(ctx, grad_output):
        bsig, = ctx.saved_tensors
        grad = branched_sig_to_log_sig_backprop(
            bsig, grad_output, ctx.dimension, ctx.degree,
            time_aug=ctx.time_aug, lead_lag=ctx.lead_lag,
            planar=ctx.planar, n_jobs=ctx.n_jobs)
        return grad, None, None, None, None, None, None


def branched_sig_to_log_sig(
        bsig: Union[np.ndarray, torch.Tensor],
        dimension: int,
        degree: int,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        planar: bool = False,
        n_jobs: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    return BranchedSigToLogSig.apply(
        bsig, dimension, degree, time_aug, lead_lag, planar, n_jobs)


branched_sig_to_log_sig.__doc__ = branched_sig_to_log_sig_forward.__doc__


def branched_log_sig(
        path: Union[np.ndarray, torch.Tensor],
        degree: int,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.0,
        planar: bool = False,
        scalar_term: bool = False,
        correction = None,
        n_jobs: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    bsig = branched_sig(
        path, degree, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
        planar=planar, scalar_term=scalar_term, correction=correction, n_jobs=n_jobs)
    dimension = path.shape[-1]
    return branched_sig_to_log_sig(
        bsig, dimension, degree, time_aug=time_aug, lead_lag=lead_lag,
        planar=planar, n_jobs=n_jobs)


branched_log_sig.__doc__ = branched_log_sig_forward.__doc__


class LogSigToSig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_sig, dimension, degree, time_aug, lead_lag, method, scalar_term, n_jobs):
        sig_ = logsig_to_sig_forward(log_sig, dimension, degree, scalar_term=scalar_term, time_aug=time_aug, lead_lag=lead_lag, method=method, n_jobs=n_jobs)

        ctx.save_for_backward(log_sig)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.scalar_term = scalar_term
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.method = method
        ctx.n_jobs = n_jobs

        return sig_

    @staticmethod
    def backward(ctx, grad_output):
        log_sig_ = ctx.saved_tensors[0]
        # scalar_term is only honored by logsig_to_sig_backprop for methods 1, 2.
        # For method 0 it's auto-detected from log_sig input shape.
        grad = logsig_to_sig_backprop(log_sig_, grad_output, ctx.dimension, ctx.degree, scalar_term=ctx.scalar_term, time_aug=ctx.time_aug, lead_lag=ctx.lead_lag, method=ctx.method, n_jobs=ctx.n_jobs)
        return grad, None, None, None, None, None, None, None

def logsig_to_sig(
        log_sig : Union[np.ndarray, torch.Tensor],
        dimension : int,
        degree : int,
        *,
        time_aug : bool = False,
        lead_lag : bool = False,
        method : int = 1,
        scalar_term : bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    return LogSigToSig.apply(log_sig, dimension, degree, time_aug, lead_lag, method, scalar_term, n_jobs)

logsig_to_sig.__doc__ = logsig_to_sig_forward.__doc__


def linear_sig(
        displacement : Union[np.ndarray, torch.Tensor],
        dimension : int,
        degree : int,
        *,
        scalar_term : bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
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
    return sig(path, degree, scalar_term=scalar_term, n_jobs=n_jobs)

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
        sig : Union[np.ndarray, torch.Tensor],
        displacement : Union[np.ndarray, torch.Tensor],
        dimension : int,
        degree : int,
        *,
        prepend : bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    return SigJoin.apply(sig, displacement, dimension, degree, int(prepend), n_jobs)

sig_join.__doc__ = sig_join_forward.__doc__

class LogSigJoin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_sig, displacement, dimension, degree, n_jobs):
        result = log_sig_join_forward(log_sig, displacement, dimension, degree, n_jobs=n_jobs)

        ctx.save_for_backward(log_sig, displacement)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.n_jobs = n_jobs

        return result

    @staticmethod
    def backward(ctx, grad_output):
        log_sig, displacement = ctx.saved_tensors
        d_logsig, d_displacement = log_sig_join_backprop(grad_output, log_sig, displacement, ctx.dimension, ctx.degree, n_jobs=ctx.n_jobs)
        return d_logsig, d_displacement, None, None, None

def log_sig_join(
        log_sig : Union[np.ndarray, torch.Tensor],
        displacement : Union[np.ndarray, torch.Tensor],
        dimension : int,
        degree : int,
        *,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    return LogSigJoin.apply(log_sig, displacement, dimension, degree, n_jobs)

log_sig_join.__doc__ = log_sig_join_forward.__doc__
