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

from typing import Optional, Tuple, Union
from ctypes import POINTER, cast
from math import prod

import numpy as np
import torch

from .branched_sig_kernel import branched_sig_kernel
from .data_handlers import (
    GridOutputHandler,
    MultiplePathInputHandler,
    PathInputHandler,
    ScalarInputHandler,
)
from .dtypes import (
    CPSIG_BRANCHED_SIG_KERNEL_BACKPROP,
    CUSIG_BRANCHED_SIG_KERNEL_BACKPROP,
    DTYPES,
)
from .error_codes import err_msg
from .param_checks import check_non_neg, check_n_jobs, check_type, dyadic_grid_length, parse_dyadic_order
from .sig_kernel import _ensure_3d
from .static_kernels import Context, LinearKernel, StaticKernel
from .transform_path import transform_path
from .transform_path_backprop import transform_path_backprop


def branched_gram_deriv(
        derivs_data,
        data,
        gram: Union[np.ndarray, torch.Tensor],
        k_stack_data,
        depth: int,
        dyadic_order_1: int,
        dyadic_order_2: int,
        return_grid: bool = False,
        n_jobs: int = 1,
) -> torch.Tensor:
    result = GridOutputHandler(data.length[0] - 1, data.length[1] - 1, derivs_data)
    gram_ptr = cast(gram.data_ptr(), POINTER(DTYPES[str(gram.dtype)[6:]]))
    k_stack_ptr = None if k_stack_data is None else k_stack_data.data_ptr

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_KERNEL_BACKPROP[data.dtype](
            gram_ptr, result.data_ptr, derivs_data.data_ptr, k_stack_ptr,
            data.batch_size, data.dimension, data.length[0], data.length[1],
            depth, dyadic_order_1, dyadic_order_2, return_grid, n_jobs)
    else:
        err_code = CUSIG_BRANCHED_SIG_KERNEL_BACKPROP[data.dtype](
            gram_ptr, result.data_ptr, derivs_data.data_ptr, k_stack_ptr,
            data.batch_size, data.dimension, data.length[0], data.length[1],
            depth, dyadic_order_1, dyadic_order_2, return_grid)
    if err_code:
        raise Exception("Error in pysiglib.branched_sig_kernel_backprop: " + err_msg(err_code))
    return torch.as_tensor(result.data)


def branched_sig_kernel_backprop(
        derivs: Union[np.ndarray, torch.Tensor],
        path1: Union[np.ndarray, torch.Tensor],
        path2: Union[np.ndarray, torch.Tensor],
        depth: int,
        dyadic_order: Union[int, tuple],
        *,
        static_kernel: Optional[StaticKernel] = None,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.,
        left_deriv: bool = True,
        right_deriv: bool = False,
        k_stack: Union[np.ndarray, torch.Tensor] = None,
        n_jobs: int = 1,
        return_grid: bool = False,
) -> Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Backpropagates through :func:`pysiglib.branched_sig_kernel`.

    :param derivs: Derivatives with respect to a scalar branched kernel or
        the final-depth grid when ``return_grid=True``.
    :type derivs: numpy.ndarray | torch.Tensor
    :param path1: First path or batch of paths.
    :type path1: numpy.ndarray | torch.Tensor
    :param path2: Second path or batch of paths.
    :type path2: numpy.ndarray | torch.Tensor
    :param depth: Forest depth truncation used in the forward pass.
    :type depth: int
    :param dyadic_order: Dyadic refinement order, or a pair of orders.
    :type dyadic_order: int | tuple
    :param k_stack: Optional internal all-depth grid stack. If omitted, it is
        reconstructed from the static-kernel increments.
    :type k_stack: None | numpy.ndarray | torch.Tensor
    :param return_grid: If ``True``, ``derivs`` is interpreted as final-grid derivatives.
    :type return_grid: bool
    :return: Derivatives with respect to one or both paths.
    :rtype: tuple
    """
    check_type(depth, "depth", int)
    check_non_neg(depth, "depth")
    check_n_jobs(n_jobs)
    check_type(left_deriv, "left_deriv", bool)
    check_type(right_deriv, "right_deriv", bool)
    if not (left_deriv or right_deriv):
        return None, None

    dyadic_order_1, dyadic_order_2 = parse_dyadic_order(dyadic_order)

    if path1.ndim > 3 or path2.ndim > 3:
        if tuple(path1.shape[:-2]) != tuple(path2.shape[:-2]):
            raise ValueError(
                "path1 and path2 must have matching leading batch dimensions; got "
                f"{tuple(path1.shape[:-2])} and {tuple(path2.shape[:-2])}."
            )
        leading_shape = tuple(path1.shape[:-2])
        lead_size = prod(leading_shape)
        flat_derivs = derivs.reshape(lead_size, *derivs.shape[-2:]) if return_grid else derivs.reshape(lead_size)
        flat_k_stack = None if k_stack is None else k_stack.reshape(lead_size, *k_stack.shape[-3:])

        ld, rd = branched_sig_kernel_backprop(
            flat_derivs, _ensure_3d(path1), _ensure_3d(path2), depth, dyadic_order,
            static_kernel=static_kernel,
            time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
            left_deriv=left_deriv, right_deriv=right_deriv,
            k_stack=flat_k_stack, n_jobs=n_jobs, return_grid=return_grid,
        )
        if ld is not None:
            ld = ld.reshape(*leading_shape, *ld.shape[-2:])
        if rd is not None:
            rd = rd.reshape(*leading_shape, *rd.shape[-2:])
        return ld, rd

    if time_aug or lead_lag:
        path1 = transform_path(path1, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)
        path2 = transform_path(path2, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)

    data = MultiplePathInputHandler([path1, path2], False, False, end_time, ["path1", "path2"])

    if data.batch_size == 0:
        from .data_handlers import PathOutputHandler
        ld = PathOutputHandler(data.data[0].data_length, data.data[0].data_dimension, data.data[0]).data
        rd = PathOutputHandler(data.data[1].data_length, data.data[1].data_dimension, data.data[1]).data
        return (ld if left_deriv else None), (rd if right_deriv else None)

    if return_grid:
        derivs_data = PathInputHandler(derivs, False, False, 0., "derivs")
    else:
        derivs_data = ScalarInputHandler(derivs, bool(data.batch_shape), "derivs")

    if not (derivs_data.type_ == data.type_ and derivs_data.device == data.device):
        raise ValueError("derivs, path1 and path2 must all be numpy arrays or all torch tensors on the same device")
    if not return_grid and data.batch_size != derivs_data.batch_size:
        raise ValueError("batch size for derivs does not match batch size of paths")
    grid_len_1 = dyadic_grid_length(data.length[0], dyadic_order_1)
    grid_len_2 = dyadic_grid_length(data.length[1], dyadic_order_2)
    if return_grid:
        expected = data.batch_shape + (grid_len_1, grid_len_2)
        if tuple(derivs.shape) != expected:
            raise ValueError(f"derivs must have shape {expected}, got {tuple(derivs.shape)}")

    torch_path1 = _ensure_3d(torch.as_tensor(data.path[0]))
    torch_path2 = _ensure_3d(torch.as_tensor(data.path[1]))

    ctx = Context()
    if static_kernel is None:
        static_kernel = LinearKernel()
    elif not isinstance(static_kernel, StaticKernel):
        raise ValueError("kernel must be a child class of pysiglib.StaticKernel")

    gram = static_kernel(ctx, torch_path1, torch_path2)

    k_stack_data = None
    if k_stack is not None:
        k_stack_data = PathInputHandler(k_stack, False, False, 0., "k_stack")
        if not (k_stack_data.type_ == data.type_ and k_stack_data.device == data.device):
            raise ValueError("k_stack, derivs, path1 and path2 must be on the same backend and device")
        expected = data.batch_shape + (depth + 1, grid_len_1, grid_len_2)
        if tuple(k_stack.shape) != expected:
            raise ValueError(f"k_stack must have shape {expected}, got {tuple(k_stack.shape)}")

    gram_derivs = branched_gram_deriv(
        derivs_data, data, gram, k_stack_data, depth,
        dyadic_order_1, dyadic_order_2, return_grid, n_jobs)

    ld = static_kernel.grad_x(ctx, gram_derivs) if left_deriv else None
    rd = static_kernel.grad_y(ctx, gram_derivs) if right_deriv else None

    if lead_lag or time_aug:
        ld = transform_path_backprop(ld, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs) if left_deriv else None
        rd = transform_path_backprop(rd, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs) if right_deriv else None

    if data.type_ == "numpy":
        ld = ld.numpy() if left_deriv else None
        rd = rd.numpy() if right_deriv else None

    return ld, rd


def branched_sig_kernel_gram_backprop(
        derivs: Union[np.ndarray, torch.Tensor],
        path1: Union[np.ndarray, torch.Tensor],
        path2: Union[np.ndarray, torch.Tensor],
        depth: int,
        dyadic_order: Union[int, tuple],
        *,
        static_kernel: Optional[StaticKernel] = None,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.,
        left_deriv: bool = True,
        right_deriv: bool = False,
        k_stack: Union[np.ndarray, torch.Tensor] = None,
        n_jobs: int = 1,
        return_grid: bool = False,
        max_batch: int = -1,
) -> Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Backpropagates through :func:`pysiglib.branched_sig_kernel_gram`.
    """
    check_type(depth, "depth", int)
    check_non_neg(depth, "depth")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(left_deriv, "left_deriv", bool)
    check_type(right_deriv, "right_deriv", bool)
    if not (left_deriv or right_deriv):
        return None, None

    check_type(max_batch, "max_batch", int)
    if max_batch == 0 or max_batch < -1:
        raise ValueError("max_batch must be a positive integer or -1")

    symmetric = path1 is path2
    batch_shape_1 = tuple(path1.shape[:-2])
    batch_shape_2 = batch_shape_1 if symmetric else tuple(path2.shape[:-2])
    n1, n2 = len(batch_shape_1), len(batch_shape_2)

    path1 = _ensure_3d(path1)
    path2 = path1 if symmetric else _ensure_3d(path2)

    if n1 != 1 or n2 != 1:
        flat_B1, flat_B2 = path1.shape[0], path2.shape[0]
        derivs = derivs.reshape(flat_B1, flat_B2, *derivs.shape[n1 + n2:]) if return_grid else derivs.reshape(flat_B1, flat_B2)
        if k_stack is not None:
            k_stack = k_stack.reshape(flat_B1, flat_B2, *k_stack.shape[n1 + n2:])

    data = MultiplePathInputHandler([path1, path2], time_aug, lead_lag, end_time, ["path1", "path2"], False)

    derivs = torch.as_tensor(derivs)
    path1 = torch.as_tensor(data.path[0])
    path2 = torch.as_tensor(data.path[1])
    if k_stack is not None:
        k_stack = torch.as_tensor(k_stack)

    batch1 = path1.shape[0]
    batch2 = path2.shape[0]

    ld = torch.zeros(path1.shape, dtype=path1.dtype, device=path1.device) if left_deriv else None
    rd = torch.zeros(path2.shape, dtype=path2.dtype, device=path2.device) if right_deriv else None

    if batch1 == 0 or batch2 == 0:
        if ld is not None:
            ld = ld.reshape(*batch_shape_1, *ld.shape[-2:])
        if rd is not None:
            rd = rd.reshape(*batch_shape_2, *rd.shape[-2:])
        if data.type_ == "numpy":
            return (ld.numpy() if ld is not None else None), (rd.numpy() if rd is not None else None)
        return ld, rd

    if max_batch == -1:
        max_batch = max(batch1, batch2)

    if symmetric:
        idx_i, idx_j = torch.triu_indices(batch1, batch1, device=path1.device)
    else:
        idx_i = torch.arange(batch1, device=path1.device).repeat_interleave(batch2)
        idx_j = torch.arange(batch2, device=path2.device).repeat(batch1)

    src2 = path1 if symmetric else path2
    n_pairs = idx_i.shape[0]
    chunk_size = max_batch * max_batch

    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        ci = idx_i[start:end]
        cj = idx_j[start:end]

        path1_ = path1[ci]
        path2_ = src2[cj]
        derivs_ = derivs[ci, cj]
        k_stack_ = None if k_stack is None else k_stack[ci, cj]

        ld_, rd_ = branched_sig_kernel_backprop(
            derivs_, path1_, path2_, depth, dyadic_order,
            static_kernel=static_kernel,
            time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
            left_deriv=left_deriv, right_deriv=right_deriv,
            k_stack=k_stack_, n_jobs=n_jobs, return_grid=return_grid)

        if left_deriv:
            ld.index_add_(0, ci, ld_.to(ld.dtype))
        if right_deriv:
            rd.index_add_(0, cj, rd_.to(rd.dtype))

        if symmetric:
            off = ci != cj
            if off.any():
                ci_off = ci[off]
                cj_off = cj[off]
                path1_t = src2[cj_off]
                path2_t = path1[ci_off]
                derivs_t = derivs[cj_off, ci_off]
                k_stack_t = None if k_stack is None else k_stack[cj_off, ci_off]

                ld_t, rd_t = branched_sig_kernel_backprop(
                    derivs_t, path1_t, path2_t, depth, dyadic_order,
                    static_kernel=static_kernel,
                    time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
                    left_deriv=left_deriv, right_deriv=right_deriv,
                    k_stack=k_stack_t, n_jobs=n_jobs, return_grid=return_grid)

                if left_deriv:
                    ld.index_add_(0, cj_off, ld_t.to(ld.dtype))
                if right_deriv:
                    rd.index_add_(0, ci_off, rd_t.to(rd.dtype))

    if ld is not None:
        ld = ld.reshape(*batch_shape_1, *ld.shape[-2:])
    if rd is not None:
        rd = rd.reshape(*batch_shape_2, *rd.shape[-2:])

    if data.type_ == "numpy":
        return (ld.numpy() if ld is not None else None), (rd.numpy() if rd is not None else None)
    return ld, rd
