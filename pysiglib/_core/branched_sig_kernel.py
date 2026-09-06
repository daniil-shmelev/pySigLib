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

from __future__ import annotations
from array_api_compat import device
from ._array import NativeArrayT as Array
from ._storage import pair_indices, pointer

from typing import Optional, Union
import warnings


from .data_handlers import MultiplePathInputHandler, ScalarOutputHandler, GridOutputHandler
from .dtypes import (
    CPSIG_BRANCHED_SIG_KERNEL,
    CUSIG_BRANCHED_SIG_KERNEL,
)
from .error_codes import err_msg
from .param_checks import (
    check_non_neg,
    check_n_jobs,
    check_type,
    dyadic_grid_length,
    parse_dyadic_order,
)
from .sig_kernel import _ensure_3d, _safe_normalize
from .static_kernels import Context, LinearKernel, StaticKernel
from .transform_path import transform_path


def branched_sig_kernel(
    path1: Array,
    path2: Array,
    depth: int,
    dyadic_order: Union[int, tuple],
    *,
    static_kernel: Optional[StaticKernel] = None,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    n_jobs: int = 1,
    return_grid: bool = False,
    normalize: bool = False,
) -> Array:
    """
    Computes a single branched signature kernel or a batch of branched
    signature kernels.

    This is the non-planar BCK branched rough path kernel of Chevyrev and
    Oberhauser. It uses the depth recursion
    ``K_0 = 1`` and ``K_m = exp(integral K_{m-1} dk)`` with the same
    static-kernel increment abstraction as :func:`pysiglib.sig_kernel`.
    The parameter ``depth`` truncates forests by tree depth, not by the
    number of nodes.

    :param path1: First path or batch of paths, shape ``(..., length_1, dimension)``.
    :type path1: Array
    :param path2: Second path or batch of paths, shape ``(..., length_2, dimension)``.
    :type path2: Array
    :param depth: Forest depth truncation for the branched kernel.
    :type depth: int
    :param dyadic_order: Dyadic refinement order, or a pair of orders.
    :type dyadic_order: int | tuple
    :param static_kernel: Static kernel. If ``None``, the linear kernel is used.
    :type static_kernel: None | pysiglib.StaticKernel
    :param time_aug: Whether to time augment the paths.
    :type time_aug: bool
    :param lead_lag: Whether to apply the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time augmentation.
    :type end_time: float
    :param n_jobs: Number of CPU worker threads.
    :type n_jobs: int
    :param return_grid: If ``True``, returns the final-depth grid.
    :type return_grid: bool
    :param normalize: If ``True``, normalizes by
        ``sqrt(K(x, x) * K(y, y))``. Cannot be used with ``return_grid=True``.
    :type normalize: bool
    :return: Branched signature kernel value or final-depth grid.
    :rtype: Array
    """
    check_type(depth, "depth", int)
    check_non_neg(depth, "depth")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_n_jobs(n_jobs)
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")

    dyadic_order_1, dyadic_order_2 = parse_dyadic_order(dyadic_order)

    if time_aug or lead_lag:
        path1 = transform_path(path1, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)
        path2 = transform_path(path2, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)

    data = MultiplePathInputHandler([path1, path2], False, False, 0., ["path1", "path2"])

    if not return_grid:
        result = ScalarOutputHandler(data)
    else:
        dyadic_len_1 = dyadic_grid_length(data.length[0], dyadic_order_1)
        dyadic_len_2 = dyadic_grid_length(data.length[1], dyadic_order_2)
        result = GridOutputHandler(dyadic_len_1, dyadic_len_2, data)

    if data.batch_size == 0:
        return result.data

    kernel_path1 = _ensure_3d(data.path[0])
    kernel_path2 = _ensure_3d(data.path[1])

    ctx = Context()
    if static_kernel is None:
        static_kernel = LinearKernel()
    elif not isinstance(static_kernel, StaticKernel):
        raise ValueError("kernel must be a child class of pysiglib.StaticKernel")

    gram = static_kernel(ctx, kernel_path1, kernel_path2)
    gram_ptr = pointer(gram)

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_KERNEL[data.dtype](
            gram_ptr, result.data_ptr, data.batch_size, data.dimension,
            data.length[0], data.length[1], depth,
            dyadic_order_1, dyadic_order_2, return_grid, n_jobs)
    else:
        err_code = CUSIG_BRANCHED_SIG_KERNEL[data.dtype](
            gram_ptr, result.data_ptr, data.batch_size, data.dimension,
            data.length[0], data.length[1], depth,
            dyadic_order_1, dyadic_order_2, return_grid)
    if err_code:
        raise Exception(
            "Error in pysiglib.branched_sig_kernel: "
            + err_msg(err_code, result.device))

    has_bad = not data.xp.all(data.xp.isfinite(result.data))

    if has_bad:
        warnings.warn(
            "branched_sig_kernel produced NaN or Inf values. This is typically "
            "caused by large static-kernel increments. Consider scaling paths "
            "or using a bounded static kernel.",
            RuntimeWarning,
            stacklevel=2,
        )

    if normalize:
        k1 = branched_sig_kernel(path1, path1, depth, dyadic_order, static_kernel=static_kernel, n_jobs=n_jobs)
        k2 = branched_sig_kernel(path2, path2, depth, dyadic_order, static_kernel=static_kernel, n_jobs=n_jobs)
        result.data = _safe_normalize(result.data, k1, k2, "branched_sig_kernel(normalize=True)")

    return result.data


def branched_sig_kernel_gram(
    path1: Array,
    path2: Array,
    depth: int,
    dyadic_order: Union[int, tuple],
    *,
    static_kernel: Optional[StaticKernel] = None,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    n_jobs: int = 1,
    max_batch: int = -1,
    return_grid: bool = False,
    normalize: bool = False,
) -> Array:
    """
    Computes a Gram matrix of branched signature kernels.

    :param path1: First path batch, shape ``(*batch_shape_1, length_1, dimension)``.
    :type path1: Array
    :param path2: Second path batch, shape ``(*batch_shape_2, length_2, dimension)``.
    :type path2: Array
    :param depth: Forest depth truncation for the branched kernel.
    :type depth: int
    :param dyadic_order: Dyadic refinement order, or a pair of orders.
    :type dyadic_order: int | tuple
    :param max_batch: Maximum side length of pair chunks. ``-1`` uses all pairs.
    :type max_batch: int
    :param return_grid: If ``True``, returns final-depth grids per pair.
    :type return_grid: bool
    :param normalize: If ``True``, normalizes the Gram matrix.
    :type normalize: bool
    :return: Gram matrix of branched signature kernels.
    :rtype: Array
    """
    check_type(depth, "depth", int)
    check_non_neg(depth, "depth")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(max_batch, "max_batch", int)
    if max_batch == 0 or max_batch < -1:
        raise ValueError("max_batch must be a positive integer or -1")
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")

    symmetric = path1 is path2
    batch_shape_1 = tuple(path1.shape[:-2])
    batch_shape_2 = batch_shape_1 if symmetric else tuple(path2.shape[:-2])

    path1 = _ensure_3d(path1)
    path2 = path1 if symmetric else _ensure_3d(path2)

    data = MultiplePathInputHandler([path1, path2], time_aug, lead_lag, end_time, ["path1", "path2"], False)

    xp = data.xp
    path1 = data.path[0]
    path2 = data.path[1]

    batch1 = path1.shape[0]
    batch2 = path2.shape[0]

    do1, do2 = parse_dyadic_order(dyadic_order)

    if return_grid:
        gl1 = dyadic_grid_length(data.length[0], do1)
        gl2 = dyadic_grid_length(data.length[1], do2)
        if symmetric and gl1 != gl2:
            symmetric = False

    if batch1 == 0 or batch2 == 0:
        out_shape = batch_shape_1 + batch_shape_2
        if return_grid:
            out_shape = out_shape + (gl1, gl2)
        res = xp.empty(out_shape, dtype=path1.dtype, device=device(path1))
        return res

    if max_batch == -1:
        max_batch = max(batch1, batch2)

    idx_i, idx_j = pair_indices(batch1, batch2, symmetric, path1)
    src1, src2 = path1, path1 if symmetric else path2

    if return_grid:
        res = xp.empty((batch1, batch2, gl1, gl2), dtype=path1.dtype, device=device(path1))
    else:
        res = xp.empty((batch1, batch2), dtype=path1.dtype, device=device(path1))

    chunk_size = max_batch * max_batch
    for start in range(0, idx_i.shape[0], chunk_size):
        end = min(start + chunk_size, idx_i.shape[0])
        ci = idx_i[start:end]
        cj = idx_j[start:end]

        k = branched_sig_kernel(
            src1[ci], src2[cj], depth, dyadic_order,
            static_kernel=static_kernel, time_aug=time_aug,
            lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs,
            return_grid=return_grid)
        res[ci, cj] = k

        if symmetric:
            off = ci != cj
            if off.any():
                k_mirror = k[off]
                if return_grid:
                    k_mirror = xp.matrix_transpose(k_mirror)
                res[cj[off], ci[off]] = k_mirror

    if normalize:
        d1 = branched_sig_kernel(path1, path1, depth, dyadic_order, static_kernel=static_kernel,
                                 time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)
        d2 = d1 if symmetric else branched_sig_kernel(
            path2, path2, depth, dyadic_order, static_kernel=static_kernel,
            time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)
        res = _safe_normalize(res, xp.expand_dims(d1, axis=1), xp.expand_dims(d2, axis=0), 'branched_sig_kernel_gram(normalize=True)')

    out_shape = batch_shape_1 + batch_shape_2
    if return_grid:
        out_shape = out_shape + (res.shape[-2], res.shape[-1])
    res = res.reshape(out_shape) if out_shape else res.squeeze()

    return res
