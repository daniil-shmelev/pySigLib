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

from __future__ import annotations
from array_api_compat import array_namespace, device
from ._array import NativeArrayT as Array, empty_like_shape
from ._storage import pair_indices, pointer

from typing import Union, Optional
import warnings


from .transform_path import transform_path
from .param_checks import check_type, parse_dyadic_order, dyadic_grid_length, check_n_jobs
from .error_codes import err_msg


def _ensure_3d(t):
    """Ensure a path tensor is exactly 3D (batch, length, dim)."""
    if t.ndim == 2:
        return t.reshape(1, t.shape[-2], t.shape[-1])
    if t.ndim > 3:
        return t.reshape(-1, t.shape[-2], t.shape[-1])
    return t


_NORMALIZE_WARNING = (
    "encountered non-positive K(x,x)*K(y,y). "
    "This indicates the PDE solver has not converged. Increase the solver order, "
    "scale down your paths, or use a bounded static kernel (e.g., RBFKernel). "
    "Affected entries will be set to NaN."
)

_SIG_KERNEL_METHODS = ("finite_difference", "polynomial")


def _parse_sig_kernel_method(method, dyadic_order, order, return_grid):
    check_type(method, "method", str)
    if method not in _SIG_KERNEL_METHODS:
        raise ValueError("method must be one of " + ", ".join(_SIG_KERNEL_METHODS))

    if method == "finite_difference":
        if dyadic_order is None:
            raise ValueError("dyadic_order is required when method='finite_difference'")
        if order is not None:
            raise ValueError("order is not supported when method='finite_difference'")
        return parse_dyadic_order(dyadic_order)

    if dyadic_order is not None:
        raise ValueError("dyadic_order is not supported when method='" + method + "'")
    if order is None:
        raise ValueError("order is required when method='" + method + "'")
    if type(order) is not int:
        raise TypeError("order must be of type int")
    if order < 2 or order > 64:
        raise ValueError("order must be between 2 and 64")
    return 0, 0


def _safe_normalize(result, k1, k2, func_name, stacklevel=2):
    """Normalize kernel values by sqrt(K(x,x)*K(y,y)), setting NaN where denom <= 0."""
    xp = array_namespace(result)
    denom = k1 * k2
    bad = denom <= 0
    if xp.any(bad):
        warnings.warn(func_name + ": " + _NORMALIZE_WARNING, RuntimeWarning, stacklevel=stacklevel + 1)
    return xp.where(bad, float("nan"), result / xp.sqrt(xp.clip(denom, min=1e-30)))


from .dtypes import (
    CPSIG_SIG_KERNEL_POLY,
    CPSIG_SIG_KERNEL,
    CUSIG_SIG_KERNEL_POLY,
    CUSIG_SIG_KERNEL,
)
from .data_handlers import MultiplePathInputHandler, ScalarOutputHandler, GridOutputHandler
from .static_kernels import StaticKernel, LinearKernel, Context


def sig_kernel(
    path1: Array,
    path2: Array,
    *,
    method: str = "finite_difference",
    dyadic_order: Optional[Union[int, tuple]] = None,
    order: Optional[int] = None,
    static_kernel: Optional[StaticKernel] = None,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    n_jobs: int = 1,
    return_grid: bool = False,
    normalize: bool = False,
    _return_state: bool = False,
) -> Array:
    """
    Computes a single signature kernel or a batch of signature kernels.
    The signature kernel of two :math:`d`-dimensional paths :math:`x,y`
    is defined as

    .. math::

        k_{x,y}(s,t) := \\left< S(x)_{[0,s]}, S(y)_{[0, t]} \\right>_{T((\\mathbb{R}^d))}

    where the inner product is defined as

    .. math::

        \\left< A, B \\right> := \\sum_{k=0}^{\\infty} \\left< A_k, B_k \\right>_{\\left(\\mathbb{R}^d\\right)^{\\otimes k}}
    .. math::

        \\left< u, v \\right>_{\\left(\\mathbb{R}^d\\right)^{\\otimes k}} := \\prod_{i=1}^k \\left< u_i, v_i \\right>_{\\mathbb{R}^d}.

    Optionally, a static kernel can be specified. For details, see the documentation on
    :doc:`static kernels </pages/signature_kernels/static_kernels>`.

    :param path1: The first underlying path or batch of paths, of shape
        ``(..., length_1, dimension)``.
    :type path1: Array
    :param path2: The second underlying path or batch of paths, of shape
        ``(..., length_2, dimension)``. Leading batch dimensions must match those of
        ``path1``.
    :type path2: Array
    :param dyadic_order: (``method="finite_difference"`` only) If set to a positive integer :math:`\\lambda`, will refine the
        paths by a factor of :math:`2^\\lambda`. If set to a tuple of positive integers
        :math:`(\\lambda_1, \\lambda_2)`, will refine the first path by :math:`2^{\\lambda_1}`
        and the second path by :math:`2^{\\lambda_2}`.
    :type dyadic_order: None | int | tuple
    :param method: Solver method. Must be ``"finite_difference"`` or ``"polynomial"``.
    :type method: str
    :param order: (``method="polynomial"`` only) Highest retained polynomial degree. Must be
        between 2 and 64. Unsupported for finite differences.
    :type order: None | int
    :param static_kernel: Static kernel. If ``None`` (default), the linear kernel will be used.
        For details, see the documentation on :doc:`static kernels </pages/signature_kernels/static_kernels>`.
    :type static_kernel: None | pysiglib.StaticKernel
    :param time_aug: If set to True, will compute the signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math:`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the signature of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param n_jobs: (Only applicable to CPU computation) Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :param return_grid: If ``True``, returns the entire PDE grid. Finite differences
        return the dyadically refined grid, while the polynomial method returns one
        value per vertex of each transformed path.
    :type return_grid: bool
    :param normalize: If ``True``, normalizes the signature kernel so that :math:`k(x, x) = 1`
        by dividing by :math:`\\sqrt{k(x, x) \\cdot k(y, y)}`. Cannot be used with ``return_grid=True``.
    :type normalize: bool
    :return: Single signature kernel or batch of signature kernels
    :rtype: Array

    Example:
    ---------

    .. code-block:: python

        import numpy as np
        import pysiglib

        path1 = np.random.random((10, 100, 5))
        path2 = np.random.random((10, 100, 5))
        k = pysiglib.sig_kernel(path1, path2, dyadic_order=2)
        print(k)

    .. code-block:: python

        # Using an RBF static kernel with time augmentation
        import numpy as np
        import pysiglib

        path1 = np.random.random((10, 100, 5))
        path2 = np.random.random((10, 100, 5))
        rbf = pysiglib.RBFKernel(sigma=1.0)
        k = pysiglib.sig_kernel(
            path1, path2,
            dyadic_order=2,
            static_kernel=rbf,
            time_aug=True,
        )
        print(k)

    .. code-block:: python

        # Asymmetric dyadic orders and returning the PDE grid
        import numpy as np
        import pysiglib

        path1 = np.random.random((100, 5))
        path2 = np.random.random((80, 5))
        grid = pysiglib.sig_kernel(
            path1, path2,
            dyadic_order=(2, 3),
            return_grid=True,
        )
        print(grid.shape)

    """
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_n_jobs(n_jobs)
    dyadic_order_1, dyadic_order_2 = _parse_sig_kernel_method(
        method, dyadic_order, order, return_grid)
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")
    if _return_state and method != "polynomial":
        raise ValueError("_return_state is only supported for method='polynomial'")
    if _return_state and normalize:
        raise ValueError("_return_state cannot be used with normalize=True")

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
        if _return_state:
            state = empty_like_shape(
                data.path[0], (0, data.length[0] - 1, data.length[1] - 1, 2, order + 1)
            )
            return result.data, state
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

    if method == "polynomial":
        state = (
            empty_like_shape(
                gram,
                (data.batch_size, data.length[0] - 1, data.length[1] - 1, 2, order + 1),
            )
            if _return_state
            else None
        )
        state_ptr = None if state is None else pointer(state)

    if method == "polynomial" and data.device == "cpu":
        err_code = CPSIG_SIG_KERNEL_POLY[data.dtype](
            gram_ptr, result.data_ptr, state_ptr, data.batch_size, data.dimension,
            data.length[0], data.length[1], order, return_grid, n_jobs)
    elif method == "polynomial":
        err_code = CUSIG_SIG_KERNEL_POLY[data.dtype](
            gram_ptr, result.data_ptr, state_ptr, data.batch_size, data.dimension,
            data.length[0], data.length[1], order, return_grid)
    elif data.device == "cpu":
        err_code = CPSIG_SIG_KERNEL[data.dtype](
            gram_ptr, result.data_ptr, data.batch_size, data.dimension,
            data.length[0], data.length[1],
            dyadic_order_1, dyadic_order_2, return_grid, n_jobs)
    else:
        err_code = CUSIG_SIG_KERNEL[data.dtype](
            gram_ptr, result.data_ptr, data.batch_size, data.dimension,
            data.length[0], data.length[1],
            dyadic_order_1, dyadic_order_2, return_grid)
    if err_code:
        raise Exception(
            "Error in pysiglib.sig_kernel: "
            + err_msg(err_code, result.device))

    has_bad = not data.xp.all(data.xp.isfinite(result.data))

    if has_bad:
        warnings.warn(
            "sig_kernel produced NaN or Inf values. This is typically caused by "
            "paths with large increments, leading to numerical overflow in the "
            "PDE solver. Consider normalizing your paths or using a static kernel "
            "(e.g., pysiglib.RBFKernel) to bound the inner products.",
            RuntimeWarning,
            stacklevel=2
        )

    if path1 is path2 and not has_bad and not return_grid:
        has_neg = data.xp.any(result.data < 0)
        if has_neg:
            warnings.warn(
                "sig_kernel produced negative K(X, X) values."
                "This is typically caused by PDE-solver instability at large "
                "dyadic_order or large path increments. Consider reducing "
                "dyadic_order, scaling paths down, or using a bounded static "
                "kernel (e.g., pysiglib.RBFKernel).",
                RuntimeWarning,
                stacklevel=2
            )

    if normalize:
        k1 = sig_kernel(path1, path1, dyadic_order=dyadic_order, method=method, order=order,
                        static_kernel=static_kernel, n_jobs=n_jobs)
        k2 = sig_kernel(path2, path2, dyadic_order=dyadic_order, method=method, order=order,
                        static_kernel=static_kernel, n_jobs=n_jobs)
        result.data = _safe_normalize(result.data, k1, k2, "sig_kernel(normalize=True)")

    if _return_state:
        return result.data, state
    return result.data


def sig_kernel_gram(
    path1: Array,
    path2: Array,
    *,
    method: str = "finite_difference",
    dyadic_order: Optional[Union[int, tuple]] = None,
    order: Optional[int] = None,
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
    Given batches of paths :math:`\\{x_i\\}_{i=1}^{B_1}` and :math:`\\{y_j\\}_{j=1}^{B_2}`, computes the gram matrix of signature kernels

    .. math::

        G = (k_{x_i, y_j})_{i = 1, j = 1}^{B_1, B_2}.

    The signature kernel of two :math:`d`-dimensional paths :math:`x,y`
    is defined as

    .. math::

        k_{x,y}(s,t) := \\left< S(x)_{[0,s]}, S(y)_{[0, t]} \\right>_{T((\\mathbb{R}^d))}

    where the inner product is defined as

    .. math::

        \\left< A, B \\right> := \\sum_{k=0}^{\\infty} \\left< A_k, B_k \\right>_{\\left(\\mathbb{R}^d\\right)^{\\otimes k}}
    .. math::

        \\left< u, v \\right>_{\\left(\\mathbb{R}^d\\right)^{\\otimes k}} := \\prod_{i=1}^k \\left< u_i, v_i \\right>_{\\mathbb{R}^d}.

    Optionally, a static kernel can be specified. For details, see the documentation on
    :doc:`static kernels </pages/signature_kernels/static_kernels>`.

    :param path1: A path or batch of paths, of shape ``(*batch_shape_1, length_1, dimension)``.
    :type path1: Array
    :param path2: A path or batch of paths, of shape ``(*batch_shape_2, length_2, dimension)``.
        Independent of ``path1``'s batch shape.
    :type path2: Array
    :param dyadic_order: (``method="finite_difference"`` only) If set to a positive integer :math:`\\lambda`, will refine the
        paths by a factor of :math:`2^\\lambda`. If set to a tuple of positive integers
        :math:`(\\lambda_1, \\lambda_2)`, will refine the first path by :math:`2^{\\lambda_1}`
        and the second path by :math:`2^{\\lambda_2}`.
    :type dyadic_order: None | int | tuple
    :param method: Solver method. Must be ``"finite_difference"`` or ``"polynomial"``.
    :type method: str
    :param order: (``method="polynomial"`` only) Highest retained polynomial degree. Must be
        between 2 and 64. Unsupported for finite differences.
    :type order: None | int
    :param static_kernel: Static kernel. If ``None`` (default), the linear kernel will be used.
        For details, see the documentation on :doc:`static kernels </pages/signature_kernels/static_kernels>`.
    :type static_kernel: None | pysiglib.StaticKernel
    :param time_aug: If set to True, will compute the signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math:`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the signature of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param n_jobs: (Only applicable to CPU computation) Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :param max_batch: Maximum batch size to run in parallel. In the JAX API, this bounds
        the number of path pairs passed to each solver invocation. If the computation is
        failing due to insufficient memory, this parameter should be decreased.
        If set to -1, no explicit batch limit is applied.
    :type max_batch: int
    :param return_grid: If ``True``, returns the entire PDE grid. Finite differences
        return the dyadically refined grid, while the polynomial method returns one
        value per vertex of each transformed path.
    :type return_grid: bool
    :param normalize: If ``True``, normalizes the gram matrix so that :math:`K(x, x) = 1` by
        dividing each entry by :math:`\\sqrt{K(x_i, x_i) \\cdot K(y_j, y_j)}`. Cannot be used with ``return_grid=True``.
    :type normalize: bool
    :return: Gram matrix of signature kernels, of shape ``(*batch_shape_1, *batch_shape_2)``.
        With ``return_grid=True``, the final two dimensions are the dyadically refined
        grid lengths for finite differences, or the transformed path vertex lengths for
        the polynomial method.
    :rtype: Array

    .. note::

        When called via ``pysiglib.torch_api``, the default behaviour is to reconstruct the
        PDE grids during backpropagation. This is done to avoid memory allocation issues for large batch sizes.

    Example:
    ---------

    .. code-block:: python

        import numpy as np
        import pysiglib

        path1 = np.random.random((10, 100, 5))
        path2 = np.random.random((8, 100, 5))
        gram = pysiglib.sig_kernel_gram(path1, path2, dyadic_order=2)
        print(gram.shape) # gram has shape (10, 8)
        print(gram)

    .. code-block:: python

        # Gram matrix with a static kernel
        import numpy as np
        import pysiglib

        path1 = np.random.random((10, 100, 5))
        path2 = np.random.random((8, 100, 5))
        rbf = pysiglib.RBFKernel(sigma=0.5)
        gram = pysiglib.sig_kernel_gram(
            path1, path2,
            dyadic_order=2,
            static_kernel=rbf,
            time_aug=True,
            lead_lag=True,
            max_batch=4,
        )
        print(gram.shape)

    .. code-block:: python

        # Multi-dim batches: leading dims are flattened and the result has shape
        # (*batch_shape_1, *batch_shape_2)
        import numpy as np
        import pysiglib

        path1 = np.random.random((4, 10, 100, 5))  # batch_shape_1 = (4, 10)
        path2 = np.random.random((8, 100, 5))      # batch_shape_2 = (8,)
        gram = pysiglib.sig_kernel_gram(path1, path2, dyadic_order=2)
        print(gram.shape)  # gram has shape (4, 10, 8)

    """
    # We use sig_kernel for simplicity, rather than directly calling
    # the cpp function.
    # There is clearly more overhead here than is necessary, but it
    # shouldn't be significant for large computations.

    symmetric = path1 is path2

    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(max_batch, "max_batch", int)
    check_n_jobs(n_jobs)
    do1, do2 = _parse_sig_kernel_method(method, dyadic_order, order, return_grid)
    if max_batch == 0 or max_batch < -1:
        raise ValueError("max_batch must be a positive integer or -1")
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")

    batch_shape_1 = tuple(path1.shape[:-2])
    batch_shape_2 = batch_shape_1 if symmetric else tuple(path2.shape[:-2])

    path1 = _ensure_3d(path1)
    path2 = path1 if symmetric else _ensure_3d(path2)

    data = MultiplePathInputHandler([path1, path2], time_aug, lead_lag, end_time, ["path1", "path2"], False)

    # Preserve the input backend throughout batch computation.
    xp = data.xp
    path1 = data.path[0]
    path2 = data.path[1]

    batch1 = path1.shape[0]
    batch2 = path2.shape[0]

    if max_batch == -1:
        max_batch = max(batch1, batch2)

    ####################################
    # Now run computation in batches
    ####################################

    if return_grid:
        gl1 = dyadic_grid_length(data.length[0], do1)
        gl2 = dyadic_grid_length(data.length[1], do2)

    if symmetric:
        # Symmetric case: only compute upper triangle pairs, mirror to lower.
        # Cannot mirror grids when dyadic orders differ (gl1 != gl2), so fall through.
        if return_grid and gl1 != gl2:
            symmetric = False

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

        k = sig_kernel(src1[ci], src2[cj], dyadic_order=dyadic_order, method=method, order=order,
                       static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
                       end_time=end_time, n_jobs=n_jobs, return_grid=return_grid)
        res[ci, cj] = k

        if symmetric:
            off = ci != cj
            if off.any():
                k_mirror = k[off]
                if return_grid:
                    k_mirror = xp.matrix_transpose(k_mirror)
                res[cj[off], ci[off]] = k_mirror

    if normalize:
        d1 = sig_kernel(path1, path1, dyadic_order=dyadic_order, method=method, order=order,
                        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
                        end_time=end_time, n_jobs=n_jobs)
        d2 = sig_kernel(path2, path2, dyadic_order=dyadic_order, method=method, order=order,
                        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
                        end_time=end_time, n_jobs=n_jobs) if not symmetric else d1
        res = _safe_normalize(res, xp.expand_dims(d1, axis=1), xp.expand_dims(d2, axis=0), 'sig_kernel_gram(normalize=True)')

    out_shape = batch_shape_1 + batch_shape_2
    if return_grid:
        out_shape = out_shape + (res.shape[-2], res.shape[-1])
    res = res.reshape(out_shape) if out_shape else res.squeeze()

    return res
