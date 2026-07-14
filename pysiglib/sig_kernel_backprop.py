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

from typing import Union, Tuple, Optional
from ctypes import POINTER, cast
from math import prod

import numpy as np
import torch

from .transform_path import transform_path
from .transform_path_backprop import transform_path_backprop
from .sig_kernel import sig_kernel, _ensure_3d
from .param_checks import (
    check_type, parse_dyadic_order, parse_log_pde_parameters, check_n_jobs,
)
from .error_codes import err_msg
from .dtypes import (
    CPSIG_SIG_KERNEL_BACKPROP, CPSIG_SIG_KERNEL_LOG_PDE_BACKPROP,
    DTYPES, CUSIG_SIG_KERNEL_BACKPROP_CUDA,
)
from .data_handlers import (
    MultiplePathInputHandler, ScalarInputHandler, GridOutputHandler,
    PathInputHandler, PathOutputHandler,
)
from .static_kernels import StaticKernel, LinearKernel, Context

def gram_deriv(
        derivs_data,
        data,
        gram : Union[np.ndarray, torch.tensor],
        k_grid_data : Union[np.ndarray, torch.tensor],
        dyadic_order_1,
        dyadic_order_2,
        return_grid : bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:

    result = GridOutputHandler(data.length[0] - 1, data.length[1] - 1, derivs_data) #Derivatives with respect to gram matrix
    gram_ptr = cast(gram.data_ptr(), POINTER(DTYPES[str(gram.dtype)[6:]]))

    if data.device == "cpu":
        err_code = CPSIG_SIG_KERNEL_BACKPROP[data.dtype](
            gram_ptr, result.data_ptr, derivs_data.data_ptr, k_grid_data.data_ptr,
            data.batch_size, data.dimension, data.length[0], data.length[1],
            dyadic_order_1, dyadic_order_2, return_grid, n_jobs)
    else:
        err_code = CUSIG_SIG_KERNEL_BACKPROP_CUDA[data.dtype](
            gram_ptr, result.data_ptr, derivs_data.data_ptr, k_grid_data.data_ptr,
            data.batch_size, data.dimension, data.length[0], data.length[1],
            dyadic_order_1, dyadic_order_2, return_grid)
    if err_code:
        raise Exception("Error in pysiglib.sig_kernel_backprop: " + err_msg(err_code))
    return torch.as_tensor(result.data)

def sig_kernel_backprop(
        derivs : Union[np.ndarray, torch.tensor],
        path1 : Union[np.ndarray, torch.tensor],
        path2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        *,
        method : str = "pde",
        log_degree : Union[int, tuple, None] = None,
        log_steps : Union[int, tuple, None] = None,
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        left_deriv : bool = True,
        right_deriv : bool = False,
        k_grid : Union[np.ndarray, torch.tensor] = None,
        n_jobs : int = 1,
        return_grid: bool = False
) -> Union[np.ndarray, torch.tensor, Tuple[np.ndarray, np.ndarray], Tuple[torch.tensor, torch.tensor]]:
    """
    This function is required to backpropagate through ``pysiglib.sig_kernel``.
    Given the derivatives of a scalar function :math:`F` with respect to a
    signature kernel, :math:`\\partial F / \\left< S(x), S(y) \\right>`,
    returns the derivatives of :math:`F` with respect to one or both of the
    underlying paths, :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^{L_1}` and
    :math:`\\{\\partial F / y_{t_i}\\}_{i=0}^{L_2}`.

    :param derivs: Derivatives with respect to a signature kernel or batch
        of signature kernels, :math:`\\partial F / \\left< S(x), S(y) \\right>`.
        If ``return_grid=False``, this should be of shape ``(...)`` matching the leading
        batch dimensions of the paths. If ``return_grid=True``, this should have the same
        shape as the PDE grid returned by ``pysiglib.sig_kernel(..., return_grid=True)``.
    :type derivs: numpy.ndarray | torch.tensor
    :param path1: The first underlying path or batch of paths, of shape
        ``(..., length_1, dimension)``.
    :type path1: numpy.ndarray | torch.tensor
    :param path2: The second underlying path or batch of paths, of shape
        ``(..., length_2, dimension)``. Leading batch dimensions must match those of
        ``path1``.
    :type path2: numpy.ndarray | torch.tensor
    :param dyadic_order: The dyadic order(s) used to compute the signature kernels.
    :type dyadic_order: int | tuple
    :param method: Forward PDE method. Use ``"pde"`` for the standard Goursat
        solver or ``"log_pde"`` for the higher-order log-PDE method. The log-PDE
        method is currently CPU-only and supports only the linear static kernel.
    :type method: str
    :param log_degree: Tensor-log truncation degree. Required for
        ``method="log_pde"``. An integer applies to both paths; a pair applies
        separately to the first and second paths.
    :type log_degree: int | tuple | None
    :param log_steps: Number of original path intervals per tensor-log block for
        ``method="log_pde"``. Required for that method. Each value must divide
        its path's interval count. An integer applies to both paths; a pair applies
        separately. With ``lead_lag=True``, values still count original intervals.
    :type log_steps: int | tuple | None
    :param static_kernel: Static kernel. If ``None`` (default), the linear kernel will be used.
        For details, see the documentation on :doc:`static kernels </pages/signature_kernels/static_kernels>`.
    :type static_kernel: None | pysiglib.StaticKernel
    :param time_aug: Whether the signature kernels were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the signature kernels were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param left_deriv: If ``True``, returns :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^{L_1}`.
        At least one of ``left_deriv`` and ``right_deriv`` must be ``True``. If both are
        ``True``, returns both derivatives as a tuple.
    :type left_deriv: bool
    :param right_deriv: If ``True``, returns :math:`\\{\\partial F / y_{t_i}\\}_{i=0}^{L_2}`.
        At least one of ``left_deriv`` and ``right_deriv`` must be ``True``. If both are
        ``True``, returns both derivatives as a tuple.
    :type right_deriv: bool
    :param k_grid: Full forward grid. If ``None``, the grid is recomputed. For
        ``method="log_pde"``, a supplied grid checkpoints the scalar kernel state;
        the auxiliary tensor states required by backpropagation are still recomputed.
    :type k_grid: numpy.ndarray | torch.tensor
    :param n_jobs: (Only applicable to CPU computation) Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :param return_grid: If ``True``, backpropagates derivatives with respect to the entire PDE grid.
    :type return_grid: bool
    :return: Tuple of derivatives of :math:`F` with respect to one or both of the
        underlying paths. If ``left_deriv`` is ``True``, the first element of
        this tuple is  :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^{L_1}`, otherwise
        it is ``None``. Similarly for ``right_deriv`` and
        :math:`\\{\\partial F / y_{t_i}\\}_{i=0}^{L_2}`.
    :rtype: numpy.ndarray | torch.tensor | Tuple[numpy.ndarray | numpy.ndarray] | Tuple[torch.tensor | torch.tensor]

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        path1 = torch.rand((10, 100, 5))
        path2 = torch.rand((10, 100, 5))
        k = pysiglib.sig_kernel(path1, path2, dyadic_order=2)
        derivs = torch.ones_like(k)
        dpath1, _ = pysiglib.sig_kernel_backprop(derivs, path1, path2, dyadic_order=2)
        print(dpath1)

    .. code-block:: python

        # Backprop with a static kernel and time augmentation
        import torch
        import pysiglib

        path1 = torch.rand((10, 100, 5))
        path2 = torch.rand((10, 100, 5))
        rbf = pysiglib.RBFKernel(sigma=1.0)
        k = pysiglib.sig_kernel(
            path1, path2, dyadic_order=2, static_kernel=rbf, time_aug=True,
        )
        derivs = torch.ones_like(k)
        dpath1, _ = pysiglib.sig_kernel_backprop(
            derivs, path1, path2,
            dyadic_order=2,
            static_kernel=rbf,
            time_aug=True,
        )
        print(dpath1)

    """
    check_n_jobs(n_jobs)
    check_type(left_deriv, "left_deriv", bool)
    check_type(right_deriv, "right_deriv", bool)
    if not (left_deriv or right_deriv):
        return None, None

    dyadic_order_1, dyadic_order_2 = parse_dyadic_order(dyadic_order)
    log_degrees, log_step_sizes = parse_log_pde_parameters(method, log_degree, log_steps)
    if method == "log_pde" and lead_lag:
        log_step_sizes = tuple(2 * step for step in log_step_sizes)

    if path1.ndim > 3 or path2.ndim > 3:
        if tuple(path1.shape[:-2]) != tuple(path2.shape[:-2]):
            raise ValueError(
                "path1 and path2 must have matching leading batch dimensions; got "
                f"{tuple(path1.shape[:-2])} and {tuple(path2.shape[:-2])}."
            )
        leading_shape = tuple(path1.shape[:-2])
        lead_size = prod(leading_shape)
        flat_derivs = derivs.reshape(lead_size, *derivs.shape[-2:]) if return_grid else derivs.reshape(lead_size)
        flat_k_grid = None if k_grid is None else k_grid.reshape(lead_size, *k_grid.shape[-2:])

        ld, rd = sig_kernel_backprop(
            flat_derivs, _ensure_3d(path1), _ensure_3d(path2), dyadic_order,
            method=method, log_degree=log_degree, log_steps=log_steps,
            static_kernel=static_kernel,
            time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
            left_deriv=left_deriv, right_deriv=right_deriv,
            k_grid=flat_k_grid, n_jobs=n_jobs, return_grid=return_grid,
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

    if method == "log_pde":
        if static_kernel is not None and not isinstance(static_kernel, LinearKernel):
            raise ValueError("method='log_pde' supports only the linear static kernel")
        if data.device != "cpu":
            raise ValueError("method='log_pde' is currently CPU-only")
        steps = []
        for length, log_step in zip(data.length, log_step_sizes):
            intervals = length - 1
            if intervals <= 0:
                raise ValueError("method='log_pde' requires paths with at least two points")
            if intervals % log_step:
                raise ValueError(
                    f"log_steps={log_step} must divide the number of path intervals {intervals}"
                )
            steps.append(intervals // log_step)
        grid_shape = data.batch_shape + (
            (steps[0] << dyadic_order_1) + 1,
            (steps[1] << dyadic_order_2) + 1,
        )
        expected_shape = grid_shape if return_grid else data.batch_shape
        if tuple(derivs.shape) != expected_shape:
            raise ValueError("derivs has the wrong shape for the log-PDE output")

        k_grid_data = None
        if k_grid is not None:
            k_grid_data = PathInputHandler(k_grid, False, False, 0., "k_grid")
            if tuple(k_grid_data.path.shape) != grid_shape:
                raise ValueError(
                    f"k_grid has shape {tuple(k_grid_data.path.shape)}, expected {grid_shape}"
                )
            if not (
                k_grid_data.type_ == data.type_ and
                k_grid_data.device == data.device and
                k_grid_data.dtype == data.dtype
            ):
                raise ValueError(
                    "k_grid, path1 and path2 must have the same array type, device and dtype"
                )

        ld_result = PathOutputHandler(
            data.data[0].data_length, data.data[0].data_dimension, data.data[0]
        )
        rd_result = PathOutputHandler(
            data.data[1].data_length, data.data[1].data_dimension, data.data[1]
        )
        err_code = CPSIG_SIG_KERNEL_LOG_PDE_BACKPROP[data.dtype](
            data.data[0].data_ptr, data.data[1].data_ptr,
            ld_result.data_ptr, rd_result.data_ptr,
            derivs_data.data_ptr,
            None if k_grid_data is None else k_grid_data.data_ptr,
            data.batch_size, data.dimension,
            data.length[0], data.length[1], log_step_sizes[0], log_step_sizes[1],
            log_degrees[0], log_degrees[1], dyadic_order_1, dyadic_order_2,
            return_grid, n_jobs,
        )
        if err_code:
            raise Exception("Error in log-PDE signature kernel backprop: " + err_msg(err_code))
        ld = ld_result.data
        rd = rd_result.data
        if lead_lag or time_aug:
            ld = transform_path_backprop(
                ld, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs
            )
            rd = transform_path_backprop(
                rd, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs
            )
        return (ld if left_deriv else None), (rd if right_deriv else None)

    torch_path1 = torch.as_tensor(data.path[0])  # Avoids data copy
    torch_path2 = torch.as_tensor(data.path[1])

    if k_grid is None:
        k_grid = sig_kernel(torch.as_tensor(path1), torch.as_tensor(path2), dyadic_order, static_kernel=static_kernel, time_aug=False, lead_lag=False, end_time=end_time, n_jobs=n_jobs, return_grid=True)

    torch_path1 = _ensure_3d(torch_path1)
    torch_path2 = _ensure_3d(torch_path2)

    ctx = Context()

    if static_kernel is None:
        static_kernel = LinearKernel()
    elif not isinstance(static_kernel, StaticKernel):
        raise ValueError("kernel must be a child class of pysiglib.StaticKernel")

    gram = static_kernel(ctx, torch_path1, torch_path2).squeeze()

    k_grid_data = PathInputHandler(k_grid, False, False, 0., "k_grid")

    gram_derivs = gram_deriv(derivs_data, data, gram, k_grid_data, dyadic_order_1, dyadic_order_2, return_grid, n_jobs)

    ld = static_kernel.grad_x(ctx, gram_derivs) if left_deriv else None
    rd = static_kernel.grad_y(ctx, gram_derivs) if right_deriv else None

    if lead_lag or time_aug:
        ld = transform_path_backprop(ld, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs) if left_deriv else None
        rd = transform_path_backprop(rd, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs) if right_deriv else None

    if data.type_ == "numpy":
        ld = ld.numpy() if left_deriv else None
        rd = rd.numpy() if right_deriv else None

    return ld, rd


def sig_kernel_gram_backprop(
        derivs : Union[np.ndarray, torch.tensor],
        path1 : Union[np.ndarray, torch.tensor],
        path2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        *,
        method : str = "pde",
        log_degree : Union[int, tuple, None] = None,
        log_steps : Union[int, tuple, None] = None,
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        left_deriv : bool = True,
        right_deriv : bool = False,
        k_grid : Union[np.ndarray, torch.tensor] = None,
        n_jobs : int = 1,
        return_grid : bool = False,
        max_batch : int = -1
) -> Union[np.ndarray, torch.tensor, Tuple[np.ndarray, np.ndarray], Tuple[torch.tensor, torch.tensor]]:
    """
    This function is required to backpropagate through ``pysiglib.sig_kernel_gram``.
    Given the derivatives of a scalar function :math:`F` with respect to a
    gram matrix of signature kernels, :math:`\\partial F / G`,
    returns the derivatives of :math:`F` with respect to one or both of the
    underlying paths, :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^{L_1}` and
    :math:`\\{\\partial F / y_{t_i}\\}_{i=0}^{L_2}`.

    :param derivs: Derivatives with respect to a gram matrix of signature kernels,
        :math:`\\partial F / G`. Should have the same shape as the output of
        ``pysiglib.sig_kernel_gram`` for the same inputs:
        ``(*batch_shape_1, *batch_shape_2)`` if ``return_grid=False``, or
        ``(*batch_shape_1, *batch_shape_2, dyadic_length_1, dyadic_length_2)`` if
        ``return_grid=True``.
    :type derivs: numpy.ndarray | torch.tensor
    :param path1: A path or batch of paths, of shape ``(*batch_shape_1, length_1, dimension)``.
    :type path1: numpy.ndarray | torch.tensor
    :param path2: A path or batch of paths, of shape ``(*batch_shape_2, length_2, dimension)``.
        Independent of ``path1``'s batch shape.
    :type path2: numpy.ndarray | torch.tensor
    :param dyadic_order: The dyadic order(s) used to compute the signature kernels.
    :type dyadic_order: int | tuple
    :param method: Forward PDE method. Use ``"pde"`` for the standard Goursat
        solver or ``"log_pde"`` for the higher-order log-PDE method. The log-PDE
        method is currently CPU-only and supports only the linear static kernel.
    :type method: str
    :param log_degree: Tensor-log truncation degree. Required for
        ``method="log_pde"``. An integer applies to both paths; a pair applies
        separately to the first and second paths.
    :type log_degree: int | tuple | None
    :param log_steps: Number of original path intervals per tensor-log block for
        ``method="log_pde"``. Required for that method. Each value must divide
        its path's interval count. An integer applies to both paths; a pair applies
        separately. With ``lead_lag=True``, values still count original intervals.
    :type log_steps: int | tuple | None
    :param static_kernel: Static kernel. If ``None`` (default), the linear kernel will be used.
        For details, see the documentation on :doc:`static kernels </pages/signature_kernels/static_kernels>`.
    :type static_kernel: None | pysiglib.StaticKernel
    :param time_aug: If ``True``, assumes the paths were time augmented.
    :type time_aug: bool
    :param lead_lag: If ``True``, assumes the lead-lag transform was applied.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param left_deriv: If ``True``, returns :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^{L_1}`.
        At least one of ``left_deriv`` and ``right_deriv`` must be ``True``. If both are
        ``True``, returns both derivatives as a tuple.
    :type left_deriv: bool
    :param right_deriv: If ``True``, returns :math:`\\{\\partial F / y_{t_i}\\}_{i=0}^{L_2}`.
        At least one of ``left_deriv`` and ``right_deriv`` must be ``True``. If both are
        ``True``, returns both derivatives as a tuple.
    :type right_deriv: bool
    :param k_grid: Full forward Gram grids. If ``None``, the grids are recomputed.
        For ``method="log_pde"``, supplied grids checkpoint the scalar kernel states;
        the auxiliary tensor states required by backpropagation are still recomputed.
    :type k_grid: numpy.ndarray | torch.tensor
    :param n_jobs: (Only applicable to CPU computation) Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :param return_grid: If ``True``, backpropagates derivatives with respect to the entire PDE grid.
    :type return_grid: bool
    :param max_batch: Maximum batch size to run in parallel. If the computation is failing
        due to insufficient memory, this parameter should be decreased.
        If set to -1, the entire batch is computed in parallel.
    :type max_batch: int
    :return: Tuple of derivatives of :math:`F` with respect to one or both of the
        underlying paths. If ``left_deriv`` is ``True``, the first element of
        this tuple is  :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^{L_1}` with shape
        matching ``path1``, otherwise it is ``None``. Similarly for ``right_deriv``
        and :math:`\\{\\partial F / y_{t_i}\\}_{i=0}^{L_2}` with shape matching ``path2``.
    :rtype: numpy.ndarray | torch.tensor | Tuple[numpy.ndarray | numpy.ndarray] | Tuple[torch.tensor | torch.tensor]

    .. note::

        When called via ``pysiglib.torch_api``, the default behaviour is to pass ``k_grid = None`` and reconstruct the
        PDE grids. This is done to avoid memory allocation issues for large batch sizes.

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        path1 = torch.rand((10, 100, 5))
        path2 = torch.rand((8, 100, 5))
        gram = pysiglib.sig_kernel_gram(path1, path2, dyadic_order=2)
        derivs = torch.ones_like(gram)
        dpath1, _ = pysiglib.sig_kernel_gram_backprop(derivs, path1, path2, dyadic_order=2)
        print(dpath1)

    .. code-block:: python

        # Gram backprop with a static kernel
        import torch
        import pysiglib

        path1 = torch.rand((10, 100, 5))
        path2 = torch.rand((8, 100, 5))
        rbf = pysiglib.RBFKernel(sigma=0.5)
        gram = pysiglib.sig_kernel_gram(
            path1, path2, dyadic_order=2, static_kernel=rbf, time_aug=True,
        )
        derivs = torch.ones_like(gram)
        dpath1, dpath2 = pysiglib.sig_kernel_gram_backprop(
            derivs, path1, path2,
            dyadic_order=2,
            static_kernel=rbf,
            time_aug=True,
            left_deriv=True,
            right_deriv=True,
            max_batch=4,
        )
        print(dpath1)
        print(dpath2)

    """
    # We use sig_kernel_backprop for simplicity, rather than directly calling
    # the cpp function.
    # There is clearly more overhead here than is necessary, but it
    # shouldn't be significant for large computations.

    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(left_deriv, "left_deriv", bool)
    check_type(right_deriv, "right_deriv", bool)
    if not (left_deriv or right_deriv):
        return None, None

    check_type(max_batch, "max_batch", int)
    if max_batch == 0 or max_batch < -1:
        raise ValueError("max_batch must be a positive integer or -1")

    log_degrees, log_step_sizes = parse_log_pde_parameters(method, log_degree, log_steps)
    if method == "log_pde" and static_kernel is not None and not isinstance(static_kernel, LinearKernel):
        raise ValueError("method='log_pde' supports only the linear static kernel")

    do1, do2 = parse_dyadic_order(dyadic_order)
    symmetric = path1 is path2
    if method == "log_pde" and (
        do1 != do2 or
        log_degrees[0] != log_degrees[1] or log_step_sizes[0] != log_step_sizes[1]
    ):
        symmetric = False

    batch_shape_1 = tuple(path1.shape[:-2])
    batch_shape_2 = batch_shape_1 if symmetric else tuple(path2.shape[:-2])
    n1, n2 = len(batch_shape_1), len(batch_shape_2)

    path1 = _ensure_3d(path1)
    path2 = path1 if symmetric else _ensure_3d(path2)

    if n1 != 1 or n2 != 1:
        flat_B1, flat_B2 = path1.shape[0], path2.shape[0]
        derivs = derivs.reshape(flat_B1, flat_B2, *derivs.shape[n1 + n2:]) if return_grid else derivs.reshape(flat_B1, flat_B2)
        if k_grid is not None:
            k_grid = k_grid.reshape(flat_B1, flat_B2, *k_grid.shape[n1 + n2:])

    data = MultiplePathInputHandler([path1, path2], time_aug, lead_lag, end_time, ["path1", "path2"], False)

    derivs = torch.as_tensor(derivs)

    # Use torch for simplicity
    path1 = torch.as_tensor(data.path[0])
    path2 = torch.as_tensor(data.path[1])
    if k_grid is not None:
        k_grid = torch.as_tensor(k_grid)

    batch1 = path1.shape[0]
    batch2 = path2.shape[0]

    if max_batch == -1:
        max_batch = max(batch1, batch2)

    ld = torch.zeros(path1.shape, dtype=torch.float64, device=path1.device) if left_deriv else None
    rd = torch.zeros(path2.shape, dtype=torch.float64, device=path1.device) if right_deriv else None

    ####################################
    # Now run computation in batches
    ####################################

    if symmetric:
        idx_i, idx_j = torch.triu_indices(batch1, batch1, device=path1.device)
    else:
        idx_i = torch.arange(batch1, device=path1.device).repeat_interleave(batch2)
        idx_j = torch.arange(batch2, device=path2.device).repeat(batch1)

    src2 = path1 if symmetric else path2
    n_pairs = idx_i.shape[0]
    chunk_size = max_batch * max_batch

    # Check if k_grid can be transposed for symmetric off-diagonal pairs
    can_transpose_k = (do1 == do2)

    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        ci = idx_i[start:end]
        cj = idx_j[start:end]

        path1_ = path1[ci]
        path2_ = src2[cj]

        if k_grid is None and method == "pde":
            k = sig_kernel(path1_, path2_, dyadic_order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, return_grid=True)
        elif k_grid is None:
            k = None
        else:
            k = k_grid[ci, cj]

        derivs_ = derivs[ci, cj]

        ld_, rd_ = sig_kernel_backprop(
            derivs_, path1_, path2_, dyadic_order,
            method=method, log_degree=log_degree, log_steps=log_steps,
            static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
            end_time=end_time, left_deriv=left_deriv, right_deriv=right_deriv,
            k_grid=k, n_jobs=n_jobs, return_grid=return_grid,
        )

        if left_deriv:
            ld.index_add_(0, ci, ld_.to(ld.dtype))
        if right_deriv:
            rd.index_add_(0, cj, rd_.to(rd.dtype))

        # Symmetric: handle transposed off-diagonal entries G[j,i]
        if symmetric:
            off = ci != cj
            if off.any():
                ci_off = ci[off]
                cj_off = cj[off]

                path1_t = src2[cj_off]
                path2_t = path1[ci_off]

                if k_grid is not None:
                    k_t = k_grid[cj_off, ci_off]
                elif can_transpose_k and k is not None:
                    k_t = k[off].transpose(-2, -1)
                elif method == "pde":
                    k_t = sig_kernel(path1_t, path2_t, dyadic_order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, return_grid=True)
                else:
                    k_t = None

                derivs_t = derivs[cj_off, ci_off]

                ld_t, rd_t = sig_kernel_backprop(
                    derivs_t, path1_t, path2_t, dyadic_order,
                    method=method, log_degree=log_degree, log_steps=log_steps,
                    static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
                    end_time=end_time, left_deriv=left_deriv, right_deriv=right_deriv,
                    k_grid=k_t, n_jobs=n_jobs, return_grid=return_grid,
                )

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

