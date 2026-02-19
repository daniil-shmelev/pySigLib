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

import numpy as np
import torch

from .transform_path import transform_path
from .transform_path_backprop import transform_path_backprop
from .sig_kernel import sig_kernel
from .load_siglib import BUILT_WITH_CUDA
from .param_checks import check_type
from .error_codes import err_msg
from .dtypes import CPSIG_BATCH_SIG_KERNEL_BACKPROP, DTYPES, CUSIG_BATCH_SIG_KERNEL_BACKPROP_CUDA
from .data_handlers import MultiplePathInputHandler, ScalarInputHandler, GridOutputHandler, PathInputHandler
from .static_kernels import StaticKernel, LinearKernel, Context

def sig_kernel_backprop_(data, derivs_data, result, gram, k_grid_data, dyadic_order_1, dyadic_order_2, n_jobs):

    err_code = CPSIG_BATCH_SIG_KERNEL_BACKPROP[data.dtype](
        cast(gram.data_ptr(), POINTER(DTYPES[str(gram.dtype)[6:]])),
        result.data_ptr,
        derivs_data.data_ptr,
        k_grid_data.data_ptr,
        data.batch_size,
        data.dimension,
        data.length[0],
        data.length[1],
        dyadic_order_1,
        dyadic_order_2,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_kernel_backprop: " + err_msg(err_code))#

def sig_kernel_backprop_cuda_(data, derivs_data, result, gram, k_grid_data, dyadic_order_1, dyadic_order_2):
    err_code = CUSIG_BATCH_SIG_KERNEL_BACKPROP_CUDA[data.dtype](
        cast(gram.data_ptr(), POINTER(DTYPES[str(gram.dtype)[6:]])),
        result.data_ptr,
        derivs_data.data_ptr,
        k_grid_data.data_ptr,
        data.batch_size,
        data.dimension,
        data.length[0],
        data.length[1],
        dyadic_order_1,
        dyadic_order_2
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_kernel_backprop: " + err_msg(err_code))

def gram_deriv(
        derivs_data,
        data,
        gram : Union[np.ndarray, torch.tensor],
        k_grid_data : Union[np.ndarray, torch.tensor],
        dyadic_order_1,
        dyadic_order_2,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:

    result = GridOutputHandler(data.length[0] - 1, data.length[1] - 1, derivs_data) #Derivatives with respect to gram matrix

    if data.device == "cpu":
        sig_kernel_backprop_(data, derivs_data, result, gram, k_grid_data, dyadic_order_1, dyadic_order_2, n_jobs)
    else:
        if not BUILT_WITH_CUDA:
            raise RuntimeError("pySigLib was built without CUDA - data must be moved to CPU.")
        sig_kernel_backprop_cuda_(data, derivs_data, result, gram, k_grid_data, dyadic_order_1, dyadic_order_2)

    return result.data

def sig_kernel_backprop(
        derivs : Union[np.ndarray, torch.tensor],
        path1 : Union[np.ndarray, torch.tensor],
        path2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        left_deriv : bool = True,
        right_deriv : bool = False,
        k_grid : Union[np.ndarray, torch.tensor] = None,
        n_jobs : int = 1
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
    :type derivs: numpy.ndarray | torch.tensor
    :param path1: The first underlying path or batch of paths, given as a `numpy.ndarray` or
        `torch.tensor`. For a single path, this must be of shape ``(length_1, dimension)``. For a
        batch of paths, this must be of shape ``(batch_size, length_1, dimension)``.
    :type path1: numpy.ndarray | torch.tensor
    :param path2: The second underlying path or batch of paths, given as a `numpy.ndarray`
        or `torch.tensor`. For a single path, this must be of shape ``(length_2, dimension)``.
        For a batch of paths, this must be of shape ``(batch_size, length_2, dimension)``.
    :type path2: numpy.ndarray | torch.tensor
    :param dyadic_order: The dyadic order(s) used to compute the signature kernels.
    :type dyadic_order: int | tuple
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
    :param k_grid: Signature kernel PDE grid. If ``None``, the grid will be recomputed.
    :type k_grid: numpy.ndarray | torch.tensor
    :param n_jobs: (Only applicable to CPU computation) Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
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

    """
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    check_type(left_deriv, "left_deriv", bool)
    check_type(right_deriv, "right_deriv", bool)
    if not (left_deriv or right_deriv):
        return None, None

    if isinstance(dyadic_order, tuple) and len(dyadic_order) == 2:
        dyadic_order_1 = dyadic_order[0]
        dyadic_order_2 = dyadic_order[1]
    elif isinstance(dyadic_order, int):
        dyadic_order_1 = dyadic_order
        dyadic_order_2 = dyadic_order
    else:
        raise TypeError("dyadic_order must be an integer or a tuple of length 2")

    if dyadic_order_1 < 0 or dyadic_order_2 < 0:
        raise ValueError("dyadic_order must be a non-negative integer or tuple of non-negative integers")

    if time_aug or lead_lag:
        path1 = transform_path(path1, time_aug, lead_lag, end_time, n_jobs)
        path2 = transform_path(path2, time_aug, lead_lag, end_time, n_jobs)

    data = MultiplePathInputHandler([path1, path2], False, False, end_time, ["path1", "path2"])

    derivs = torch.as_tensor(derivs)
    derivs_data = ScalarInputHandler(derivs, data.is_batch, "derivs")

    if not (derivs_data.type_ == data.type_ and derivs_data.device == data.device):
        raise ValueError("derivs, path1 and path2 must all be numpy arrays or all torch tensors on the same device")
    if data.batch_size != derivs_data.batch_size:
        raise ValueError("batch size for derivs does not match batch size of paths")

    torch_path1 = torch.as_tensor(data.path[0])  # Avoids data copy
    torch_path2 = torch.as_tensor(data.path[1])

    if k_grid is None:
        k_grid = sig_kernel(torch.as_tensor(path1), torch.as_tensor(path2), dyadic_order, static_kernel, False, False, end_time, n_jobs, True)

    if not data.is_batch:
        torch_path1 = torch_path1.unsqueeze(0)
        torch_path2 = torch_path2.unsqueeze(0)

    ctx = Context()

    if static_kernel is None:
        static_kernel = LinearKernel()
    elif not isinstance(static_kernel, StaticKernel):
        raise ValueError("kernel must be a child class of pysiglib.StaticKernel")

    gram = static_kernel(ctx, torch_path1, torch_path2).squeeze()

    k_grid_data = PathInputHandler(k_grid, False, False, 0., "k_grid")
    gram_derivs = gram_deriv(derivs_data, data, gram, k_grid_data, dyadic_order_1, dyadic_order_2, n_jobs)

    ld = static_kernel.grad_x(ctx, gram_derivs) if left_deriv else None
    rd = static_kernel.grad_y(ctx, gram_derivs) if right_deriv else None

    if lead_lag or time_aug:
        ld = transform_path_backprop(ld, time_aug, lead_lag, end_time, n_jobs)
        rd = transform_path_backprop(rd, time_aug, lead_lag, end_time, n_jobs)

    if data.type_ == "numpy":
        ld = ld.numpy()
        rd = rd.numpy()

    return ld, rd


def sig_kernel_gram_backprop(
        derivs : Union[np.ndarray, torch.tensor],
        path1 : Union[np.ndarray, torch.tensor],
        path2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        static_kernel : Optional[StaticKernel] = None,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        left_deriv : bool = True,
        right_deriv : bool = False,
        k_grid : Union[np.ndarray, torch.tensor] = None,
        n_jobs : int = 1,
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
        :math:`\\partial F / G`.
    :type derivs: numpy.ndarray | torch.tensor
    :param path1: The first underlying path or batch of paths, given as a `numpy.ndarray` or
        `torch.tensor`. For a single path, this must be of shape ``(length_1, dimension)``. For a
        batch of paths, this must be of shape ``(batch_size_1, length_1, dimension)``.
    :type path1: numpy.ndarray | torch.tensor
    :param path2: The second underlying path or batch of paths, given as a `numpy.ndarray`
        or `torch.tensor`. For a single path, this must be of shape ``(length_2, dimension)``.
        For a batch of paths, this must be of shape ``(batch_size_2, length_2, dimension)``.
    :type path2: numpy.ndarray | torch.tensor
    :param dyadic_order: The dyadic order(s) used to compute the signature kernels.
    :type dyadic_order: int | tuple
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
    :param k_grid: Signature kernel PDE grid. If ``None``, the grid will be recomputed.
    :type k_grid: numpy.ndarray | torch.tensor
    :param n_jobs: (Only applicable to CPU computation) Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :param max_batch: Maximum batch size to run in parallel. If the computation is failing
        due to insufficient memory, this parameter should be decreased.
        If set to -1, the entire batch is computed in parallel.
    :type max_batch: int
    :return: Tuple of derivatives of :math:`F` with respect to one or both of the
        underlying paths. If ``left_deriv`` is ``True``, the first element of
        this tuple is  :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^{L_1}`, otherwise
        it is ``None``. Similarly for ``right_deriv`` and
        :math:`\\{\\partial F / y_{t_i}\\}_{i=0}^{L_2}`.
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

    ld = torch.zeros(path1.shape, dtype = torch.float64, device = path1.device) if left_deriv else None
    rd = torch.zeros(path2.shape, dtype = torch.float64, device = path1.device) if right_deriv else None

    ####################################
    # Now run computation in batches
    ####################################

    for i in range(0, batch1, max_batch):
        batch1_ = min(max_batch, batch1 - i)
        for j in range(0, batch2, max_batch):
            batch2_ = min(max_batch, batch2 - j)

            path1_ = path1[i:i + batch1_, :, :].repeat_interleave(batch2_, 0).contiguous().clone()
            path2_ = path2[j:j + batch2_, :, :].repeat(batch1_, 1, 1).contiguous().clone()

            if k_grid is None:
                k = sig_kernel(path1_, path2_, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, True)
            else:
                k = k_grid[i:i + batch1_, j:j + batch2_, :, :].contiguous().clone()

            derivs_ = derivs[i:i + batch1_, j:j + batch2_].flatten().contiguous().clone()

            ld_, rd_ = sig_kernel_backprop(derivs_, path1_, path2_, dyadic_order, static_kernel, time_aug, lead_lag, end_time, left_deriv, right_deriv, k, n_jobs)

            if left_deriv:
                ld_ = ld_.reshape((batch1_, batch2_) + ld_.shape[1:])
                ld_ = ld_.sum(1)
                ld[i:i + batch1_, :, :] += ld_
            if right_deriv:
                rd_ = rd_.reshape((batch1_, batch2_) + rd_.shape[1:])
                rd_ = rd_.permute(1, 0, 2, 3).sum(1)
                rd[j:j + batch2_, :, :] += rd_

    if data.type_ == "numpy":
        return ld.numpy(), rd.numpy()
    return ld, rd

