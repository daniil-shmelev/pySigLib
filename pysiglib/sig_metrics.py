from typing import Union, Optional
import numpy as np
import torch

from .param_checks import check_type_multiple
from .data_handlers import MultiplePathInputHandler

from .static_kernels import StaticKernel
from .sig_kernel import sig_kernel_gram

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
    """
    Computes the (generalised) signature kernel score

    .. math::

        \\phi_{\\text{sig}}(\\mu, y) := \\lambda \\mathbb{E}_{x,x' \\sim \\mu}[k(x,x')] - 2\\mathbb{E}_{x\\sim \\mu}[k(x,y)].

    Given a batch of sample paths :math:`\\{x_i\\}_{i=1}^m \\sim \\mu` and a path :math:`y`,
    the score is computed using the consistent and unbiased estimator

    .. math::

        \\widehat{\\phi}_{\\text{sig}}(\\mu, y) := \\frac{\\lambda }{m(m-1)} \\sum_{j \\neq i} k(x_i, x_j) - \\frac{2}{m} \\sum_i k(x_i, y).

    Optionally, a static kernel can be specified. For details, see the documentation on
    :doc:`static kernels </pages/signature_kernels/static_kernels>`.

    :param sample: The batch of sample paths drawn from :math:`\\mu`, given as a `numpy.ndarray` or
        `torch.tensor`. This must be of shape ``(batch_size_1, length_1, dimension)``.
    :type sample: numpy.ndarray | torch.tensor
    :param y: The path(s) y, given as a `numpy.ndarray` or `torch.tensor`. For a single path,
        this must be of shape ``(length_2, dimension)``. For a batch of paths, this must be of shape
        ``(batch_size_2, length_2, dimension)``.
    :type y: numpy.ndarray | torch.tensor
    :param dyadic_order: If set to a positive integer :math:`\\lambda`, will refine the
        paths by a factor of :math:`2^\\lambda`. If set to a tuple of positive integers
        :math:`(\\lambda_1, \\lambda_2)`, will refine the first path by :math:`2^{\\lambda_1}`
        and the second path by :math:`2^{\\lambda_2}`.
    :type dyadic_order: int | tuple
    :param lam: The parameter :math:`\\lambda` of the generalised signature kernel score (default = 1.0).
    :type lam: float
    :param static_kernel: Static kernel passed to the signature kernel computation. If ``None`` (default), the
        linear kernel will be used. For details, see the documentation on
        :doc:`static kernels </pages/signature_kernels/static_kernels>`.
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
    :param max_batch: Maximum batch size to run in parallel. If the computation is failing
        due to insufficient memory, this parameter should be decreased.
        If set to -1, the entire batch is computed in parallel.
    :type max_batch: int
    :return: Signature kernel score
    :rtype: numpy.ndarray | torch.tensor

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        sample = torch.rand((20, 100, 5))
        y = torch.rand((100, 5))
        score = pysiglib.sig_score(sample, y, dyadic_order=2)
        print(score)

    """

    check_type_multiple(sample, "sample", (np.ndarray, torch.Tensor))
    check_type_multiple(y, "y", (np.ndarray, torch.Tensor))

    # Use torch for simplicity
    sample = torch.as_tensor(sample)
    y = torch.as_tensor(y)
    if len(y.shape) == 2:
        y = y.unsqueeze(0).contiguous().clone()

    data = MultiplePathInputHandler([sample, y], time_aug, lead_lag, end_time, ["sample_paths", "y"], False)

    B = sample.shape[0]

    xx = sig_kernel_gram(sample, sample, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    xy = sig_kernel_gram(sample, y, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, False)

    xx_sum = (torch.sum(xx) - torch.sum(torch.diag(xx))) / (B * (B - 1.))
    xy_sum = torch.sum(xy, dim = 0) * (2. / B)

    res = lam * xx_sum - xy_sum

    if data.type_ == "numpy":
        return res.numpy()
    return res

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
    """
    Computes the expected (generalised) signature kernel score

    .. math::

        \\mathbb{E}_{y \\sim \\nu}[\\phi_{\\text{sig}}(\\mu, y)] := \\lambda \\mathbb{E}_{x,x' \\sim \\mu}[k(x,x')] - 2\\mathbb{E}_{x,y\\sim \\mu \\times \\nu}[k(x,y)].

    Given a batch of sample paths :math:`\\{x_i\\}_{i=1}^m \\sim \\mu` and :math:`\\{y_j\\}_{j=1}^n \\sim \\nu`,
    the score is computed using the unbiased estimator

    .. math::

        \\frac{\\lambda }{m(m-1)} \\sum_{j \\neq i} k(x_i, x_j) - \\frac{2}{mn} \\sum_{i,j} k(x_i, y_j).

    Optionally, a static kernel can be specified. For details, see the documentation on
    :doc:`static kernels </pages/signature_kernels/static_kernels>`.

    :param sample1: The batch of sample paths drawn from :math:`\\mu`, given as a `numpy.ndarray` or
        `torch.tensor`. This must be of shape ``(batch_size_1, length_1, dimension)``.
    :type sample1: numpy.ndarray | torch.tensor
    :param sample2: The batch of sample paths drawn from :math:`\\nu`, given as a `numpy.ndarray` or
        `torch.tensor`. This must be of shape ``(batch_size_2, length_2, dimension)``.
    :type sample2: numpy.ndarray | torch.tensor
    :param dyadic_order: If set to a positive integer :math:`\\lambda`, will refine the
        paths by a factor of :math:`2^\\lambda`. If set to a tuple of positive integers
        :math:`(\\lambda_1, \\lambda_2)`, will refine the first path by :math:`2^{\\lambda_1}`
        and the second path by :math:`2^{\\lambda_2}`.
    :type dyadic_order: int | tuple
    :param lam: The parameter :math:`\\lambda` of the generalised signature kernel score (default = 1.0).
    :type lam: float
    :param static_kernel: Static kernel passed to the signature kernel computation. If ``None`` (default), the
        linear kernel will be used. For details, see the documentation on
        :doc:`static kernels </pages/signature_kernels/static_kernels>`.
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
    :param max_batch: Maximum batch size to run in parallel. If the computation is failing
        due to insufficient memory, this parameter should be decreased.
        If set to -1, the entire batch is computed in parallel.
    :type max_batch: int
    :return: Expected signature kernel score
    :rtype: numpy.ndarray | torch.tensor

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        sample1 = torch.rand((20, 100, 5))
        sample2 = torch.rand((15, 100, 5))
        score = pysiglib.expected_sig_score(sample1, sample2, dyadic_order=2)
        print(score)

    """

    res = sig_score(sample1, sample2, dyadic_order, lam, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch)

    if isinstance(res, torch.Tensor):
        res = torch.mean(res, 0, True)
    else:
        res = np.mean(res, keepdims=True)

    return res

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
    """
    Computes the squared maximum mean discrepancy (MMD)

    .. math::

        d(\\mu, \\nu)^2 := \\sup_f(\\mathbb{E}_{x \\sim \\mu}[f(x)] - \\mathbb{E}_{y \\sim \\nu}[f(y)]).

    .. math::

        = \\mathbb{E}_{xx' \\sim \\mu}[k(x,x')] - 2\\mathbb{E}_{x,y \\sim \\mu \\times \\nu}[k(x,y)] + \\mathbb{E}_{y,y' \\sim \\nu}[k(y,y')].

    Given a batch of sample paths :math:`\\{x_i\\}_{i=1}^m \\sim \\mu` and :math:`\\{y_j\\}_{j=1}^n \\sim \\nu`,
    the MMD is computed using the unbiased estimator

    .. math::

        \\widehat{d}(\\mu, \\nu)^2 = \\frac{1}{m(m-1)}\\sum_{j \\neq i}k(x_i, x_j) - \\frac{2}{mn}\\sum_{i,j}k(x_i, x_j) + \\frac{1}{n(n-1)}\\sum_{j \\neq i} k(y_i, y_j).

    Optionally, a static kernel can be specified. For details, see the documentation on
    :doc:`static kernels </pages/signature_kernels/static_kernels>`.

    :param sample1: The batch of sample paths drawn from :math:`\\mu`, given as a `numpy.ndarray` or
        `torch.tensor`. This must be of shape ``(batch_size_1, length_1, dimension)``.
    :type sample1: numpy.ndarray | torch.tensor
    :param sample2: The batch of sample paths drawn from :math:`\\nu`, given as a `numpy.ndarray` or
        `torch.tensor`. This must be of shape ``(batch_size_2, length_2, dimension)``.
    :type sample2: numpy.ndarray | torch.tensor
    :param dyadic_order: If set to a positive integer :math:`\\lambda`, will refine the
        paths by a factor of :math:`2^\\lambda`. If set to a tuple of positive integers
        :math:`(\\lambda_1, \\lambda_2)`, will refine the first path by :math:`2^{\\lambda_1}`
        and the second path by :math:`2^{\\lambda_2}`.
    :type dyadic_order: int | tuple
    :param static_kernel: Static kernel passed to the signature kernel computation. If ``None`` (default), the
        linear kernel will be used. For details, see the documentation on
        :doc:`static kernels </pages/signature_kernels/static_kernels>`.
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
    :param max_batch: Maximum batch size to run in parallel. If the computation is failing
        due to insufficient memory, this parameter should be decreased.
        If set to -1, the entire batch is computed in parallel.
    :type max_batch: int
    :return: Signature MMD
    :rtype: numpy.ndarray | torch.tensor

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        sample1 = torch.rand((20, 100, 5))
        sample2 = torch.rand((15, 100, 5))
        mmd = pysiglib.sig_mmd(sample1, sample2, dyadic_order=2)
        print(mmd)

    """
    data = MultiplePathInputHandler([sample1, sample2], time_aug, lead_lag, end_time, ["sample1", "sample2"], False)

    # Use torch for simplicity
    sample1 = torch.as_tensor(data.path[0])
    sample2 = torch.as_tensor(data.path[1])

    m = sample1.shape[0]
    n = sample2.shape[0]

    xx = sig_kernel_gram(sample1, sample1, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    xy = sig_kernel_gram(sample1, sample2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    yy = sig_kernel_gram(sample2, sample2, dyadic_order, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch, False)

    xx_sum = (torch.sum(xx) - torch.sum(torch.diag(xx))) / (m * (m - 1))
    xy_sum = 2. * torch.mean(xy)
    yy_sum = (torch.sum(yy) - torch.sum(torch.diag(yy))) / (n * (n - 1))

    res = xx_sum - xy_sum + yy_sum

    if data.type_ == "numpy":
        return res.numpy()
    return res
