from typing import Union, Optional
import numpy as np
import torch

from .param_checks import (
    check_type_multiple, parse_dyadic_order, parse_log_pde_parameters,
)
from .static_kernels import StaticKernel
from .sig_kernel import sig_kernel_gram, _ensure_3d

def sig_score(
        sample : Union[np.ndarray, torch.tensor],
        y : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        *,
        lam : float = 1.,
        method : str = "pde",
        log_degree : Union[int, tuple, None] = None,
        log_steps : Union[int, tuple, None] = None,
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

    :param sample: The batch of sample paths drawn from :math:`\\mu`, of shape
        ``(*batch_shape, length_1, dimension)``.
    :type sample: numpy.ndarray | torch.tensor
    :param y: The path(s) :math:`y`, of shape ``(*batch_shape_y, length_2, dimension)``.
        ``batch_shape_y`` may be empty (single ``y``) or arbitrary; the score is computed
        independently for each ``y``. Independent of ``sample``'s batch shape.
    :type y: numpy.ndarray | torch.tensor
    :param dyadic_order: If set to a positive integer :math:`\\lambda`, will refine the
        paths by a factor of :math:`2^\\lambda`. If set to a tuple of positive integers
        :math:`(\\lambda_1, \\lambda_2)`, will refine the first path by :math:`2^{\\lambda_1}`
        and the second path by :math:`2^{\\lambda_2}`.
    :type dyadic_order: int | tuple
    :param lam: The parameter :math:`\\lambda` of the generalised signature kernel score (default = 1.0).
    :type lam: float
    :param method: PDE method. Use ``"pde"`` for the standard Goursat solver or
        ``"log_pde"`` for the higher-order log-PDE method. The log-PDE method is
        currently CPU-only and supports only the linear static kernel.
    :type method: str
    :param log_degree: Tensor-log truncation degree. Required for
        ``method="log_pde"``. An integer applies to both path collections; a pair
        applies separately to the first and second collections.
    :type log_degree: int | tuple | None
    :param log_steps: Number of original path intervals per tensor-log block for
        ``method="log_pde"``. Required for that method. Each value must divide
        its path's interval count. An integer applies to both path collections; a
        pair applies separately. With ``lead_lag=True``, values still count original
        intervals.
    :type log_steps: int | tuple | None
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
    :return: Signature kernel score, of shape ``batch_shape_y`` (or ``(1,)`` if ``y``
        is a single 2D path).
    :rtype: numpy.ndarray | torch.tensor

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        sample = torch.rand((20, 100, 5))
        y = torch.rand((100, 5))
        score = pysiglib.sig_score(sample, y, dyadic_order=2)
        print(score.shape)  # (1,)

    .. code-block:: python

        # Using a static kernel with regularisation and time augmentation
        import torch
        import pysiglib

        sample = torch.rand((20, 100, 5))
        y = torch.rand((100, 5))
        rbf = pysiglib.RBFKernel(sigma=1.0)
        score = pysiglib.sig_score(
            sample, y,
            dyadic_order=2,
            lam=0.1,
            static_kernel=rbf,
            time_aug=True,
            max_batch=8,
        )
        print(score)

    .. code-block:: python

        # Multi-dim sample batch and a batch of y paths
        # one score is returned per y
        import torch
        import pysiglib

        sample = torch.rand((4, 5, 100, 5))  # 4 * 5 = 20 sample paths total
        y = torch.rand((3, 100, 5))          # batch of 3 target paths
        score = pysiglib.sig_score(sample, y, dyadic_order=2)
        print(score.shape)  # (3,)

    """

    check_type_multiple(sample, "sample", (np.ndarray, torch.Tensor))
    check_type_multiple(y, "y", (np.ndarray, torch.Tensor))

    is_numpy = isinstance(sample, np.ndarray)
    sample = torch.as_tensor(sample)
    y = torch.as_tensor(y)

    batch_shape_y = tuple(y.shape[:-2])
    sample = _ensure_3d(sample)
    y = _ensure_3d(y)

    B = sample.shape[0]
    if B < 2:
        raise ValueError("sig_score requires at least 2 sample paths (got {}).".format(B))

    do1, do2 = parse_dyadic_order(dyadic_order)
    log_degrees, log_step_sizes = parse_log_pde_parameters(
        method, log_degree, log_steps
    )
    left_degree = log_degrees[0] if log_degrees is not None else None
    left_steps = log_step_sizes[0] if log_step_sizes is not None else None

    xx = sig_kernel_gram(
        sample, sample, (do1, do1), method=method,
        log_degree=left_degree, log_steps=left_steps,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False,
    )
    xy = sig_kernel_gram(
        sample, y, (do1, do2), method=method,
        log_degree=log_degrees, log_steps=log_step_sizes,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False,
    )

    xx_sum = (torch.sum(xx) - torch.trace(xx)) / (B * (B - 1.))
    xy_sum = torch.sum(xy, dim=0) * (2. / B)

    res = lam * xx_sum - xy_sum
    if batch_shape_y:
        res = res.reshape(*batch_shape_y)

    if is_numpy:
        return res.numpy()
    return res

def expected_sig_score(
        sample1 : Union[np.ndarray, torch.tensor],
        sample2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        *,
        lam : float = 1.,
        method : str = "pde",
        log_degree : Union[int, tuple, None] = None,
        log_steps : Union[int, tuple, None] = None,
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

    :param sample1: The batch of sample paths drawn from :math:`\\mu`, of shape
        ``(*batch_shape, length_1, dimension)``.
    :type sample1: numpy.ndarray | torch.tensor
    :param sample2: The batch of sample paths drawn from :math:`\\nu`, of shape
        ``(*batch_shape, length_2, dimension)``. Independent of ``sample1``'s batch
        shape.
    :type sample2: numpy.ndarray | torch.tensor
    :param dyadic_order: If set to a positive integer :math:`\\lambda`, will refine the
        paths by a factor of :math:`2^\\lambda`. If set to a tuple of positive integers
        :math:`(\\lambda_1, \\lambda_2)`, will refine the first path by :math:`2^{\\lambda_1}`
        and the second path by :math:`2^{\\lambda_2}`.
    :type dyadic_order: int | tuple
    :param lam: The parameter :math:`\\lambda` of the generalised signature kernel score (default = 1.0).
    :type lam: float
    :param method: PDE method. Use ``"pde"`` for the standard Goursat solver or
        ``"log_pde"`` for the higher-order log-PDE method. The log-PDE method is
        currently CPU-only and supports only the linear static kernel.
    :type method: str
    :param log_degree: Tensor-log truncation degree. Required for
        ``method="log_pde"``. An integer applies to both path collections; a pair
        applies separately to the first and second collections.
    :type log_degree: int | tuple | None
    :param log_steps: Number of original path intervals per tensor-log block for
        ``method="log_pde"``. Required for that method. Each value must divide
        its path's interval count. An integer applies to both path collections; a
        pair applies separately. With ``lead_lag=True``, values still count original
        intervals.
    :type log_steps: int | tuple | None
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
    :return: Expected signature kernel score, of shape ``(1,)``.
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

    .. code-block:: python

        # Expected score with lead-lag and a static kernel
        import torch
        import pysiglib

        sample1 = torch.rand((20, 100, 5))
        sample2 = torch.rand((15, 100, 5))
        rbf = pysiglib.RBFKernel(sigma=1.0)
        score = pysiglib.expected_sig_score(
            sample1, sample2,
            dyadic_order=2,
            static_kernel=rbf,
            lead_lag=True,
            max_batch=8,
        )
        print(score)

    """

    res = sig_score(
        sample1, sample2, dyadic_order, lam=lam, method=method,
        log_degree=log_degree, log_steps=log_steps,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch,
    )
    return res.mean().reshape(1)

def sig_mmd(
        sample1 : Union[np.ndarray, torch.tensor],
        sample2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        *,
        method : str = "pde",
        log_degree : Union[int, tuple, None] = None,
        log_steps : Union[int, tuple, None] = None,
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

    :param sample1: The batch of sample paths drawn from :math:`\\mu`, of shape
        ``(*batch_shape, length_1, dimension)``. All dimensions before ``length_1`` are
        batch dimensions and are flattened to a single batch :math:`m` of paths.
    :type sample1: numpy.ndarray | torch.tensor
    :param sample2: The batch of sample paths drawn from :math:`\\nu`, of shape
        ``(*batch_shape, length_2, dimension)``. All dimensions before ``length_2`` are
        flattened to a single batch :math:`n` of paths. Independent of ``sample1``'s batch
        shape.
    :type sample2: numpy.ndarray | torch.tensor
    :param dyadic_order: If set to a positive integer :math:`\\lambda`, will refine the
        paths by a factor of :math:`2^\\lambda`. If set to a tuple of positive integers
        :math:`(\\lambda_1, \\lambda_2)`, will refine the first path by :math:`2^{\\lambda_1}`
        and the second path by :math:`2^{\\lambda_2}`.
    :type dyadic_order: int | tuple
    :param method: PDE method. Use ``"pde"`` for the standard Goursat solver or
        ``"log_pde"`` for the higher-order log-PDE method. The log-PDE method is
        currently CPU-only and supports only the linear static kernel.
    :type method: str
    :param log_degree: Tensor-log truncation degree. Required for
        ``method="log_pde"``. An integer applies to both path collections; a pair
        applies separately to the first and second collections.
    :type log_degree: int | tuple | None
    :param log_steps: Number of original path intervals per tensor-log block for
        ``method="log_pde"``. Required for that method. Each value must divide
        its path's interval count. An integer applies to both path collections; a
        pair applies separately. With ``lead_lag=True``, values still count original
        intervals.
    :type log_steps: int | tuple | None
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

    .. code-block:: python

        # MMD with a static kernel and time augmentation
        import torch
        import pysiglib

        sample1 = torch.rand((20, 100, 5))
        sample2 = torch.rand((15, 100, 5))
        rbf = pysiglib.RBFKernel(sigma=0.5)
        mmd = pysiglib.sig_mmd(
            sample1, sample2,
            dyadic_order=2,
            static_kernel=rbf,
            time_aug=True,
            max_batch=8,
        )
        print(mmd)

    """
    is_numpy = isinstance(sample1, np.ndarray)
    sample1 = torch.as_tensor(sample1)
    sample2 = torch.as_tensor(sample2)

    sample1 = _ensure_3d(sample1)
    sample2 = _ensure_3d(sample2)

    m = sample1.shape[0]
    n = sample2.shape[0]
    if m < 2:
        raise ValueError("sig_mmd requires at least 2 paths in sample1 (got {}).".format(m))
    if n < 2:
        raise ValueError("sig_mmd requires at least 2 paths in sample2 (got {}).".format(n))

    do1, do2 = parse_dyadic_order(dyadic_order)
    log_degrees, log_step_sizes = parse_log_pde_parameters(
        method, log_degree, log_steps
    )
    left_degree = log_degrees[0] if log_degrees is not None else None
    right_degree = log_degrees[1] if log_degrees is not None else None
    left_steps = log_step_sizes[0] if log_step_sizes is not None else None
    right_steps = log_step_sizes[1] if log_step_sizes is not None else None

    xx = sig_kernel_gram(
        sample1, sample1, (do1, do1), method=method,
        log_degree=left_degree, log_steps=left_steps,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False,
    )
    xy = sig_kernel_gram(
        sample1, sample2, (do1, do2), method=method,
        log_degree=log_degrees, log_steps=log_step_sizes,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False,
    )
    yy = sig_kernel_gram(
        sample2, sample2, (do2, do2), method=method,
        log_degree=right_degree, log_steps=right_steps,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False,
    )

    xx_sum = (torch.sum(xx) - torch.trace(xx)) / (m * (m - 1))
    xy_sum = 2. * torch.mean(xy)
    yy_sum = (torch.sum(yy) - torch.trace(yy)) / (n * (n - 1))

    res = xx_sum - xy_sum + yy_sum

    if is_numpy:
        return res.numpy()
    return res
