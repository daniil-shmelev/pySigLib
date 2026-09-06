from __future__ import annotations
from array_api_compat import array_namespace
from ._array import NativeArrayT as Array
from ._storage import check_native_array

from typing import Union, Optional

from .static_kernels import StaticKernel
from .sig_kernel import sig_kernel_gram, _ensure_3d


def sig_score(
    sample: Array,
    y: Array,
    *,
    method: str = "finite_difference",
    dyadic_order: Optional[Union[int, tuple]] = None,
    order: Optional[int] = None,
    lam: float = 1.0,
    static_kernel: Optional[StaticKernel] = None,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    n_jobs: int = 1,
    max_batch: int = -1,
) -> Array:
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
    :type sample: Array
    :param y: The path(s) :math:`y`, of shape ``(*batch_shape_y, length_2, dimension)``.
        ``batch_shape_y`` may be empty (single ``y``) or arbitrary; the score is computed
        independently for each ``y``. Independent of ``sample``'s batch shape.
    :type y: Array
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
        If set to -1, no explicit batch limit is applied.
    :type max_batch: int
    :return: Signature kernel score, of shape ``batch_shape_y`` (or ``(1,)`` if ``y``
        is a single 2D path).
    :rtype: Array

    Example:
    ---------

    .. code-block:: python

        import numpy as np
        import pysiglib

        sample = np.random.random((20, 100, 5))
        y = np.random.random((100, 5))
        score = pysiglib.sig_score(sample, y, dyadic_order=2)
        print(score.shape)  # (1,)

    .. code-block:: python

        # Using a static kernel with regularisation and time augmentation
        import numpy as np
        import pysiglib

        sample = np.random.random((20, 100, 5))
        y = np.random.random((100, 5))
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
        import numpy as np
        import pysiglib

        sample = np.random.random((4, 5, 100, 5))  # 4 * 5 = 20 sample paths total
        y = np.random.random((3, 100, 5))          # batch of 3 target paths
        score = pysiglib.sig_score(sample, y, dyadic_order=2)
        print(score.shape)  # (3,)

    """

    check_native_array(sample, 'sample')
    check_native_array(y, 'y')


    batch_shape_y = tuple(y.shape[:-2])
    sample = _ensure_3d(sample)
    y = _ensure_3d(y)

    B = sample.shape[0]
    if B < 2:
        raise ValueError("sig_score requires at least 2 sample paths (got {}).".format(B))

    xx = sig_kernel_gram(sample, sample, dyadic_order=dyadic_order, method=method, order=order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False)
    xy = sig_kernel_gram(sample, y, dyadic_order=dyadic_order, method=method, order=order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False)

    xp = array_namespace(xx)
    xx_sum = (xp.sum(xx) - xp.trace(xx)) / (B * (B - 1.0))
    xy_sum = xp.sum(xy, axis=0) * (2.0 / B)

    res = lam * xx_sum - xy_sum
    if batch_shape_y:
        res = res.reshape(*batch_shape_y)

    return res


def expected_sig_score(
    sample1: Array,
    sample2: Array,
    *,
    method: str = "finite_difference",
    dyadic_order: Optional[Union[int, tuple]] = None,
    order: Optional[int] = None,
    lam: float = 1.0,
    static_kernel: Optional[StaticKernel] = None,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    n_jobs: int = 1,
    max_batch: int = -1,
) -> Array:
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
    :type sample1: Array
    :param sample2: The batch of sample paths drawn from :math:`\\nu`, of shape
        ``(*batch_shape, length_2, dimension)``. Independent of ``sample1``'s batch
        shape.
    :type sample2: Array
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
        If set to -1, no explicit batch limit is applied.
    :type max_batch: int
    :return: Expected signature kernel score, of shape ``(1,)``.
    :rtype: Array

    Example:
    ---------

    .. code-block:: python

        import numpy as np
        import pysiglib

        sample1 = np.random.random((20, 100, 5))
        sample2 = np.random.random((15, 100, 5))
        score = pysiglib.expected_sig_score(sample1, sample2, dyadic_order=2)
        print(score)

    .. code-block:: python

        # Expected score with lead-lag and a static kernel
        import numpy as np
        import pysiglib

        sample1 = np.random.random((20, 100, 5))
        sample2 = np.random.random((15, 100, 5))
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

    res = sig_score(sample1, sample2, dyadic_order=dyadic_order, method=method, order=order, lam=lam, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch)
    return res.mean().reshape(1)


def sig_mmd(
    sample1: Array,
    sample2: Array,
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
) -> Array:
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
    :type sample1: Array
    :param sample2: The batch of sample paths drawn from :math:`\\nu`, of shape
        ``(*batch_shape, length_2, dimension)``. All dimensions before ``length_2`` are
        flattened to a single batch :math:`n` of paths. Independent of ``sample1``'s batch
        shape.
    :type sample2: Array
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
        If set to -1, no explicit batch limit is applied.
    :type max_batch: int
    :return: Signature MMD
    :rtype: Array

    Example:
    ---------

    .. code-block:: python

        import numpy as np
        import pysiglib

        sample1 = np.random.random((20, 100, 5))
        sample2 = np.random.random((15, 100, 5))
        mmd = pysiglib.sig_mmd(sample1, sample2, dyadic_order=2)
        print(mmd)

    .. code-block:: python

        # MMD with a static kernel and time augmentation
        import numpy as np
        import pysiglib

        sample1 = np.random.random((20, 100, 5))
        sample2 = np.random.random((15, 100, 5))
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

    sample1 = _ensure_3d(sample1)
    sample2 = _ensure_3d(sample2)

    m = sample1.shape[0]
    n = sample2.shape[0]
    if m < 2:
        raise ValueError("sig_mmd requires at least 2 paths in sample1 (got {}).".format(m))
    if n < 2:
        raise ValueError("sig_mmd requires at least 2 paths in sample2 (got {}).".format(n))

    xx = sig_kernel_gram(sample1, sample1, dyadic_order=dyadic_order, method=method, order=order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False)
    xy = sig_kernel_gram(sample1, sample2, dyadic_order=dyadic_order, method=method, order=order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False)
    yy = sig_kernel_gram(sample2, sample2, dyadic_order=dyadic_order, method=method, order=order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=False)

    xp = array_namespace(xx)
    xx_sum = (xp.sum(xx) - xp.trace(xx)) / (m * (m - 1))
    xy_sum = 2.0 * xp.mean(xy)
    yy_sum = (xp.sum(yy) - xp.trace(yy)) / (n * (n - 1))

    res = xx_sum - xy_sum + yy_sum

    return res
