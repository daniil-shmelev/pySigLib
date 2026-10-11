# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Depth-truncated branched kernel scores and unbiased squared MMD."""
from functools import partial
import numpy as np
import torch
from .branched_sig_kernel import branched_sig_kernel_gram
from .sig_kernel import _ensure_3d


def _branched_sig_score(kernel_gram, asarray, sample1, sample2, depth, lam=1.0, **kwargs):
    sample1, sample2 = asarray(sample1), asarray(sample2)
    target_shape = tuple(sample2.shape[:-2])
    sample1, sample2 = _ensure_3d(sample1), _ensure_3d(sample2)
    m, n = sample1.shape[0], sample2.shape[0]
    if m < 2:
        raise ValueError("branched_sig_score requires at least two paths in sample1")
    if n == 0:
        raise ValueError("branched_sig_score requires a nonempty second sample")
    kernel = partial(kernel_gram, depth=depth, **kwargs)
    xx, xy = kernel(sample1, sample1), kernel(sample1, sample2)
    xx_mean = (xx.sum() - xx.diagonal().sum()) / (m * (m - 1))
    result = lam * xx_mean - 2 * xy.mean(axis=0)
    return result.reshape(target_shape) if target_shape else result.reshape(1)


def _branched_sig_mmd(kernel_gram, asarray, sample1, sample2, depth, **kwargs):
    sample1, sample2 = _ensure_3d(asarray(sample1)), _ensure_3d(asarray(sample2))
    m, n = sample1.shape[0], sample2.shape[0]
    if m < 2 or n < 2:
        raise ValueError("branched_sig_mmd requires at least two paths in each sample")
    kernel = partial(kernel_gram, depth=depth, **kwargs)
    xx, xy, yy = kernel(sample1, sample1), kernel(sample1, sample2), kernel(sample2, sample2)
    xx_mean = (xx.sum() - xx.diagonal().sum()) / (m * (m - 1))
    yy_mean = (yy.sum() - yy.diagonal().sum()) / (n * (n - 1))
    return xx_mean - 2 * xy.mean() + yy_mean


def branched_sig_score(
    sample1, sample2, depth, dyadic_order=None, *,
    method="finite_difference", order=None,
    lam=1.0,
    static_kernel=None, time_aug=False, lead_lag=False, end_time=1.0,
    n_jobs=1, max_batch=-1,
):
    """Compute the kernel score for each target path with the depth-truncated BCK kernel.

    ``depth`` truncates tree depth. ``method``, ``order``, ``dyadic_order``,
    static kernels and path transforms follow :func:`branched_sig_kernel`.
    Leading batch dimensions are flattened as distribution samples.
    The first sample must contain at least two paths.
    """
    result = _branched_sig_score(
        branched_sig_kernel_gram, torch.as_tensor, sample1, sample2, depth,
        dyadic_order=dyadic_order, method=method, order=order, lam=lam,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch)
    return result.numpy() if isinstance(sample1, np.ndarray) else result


def expected_branched_sig_score(
    sample1, sample2, depth, dyadic_order=None, *,
    method="finite_difference", order=None,
    lam=1.0,
    static_kernel=None, time_aug=False, lead_lag=False, end_time=1.0,
    n_jobs=1, max_batch=-1,
):
    """Compute the expected kernel score with the depth-truncated BCK kernel.

    ``depth`` truncates tree depth. ``method``, ``order``, ``dyadic_order``,
    static kernels and path transforms follow :func:`branched_sig_kernel`.
    Leading batch dimensions are flattened as distribution samples.
    The first sample must contain at least two paths.
    """
    return branched_sig_score(
        sample1, sample2, depth, dyadic_order, method=method, order=order, lam=lam,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch).mean().reshape(1)


def branched_sig_mmd(
    sample1, sample2, depth, dyadic_order=None, *,
    method="finite_difference", order=None,
    static_kernel=None, time_aug=False, lead_lag=False, end_time=1.0,
    n_jobs=1, max_batch=-1,
):
    """Compute unbiased squared MMD with the depth-truncated BCK kernel.

    ``depth`` truncates tree depth. ``method``, ``order``, ``dyadic_order``,
    static kernels and path transforms follow :func:`branched_sig_kernel`.
    Leading batch dimensions are flattened as distribution samples.
    Both samples must contain at least two paths; the unbiased estimate can be negative.
    """
    result = _branched_sig_mmd(
        branched_sig_kernel_gram, torch.as_tensor, sample1, sample2, depth,
        dyadic_order=dyadic_order, method=method, order=order,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch)
    return result.numpy() if isinstance(sample1, np.ndarray) else result
