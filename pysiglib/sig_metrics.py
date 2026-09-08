from __future__ import annotations

from typing import Optional, Union
import numpy as np
from ._core import sig_metrics as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc
from .static_kernels import StaticKernel


def sig_score(sample: np.ndarray, y: np.ndarray, *, method: str = 'finite_difference',
        dyadic_order: Optional[Union[int, tuple]] = None, order: Optional[int] = None,
        lam: float = 1.0, static_kernel: Optional[StaticKernel] = None, time_aug: bool = False,
        lead_lag: bool = False, end_time: float = 1.0, n_jobs: int = 1, max_batch: int = -1) -> np.ndarray:
    _require_array(sample, np.ndarray, "sample")
    _require_array(y, np.ndarray, "y")
    return np.asarray(
        _impl.sig_score(sample, y, method=method, dyadic_order=dyadic_order, order=order,
        lam=lam, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch)
    )


sig_score.__doc__ = _backend_doc(_impl.sig_score.__doc__, "numpy")


def expected_sig_score(sample1: np.ndarray, sample2: np.ndarray, *,
        method: str = 'finite_difference', dyadic_order: Optional[Union[int, tuple]] = None,
        order: Optional[int] = None, lam: float = 1.0,
        static_kernel: Optional[StaticKernel] = None, time_aug: bool = False,
        lead_lag: bool = False, end_time: float = 1.0, n_jobs: int = 1, max_batch: int = -1) -> np.ndarray:
    _require_array(sample1, np.ndarray, "sample1")
    _require_array(sample2, np.ndarray, "sample2")
    return np.asarray(
        _impl.expected_sig_score(sample1, sample2, method=method, dyadic_order=dyadic_order,
        order=order, lam=lam, static_kernel=static_kernel, time_aug=time_aug,
        lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs, max_batch=max_batch)
    )


expected_sig_score.__doc__ = _backend_doc(_impl.expected_sig_score.__doc__, "numpy")


def sig_mmd(sample1: np.ndarray, sample2: np.ndarray, *, method: str = 'finite_difference',
        dyadic_order: Optional[Union[int, tuple]] = None, order: Optional[int] = None,
        static_kernel: Optional[StaticKernel] = None, time_aug: bool = False,
        lead_lag: bool = False, end_time: float = 1.0, n_jobs: int = 1, max_batch: int = -1) -> np.ndarray:
    _require_array(sample1, np.ndarray, "sample1")
    _require_array(sample2, np.ndarray, "sample2")
    return np.asarray(
        _impl.sig_mmd(sample1, sample2, method=method, dyadic_order=dyadic_order, order=order,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch)
    )


sig_mmd.__doc__ = _backend_doc(_impl.sig_mmd.__doc__, "numpy")
