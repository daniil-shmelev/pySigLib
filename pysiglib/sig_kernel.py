from __future__ import annotations

from typing import Optional, Union
import numpy as np
from ._core import sig_kernel as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc
from .static_kernels import StaticKernel


def sig_kernel(path1: np.ndarray, path2: np.ndarray, *, method: str = 'finite_difference',
        dyadic_order: Optional[Union[int, tuple]] = None, order: Optional[int] = None,
        static_kernel: Optional[StaticKernel] = None, time_aug: bool = False,
        lead_lag: bool = False, end_time: float = 1.0, n_jobs: int = 1, return_grid: bool = False,
        normalize: bool = False) -> np.ndarray:
    _require_array(path1, np.ndarray, "path1")
    _require_array(path2, np.ndarray, "path2")
    return _impl.sig_kernel(path1, path2, method=method, dyadic_order=dyadic_order, order=order,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, return_grid=return_grid, normalize=normalize)


sig_kernel.__doc__ = _backend_doc(_impl.sig_kernel.__doc__, "numpy")


def sig_kernel_gram(path1: np.ndarray, path2: np.ndarray, *, method: str = 'finite_difference',
        dyadic_order: Optional[Union[int, tuple]] = None, order: Optional[int] = None,
        static_kernel: Optional[StaticKernel] = None, time_aug: bool = False,
        lead_lag: bool = False, end_time: float = 1.0, n_jobs: int = 1, max_batch: int = -1,
        return_grid: bool = False, normalize: bool = False) -> np.ndarray:
    _require_array(path1, np.ndarray, "path1")
    _require_array(path2, np.ndarray, "path2")
    return _impl.sig_kernel_gram(path1, path2, method=method, dyadic_order=dyadic_order,
        order=order, static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs, max_batch=max_batch, return_grid=return_grid,
        normalize=normalize)


sig_kernel_gram.__doc__ = _backend_doc(_impl.sig_kernel_gram.__doc__, "numpy")
