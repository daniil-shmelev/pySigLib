from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
from ._core import sig_kernel_backprop as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc
from .static_kernels import StaticKernel


def sig_kernel_backprop(derivs: np.ndarray, path1: np.ndarray, path2: np.ndarray, *,
        method: str = 'finite_difference', dyadic_order: Optional[Union[int, tuple]] = None,
        order: Optional[int] = None, static_kernel: Optional[StaticKernel] = None,
        time_aug: bool = False, lead_lag: bool = False, end_time: float = 1.0,
        left_deriv: bool = True, right_deriv: bool = False, k_grid: Optional[np.ndarray] = None,
        n_jobs: int = 1, return_grid: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    _require_array(derivs, np.ndarray, "derivs")
    _require_array(path1, np.ndarray, "path1")
    _require_array(path2, np.ndarray, "path2")
    if k_grid is not None:
        _require_array(k_grid, np.ndarray, "k_grid")
    return _impl.sig_kernel_backprop(derivs, path1, path2, method=method,
        dyadic_order=dyadic_order, order=order, static_kernel=static_kernel,
        time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, left_deriv=left_deriv,
        right_deriv=right_deriv, k_grid=k_grid, n_jobs=n_jobs, return_grid=return_grid)


sig_kernel_backprop.__doc__ = _backend_doc(_impl.sig_kernel_backprop.__doc__, "numpy")


def sig_kernel_gram_backprop(derivs: np.ndarray, path1: np.ndarray, path2: np.ndarray, *,
        method: str = 'finite_difference', dyadic_order: Optional[Union[int, tuple]] = None,
        order: Optional[int] = None, static_kernel: Optional[StaticKernel] = None,
        time_aug: bool = False, lead_lag: bool = False, end_time: float = 1.0,
        left_deriv: bool = True, right_deriv: bool = False, k_grid: Optional[np.ndarray] = None,
        n_jobs: int = 1, return_grid: bool = False, max_batch: int = -1) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    _require_array(derivs, np.ndarray, "derivs")
    _require_array(path1, np.ndarray, "path1")
    _require_array(path2, np.ndarray, "path2")
    if k_grid is not None:
        _require_array(k_grid, np.ndarray, "k_grid")
    return _impl.sig_kernel_gram_backprop(derivs, path1, path2, method=method,
        dyadic_order=dyadic_order, order=order, static_kernel=static_kernel,
        time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, left_deriv=left_deriv,
        right_deriv=right_deriv, k_grid=k_grid, n_jobs=n_jobs, return_grid=return_grid,
        max_batch=max_batch)


sig_kernel_gram_backprop.__doc__ = _backend_doc(
    _impl.sig_kernel_gram_backprop.__doc__, "numpy"
)
