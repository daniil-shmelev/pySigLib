from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
from ._core import branched_sig_kernel_backprop as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc
from .static_kernels import StaticKernel


def branched_sig_kernel_backprop(derivs: np.ndarray, path1: np.ndarray, path2: np.ndarray,
        depth: int, dyadic_order: Union[int, tuple], *,
        static_kernel: Optional[StaticKernel] = None, time_aug: bool = False,
        lead_lag: bool = False, end_time: float = 1.0, left_deriv: bool = True,
        right_deriv: bool = False, k_stack: Optional[np.ndarray] = None, n_jobs: int = 1,
        return_grid: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    _require_array(derivs, np.ndarray, "derivs")
    _require_array(path1, np.ndarray, "path1")
    _require_array(path2, np.ndarray, "path2")
    if k_stack is not None:
        _require_array(k_stack, np.ndarray, "k_stack")
    return _impl.branched_sig_kernel_backprop(derivs, path1, path2, depth, dyadic_order,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, left_deriv=left_deriv, right_deriv=right_deriv,
        k_stack=k_stack, n_jobs=n_jobs, return_grid=return_grid)


branched_sig_kernel_backprop.__doc__ = _backend_doc(
    _impl.branched_sig_kernel_backprop.__doc__, "numpy"
)


def branched_sig_kernel_gram_backprop(derivs: np.ndarray, path1: np.ndarray, path2: np.ndarray,
        depth: int, dyadic_order: Union[int, tuple], *,
        static_kernel: Optional[StaticKernel] = None, time_aug: bool = False,
        lead_lag: bool = False, end_time: float = 1.0, left_deriv: bool = True,
        right_deriv: bool = False, k_stack: Optional[np.ndarray] = None, n_jobs: int = 1,
        return_grid: bool = False, max_batch: int = -1) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    _require_array(derivs, np.ndarray, "derivs")
    _require_array(path1, np.ndarray, "path1")
    _require_array(path2, np.ndarray, "path2")
    if k_stack is not None:
        _require_array(k_stack, np.ndarray, "k_stack")
    return _impl.branched_sig_kernel_gram_backprop(derivs, path1, path2, depth, dyadic_order,
        static_kernel=static_kernel, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, left_deriv=left_deriv, right_deriv=right_deriv,
        k_stack=k_stack, n_jobs=n_jobs, return_grid=return_grid, max_batch=max_batch)


branched_sig_kernel_gram_backprop.__doc__ = _backend_doc(
    _impl.branched_sig_kernel_gram_backprop.__doc__, "numpy"
)
