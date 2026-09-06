from __future__ import annotations

from typing import Tuple
import numpy as np
from ._core import log_sig_combine as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def log_sig_combine(log_sig1: np.ndarray, log_sig2: np.ndarray, dimension: int, degree: int, *,
        time_aug: bool = False, lead_lag: bool = False, n_jobs: int = 1) -> np.ndarray:
    _require_array(log_sig1, np.ndarray, "log_sig1")
    _require_array(log_sig2, np.ndarray, "log_sig2")
    return _impl.log_sig_combine(log_sig1, log_sig2, dimension, degree, time_aug=time_aug,
        lead_lag=lead_lag, n_jobs=n_jobs)


log_sig_combine.__doc__ = _backend_doc(_impl.log_sig_combine.__doc__, "numpy")


def log_sig_combine_backprop(deriv: np.ndarray, ls1: np.ndarray, ls2: np.ndarray, dimension: int,
        degree: int, *, time_aug: bool = False, lead_lag: bool = False, n_jobs: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    _require_array(deriv, np.ndarray, "deriv")
    _require_array(ls1, np.ndarray, "ls1")
    _require_array(ls2, np.ndarray, "ls2")
    return _impl.log_sig_combine_backprop(deriv, ls1, ls2, dimension, degree, time_aug=time_aug,
        lead_lag=lead_lag, n_jobs=n_jobs)


log_sig_combine_backprop.__doc__ = _backend_doc(
    _impl.log_sig_combine_backprop.__doc__, "numpy"
)
