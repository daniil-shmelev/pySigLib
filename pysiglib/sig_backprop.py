from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from ._core import sig_backprop as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def sig_backprop(path: np.ndarray, sig: np.ndarray, sig_derivs: np.ndarray, degree: int, *,
        time_aug: bool = False, lead_lag: bool = False, end_time: float = 1.0,
        correction: Optional[np.ndarray] = None, n_jobs: int = 1) -> np.ndarray:
    _require_array(path, np.ndarray, "path")
    _require_array(sig, np.ndarray, "sig")
    _require_array(sig_derivs, np.ndarray, "sig_derivs")
    if correction is not None:
        _require_array(correction, np.ndarray, "correction")
    return _impl.sig_backprop(path, sig, sig_derivs, degree, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, correction=correction, n_jobs=n_jobs)


sig_backprop.__doc__ = _backend_doc(_impl.sig_backprop.__doc__, "numpy")


def sig_combine_backprop(deriv: np.ndarray, sig1: np.ndarray, sig2: np.ndarray, dimension: int,
        degree: int, *, time_aug: bool = False, lead_lag: bool = False, n_jobs: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    _require_array(deriv, np.ndarray, "deriv")
    _require_array(sig1, np.ndarray, "sig1")
    _require_array(sig2, np.ndarray, "sig2")
    return _impl.sig_combine_backprop(deriv, sig1, sig2, dimension, degree, time_aug=time_aug,
        lead_lag=lead_lag, n_jobs=n_jobs)


sig_combine_backprop.__doc__ = _backend_doc(_impl.sig_combine_backprop.__doc__, "numpy")
