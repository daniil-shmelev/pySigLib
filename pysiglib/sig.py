from __future__ import annotations

from typing import Optional
import numpy as np
from ._core import sig as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def sig_combine(sig1: np.ndarray, sig2: np.ndarray, dimension: int, degree: int, *,
        time_aug: bool = False, lead_lag: bool = False, n_jobs: int = 1) -> np.ndarray:
    _require_array(sig1, np.ndarray, "sig1")
    _require_array(sig2, np.ndarray, "sig2")
    return _impl.sig_combine(sig1, sig2, dimension, degree, time_aug=time_aug, lead_lag=lead_lag,
        n_jobs=n_jobs)


sig_combine.__doc__ = _backend_doc(_impl.sig_combine.__doc__, "numpy")


def sig(path: np.ndarray, degree: int, *, time_aug: bool = False, lead_lag: bool = False,
        end_time: float = 1.0, horner: bool = True, scalar_term: bool = False,
        correction: Optional[np.ndarray] = None, n_jobs: int = 1) -> np.ndarray:
    _require_array(path, np.ndarray, "path")
    if correction is not None:
        _require_array(correction, np.ndarray, "correction")
    return _impl.sig(path, degree, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
        horner=horner, scalar_term=scalar_term, correction=correction, n_jobs=n_jobs)


sig.__doc__ = _backend_doc(_impl.sig.__doc__, "numpy")
