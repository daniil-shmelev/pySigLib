from __future__ import annotations

from typing import Optional
import numpy as np
from ._core import branched_sig_coef_backprop as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def branched_sig_coef_backprop(path: np.ndarray, trees, coefs: np.ndarray, derivs: np.ndarray, *,
        time_aug: bool = False, lead_lag: bool = False, end_time: float = 1.0,
        planar: bool = False, correction: Optional[np.ndarray] = None, n_jobs: int = 1) -> np.ndarray:
    _require_array(path, np.ndarray, "path")
    _require_array(coefs, np.ndarray, "coefs")
    _require_array(derivs, np.ndarray, "derivs")
    if correction is not None:
        _require_array(correction, np.ndarray, "correction")
    return _impl.branched_sig_coef_backprop(path, trees, coefs, derivs, time_aug=time_aug,
        lead_lag=lead_lag, end_time=end_time, planar=planar, correction=correction,
        n_jobs=n_jobs)


branched_sig_coef_backprop.__doc__ = _backend_doc(
    _impl.branched_sig_coef_backprop.__doc__, "numpy"
)
