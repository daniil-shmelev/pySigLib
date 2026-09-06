from __future__ import annotations

import numpy as np
from ._core import log_sig_backprop as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def sig_to_log_sig_backprop(sig: np.ndarray, log_sig_derivs: np.ndarray, dimension: int,
        degree: int, *, time_aug: bool = False, lead_lag: bool = False, method: int = 1,
        n_jobs: int = 1) -> np.ndarray:
    _require_array(sig, np.ndarray, "sig")
    _require_array(log_sig_derivs, np.ndarray, "log_sig_derivs")
    return _impl.sig_to_log_sig_backprop(sig, log_sig_derivs, dimension, degree, time_aug=time_aug,
        lead_lag=lead_lag, method=method, n_jobs=n_jobs)


sig_to_log_sig_backprop.__doc__ = _backend_doc(
    _impl.sig_to_log_sig_backprop.__doc__, "numpy"
)
