from __future__ import annotations

from typing import Optional
import numpy as np
from ._core import log_sig as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc

from ._core.log_sig import set_cache_dir as set_cache_dir

from ._core.log_sig import prepare_log_sig as prepare_log_sig

from ._core.log_sig import clear_cache as clear_cache


def sig_to_log_sig(sig: np.ndarray, dimension: int, degree: int, *, time_aug: bool = False,
        lead_lag: bool = False, method: int = 1, n_jobs: int = 1) -> np.ndarray:
    _require_array(sig, np.ndarray, "sig")
    return _impl.sig_to_log_sig(sig, dimension, degree, time_aug=time_aug, lead_lag=lead_lag,
        method=method, n_jobs=n_jobs)


sig_to_log_sig.__doc__ = _backend_doc(_impl.sig_to_log_sig.__doc__, "numpy")


def log_sig(path: np.ndarray, degree: int, *, time_aug: bool = False, lead_lag: bool = False,
        end_time: float = 1.0, method: int = 1, scalar_term: bool = False,
        correction: Optional[np.ndarray] = None, n_jobs: int = 1) -> np.ndarray:
    _require_array(path, np.ndarray, "path")
    if correction is not None:
        _require_array(correction, np.ndarray, "correction")
    return _impl.log_sig(path, degree, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
        method=method, scalar_term=scalar_term, correction=correction, n_jobs=n_jobs)


log_sig.__doc__ = _backend_doc(_impl.log_sig.__doc__, "numpy")
