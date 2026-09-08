from __future__ import annotations

from typing import Optional
import numpy as np
from ._core import branched_log_sig as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def branched_log_sig(path: np.ndarray, degree: int, *, time_aug: bool = False,
        lead_lag: bool = False, end_time: float = 1.0, planar: bool = False,
        scalar_term: bool = False, method: Optional[int] = None,
        correction: Optional[np.ndarray] = None, n_jobs: int = 1) -> np.ndarray:
    _require_array(path, np.ndarray, "path")
    if correction is not None:
        _require_array(correction, np.ndarray, "correction")
    return _impl.branched_log_sig(path, degree, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, planar=planar, scalar_term=scalar_term, method=method,
        correction=correction, n_jobs=n_jobs)


branched_log_sig.__doc__ = _backend_doc(_impl.branched_log_sig.__doc__, "numpy")


from ._core.branched_log_sig import branched_log_sig_length as branched_log_sig_length


def branched_sig_to_log_sig(bsig: np.ndarray, dimension: int, degree: int, *,
        time_aug: bool = False, lead_lag: bool = False, planar: bool = False,
        method: Optional[int] = None, n_jobs: int = 1) -> np.ndarray:
    _require_array(bsig, np.ndarray, "bsig")
    return _impl.branched_sig_to_log_sig(bsig, dimension, degree, time_aug=time_aug,
        lead_lag=lead_lag, planar=planar, method=method, n_jobs=n_jobs)


branched_sig_to_log_sig.__doc__ = _backend_doc(
    _impl.branched_sig_to_log_sig.__doc__, "numpy"
)


from ._core.branched_log_sig import prepare_branched_log_sig as prepare_branched_log_sig
