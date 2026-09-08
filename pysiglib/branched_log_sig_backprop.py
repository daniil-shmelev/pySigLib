from __future__ import annotations

from typing import Optional
import numpy as np
from ._core import branched_log_sig_backprop as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def branched_sig_to_log_sig_backprop(bsig: np.ndarray, blogsig_derivs: np.ndarray, dimension: int,
        degree: int, *, time_aug: bool = False, lead_lag: bool = False, planar: bool = False,
        method: Optional[int] = None, n_jobs: int = 1) -> np.ndarray:
    _require_array(bsig, np.ndarray, "bsig")
    _require_array(blogsig_derivs, np.ndarray, "blogsig_derivs")
    return _impl.branched_sig_to_log_sig_backprop(bsig, blogsig_derivs, dimension, degree,
        time_aug=time_aug, lead_lag=lead_lag, planar=planar, method=method,
        n_jobs=n_jobs)


branched_sig_to_log_sig_backprop.__doc__ = _backend_doc(
    _impl.branched_sig_to_log_sig_backprop.__doc__, "numpy"
)
