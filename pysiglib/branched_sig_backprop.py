from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from ._core import branched_sig_backprop as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def branched_sig_backprop(path: np.ndarray, bsig: np.ndarray, bsig_derivs: np.ndarray, degree: int,
        *, time_aug: bool = False, lead_lag: bool = False, end_time: float = 1.0,
        planar: bool = False, correction: Optional[np.ndarray] = None, n_jobs: int = 1) -> np.ndarray:
    _require_array(path, np.ndarray, "path")
    _require_array(bsig, np.ndarray, "bsig")
    _require_array(bsig_derivs, np.ndarray, "bsig_derivs")
    if correction is not None:
        _require_array(correction, np.ndarray, "correction")
    return _impl.branched_sig_backprop(path, bsig, bsig_derivs, degree, time_aug=time_aug,
        lead_lag=lead_lag, end_time=end_time, planar=planar, correction=correction,
        n_jobs=n_jobs)


branched_sig_backprop.__doc__ = _backend_doc(
    _impl.branched_sig_backprop.__doc__, "numpy"
)


def branched_sig_combine_backprop(derivs: np.ndarray, bsig1: np.ndarray, bsig2: np.ndarray,
        dimension: int, degree: int, *, planar: bool = False, n_jobs: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    _require_array(derivs, np.ndarray, "derivs")
    _require_array(bsig1, np.ndarray, "bsig1")
    _require_array(bsig2, np.ndarray, "bsig2")
    return _impl.branched_sig_combine_backprop(derivs, bsig1, bsig2, dimension, degree,
        planar=planar, n_jobs=n_jobs)


branched_sig_combine_backprop.__doc__ = _backend_doc(
    _impl.branched_sig_combine_backprop.__doc__, "numpy"
)
