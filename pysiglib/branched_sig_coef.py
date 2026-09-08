from __future__ import annotations

from typing import Optional
import numpy as np
from ._core import branched_sig_coef as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def extract_branched_sig_coef(bsig: np.ndarray, trees, dimension: int, *, time_aug: bool = False,
        lead_lag: bool = False, planar: bool = False, scalar_term: bool = False) -> np.ndarray:
    _require_array(bsig, np.ndarray, "bsig")
    return _impl.extract_branched_sig_coef(bsig, trees, dimension, time_aug=time_aug,
        lead_lag=lead_lag, planar=planar, scalar_term=scalar_term)


extract_branched_sig_coef.__doc__ = _backend_doc(
    _impl.extract_branched_sig_coef.__doc__, "numpy"
)


from ._core.branched_sig_coef import (
    prepare_branched_sig_coef as prepare_branched_sig_coef,
)


def branched_sig_coef(path: np.ndarray, trees, *, time_aug: bool = False, lead_lag: bool = False,
        end_time: float = 1.0, planar: bool = False, correction: Optional[np.ndarray] = None,
        n_jobs: int = 1) -> np.ndarray:
    _require_array(path, np.ndarray, "path")
    if correction is not None:
        _require_array(correction, np.ndarray, "correction")
    return _impl.branched_sig_coef(path, trees, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, planar=planar, correction=correction, n_jobs=n_jobs)


branched_sig_coef.__doc__ = _backend_doc(_impl.branched_sig_coef.__doc__, "numpy")
