from __future__ import annotations

from typing import Optional
import numpy as np
from ._core import branched_sig as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc

from ._core.branched_sig import prepare_branched_sig as prepare_branched_sig


def branched_sig(path: np.ndarray, degree: int, *, time_aug: bool = False, lead_lag: bool = False,
        end_time: float = 1.0, planar: bool = False, scalar_term: bool = False,
        correction: Optional[np.ndarray] = None, n_jobs: int = 1) -> np.ndarray:
    _require_array(path, np.ndarray, "path")
    if correction is not None:
        _require_array(correction, np.ndarray, "correction")
    return _impl.branched_sig(path, degree, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, planar=planar, scalar_term=scalar_term,
        correction=correction, n_jobs=n_jobs)


branched_sig.__doc__ = _backend_doc(_impl.branched_sig.__doc__, "numpy")


def branched_sig_combine(bsig1: np.ndarray, bsig2: np.ndarray, dimension: int, degree: int, *,
        planar: bool = False, n_jobs: int = 1) -> np.ndarray:
    _require_array(bsig1, np.ndarray, "bsig1")
    _require_array(bsig2, np.ndarray, "bsig2")
    return _impl.branched_sig_combine(bsig1, bsig2, dimension, degree, planar=planar, n_jobs=n_jobs)


branched_sig_combine.__doc__ = _backend_doc(_impl.branched_sig_combine.__doc__, "numpy")


from ._core.branched_sig import branched_sig_length as branched_sig_length
