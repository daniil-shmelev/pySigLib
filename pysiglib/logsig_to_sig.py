from __future__ import annotations

import numpy as np
from ._core import logsig_to_sig as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def logsig_to_sig(log_sig: np.ndarray, dimension: int, degree: int, *, time_aug: bool = False,
        lead_lag: bool = False, method: int = 1, scalar_term: bool = False, n_jobs: int = 1) -> np.ndarray:
    _require_array(log_sig, np.ndarray, "log_sig")
    return _impl.logsig_to_sig(log_sig, dimension, degree, time_aug=time_aug, lead_lag=lead_lag,
        method=method, scalar_term=scalar_term, n_jobs=n_jobs)


logsig_to_sig.__doc__ = _backend_doc(_impl.logsig_to_sig.__doc__, "numpy")
