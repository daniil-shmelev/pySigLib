from __future__ import annotations

import numpy as np
from ._core import transform_path_backprop as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def transform_path_backprop(derivs: np.ndarray, *, time_aug: bool = False, lead_lag: bool = False,
        end_time: float = 1.0, n_jobs: int = 1) -> np.ndarray:
    _require_array(derivs, np.ndarray, "derivs")
    return _impl.transform_path_backprop(derivs, time_aug=time_aug, lead_lag=lead_lag,
        end_time=end_time, n_jobs=n_jobs)


transform_path_backprop.__doc__ = _backend_doc(
    _impl.transform_path_backprop.__doc__, "numpy"
)
