from __future__ import annotations

import numpy as np
from ._core import transform_path as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def transform_path(path: np.ndarray, *, time_aug: bool = False, lead_lag: bool = False,
        end_time: float = 1.0, n_jobs: int = 1) -> np.ndarray:
    _require_array(path, np.ndarray, "path")
    return _impl.transform_path(path, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
        n_jobs=n_jobs)


transform_path.__doc__ = _backend_doc(_impl.transform_path.__doc__, "numpy")
