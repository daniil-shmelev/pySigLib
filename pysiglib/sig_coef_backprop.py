from __future__ import annotations

from typing import Union
import numpy as np
from ._core import sig_coef_backprop as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def sig_coef_backprop(path: np.ndarray, words: Union[tuple[int, ...], list[tuple[int, ...]]],
        coefs: np.ndarray, derivs: np.ndarray, *, time_aug: bool = False, lead_lag: bool = False,
        end_time: float = 1.0, n_jobs: int = 1) -> np.ndarray:
    _require_array(path, np.ndarray, "path")
    _require_array(coefs, np.ndarray, "coefs")
    _require_array(derivs, np.ndarray, "derivs")
    return _impl.sig_coef_backprop(path, words, coefs, derivs, time_aug=time_aug,
        lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)


sig_coef_backprop.__doc__ = _backend_doc(_impl.sig_coef_backprop.__doc__, "numpy")
