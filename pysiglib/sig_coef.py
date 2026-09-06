from __future__ import annotations

from typing import Union
import numpy as np
from ._core import sig_coef as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def extract_sig_coef(sig: np.ndarray, words: Union[tuple[int, ...], list[tuple[int, ...]]],
        dimension: int, *, time_aug: bool = False, lead_lag: bool = False,
        scalar_term: bool = False) -> np.ndarray:
    _require_array(sig, np.ndarray, "sig")
    return _impl.extract_sig_coef(sig, words, dimension, time_aug=time_aug, lead_lag=lead_lag,
        scalar_term=scalar_term)


extract_sig_coef.__doc__ = _backend_doc(_impl.extract_sig_coef.__doc__, "numpy")


def sig_coef(path: np.ndarray, words: Union[tuple[int, ...], list[tuple[int, ...]]], *,
        time_aug: bool = False, lead_lag: bool = False, end_time: float = 1.0,
        prefixes: bool = False, n_jobs: int = 1) -> np.ndarray:
    _require_array(path, np.ndarray, "path")
    return _impl.sig_coef(path, words, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
        prefixes=prefixes, n_jobs=n_jobs)


sig_coef.__doc__ = _backend_doc(_impl.sig_coef.__doc__, "numpy")
