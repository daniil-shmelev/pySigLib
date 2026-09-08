from __future__ import annotations

from typing import Tuple
import numpy as np
from ._core import sig_join_backprop as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def sig_join_backprop(d_out: np.ndarray, sig: np.ndarray, displacement: np.ndarray, dimension: int,
        degree: int, *, prepend: bool = False, n_jobs: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    _require_array(d_out, np.ndarray, "d_out")
    _require_array(sig, np.ndarray, "sig")
    _require_array(displacement, np.ndarray, "displacement")
    return _impl.sig_join_backprop(d_out, sig, displacement, dimension, degree, prepend=prepend,
        n_jobs=n_jobs)


sig_join_backprop.__doc__ = _backend_doc(_impl.sig_join_backprop.__doc__, "numpy")
