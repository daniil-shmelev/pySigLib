from __future__ import annotations

from typing import Tuple
import numpy as np
from ._core import log_sig_join_backprop as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def log_sig_join_backprop(d_out: np.ndarray, log_sig: np.ndarray, displacement: np.ndarray,
        dimension: int, degree: int, *, n_jobs: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    _require_array(d_out, np.ndarray, "d_out")
    _require_array(log_sig, np.ndarray, "log_sig")
    _require_array(displacement, np.ndarray, "displacement")
    return _impl.log_sig_join_backprop(d_out, log_sig, displacement, dimension, degree,
        n_jobs=n_jobs)


log_sig_join_backprop.__doc__ = _backend_doc(
    _impl.log_sig_join_backprop.__doc__, "numpy"
)
