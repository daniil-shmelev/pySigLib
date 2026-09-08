from __future__ import annotations

import numpy as np
from ._core import sig_join as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def sig_join(sig: np.ndarray, displacement: np.ndarray, dimension: int, degree: int, *,
        prepend: bool = False, n_jobs: int = 1) -> np.ndarray:
    _require_array(sig, np.ndarray, "sig")
    _require_array(displacement, np.ndarray, "displacement")
    return _impl.sig_join(sig, displacement, dimension, degree, prepend=prepend, n_jobs=n_jobs)


sig_join.__doc__ = _backend_doc(_impl.sig_join.__doc__, "numpy")
