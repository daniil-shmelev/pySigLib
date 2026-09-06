from __future__ import annotations

import numpy as np
from ._core import linear_sig as _impl
from ._core._array import require_array as _require_array
from ._core._docs import backend_doc as _backend_doc


def linear_sig(displacement: np.ndarray, dimension: int, degree: int, *, scalar_term: bool = False,
        n_jobs: int = 1) -> np.ndarray:
    _require_array(displacement, np.ndarray, "displacement")
    return _impl.linear_sig(displacement, dimension, degree, scalar_term=scalar_term, n_jobs=n_jobs)


linear_sig.__doc__ = _backend_doc(_impl.linear_sig.__doc__, "numpy")
