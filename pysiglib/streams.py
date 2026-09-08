"""Streaming signature computations on NumPy arrays."""

import numpy as np
from ._core import streams as _impl
from ._core._docs import backend_doc as _backend_doc


class SigStream(_impl.SigStream[np.ndarray]):
    _array_type = np.ndarray
    __doc__ = _backend_doc(_impl.SigStream.__doc__, "numpy")


class LogSigStream(_impl.LogSigStream[np.ndarray]):
    _array_type = np.ndarray
    __doc__ = _backend_doc(_impl.LogSigStream.__doc__, "numpy")


class SigWindowStream(_impl.SigWindowStream[np.ndarray]):
    _array_type = np.ndarray
    __doc__ = _backend_doc(_impl.SigWindowStream.__doc__, "numpy")


class LogSigWindowStream(_impl.LogSigWindowStream[np.ndarray]):
    _array_type = np.ndarray
    __doc__ = _backend_doc(_impl.LogSigWindowStream.__doc__, "numpy")
