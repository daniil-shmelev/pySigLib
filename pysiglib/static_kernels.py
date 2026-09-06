"""Static kernels for numpy arrays."""

import numpy as np
from ._core import static_kernels as _impl
from ._core._docs import backend_doc as _backend_doc


class Context(_impl.Context[np.ndarray]):
    __doc__ = _backend_doc(_impl.Context.__doc__, "numpy")


class StaticKernel(_impl.StaticKernel[np.ndarray]):
    _array_type = np.ndarray
    __doc__ = _backend_doc(_impl.StaticKernel.__doc__, "numpy")


class LinearKernel(_impl.LinearKernel[np.ndarray], StaticKernel):
    __doc__ = _backend_doc(_impl.LinearKernel.__doc__, "numpy")


class ScaledLinearKernel(_impl.ScaledLinearKernel[np.ndarray], StaticKernel):
    __doc__ = _backend_doc(_impl.ScaledLinearKernel.__doc__, "numpy")


class RBFKernel(_impl.RBFKernel[np.ndarray], StaticKernel):
    __doc__ = _backend_doc(_impl.RBFKernel.__doc__, "numpy")


class PolynomialKernel(_impl.PolynomialKernel[np.ndarray], StaticKernel):
    __doc__ = _backend_doc(_impl.PolynomialKernel.__doc__, "numpy")


class Matern12Kernel(_impl.Matern12Kernel[np.ndarray], StaticKernel):
    __doc__ = _backend_doc(_impl.Matern12Kernel.__doc__, "numpy")


class Matern32Kernel(_impl.Matern32Kernel[np.ndarray], StaticKernel):
    __doc__ = _backend_doc(_impl.Matern32Kernel.__doc__, "numpy")


class Matern52Kernel(_impl.Matern52Kernel[np.ndarray], StaticKernel):
    __doc__ = _backend_doc(_impl.Matern52Kernel.__doc__, "numpy")


class RationalQuadraticKernel(_impl.RationalQuadraticKernel[np.ndarray], StaticKernel):
    __doc__ = _backend_doc(_impl.RationalQuadraticKernel.__doc__, "numpy")
