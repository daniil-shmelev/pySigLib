"""Static kernels for torch arrays."""

import torch
from .._core import static_kernels as _impl
from .._core._docs import backend_doc as _backend_doc


class Context(_impl.Context[torch.Tensor]):
    __doc__ = _backend_doc(_impl.Context.__doc__, "torch")


class StaticKernel(_impl.StaticKernel[torch.Tensor]):
    _array_type = torch.Tensor
    __doc__ = _backend_doc(_impl.StaticKernel.__doc__, "torch")


class LinearKernel(_impl.LinearKernel[torch.Tensor], StaticKernel):
    __doc__ = _backend_doc(_impl.LinearKernel.__doc__, "torch")


class ScaledLinearKernel(_impl.ScaledLinearKernel[torch.Tensor], StaticKernel):
    __doc__ = _backend_doc(_impl.ScaledLinearKernel.__doc__, "torch")


class RBFKernel(_impl.RBFKernel[torch.Tensor], StaticKernel):
    __doc__ = _backend_doc(_impl.RBFKernel.__doc__, "torch")


class PolynomialKernel(_impl.PolynomialKernel[torch.Tensor], StaticKernel):
    __doc__ = _backend_doc(_impl.PolynomialKernel.__doc__, "torch")


class Matern12Kernel(_impl.Matern12Kernel[torch.Tensor], StaticKernel):
    __doc__ = _backend_doc(_impl.Matern12Kernel.__doc__, "torch")


class Matern32Kernel(_impl.Matern32Kernel[torch.Tensor], StaticKernel):
    __doc__ = _backend_doc(_impl.Matern32Kernel.__doc__, "torch")


class Matern52Kernel(_impl.Matern52Kernel[torch.Tensor], StaticKernel):
    __doc__ = _backend_doc(_impl.Matern52Kernel.__doc__, "torch")


class RationalQuadraticKernel(
    _impl.RationalQuadraticKernel[torch.Tensor], StaticKernel
):
    __doc__ = _backend_doc(_impl.RationalQuadraticKernel.__doc__, "torch")
