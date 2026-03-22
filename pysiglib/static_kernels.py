# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from abc import ABC, abstractmethod
import torch

class Context:
    """
    Provides context for backpropagation through static kernels.
    It is not generally necessary to create instances of this class
    manually; documentation for this class is provided purely for
    reference when constructing custom-made static kernels.
    """

    def __init__(self):
        self.saved_tensors = ()
        self.saved_for_y = ()

    def save_for_backward(self, *args):
        """
        Save objects from the forward pass to be re-used on the backward pass.
        """
        self.saved_tensors = args

    def save_for_grad_y(self, *args):
        """
        Save objects from the computation of the gradient with respect to x
        to be re-used for that of the gradient with respect to y.
        """
        self.saved_for_y = args

class StaticKernel(ABC):

    @abstractmethod
    def __call__(self, ctx : Context, x : torch.Tensor, y : torch.Tensor):
        """
        Returns the gram matrix of static kernels:

        .. math::

            \\{ \\kappa(x_s, y_t) - \\kappa(x_{s-1}, x_t) - \\kappa(x_s, y_{t-1}) + \\kappa(x_{s-1}, y_{t-1}) \\}_{0 \\leq s \\leq L_1, 0 \\leq t \\leq L_2}

        as a tensor of shape ``(batch_size, length_1 - 1, length_2 - 1)``, where
        ``length_1`` is the length of :math:`x` and ``length_2``
        is the length of :math:`y`.

        :param ctx: ``pysiglib.Context`` object for backpropagation
        :type ctx: pysiglib.Context
        :param x: Path :math:`x` of shape ``(batch_size, length_1, dimension)``.
        :type x: torch.Tensor
        :param y: Path :math:`y` of shape ``(batch_size, length_2, dimension)``.
        :type y: torch.Tensor
        :return: Batch of gram matrices of shape ``(batch_size, length_1 - 1, length_2 - 1)``.
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def grad_x(self, ctx : Context, derivs : torch.Tensor):
        """
        Backpropagates ``derivs`` through the static kernel computation and returns the
        derivatives with respect to the path :math:`x`.

        :param ctx: ``pysiglib.Context`` object for backpropagation
        :type ctx: pysiglib.Context
        :param derivs: Derivatives with respect to the gram matrices outputted by ``__call__``, of
            shape ``(batch_size, length_1 - 1, length_2 - 1)``.
        :return: Derivatives with respect to the path :math:`x` of shape ``(batch_size, length_1, dimension)``.
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def grad_y(self, ctx : Context, derivs : torch.Tensor):
        """
        Backpropagates ``derivs`` through the static kernel computation and returns the
        derivatives with respect to the path :math:`y`.

        :param ctx: ``pysiglib.Context`` object for backpropagation
        :type ctx: pysiglib.Context
        :param derivs: Derivatives with respect to the gram matrices outputted by ``__call__``, of
            shape ``(batch_size, length_1 - 1, length_2 - 1)``.
        :return: Derivatives with respect to the path :math:`y` of shape ``(batch_size, length_2, dimension)``.
        :rtype: torch.Tensor
        """
        pass

class LinearKernel(StaticKernel):
    """
    The linear kernel, defined by :math:`\\kappa(x, y) = \\langle x, y \\rangle`.
    """

    def __call__(self, ctx : Context, x : torch.Tensor, y : torch.Tensor):
        dx = torch.diff(x, dim=1)
        dy = torch.diff(y, dim=1)
        ctx.save_for_backward(dx, dy)
        return torch.bmm(dx, dy.permute(0, 2, 1))

    def grad_x(self, ctx : Context, derivs : torch.Tensor):
        dx, dy = ctx.saved_tensors
        out = torch.empty((dx.shape[0], dx.shape[1] + 1, dy.shape[1]), dtype=dx.dtype, device=derivs.device)
        out[:, 0, :] = 0
        out[:, 1:, :] = derivs
        out[:, :-1, :] -= derivs
        return torch.bmm(out, dy)

    def grad_y(self, ctx : Context, derivs : torch.Tensor):
        dx, dy = ctx.saved_tensors
        out = torch.empty((dx.shape[0], dx.shape[1], dy.shape[1] + 1), dtype=dx.dtype, device=derivs.device)
        out[:, :, 0] = 0
        out[:, :, 1:] = derivs
        out[:, :, :-1] -= derivs
        return torch.bmm(out.permute(0, 2, 1), dx)

class ScaledLinearKernel(StaticKernel):
    """
    The scaled linear kernel, defined by :math:`\\kappa(x, y) = \\langle \\alpha x, \\alpha y \\rangle = \\alpha^2 \\langle x, y \\rangle`,
    where :math:`\\alpha` is given by the parameter ``scale``. A choice of ``scale=1.0`` corresponds to the standard
    linear kernel.
    """

    def __init__(self, scale : float = 1.):
        self.linear_kernel = LinearKernel()
        self.scale = scale
        self._scale_sq = scale ** 2

    def __call__(self, ctx : Context, x : torch.Tensor, y : torch.Tensor):
        return self.linear_kernel(ctx, x * self._scale_sq, y)

    def grad_x(self, ctx : Context, derivs : torch.Tensor):
        return self.linear_kernel.grad_x(ctx, derivs) * self._scale_sq

    def grad_y(self, ctx : Context, derivs : torch.Tensor):
        return self.linear_kernel.grad_y(ctx, derivs)

class RBFKernel(StaticKernel):
    """
    The RBF kernel, defined by :math:`\\kappa(x, y) = \\exp\\left( -\\frac{\\lVert x - y \\rVert^2}{\\sigma} \\right)`.
    """

    def __init__(self, sigma : float):
        self.sigma = sigma
        self._one_over_sigma = 1. / sigma
        self._scale = 2 * self._one_over_sigma

    def __call__(self, ctx : Context, x : torch.Tensor, y : torch.Tensor):
        dist = torch.bmm(x * self._scale, y.permute(0, 2, 1))

        x2 = torch.pow(x, 2)
        y2 = torch.pow(y, 2)
        x2 = torch.sum(x2, dim=2) * self._one_over_sigma
        y2 = torch.sum(y2, dim=2) * self._one_over_sigma

        dist = dist - (torch.reshape(x2, (x.shape[0], x.shape[1], 1)) + torch.reshape(y2, (x.shape[0], 1, y.shape[1])))
        dist = torch.exp(dist)

        ctx.save_for_backward(x, y, dist.clone())

        buff = torch.diff(dist, dim=1)
        result = torch.diff(buff, dim=2)
        return result

    def grad_x(self, ctx : Context, derivs : torch.Tensor):
        x, y, out = ctx.saved_tensors

        dout = torch.zeros_like(out)
        dout[:, 1:, 1:] += derivs
        dout[:, :-1, :-1] += derivs
        dout[:, 1:, :-1] -= derivs
        dout[:, :-1, 1:] -= derivs

        dout *= out
        dout *= 2. * self._one_over_sigma

        ctx.save_for_grad_y(x, y, dout)
        return torch.bmm(dout, y) - x * torch.sum(dout, dim=2).unsqueeze(-1)

    def grad_y(self, ctx : Context, derivs : torch.Tensor):

        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y#
        else:
            x, y, out = ctx.saved_tensors

            dout = torch.zeros_like(out)
            dout[:, 1:, 1:] += derivs
            dout[:, :-1, :-1] += derivs
            dout[:, 1:, :-1] -= derivs
            dout[:, :-1, 1:] -= derivs

            dout *= out
            dout *= 2. * self._one_over_sigma

        return torch.bmm(dout.permute(0, 2, 1), x) - y * torch.sum(dout, dim=1).unsqueeze(-1)