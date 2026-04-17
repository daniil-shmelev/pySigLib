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
import math
import warnings
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

        dout = _undo_double_diff(derivs, out)
        dout *= out
        dout *= 2. * self._one_over_sigma

        ctx.save_for_grad_y(x, y, dout)
        return torch.bmm(dout, y) - x * torch.sum(dout, dim=2, keepdim=True)

    def grad_y(self, ctx : Context, derivs : torch.Tensor):

        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, out = ctx.saved_tensors

            dout = _undo_double_diff(derivs, out)
            dout *= out
            dout *= 2. * self._one_over_sigma

        return torch.bmm(dout.permute(0, 2, 1), x) - y * torch.sum(dout, dim=1).unsqueeze(-1)

def _squared_dist(x, y):
    x2 = torch.sum(x * x, dim=2).unsqueeze(2)
    y2 = torch.sum(y * y, dim=2).unsqueeze(1)
    return torch.clamp(x2 - 2 * torch.bmm(x, y.permute(0, 2, 1)) + y2, min=0)

def _undo_double_diff(derivs, like):
    dout = torch.zeros_like(like)
    dout[:, 1:, 1:] = derivs
    dout[:, :-1, :-1] += derivs
    dout[:, 1:, :-1] -= derivs
    dout[:, :-1, 1:] -= derivs
    return dout

class PolynomialKernel(StaticKernel):
    """
    The polynomial kernel, defined by :math:`\\kappa(x, y) = \\text{scale} \\cdot \\left( \\langle x, y \\rangle + \\gamma \\right)^d`,
    where :math:`d` is the ``degree`` parameter.
    """

    def __init__(self, degree : float = 3., gamma : float = 1., scale : float = 1.):
        self.degree = degree
        self.gamma = gamma
        self.scale = scale
        self._int_degree = int(degree) if degree == int(degree) and 1 <= degree <= 5 else None
        self._needs_base_clamp = self._int_degree is None and degree != 0
        self._warned_negative_base = False

    def _pow(self, base, exp):
        if self._int_degree is not None and exp == int(exp) and 0 <= exp <= 4:
            n = int(exp)
            if n == 0: return torch.ones_like(base)
            if n == 1: return base
            if n == 2: return base * base
            b2 = base * base
            if n == 3: return b2 * base
            if n == 4: return b2 * b2
        return torch.pow(base, exp)

    def __call__(self, ctx : Context, x : torch.Tensor, y : torch.Tensor):
        inner = torch.bmm(x, y.permute(0, 2, 1))
        base = inner + self.gamma
        if self._needs_base_clamp:
            if not self._warned_negative_base and (base < 0).any():
                self._warned_negative_base = True
                warnings.warn(
                    "PolynomialKernel: non-integer degree with negative base values "
                    "(<x, y> + gamma < 0). These entries are clamped to 0. Consider "
                    "increasing gamma to ensure all base values are non-negative.",
                    RuntimeWarning, stacklevel=2
                )
            base = torch.clamp(base, min=0)
        K = self.scale * self._pow(base, self.degree)
        ctx.save_for_backward(x, y, base)
        return torch.diff(torch.diff(K, dim=1), dim=2)

    def grad_x(self, ctx : Context, derivs : torch.Tensor):
        if self.degree == 0:
            return torch.zeros_like(ctx.saved_tensors[0])
        x, y, base = ctx.saved_tensors
        dout = _undo_double_diff(derivs, base)
        dout *= self.scale * self.degree
        dout *= self._pow(base, self.degree - 1)
        ctx.save_for_grad_y(x, y, dout)
        return torch.bmm(dout, y)

    def grad_y(self, ctx : Context, derivs : torch.Tensor):
        if self.degree == 0:
            return torch.zeros_like(ctx.saved_tensors[1])
        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, base = ctx.saved_tensors
            dout = _undo_double_diff(derivs, base)
            dout *= self.scale * self.degree
            dout *= self._pow(base, self.degree - 1)
        return torch.bmm(dout.permute(0, 2, 1), x)

class Matern12Kernel(StaticKernel):
    """
    The Matern-1/2 kernel (exponential kernel), defined by :math:`\\kappa(x, y) = \\exp\\left( -\\frac{\\lVert x - y \\rVert}{\\sigma} \\right)`.
    """

    def __init__(self, sigma : float):
        self.sigma = sigma
        self._one_over_sigma = 1. / sigma

    def __call__(self, ctx : Context, x : torch.Tensor, y : torch.Tensor):
        dist2 = _squared_dist(x, y)
        dist = torch.sqrt(dist2 + 1e-30)
        K = torch.exp(-dist * self._one_over_sigma)
        ctx.save_for_backward(x, y, dist)
        return torch.diff(torch.diff(K, dim=1), dim=2)

    def grad_x(self, ctx : Context, derivs : torch.Tensor):
        x, y, dist = ctx.saved_tensors
        K = torch.exp(-dist * self._one_over_sigma)
        dout = _undo_double_diff(derivs, dist)
        dout *= K
        dout *= self._one_over_sigma / torch.clamp(dist, min=1e-15)
        ctx.save_for_grad_y(x, y, dout)
        return torch.bmm(dout, y) - x * torch.sum(dout, dim=2, keepdim=True)

    def grad_y(self, ctx : Context, derivs : torch.Tensor):
        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, dist = ctx.saved_tensors
            K = torch.exp(-dist * self._one_over_sigma)
            dout = _undo_double_diff(derivs, dist)
            dout *= K
            dout *= self._one_over_sigma / torch.clamp(dist, min=1e-15)
        return torch.bmm(dout.permute(0, 2, 1), x) - y * torch.sum(dout, dim=1).unsqueeze(-1)

class Matern32Kernel(StaticKernel):
    """
    The Matern-3/2 kernel, defined by :math:`\\kappa(x, y) = \\left(1 + \\frac{\\sqrt{3} \\lVert x - y \\rVert}{\\sigma}\\right) \\exp\\left( -\\frac{\\sqrt{3} \\lVert x - y \\rVert}{\\sigma} \\right)`.
    """

    def __init__(self, sigma : float):
        self.sigma = sigma
        self._sqrt3_over_sigma = math.sqrt(3.) / sigma
        self._3_over_sigma_sq = 3. / (sigma ** 2)

    def __call__(self, ctx : Context, x : torch.Tensor, y : torch.Tensor):
        D_scaled = torch.sqrt(_squared_dist(x, y) + 1e-30) * self._sqrt3_over_sigma
        exp_term = torch.exp(-D_scaled)
        K = (1. + D_scaled) * exp_term
        ctx.save_for_backward(x, y, exp_term)
        return torch.diff(torch.diff(K, dim=1), dim=2)

    def grad_x(self, ctx : Context, derivs : torch.Tensor):
        x, y, exp_term = ctx.saved_tensors
        dout = _undo_double_diff(derivs, exp_term)
        dout *= exp_term
        dout *= self._3_over_sigma_sq
        ctx.save_for_grad_y(x, y, dout)
        return torch.bmm(dout, y) - x * torch.sum(dout, dim=2, keepdim=True)

    def grad_y(self, ctx : Context, derivs : torch.Tensor):
        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, exp_term = ctx.saved_tensors
            dout = _undo_double_diff(derivs, exp_term)
            dout *= exp_term
            dout *= self._3_over_sigma_sq
        return torch.bmm(dout.permute(0, 2, 1), x) - y * torch.sum(dout, dim=1).unsqueeze(-1)

class Matern52Kernel(StaticKernel):
    """
    The Matern-5/2 kernel, defined by :math:`\\kappa(x, y) = \\left(1 + \\frac{\\sqrt{5} \\lVert x - y \\rVert}{\\sigma} + \\frac{5 \\lVert x - y \\rVert^2}{3\\sigma^2}\\right) \\exp\\left( -\\frac{\\sqrt{5} \\lVert x - y \\rVert}{\\sigma} \\right)`.
    """

    def __init__(self, sigma : float):
        self.sigma = sigma
        self._sqrt5_over_sigma = math.sqrt(5.) / sigma
        self._5_over_3sigma_sq = 5. / (3. * sigma ** 2)

    def __call__(self, ctx : Context, x : torch.Tensor, y : torch.Tensor):
        u = torch.sqrt(_squared_dist(x, y) + 1e-30) * self._sqrt5_over_sigma
        exp_term = torch.exp(-u)
        K = (1. + u + u * u / 3.) * exp_term
        ctx.save_for_backward(x, y, u, exp_term)
        return torch.diff(torch.diff(K, dim=1), dim=2)

    def grad_x(self, ctx : Context, derivs : torch.Tensor):
        x, y, u, exp_term = ctx.saved_tensors
        dout = _undo_double_diff(derivs, exp_term)
        dout *= exp_term
        dout *= 1. + u
        dout *= self._5_over_3sigma_sq
        ctx.save_for_grad_y(x, y, dout)
        return torch.bmm(dout, y) - x * torch.sum(dout, dim=2, keepdim=True)

    def grad_y(self, ctx : Context, derivs : torch.Tensor):
        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, u, exp_term = ctx.saved_tensors
            dout = _undo_double_diff(derivs, exp_term)
            dout *= exp_term
            dout *= 1. + u
            dout *= self._5_over_3sigma_sq
        return torch.bmm(dout.permute(0, 2, 1), x) - y * torch.sum(dout, dim=1).unsqueeze(-1)

class RationalQuadraticKernel(StaticKernel):
    """
    The rational quadratic kernel, defined by :math:`\\kappa(x, y) = \\left(1 + \\frac{\\lVert x - y \\rVert^2}{2 \\alpha \\sigma^2}\\right)^{-\\alpha}`.
    """

    def __init__(self, sigma : float, alpha : float = 1.):
        self.sigma = sigma
        self.alpha = alpha
        self._c = 2. * alpha * sigma ** 2
        self._one_over_sigma_sq = 1. / (sigma ** 2)

    def __call__(self, ctx : Context, x : torch.Tensor, y : torch.Tensor):
        dist2 = _squared_dist(x, y)
        base = 1. + dist2 / self._c
        K = torch.pow(base, -self.alpha)
        ctx.save_for_backward(x, y, K / base)
        return torch.diff(torch.diff(K, dim=1), dim=2)

    def grad_x(self, ctx : Context, derivs : torch.Tensor):
        x, y, weight = ctx.saved_tensors
        dout = _undo_double_diff(derivs, weight)
        dout *= weight
        dout *= self._one_over_sigma_sq
        ctx.save_for_grad_y(x, y, dout)
        return torch.bmm(dout, y) - x * torch.sum(dout, dim=2, keepdim=True)

    def grad_y(self, ctx : Context, derivs : torch.Tensor):
        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, weight = ctx.saved_tensors
            dout = _undo_double_diff(derivs, weight)
            dout *= weight
            dout *= self._one_over_sigma_sq
        return torch.bmm(dout.permute(0, 2, 1), x) - y * torch.sum(dout, dim=1).unsqueeze(-1)
