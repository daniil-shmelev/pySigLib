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

from __future__ import annotations
from array_api_compat import array_namespace, device
from ._array import ArrayT, copy_array, require_array

from typing import Generic

from abc import ABC, abstractmethod
import math
import warnings


class Context(Generic[ArrayT]):
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


class StaticKernel(ABC, Generic[ArrayT]):
    _array_type = None

    @abstractmethod
    def __call__(self, ctx: Context[ArrayT], x: ArrayT, y: ArrayT) -> ArrayT:
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
        :type x: Array
        :param y: Path :math:`y` of shape ``(batch_size, length_2, dimension)``.
        :type y: Array
        :return: Batch of gram matrices of shape ``(batch_size, length_1 - 1, length_2 - 1)``.
        :rtype: Array
        """
        if self._array_type is not None:
            require_array(x, self._array_type, "x")
            require_array(y, self._array_type, "y")
        pass

    @abstractmethod
    def grad_x(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        """
        Backpropagates ``derivs`` through the static kernel computation and returns the
        derivatives with respect to the path :math:`x`.

        :param ctx: ``pysiglib.Context`` object for backpropagation
        :type ctx: pysiglib.Context
        :param derivs: Derivatives with respect to the gram matrices outputted by ``__call__``, of
            shape ``(batch_size, length_1 - 1, length_2 - 1)``.
        :type derivs: Array
        :return: Derivatives with respect to the path :math:`x` of shape ``(batch_size, length_1, dimension)``.
        :rtype: Array
        """
        pass

    @abstractmethod
    def grad_y(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        """
        Backpropagates ``derivs`` through the static kernel computation and returns the
        derivatives with respect to the path :math:`y`.

        :param ctx: ``pysiglib.Context`` object for backpropagation
        :type ctx: pysiglib.Context
        :param derivs: Derivatives with respect to the gram matrices outputted by ``__call__``, of
            shape ``(batch_size, length_1 - 1, length_2 - 1)``.
        :type derivs: Array
        :return: Derivatives with respect to the path :math:`y` of shape ``(batch_size, length_2, dimension)``.
        :rtype: Array
        """
        pass


class LinearKernel(StaticKernel[ArrayT]):
    """
    The linear kernel, defined by :math:`\\kappa(x, y) = \\langle x, y \\rangle`.
    """

    def __call__(self, ctx: Context[ArrayT], x: ArrayT, y: ArrayT) -> ArrayT:
        xp = array_namespace(x)
        if self._array_type is not None:
            require_array(x, self._array_type, "x")
            require_array(y, self._array_type, "y")
        dx = xp.diff(x, axis=1)
        dy = xp.diff(y, axis=1)
        ctx.save_for_backward(dx, dy)
        return xp.matmul(dx, xp.matrix_transpose(dy))

    def grad_x(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        dx, dy = ctx.saved_tensors
        out = xp.empty((dx.shape[0], dx.shape[1] + 1, dy.shape[1]), dtype=dx.dtype, device=device(derivs))
        out[:, 0, :] = 0
        out[:, 1:, :] = derivs
        out[:, :-1, :] -= derivs
        return xp.matmul(out, dy)

    def grad_y(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        dx, dy = ctx.saved_tensors
        out = xp.empty((dx.shape[0], dx.shape[1], dy.shape[1] + 1), dtype=dx.dtype, device=device(derivs))
        out[:, :, 0] = 0
        out[:, :, 1:] = derivs
        out[:, :, :-1] -= derivs
        return xp.matmul(xp.matrix_transpose(out), dx)


class ScaledLinearKernel(StaticKernel[ArrayT]):
    """
    The scaled linear kernel, defined by :math:`\\kappa(x, y) = \\langle \\alpha x, \\alpha y \\rangle = \\alpha^2 \\langle x, y \\rangle`,
    where :math:`\\alpha` is given by the parameter ``scale``. A choice of ``scale=1.0`` corresponds to the standard
    linear kernel.
    """

    def __init__(self, scale : float = 1.):
        if scale < 0:
            raise ValueError(f"ScaledLinearKernel: scale must be >= 0, got {scale}")
        self.linear_kernel = LinearKernel()
        self.scale = scale
        self._scale_sq = scale ** 2

    def __call__(self, ctx: Context[ArrayT], x: ArrayT, y: ArrayT) -> ArrayT:
        if self._array_type is not None:
            require_array(x, self._array_type, "x")
            require_array(y, self._array_type, "y")
        return self.linear_kernel(ctx, x * self._scale_sq, y)

    def grad_x(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        return self.linear_kernel.grad_x(ctx, derivs) * self._scale_sq

    def grad_y(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        return self.linear_kernel.grad_y(ctx, derivs)


class RBFKernel(StaticKernel[ArrayT]):
    """
    The RBF kernel, defined by :math:`\\kappa(x, y) = \\exp\\left( -\\frac{\\lVert x - y \\rVert^2}{\\sigma} \\right)`.
    """

    def __init__(self, sigma : float):
        if sigma <= 0:
            raise ValueError(f"RBFKernel: sigma must be > 0, got {sigma}")
        self.sigma = sigma
        self._one_over_sigma = 1. / sigma
        self._scale = 2 * self._one_over_sigma

    def __call__(self, ctx: Context[ArrayT], x: ArrayT, y: ArrayT) -> ArrayT:
        xp = array_namespace(x)
        if self._array_type is not None:
            require_array(x, self._array_type, "x")
            require_array(y, self._array_type, "y")
        dist = xp.matmul(x * self._scale, xp.matrix_transpose(y))

        x2 = x**2
        y2 = y**2
        x2 = xp.sum(x2, axis=2) * self._one_over_sigma
        y2 = xp.sum(y2, axis=2) * self._one_over_sigma

        dist = dist - (xp.reshape(x2, (x.shape[0], x.shape[1], 1)) + xp.reshape(y2, (x.shape[0], 1, y.shape[1])))
        dist = xp.exp(dist)

        ctx.save_for_backward(x, y, copy_array(dist))

        buff = xp.diff(dist, axis=1)
        result = xp.diff(buff, axis=2)
        return result

    def grad_x(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        x, y, out = ctx.saved_tensors

        dout = _undo_double_diff(derivs, out)
        dout *= out
        dout *= 2. * self._one_over_sigma

        ctx.save_for_grad_y(x, y, dout)
        return xp.matmul(dout, y) - x * xp.sum(dout, axis=2, keepdims=True)

    def grad_y(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:

        xp = array_namespace(derivs)
        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, out = ctx.saved_tensors

            dout = _undo_double_diff(derivs, out)
            dout *= out
            dout *= 2. * self._one_over_sigma

        return xp.matmul(xp.matrix_transpose(dout), x) - y * xp.expand_dims(xp.sum(dout, axis=1), axis=-1)


def _squared_dist(x, y):
    xp = array_namespace(x)
    x2 = xp.expand_dims(xp.sum(x * x, axis=2), axis=2)
    y2 = xp.expand_dims(xp.sum(y * y, axis=2), axis=1)
    return xp.clip(x2 - 2 * xp.matmul(x, xp.matrix_transpose(y)) + y2, min=0)


def _undo_double_diff(derivs, like):
    xp = array_namespace(derivs)
    dout = xp.zeros_like(like)
    dout[:, 1:, 1:] = derivs
    dout[:, :-1, :-1] += derivs
    dout[:, 1:, :-1] -= derivs
    dout[:, :-1, 1:] -= derivs
    return dout


class PolynomialKernel(StaticKernel[ArrayT]):
    """
    The polynomial kernel, defined by :math:`\\kappa(x, y) = \\text{scale} \\cdot \\left( \\langle x, y \\rangle + \\gamma \\right)^d`,
    where :math:`d` is the ``degree`` parameter.
    """

    def __init__(self, degree : float = 3., gamma : float = 1., scale : float = 1.):
        if degree < 0:
            raise ValueError(f"PolynomialKernel: degree must be >= 0, got {degree}")
        if scale < 0:
            raise ValueError(f"PolynomialKernel: scale must be >= 0, got {scale}")
        self.degree = degree
        self.gamma = gamma
        self.scale = scale
        self._int_degree = int(degree) if degree == int(degree) and 1 <= degree <= 5 else None
        self._needs_base_clamp = self._int_degree is None and degree != 0
        self._warned_negative_base = False

    def _pow(self, base, exp):
        xp = array_namespace(base)
        if self._int_degree is not None and exp == int(exp) and 0 <= exp <= 4:
            n = int(exp)
            if n == 0:
                return xp.ones_like(base)
            if n == 1: return base
            if n == 2: return base * base
            b2 = base * base
            if n == 3: return b2 * base
            if n == 4: return b2 * b2
        return base**exp

    def __call__(self, ctx: Context[ArrayT], x: ArrayT, y: ArrayT) -> ArrayT:
        xp = array_namespace(x)
        if self._array_type is not None:
            require_array(x, self._array_type, "x")
            require_array(y, self._array_type, "y")
        inner = xp.matmul(x, xp.matrix_transpose(y))
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
            base = xp.clip(base, min=0)
        K = self.scale * self._pow(base, self.degree)
        ctx.save_for_backward(x, y, base)
        return xp.diff(xp.diff(K, axis=1), axis=2)

    def grad_x(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        if self.degree == 0:
            return xp.zeros_like(ctx.saved_tensors[0])
        x, y, base = ctx.saved_tensors
        dout = _undo_double_diff(derivs, base)
        dout *= self.scale * self.degree
        dout *= self._pow(base, self.degree - 1)
        ctx.save_for_grad_y(x, y, dout)
        return xp.matmul(dout, y)

    def grad_y(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        if self.degree == 0:
            return xp.zeros_like(ctx.saved_tensors[1])
        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, base = ctx.saved_tensors
            dout = _undo_double_diff(derivs, base)
            dout *= self.scale * self.degree
            dout *= self._pow(base, self.degree - 1)
        return xp.matmul(xp.matrix_transpose(dout), x)


class Matern12Kernel(StaticKernel[ArrayT]):
    """
    The Matern-1/2 kernel (exponential kernel), defined by :math:`\\kappa(x, y) = \\exp\\left( -\\frac{\\lVert x - y \\rVert}{\\sigma} \\right)`.
    """

    def __init__(self, sigma : float):
        if sigma <= 0:
            raise ValueError(f"Matern12Kernel: sigma must be > 0, got {sigma}")
        self.sigma = sigma
        self._one_over_sigma = 1. / sigma

    def __call__(self, ctx: Context[ArrayT], x: ArrayT, y: ArrayT) -> ArrayT:
        xp = array_namespace(x)
        if self._array_type is not None:
            require_array(x, self._array_type, "x")
            require_array(y, self._array_type, "y")
        dist2 = _squared_dist(x, y)
        dist = xp.sqrt(dist2 + 1e-30)
        K = xp.exp(-dist * self._one_over_sigma)
        ctx.save_for_backward(x, y, dist)
        return xp.diff(xp.diff(K, axis=1), axis=2)

    def grad_x(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        x, y, dist = ctx.saved_tensors
        K = xp.exp(-dist * self._one_over_sigma)
        dout = _undo_double_diff(derivs, dist)
        dout *= K
        dout *= self._one_over_sigma / xp.clip(dist, min=1e-15)
        ctx.save_for_grad_y(x, y, dout)
        return xp.matmul(dout, y) - x * xp.sum(dout, axis=2, keepdims=True)

    def grad_y(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, dist = ctx.saved_tensors
            K = xp.exp(-dist * self._one_over_sigma)
            dout = _undo_double_diff(derivs, dist)
            dout *= K
            dout *= self._one_over_sigma / xp.clip(dist, min=1e-15)
        return xp.matmul(xp.matrix_transpose(dout), x) - y * xp.expand_dims(xp.sum(dout, axis=1), axis=-1)


class Matern32Kernel(StaticKernel[ArrayT]):
    """
    The Matern-3/2 kernel, defined by :math:`\\kappa(x, y) = \\left(1 + \\frac{\\sqrt{3} \\lVert x - y \\rVert}{\\sigma}\\right) \\exp\\left( -\\frac{\\sqrt{3} \\lVert x - y \\rVert}{\\sigma} \\right)`.
    """

    def __init__(self, sigma : float):
        if sigma <= 0:
            raise ValueError(f"Matern32Kernel: sigma must be > 0, got {sigma}")
        self.sigma = sigma
        self._sqrt3_over_sigma = math.sqrt(3.) / sigma
        self._3_over_sigma_sq = 3. / (sigma ** 2)

    def __call__(self, ctx: Context[ArrayT], x: ArrayT, y: ArrayT) -> ArrayT:
        xp = array_namespace(x)
        if self._array_type is not None:
            require_array(x, self._array_type, "x")
            require_array(y, self._array_type, "y")
        D_scaled = xp.sqrt(_squared_dist(x, y) + 1e-30) * self._sqrt3_over_sigma
        exp_term = xp.exp(-D_scaled)
        K = (1. + D_scaled) * exp_term
        ctx.save_for_backward(x, y, exp_term)
        return xp.diff(xp.diff(K, axis=1), axis=2)

    def grad_x(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        x, y, exp_term = ctx.saved_tensors
        dout = _undo_double_diff(derivs, exp_term)
        dout *= exp_term
        dout *= self._3_over_sigma_sq
        ctx.save_for_grad_y(x, y, dout)
        return xp.matmul(dout, y) - x * xp.sum(dout, axis=2, keepdims=True)

    def grad_y(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, exp_term = ctx.saved_tensors
            dout = _undo_double_diff(derivs, exp_term)
            dout *= exp_term
            dout *= self._3_over_sigma_sq
        return xp.matmul(xp.matrix_transpose(dout), x) - y * xp.expand_dims(xp.sum(dout, axis=1), axis=-1)


class Matern52Kernel(StaticKernel[ArrayT]):
    """
    The Matern-5/2 kernel, defined by :math:`\\kappa(x, y) = \\left(1 + \\frac{\\sqrt{5} \\lVert x - y \\rVert}{\\sigma} + \\frac{5 \\lVert x - y \\rVert^2}{3\\sigma^2}\\right) \\exp\\left( -\\frac{\\sqrt{5} \\lVert x - y \\rVert}{\\sigma} \\right)`.
    """

    def __init__(self, sigma : float):
        if sigma <= 0:
            raise ValueError(f"Matern52Kernel: sigma must be > 0, got {sigma}")
        self.sigma = sigma
        self._sqrt5_over_sigma = math.sqrt(5.) / sigma
        self._5_over_3sigma_sq = 5. / (3. * sigma ** 2)

    def __call__(self, ctx: Context[ArrayT], x: ArrayT, y: ArrayT) -> ArrayT:
        xp = array_namespace(x)
        if self._array_type is not None:
            require_array(x, self._array_type, "x")
            require_array(y, self._array_type, "y")
        u = xp.sqrt(_squared_dist(x, y) + 1e-30) * self._sqrt5_over_sigma
        exp_term = xp.exp(-u)
        K = (1. + u + u * u / 3.) * exp_term
        ctx.save_for_backward(x, y, u, exp_term)
        return xp.diff(xp.diff(K, axis=1), axis=2)

    def grad_x(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        x, y, u, exp_term = ctx.saved_tensors
        dout = _undo_double_diff(derivs, exp_term)
        dout *= exp_term
        dout *= 1. + u
        dout *= self._5_over_3sigma_sq
        ctx.save_for_grad_y(x, y, dout)
        return xp.matmul(dout, y) - x * xp.sum(dout, axis=2, keepdims=True)

    def grad_y(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, u, exp_term = ctx.saved_tensors
            dout = _undo_double_diff(derivs, exp_term)
            dout *= exp_term
            dout *= 1. + u
            dout *= self._5_over_3sigma_sq
        return xp.matmul(xp.matrix_transpose(dout), x) - y * xp.expand_dims(xp.sum(dout, axis=1), axis=-1)


class RationalQuadraticKernel(StaticKernel[ArrayT]):
    """
    The rational quadratic kernel, defined by :math:`\\kappa(x, y) = \\left(1 + \\frac{\\lVert x - y \\rVert^2}{2 \\alpha \\sigma^2}\\right)^{-\\alpha}`.
    """

    def __init__(self, sigma : float, alpha : float = 1.):
        if sigma <= 0:
            raise ValueError(f"RationalQuadraticKernel: sigma must be > 0, got {sigma}")
        if alpha <= 0:
            raise ValueError(f"RationalQuadraticKernel: alpha must be > 0, got {alpha}")
        self.sigma = sigma
        self.alpha = alpha
        self._c = 2. * alpha * sigma ** 2
        self._one_over_sigma_sq = 1. / (sigma ** 2)

    def __call__(self, ctx: Context[ArrayT], x: ArrayT, y: ArrayT) -> ArrayT:
        xp = array_namespace(x)
        if self._array_type is not None:
            require_array(x, self._array_type, "x")
            require_array(y, self._array_type, "y")
        dist2 = _squared_dist(x, y)
        base = 1. + dist2 / self._c
        K = base ** (-self.alpha)
        ctx.save_for_backward(x, y, K / base)
        return xp.diff(xp.diff(K, axis=1), axis=2)

    def grad_x(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        x, y, weight = ctx.saved_tensors
        dout = _undo_double_diff(derivs, weight)
        dout *= weight
        dout *= self._one_over_sigma_sq
        ctx.save_for_grad_y(x, y, dout)
        return xp.matmul(dout, y) - x * xp.sum(dout, axis=2, keepdims=True)

    def grad_y(self, ctx: Context[ArrayT], derivs: ArrayT) -> ArrayT:
        xp = array_namespace(derivs)
        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y
        else:
            x, y, weight = ctx.saved_tensors
            dout = _undo_double_diff(derivs, weight)
            dout *= weight
            dout *= self._one_over_sigma_sq
        return xp.matmul(xp.matrix_transpose(dout), x) - y * xp.expand_dims(xp.sum(dout, axis=1), axis=-1)
