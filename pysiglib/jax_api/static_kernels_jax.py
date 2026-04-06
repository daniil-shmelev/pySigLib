# Copyright 2026 Daniil Shmelev
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

"""
JAX-native static kernels for use with ``jax_api.sig_kernel``.

These are pure JAX implementations — backward passes are handled
automatically by JAX autodiff. Each kernel is a callable that takes
two batched paths ``(B, L, D)`` and returns the double-differenced
gram matrix ``(B, L-1, L-1)``.
"""

import math
import jax.numpy as jnp


def _double_diff(K):
    return K[:, 1:, 1:] - K[:, :-1, 1:] - K[:, 1:, :-1] + K[:, :-1, :-1]


def _squared_dist(x, y):
    x2 = jnp.sum(x * x, axis=2, keepdims=True)
    y2 = jnp.sum(y * y, axis=2, keepdims=True)
    return jnp.clip(x2 - 2 * jnp.matmul(x, jnp.swapaxes(y, -2, -1)) + jnp.swapaxes(y2, -2, -1), 0)


class LinearKernel:
    """Linear kernel: kappa(x, y) = <x, y>."""

    def __call__(self, x, y):
        dx = jnp.diff(x, axis=1)
        dy = jnp.diff(y, axis=1)
        return jnp.matmul(dx, jnp.swapaxes(dy, -2, -1))


class ScaledLinearKernel:
    """Scaled linear kernel: kappa(x, y) = scale^2 * <x, y>."""

    def __init__(self, scale: float = 1.):
        self._scale_sq = scale ** 2

    def __call__(self, x, y):
        dx = jnp.diff(x, axis=1)
        dy = jnp.diff(y, axis=1)
        return self._scale_sq * jnp.matmul(dx, jnp.swapaxes(dy, -2, -1))


class RBFKernel:
    """RBF kernel: kappa(x, y) = exp(-||x - y||^2 / sigma)."""

    def __init__(self, sigma: float):
        self._one_over_sigma = 1. / sigma

    def __call__(self, x, y):
        dist2 = _squared_dist(x, y)
        K = jnp.exp(-dist2 * self._one_over_sigma)
        return _double_diff(K)


class PolynomialKernel:
    """Polynomial kernel: kappa(x, y) = scale * (<x, y> + gamma)^degree."""

    def __init__(self, degree: float = 3., gamma: float = 1., scale: float = 1.):
        self.degree = degree
        self.gamma = gamma
        self.scale = scale

    def __call__(self, x, y):
        inner = jnp.matmul(x, jnp.swapaxes(y, -2, -1))
        base = inner + self.gamma
        K = self.scale * jnp.power(base, self.degree)
        return _double_diff(K)


class Matern12Kernel:
    """Matern-1/2 kernel: kappa(x, y) = exp(-||x - y|| / sigma)."""

    def __init__(self, sigma: float):
        self._one_over_sigma = 1. / sigma

    def __call__(self, x, y):
        dist = jnp.sqrt(_squared_dist(x, y) + 1e-30)
        K = jnp.exp(-dist * self._one_over_sigma)
        return _double_diff(K)


class Matern32Kernel:
    """Matern-3/2 kernel."""

    def __init__(self, sigma: float):
        self._sqrt3_over_sigma = math.sqrt(3.) / sigma

    def __call__(self, x, y):
        D_scaled = jnp.sqrt(_squared_dist(x, y) + 1e-30) * self._sqrt3_over_sigma
        K = (1. + D_scaled) * jnp.exp(-D_scaled)
        return _double_diff(K)


class Matern52Kernel:
    """Matern-5/2 kernel."""

    def __init__(self, sigma: float):
        self._sqrt5_over_sigma = math.sqrt(5.) / sigma

    def __call__(self, x, y):
        u = jnp.sqrt(_squared_dist(x, y) + 1e-30) * self._sqrt5_over_sigma
        K = (1. + u + u * u / 3.) * jnp.exp(-u)
        return _double_diff(K)


class RationalQuadraticKernel:
    """Rational quadratic kernel: kappa(x, y) = (1 + ||x-y||^2 / (2*alpha*sigma^2))^(-alpha)."""

    def __init__(self, sigma: float, alpha: float = 1.):
        self.alpha = alpha
        self._c = 2. * alpha * sigma ** 2

    def __call__(self, x, y):
        dist2 = _squared_dist(x, y)
        base = 1. + dist2 / self._c
        K = jnp.power(base, -self.alpha)
        return _double_diff(K)
