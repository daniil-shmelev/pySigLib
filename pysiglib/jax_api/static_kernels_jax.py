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

These are pure JAX implementations - backward passes are handled
automatically by JAX autodiff. Each kernel is a callable that takes
two batched paths ``(B, L, D)`` and returns the double-differenced
gram matrix ``(B, L-1, L-1)``.
"""

from __future__ import annotations

import math
import jax
from typing import Protocol
from .._core.static_kernels import Context as _Context


class Context(_Context[jax.Array]):
    """Array context retained for utility API consistency."""


class StaticKernel(Protocol):
    """A differentiable JAX static kernel on two batched paths."""

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array: ...


import jax.numpy as jnp

from .._core.static_kernels import (
    LinearKernel as _BaseLinearKernel,
    ScaledLinearKernel as _BaseScaledLinearKernel,
    RBFKernel as _BaseRBFKernel,
    PolynomialKernel as _BasePolynomialKernel,
    Matern12Kernel as _BaseMatern12Kernel,
    Matern32Kernel as _BaseMatern32Kernel,
    Matern52Kernel as _BaseMatern52Kernel,
    RationalQuadraticKernel as _BaseRationalQuadraticKernel,
)


def _double_diff(K):
    return K[:, 1:, 1:] - K[:, :-1, 1:] - K[:, 1:, :-1] + K[:, :-1, :-1]


def _squared_dist(x, y):
    x2 = jnp.sum(x * x, axis=2, keepdims=True)
    y2 = jnp.sum(y * y, axis=2, keepdims=True)
    return jnp.clip(x2 - 2 * jnp.matmul(x, jnp.swapaxes(y, -2, -1)) + jnp.swapaxes(y2, -2, -1), 0)


class LinearKernel:
    __doc__ = _BaseLinearKernel.__doc__

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        dx = jnp.diff(x, axis=1)
        dy = jnp.diff(y, axis=1)
        return jnp.matmul(dx, jnp.swapaxes(dy, -2, -1))


class ScaledLinearKernel:
    __doc__ = _BaseScaledLinearKernel.__doc__

    def __init__(self, scale: float = 1.):
        self._scale_sq = scale ** 2

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        dx = jnp.diff(x, axis=1)
        dy = jnp.diff(y, axis=1)
        return self._scale_sq * jnp.matmul(dx, jnp.swapaxes(dy, -2, -1))


class RBFKernel:
    __doc__ = _BaseRBFKernel.__doc__

    def __init__(self, sigma: float):
        self._one_over_sigma = 1. / sigma

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        dist2 = _squared_dist(x, y)
        K = jnp.exp(-dist2 * self._one_over_sigma)
        return _double_diff(K)


class PolynomialKernel:
    __doc__ = _BasePolynomialKernel.__doc__

    def __init__(self, degree: float = 3., gamma: float = 1., scale: float = 1.):
        self.degree = degree
        self.gamma = gamma
        self.scale = scale

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        inner = jnp.matmul(x, jnp.swapaxes(y, -2, -1))
        base = inner + self.gamma
        K = self.scale * jnp.power(base, self.degree)
        return _double_diff(K)


class Matern12Kernel:
    __doc__ = _BaseMatern12Kernel.__doc__

    def __init__(self, sigma: float):
        self._one_over_sigma = 1. / sigma

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        dist = jnp.sqrt(_squared_dist(x, y) + 1e-30)
        K = jnp.exp(-dist * self._one_over_sigma)
        return _double_diff(K)


class Matern32Kernel:
    __doc__ = _BaseMatern32Kernel.__doc__

    def __init__(self, sigma: float):
        self._sqrt3_over_sigma = math.sqrt(3.) / sigma

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        D_scaled = jnp.sqrt(_squared_dist(x, y) + 1e-30) * self._sqrt3_over_sigma
        K = (1. + D_scaled) * jnp.exp(-D_scaled)
        return _double_diff(K)


class Matern52Kernel:
    __doc__ = _BaseMatern52Kernel.__doc__

    def __init__(self, sigma: float):
        self._sqrt5_over_sigma = math.sqrt(5.) / sigma

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        u = jnp.sqrt(_squared_dist(x, y) + 1e-30) * self._sqrt5_over_sigma
        K = (1. + u + u * u / 3.) * jnp.exp(-u)
        return _double_diff(K)


class RationalQuadraticKernel:
    __doc__ = _BaseRationalQuadraticKernel.__doc__

    def __init__(self, sigma: float, alpha: float = 1.):
        self.alpha = alpha
        self._c = 2. * alpha * sigma ** 2

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        dist2 = _squared_dist(x, y)
        base = 1. + dist2 / self._c
        K = jnp.power(base, -self.alpha)
        return _double_diff(K)
