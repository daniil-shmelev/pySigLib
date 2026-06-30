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
"""Interval coefficients for the general convolution scheme (order 0).

The quadratic Volterra-Chen recursion needs the normalized interval
coefficients ``alpha(s_i, t_i, tau)`` of the kernel. On a uniform grid
``s_i = i dt``, ``t_i = (i+1) dt`` these depend only on the lag ``tau/dt - i``,
so this module precomputes the ``O(S)`` distinct lag coefficients rather than
the ``O(S^2)`` triples. The native recursion (cp_volterra_conv.cpp) consumes
them indexed by lag.

This increment covers the scalar (``q == 1``) kernels: the scalar fractional
kernel (closed-form incomplete-beta coefficients) and the Gamma kernel
(Gauss-Legendre quadrature). ``scipy`` is required and imported lazily.
"""

import numpy as np


def fractional_lag_coefficients(beta, *, dt, degree, S, dtype):
    r"""Lag coefficients for the scalar fractional kernel ``t^(beta-1)/Gamma(beta)``.

    Returns ``alpha_lag`` of shape ``(S + 1, degree)`` (lag 0 is unused/zero).
    ``alpha_lag[lag, n-1]`` is the normalized coefficient ``beta_n`` of the
    degree-``n`` term for a source interval ``lag`` steps before the readout.
    """
    from scipy.special import betainc, gammaln

    beta = float(beta)
    M = degree
    lag = np.arange(1, S + 1, dtype=np.float64)        # (S,)
    h = float(dt)
    tau_s = lag * h                                     # (S,)
    z = np.clip(h / tau_s, 0.0, 1.0)                    # (S,)

    out = np.zeros((S + 1, M), dtype=np.float64)
    for e in range(M):                                  # e = prefix degree |ell|
        total = (e + 1.0) * beta                        # ell.beta + beta_1
        a = e * beta + 1.0
        log_scale = total * np.log(tau_s) - (e + 1.0) * np.log(h) - gammaln(total + 1.0)
        out[1:, e] = np.exp(log_scale) * betainc(a, beta, z)
    return np.ascontiguousarray(out.astype(dtype))


def gamma_lag_coefficients(beta, scale, rate, quad_order, *, dt, degree, S, dtype):
    r"""Lag coefficients for the Gamma kernel by Gauss-Legendre quadrature.

    Kernel ``k(u) = scale * exp(-rate u) * u^(beta-1)/Gamma(beta)``. Returns
    ``alpha_lag`` of shape ``(S + 1, degree)`` (lag 0 unused/zero).
    """
    from scipy.special import betainc, gammaln

    beta = float(beta)
    scale = float(scale)
    rate = float(rate)
    M = degree
    h = float(dt)
    t = h
    nodes, weights = np.polynomial.legendre.leggauss(int(quad_order))
    u = 0.5 * h * (nodes + 1.0)                          # (Q,) since s = 0
    eff_w = 0.5 * h * weights                            # (Q,)

    lag = np.arange(1, S + 1, dtype=np.float64)          # (S,)
    tau = lag[:, None] * h                               # (S, 1)
    tau_u = tau - u[None, :]                             # (S, Q)
    exp_decay = np.exp(-rate * tau_u)
    x = np.clip((t - u[None, :]) / tau_u, 0.0, 1.0)      # (S, Q), degree-independent
    g1 = np.exp(gammaln(beta))

    out = np.zeros((S + 1, M), dtype=np.float64)
    for j in range(M):
        n = j + 1.0
        if n == 1.0:
            dot = scale * exp_decay * tau_u ** (beta - 1.0) / g1
        else:
            nbeta = n * beta
            dot = (scale ** n * exp_decay * tau_u ** (nbeta - 1.0)
                   * betainc((n - 1.0) * beta, beta, x) / np.exp(gammaln(nbeta)))
        kappa = np.sum(eff_w[None, :] * dot, axis=1)     # (S,)
        out[1:, j] = kappa / (h ** n)
    return np.ascontiguousarray(out.astype(dtype))


def convolution_lag_coefficients(kind, params, *, dt, degree, S, dtype):
    """Dispatch lag-coefficient builder by kernel kind. Returns ``(alpha_lag, M)``."""
    if degree <= 0:
        return np.zeros((S + 1, 0), dtype=dtype), 0
    if kind == "fractional":
        alpha = fractional_lag_coefficients(
            params["beta"], dt=dt, degree=degree, S=S, dtype=dtype)
    elif kind == "gamma":
        alpha = gamma_lag_coefficients(
            params["beta"], params["scale"], params["rate"], params["quad_order"],
            dt=dt, degree=degree, S=S, dtype=dtype)
    else:
        raise ValueError("unknown convolution kernel kind: " + str(kind))
    return alpha, degree
