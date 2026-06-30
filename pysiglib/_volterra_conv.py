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

Covers the fractional kernel (closed-form incomplete-beta coefficients, scalar
or multivariate ``q >= 1``) and the scalar Gamma kernel (Gauss-Legendre
quadrature, ``q == 1``). For ``q > 1`` the coefficients are packed in the native
multi-index order so the native shuffle-tensor evaluation can consume them.
``scipy`` is required and imported lazily.
"""

import numpy as np


def _enumerate_multiindices(q, max_degree):
    """Multi-indices of ``q`` components by total degree 0..max_degree, in the
    same order as the native ``populate_multiindex_layout`` (compositions with
    the first parts taken in descending order)."""
    out = []

    def rec(pos, remaining, current):
        if pos + 1 == q:
            current[pos] = remaining
            out.append(tuple(current))
            return
        for value in range(remaining, -1, -1):
            current[pos] = value
            rec(pos + 1, remaining - value, current)

    for level in range(max_degree + 1):
        rec(0, level, [0] * q)
    return np.asarray(out, dtype=np.int64).reshape(len(out), q)


def fractional_lag_coefficients(beta, *, dt, degree, S, dtype, rho=0.0, theta=0.0):
    r"""Lag coefficients for the fractional kernel ``k_p(u) = u^(beta_p-1)/Gamma(beta_p)``.

    ``beta`` is a scalar or a length-``q`` vector of fractional exponents.
    ``rho`` (basis exponent) and ``theta`` (interpolation offset, readout at
    ``tau = (lag+theta) dt``) drive the higher-order basis-expansion scheme;
    ``rho=0, theta=0`` recovers the standard order-0 left-point coefficients.
    Returns ``(alpha_lag, M)`` where ``alpha_lag`` has shape ``(S + 1, q, M)``
    (lag 0 unused/zero), ``M`` is the number of multi-indices of degree
    ``<= degree-1``, packed in the native multi-index order. The ``1/ell!``
    normalization is supplied by the native shuffle tensors, so it is not
    applied here.
    """
    from scipy.special import betainc, gammaln

    beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))   # (q,)
    q = beta.shape[0]
    ell = _enumerate_multiindices(q, degree - 1)               # (M, q)
    M = ell.shape[0]
    deg = ell.sum(axis=1).astype(np.float64)                   # (M,)
    prefix = ell.astype(np.float64) @ beta                     # (M,)  ell . beta

    h = float(dt)
    lag = np.arange(S + 1, dtype=np.float64)                   # 0..S
    tau = (lag + theta) * h                                    # readout at tau=(lag+theta) dt
    valid = tau >= h                                           # strict causality s < t <= tau
    tau_s = np.where(valid, tau, 1.0)
    z = np.clip(h / tau_s, 0.0, 1.0)
    log_tau_s = np.log(tau_s)[:, None]                         # (S+1, 1)
    g_rho = gammaln(rho + 1.0)

    out = np.zeros((S + 1, q, M), dtype=np.float64)
    for p in range(q):
        total = prefix + beta[p]                               # (M,)
        a = rho + prefix + 1.0                                 # (M,)
        log_scale = ((rho + total)[None, :] * log_tau_s
                     - (deg[None, :] + 1.0) * np.log(h)
                     + g_rho - gammaln(rho + total + 1.0)[None, :])
        vals = np.exp(log_scale) * betainc(a[None, :], beta[p], z[:, None])
        out[:, p, :] = np.where(valid[:, None], vals, 0.0)
    out[0] = 0.0                                               # lag 0: strict causality
    return np.ascontiguousarray(out.astype(dtype)), M


def gamma_lag_coefficients(beta, scale, rate, quad_order, *, dt, degree, S, dtype,
                           rho=0.0, theta=0.0):
    r"""Lag coefficients for the Gamma kernel by Gauss-Legendre quadrature.

    Kernel ``k(u) = scale * exp(-rate u) * u^(beta-1)/Gamma(beta)``. Returns
    ``alpha_lag`` of shape ``(S + 1, degree)`` (lag 0 unused/zero). ``rho``/
    ``theta`` drive the basis-expansion scheme (``rho=0, theta=0`` is order 0):
    the ``((u-s)/(t-s))^rho`` weight folds into the quadrature and the readout
    is at ``tau = (lag+theta) dt``.
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
    eff_w = 0.5 * h * weights * (((nodes + 1.0) * 0.5) ** rho)

    lag = np.arange(S + 1, dtype=np.float64)             # 0..S
    tau = (lag + theta) * h                              # (S+1,)
    valid = tau >= h
    tau_u = np.where(valid[:, None], tau[:, None] - u[None, :], 1.0)   # (S+1, Q)
    exp_decay = np.exp(-rate * tau_u)
    x = np.clip((t - u[None, :]) / tau_u, 0.0, 1.0)
    g1 = np.exp(gammaln(beta))
    g_rho = np.exp(gammaln(rho + 1.0))

    out = np.zeros((S + 1, M), dtype=np.float64)
    for j in range(M):
        n = j + 1.0
        if n == 1.0:
            dot = scale * exp_decay * tau_u ** (beta - 1.0) / g1
        else:
            nbeta = n * beta
            dot = (scale ** n * exp_decay * tau_u ** (nbeta - 1.0)
                   * betainc((n - 1.0) * beta, beta, x) / np.exp(gammaln(nbeta)))
        kappa = np.sum(eff_w[None, :] * dot, axis=1)     # (S+1,)
        out[:, j] = np.where(valid, g_rho * kappa / (h ** n), 0.0)
    out[0] = 0.0                                         # lag 0: strict causality
    return np.ascontiguousarray(out.astype(dtype))


def convolution_lag_coefficients(kind, params, *, dt, degree, S, dtype, rho=0.0, theta=0.0):
    """Dispatch lag-coefficient builder by kernel kind. Returns ``(alpha_lag, M)``.

    ``alpha_lag`` is packed as ``(S + 1, q, M)`` in the native multi-index order;
    for ``q == 1`` (Gamma, scalar fractional) the leading ``q`` axis is 1 and the
    flat layout matches the scalar lag-coefficient array consumed by the Horner
    path. ``rho``/``theta`` select the basis-expansion component/interpolation
    point (defaults give the order-0 coefficients).
    """
    if degree <= 0:
        return np.zeros((S + 1, 1, 0), dtype=dtype), 0
    if kind == "fractional":
        return fractional_lag_coefficients(
            params["beta"], dt=dt, degree=degree, S=S, dtype=dtype, rho=rho, theta=theta)
    if kind == "gamma":
        alpha = gamma_lag_coefficients(
            params["beta"], params["scale"], params["rate"], params["quad_order"],
            dt=dt, degree=degree, S=S, dtype=dtype, rho=rho, theta=theta)
        return alpha, degree
    raise ValueError("unknown convolution kernel kind: " + str(kind))
