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
"""Exact FSSK coefficients for a general (dense / Jordan / oscillatory) state
matrix ``Lambda``.

The kernel is ``K(t,s) = sum_p (1^T exp(-Lambda (t-s)) b_p) A_p``. Unlike the
diagonal fast path, ``Lambda`` is kept as a real matrix here and the per-step
coefficients are built from the matrix resolvent ``(zeta I + dt Lambda)^{-1}``
on the same Weideman-Trefethen parabolic contour used by the diagonal builder.
Because ``Lambda`` and ``b`` are real the conjugate-symmetric ``2 Re(.)`` form
of the contour stays valid, so the coefficients ``E``, ``psi``, ``phi`` come out
real. This handles diagonalizable matrices (real or complex spectrum) and
genuine (defective) Jordan blocks uniformly - the matrix exponential and the
shifted solves are well defined regardless of diagonalizability.

The returned coefficients are packed in the same multi-index order as the
native layout (compositions enumerated level by level), so the native general
recursion can consume them directly. ``scipy`` is required (matrix exponential
and complex solves) and is imported lazily by the caller.
"""

import numpy as np
from scipy.linalg import expm

from ._volterra_conv import _enumerate_multiindices


def general_coefficients(Lambda, b, *, dt, readout_lag, quad_order, degree, dtype):
    """Build packed real FSSK coefficients for a general ``Lambda``.

    :param Lambda: Real state matrix, shape ``(R, R)``.
    :param b: Real readout vectors, shape ``(q, R)``.
    :return: dict with real arrays ``E`` ``(R, R)``, ``psi`` ``(M, R)``,
        ``phi`` ``(q, Mphi, R, R)``, ``readout_weights`` ``(q, R)``, where
        ``M`` counts multi-indices of degree <= ``degree-1`` and ``Mphi`` of
        degree <= ``degree-2``. Arrays are cast to ``dtype``.
    """
    Lambda = np.ascontiguousarray(np.asarray(Lambda, dtype=np.float64))
    b = np.ascontiguousarray(np.atleast_2d(np.asarray(b, dtype=np.float64)))
    R = Lambda.shape[0]
    q = b.shape[0]
    real_dtype = np.dtype(dtype)

    E = expm(-Lambda * dt)
    readout_weights = (expm(-Lambda * float(readout_lag)) @ b.T).T  # (q, R)

    if degree == 0:
        return dict(
            E=E.astype(real_dtype),
            psi=np.zeros((0, R), dtype=real_dtype),
            phi=np.zeros((q, 0, R, R), dtype=real_dtype),
            readout_weights=np.ascontiguousarray(readout_weights.astype(real_dtype)),
        )

    ell = _enumerate_multiindices(q, degree - 1)   # (M, q)
    M = ell.shape[0]
    # Mphi = number of multi-indices of degree <= degree-2 (a prefix of ell).
    Mphi = int(np.count_nonzero(ell.sum(axis=1) <= degree - 2))

    # phi_1(-Lambda dt) via the augmented matrix exp.
    I = np.eye(R)
    aug = np.zeros((2 * R, 2 * R))
    aug[:R, :R] = -Lambda * dt
    aug[:R, R:] = I
    phi1 = expm(aug)[:R, R:]

    psi = np.zeros((M, R))
    phi = np.zeros((q, Mphi, R, R))
    psi[0] = np.sum(phi1, axis=0)   # 1^T phi_1(-Lambda dt)

    j = np.arange(1, quad_order + 1)
    theta = (2.0 * j - 1.0) * np.pi / (2.0 * quad_order)
    zeta = (2 * quad_order) * (0.1309 - 0.1194 * theta ** 2 + 0.25j * theta)
    slope = 0.2388j * theta + 0.25
    ez = np.exp(zeta)
    omega = ez * slope
    tilde_omega = ez * slope / zeta

    for mm in range(quad_order):
        z = zeta[mm]
        rv = np.linalg.solve(z * I + dt * Lambda.T, np.ones(R))   # (R,)
        uv = np.linalg.solve(z * I + dt * Lambda, b.T)            # (R, q)
        beta = rv @ b.T                                           # (q,)
        g_all = np.prod(beta[None, :] ** ell, axis=1)             # (M,)
        psi[1:] += 2.0 * np.real(np.outer(tilde_omega[mm] * g_all[1:], rv))
        # phi[p, mi, r0, r1] += 2 Re(omega g u[:,p] (x) r)
        outer = uv.T[:, None, :, None] * rv[None, None, None, :]  # (q, 1, R, R)
        phi += 2.0 * np.real((omega[mm] * g_all[:Mphi])[None, :, None, None] * outer)

    return dict(
        E=np.ascontiguousarray(E.astype(real_dtype)),
        psi=np.ascontiguousarray(psi.astype(real_dtype)),
        phi=np.ascontiguousarray(phi.astype(real_dtype)),
        readout_weights=np.ascontiguousarray(readout_weights.astype(real_dtype)),
    )
