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
"""FFT acceleration of the general convolution Volterra scheme.

On a uniform grid the quadratic Volterra-Chen recursion
``V(t_k) = unit + sum_{i<k} evalVtE(V(t_i), y_i, alpha(k-i))`` is a causal
convolution in the lag ``k-i``. Iterating over OUTPUT LEVELS (not time) breaks
the sequential dependency: each level's full time-history is a sum of FFT
convolutions of already-computed lower-level histories,

    hist[ell][j] = sum_{r=1..ell} sum_{p,|ell'|=r-1}
                     conv_i( hist[ell-r][i] (x) T_ell'[i] (x) y_p[i], alpha[p,ell'] )[j]

where ``T_ell'`` is the normalized shuffle tensor (shuffle power / ell!) and
``alpha[p,ell']`` are the lag coefficients. This is ``O(degree^2 * S log S)``
versus the ``O(S^2)`` quadratic recursion, with the identical result. It handles
``q >= 1`` uniformly (for ``q == 1`` the single multi-index per degree reduces
``T_ell' (x) y_p`` to the ordinary tensor power ``y^{(x)r}``).

This is a pure-NumPy path (no native DLL) and vectorizes over the path batch.
"""

import numpy as np

from ._volterra_conv import _enumerate_multiindices


def next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def fft_nfft(S):
    """FFT length for the causal convolution of ``S`` sources with ``S+1`` lags."""
    return next_pow2(S + (S + 1) - 1)


def basis_rhos(order, betas):
    """Deduplicated, sorted basis exponents for the higher-order scheme.

    order 0: {0}; order 1: {0} u {beta_p} u {1}; order 2: also {beta_p+1} u {2}.
    """
    betas = [float(b) for b in np.atleast_1d(np.asarray(betas, dtype=np.float64)).ravel()]
    if order == 0:
        return [0.0]
    cand = [0.0] + betas + [1.0]
    if order == 2:
        cand += [b + 1.0 for b in betas] + [2.0]
    return sorted(set(cand))


def cheb_lobatto(n):
    """Chebyshev-Lobatto interpolation nodes on ``[0, 1]`` (both endpoints)."""
    if n == 1:
        return np.array([0.0])
    k = np.arange(n, dtype=np.float64)
    return (1.0 - np.cos(np.pi * k / (n - 1))) / 2.0


def interp_inverse(h, thetas, rhos):
    """Inverse of the basis->point-evaluation matrix ``V[a,b] = (theta_a h)^rho_b``."""
    B = len(rhos)
    V = np.zeros((B, B), dtype=np.float64)
    for a, th in enumerate(thetas):
        for b, rho in enumerate(rhos):
            V[a, b] = 1.0 if rho == 0.0 else (th * h) ** rho
    return np.linalg.inv(V)


def _build_shuffle_monomials(y, ell, degree, q, m, dtype):
    """Normalized shuffle tensors T_ell over the batch+time axes.

    ``y`` has shape ``(B, S, q, m)``; returns a list ``T`` indexed by the packed
    multi-index, where ``T[idx]`` has shape ``(B, S, m**deg)`` built by the
    leading-letter recursion ``T_ell = sum_p y_p (x)_lead T_{ell-e_p}``.
    """
    B, S = y.shape[0], y.shape[1]
    deg = ell.sum(axis=1)
    index = {tuple(int(v) for v in ell[i]): i for i in range(ell.shape[0])}
    T = [None] * ell.shape[0]
    T[0] = np.ones((B, S, 1), dtype=dtype)
    for d in range(1, degree + 1):
        md, md1 = m ** d, m ** (d - 1)
        for idx in np.where(deg == d)[0]:
            i = int(idx)
            acc = np.zeros((B, S, md), dtype=dtype)
            accr = acc.reshape(B, S, m, md1)
            row = ell[i]
            for p in range(q):
                if row[p] == 0:
                    continue
                prev = list(int(v) for v in row)
                prev[p] -= 1
                src = T[index[tuple(prev)]]                    # (B, S, m^{d-1})
                accr += y[:, :, p, :][:, :, :, None] * src[:, :, None, :]
            T[i] = acc
    return T


def _fft_inputs(dX, A, degree, q, m, dtype):
    """Shared preamble for the FFT paths: project increments and build the
    multi-index layout and normalized shuffle monomials.

    Returns ``(y, level, T, np_dtype)`` where ``y`` is ``(B, S, q, m)``,
    ``level[d]`` lists the packed multi-indices of degree ``d``, and ``T`` are
    the shuffle monomials by multi-index.
    """
    np_dtype = np.dtype(dtype)
    dX = np.ascontiguousarray(np.asarray(dX, dtype=np_dtype))
    A = np.asarray(A, dtype=np_dtype)
    y = np.einsum("qmd,bsd->bsqm", A, dX)                      # (B, S, q, m)
    ell = _enumerate_multiindices(q, degree - 1)               # (M, q), native order
    deg = ell.sum(axis=1)
    level = [np.where(deg == dd)[0] for dd in range(degree)]   # multi-indices by degree
    T = _build_shuffle_monomials(y, ell, degree - 1, q, m, np_dtype)
    return y, level, T, np_dtype


def volterra_fft_terminal(dX, A, W, *, degree, q, m, scalar_term, dtype, nfft):
    """Terminal Volterra signature via the FFT scheme.

    :param dX: Increments, shape ``(B, S, d)``.
    :param A: Projection, shape ``(q, m, d)``.
    :param W: ``rfft`` of the lag coefficients over the lag axis, shape
        ``(nfreq, q, M)`` in native multi-index order (path-independent, so it
        is precomputed and cached by the caller).
    :param nfft: FFT length used to build ``W`` (``fft_nfft(S)``).
    :return: Flat signatures, shape ``(B, sig_len)`` (``sig_len`` includes the
        leading scalar term iff ``scalar_term``).
    """
    y, level, T, np_dtype = _fft_inputs(dX, A, degree, q, m, dtype)
    B, S = y.shape[0], y.shape[1]
    out_len = S + 1
    cplx = np.complex64 if np_dtype == np.float32 else np.complex128

    hist = [np.ones((B, out_len, 1), dtype=np_dtype)]          # level 0
    for L in range(1, degree + 1):
        mL, mL1 = m ** L, m ** (L - 1)
        # Accumulate in the frequency domain (irfft is linear), so each level
        # costs one inverse transform instead of one per (r, ell', p) term.
        freq = np.zeros((B, nfft // 2 + 1, mL), dtype=cplx)
        for r in range(1, L + 1):
            h_hn = hist[L - r][:, :S, :]                       # (B, S, m^{L-r})
            for idx in level[r - 1]:                           # |ell'| = r-1
                i = int(idx)
                base = (h_hn[:, :, :, None] * T[i][:, :, None, :]).reshape(B, S, mL1)
                for p in range(q):
                    src = (base[:, :, :, None] * y[:, :, p, :][:, :, None, :]).reshape(B, S, mL)
                    freq += np.fft.rfft(src, n=nfft, axis=1) * W[:, p, i][None, :, None]
        hist.append(np.fft.irfft(freq, n=nfft, axis=1)[:, :out_len])

    flat = np.concatenate([hist[L][:, S, :] for L in range(degree + 1)], axis=1)
    return flat if scalar_term else flat[:, 1:]


def volterra_fft_basis(dX, A, Wt, Wo, *, degree, q, m, thetas, interp_inv,
                       scalar_term, dtype, nfft):
    """Terminal Volterra signature via the higher-order basis-expansion FFT scheme.

    ``B = len(thetas)`` basis components are evaluated at the ``B`` interpolation
    points and interpolated per level; the source for output level ``L``, local
    order ``r``, basis ``b``, multi-index ``ell'`` (degree ``r-1``) and component
    ``p`` is ``C^(b)_{ell'-..} (x) T_ell' (x) y_p`` convolved with the rho_b lag
    weights at that interpolation point.

    :param Wt: ``Wt[k][b]`` is the ``rfft`` (shape ``(nfreq, q, M)``) of the lag
        coefficients for interpolation point ``theta_k`` and basis component
        ``b`` (out_len ``S``); :param Wo: ``Wo[b]`` the same at ``theta=0`` for
        the terminal readout (out_len ``S+1``). Both are sliced from the same
        ``S+1``-lag rfft, so a single ``nfft`` serves both.
    """
    y, level, T, np_dtype = _fft_inputs(dX, A, degree, q, m, dtype)
    nb, S = y.shape[0], y.shape[1]
    B = len(thetas)
    cplx = np.complex64 if np_dtype == np.float32 else np.complex128
    tables = Wt + [Wo]

    def eval_level(L, comp):
        """Readout level L at every weight table (B thetas + terminal) in one pass.

        The source ``comp[b] (x) T_ell' (x) y_p`` (and its forward rfft) is
        independent of the weight table, so it is built once and reused across
        all tables. Each table accumulates in the frequency domain (irfft is
        linear), so a level costs one inverse transform per table instead of
        one per (r, ell', p) term.
        """
        mL, mL1 = m ** L, m ** (L - 1)
        freqs = [np.zeros((nb, nfft // 2 + 1, mL), dtype=cplx) for _ in tables]
        for n in range(1, L + 1):
            for idx in level[n - 1]:                           # |ell'| = n-1
                i = int(idx)
                Ti = T[i]
                for b in range(B):
                    base = (comp[b][L - n][:, :, :, None] * Ti[:, :, None, :]).reshape(nb, S, mL1)
                    for p in range(q):
                        src = (base[:, :, :, None] * y[:, :, p, :][:, :, None, :]).reshape(nb, S, mL)
                        SRC = np.fft.rfft(src, n=nfft, axis=1)
                        for freq, Wtab in zip(freqs, tables):
                            freq += SRC * Wtab[b][:, p, i][None, :, None]
        return [np.fft.irfft(freq, n=nfft, axis=1)[:, :S + 1] for freq in freqs]

    # Per level: evaluate at the B interpolation points and the terminal
    # (theta=0) table together, solve the interpolation system for the basis
    # coefficients, and read the terminal signature at sample S.
    comp = [[np.ones((nb, S, 1), dtype=np_dtype) if b == 0
             else np.zeros((nb, S, 1), dtype=np_dtype)] for b in range(B)]
    terminal = [np.ones((nb, 1), dtype=np_dtype)]
    for L in range(1, degree + 1):
        accs = eval_level(L, comp)
        evals = np.stack([a[:, :S] for a in accs[:B]], axis=0)  # (B, nb, S, m^L)
        coeffs = np.einsum("ba,a...->b...", interp_inv, evals)
        for b in range(B):
            comp[b].append(coeffs[b])
        terminal.append(accs[B][:, S, :])
    flat = np.concatenate(terminal, axis=1)
    return flat if scalar_term else flat[:, 1:]


def build_fft_tables(conv_kind, conv_params, *, dt, degree, S, q, order, dtype):
    """Build (and return) the path-independent FFT weight tables for the scheme.

    Returns a dict with ``kind`` ``"order0"`` (keys ``W``, ``nfft`` for the
    single-pass terminal path) or ``"basis"`` (keys ``Wt``, ``Wo``, ``thetas``,
    ``interp_inv``, ``nfft`` for the basis-expansion path). The caller caches the
    result by ``S``.
    """
    from ._volterra_conv import convolution_lag_coefficients

    np_dtype = np.dtype(dtype)
    nfft = fft_nfft(S)

    def rfft_alpha(rho, theta):
        alpha, _M = convolution_lag_coefficients(
            conv_kind, conv_params, dt=dt, degree=degree, S=S, dtype=np_dtype,
            rho=rho, theta=theta)
        return np.fft.rfft(np.asarray(alpha, np_dtype).reshape(S + 1, q, -1), n=nfft, axis=0)

    if order == 0:
        return {"kind": "order0", "W": rfft_alpha(0.0, 0.0), "nfft": nfft}

    rhos = basis_rhos(order, conv_params["beta"])
    thetas = cheb_lobatto(len(rhos))
    Wt = [[rfft_alpha(rhos[b], th) for b in range(len(rhos))] for th in thetas]
    Wo = [rfft_alpha(rhos[b], 0.0) for b in range(len(rhos))]
    return {"kind": "basis", "Wt": Wt, "Wo": Wo, "thetas": thetas,
            "interp_inv": interp_inverse(float(dt), thetas, rhos), "nfft": nfft}
