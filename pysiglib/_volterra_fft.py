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
    np_dtype = np.dtype(dtype)
    dX = np.ascontiguousarray(np.asarray(dX, dtype=np_dtype))
    A = np.asarray(A, dtype=np_dtype)
    B, S, d = dX.shape
    y = np.einsum("qmd,bsd->bsqm", A, dX)                      # (B, S, q, m)
    out_len = S + 1

    ell = _enumerate_multiindices(q, degree - 1)               # (M, q), native order
    deg = ell.sum(axis=1)
    level = [np.where(deg == dd)[0] for dd in range(degree)]   # multi-indices by degree
    T = _build_shuffle_monomials(y, ell, degree - 1, q, m, np_dtype)

    hist = [np.ones((B, out_len, 1), dtype=np_dtype)]          # level 0
    for L in range(1, degree + 1):
        mL, mL1 = m ** L, m ** (L - 1)
        acc = np.zeros((B, out_len, mL), dtype=np_dtype)
        for r in range(1, L + 1):
            h_hn = hist[L - r][:, :S, :]                       # (B, S, m^{L-r})
            for idx in level[r - 1]:                           # |ell'| = r-1
                i = int(idx)
                base = (h_hn[:, :, :, None] * T[i][:, :, None, :]).reshape(B, S, mL1)
                for p in range(q):
                    src = (base[:, :, :, None] * y[:, :, p, :][:, :, None, :]).reshape(B, S, mL)
                    SRC = np.fft.rfft(src, n=nfft, axis=1)
                    acc += np.fft.irfft(SRC * W[:, p, i][None, :, None], n=nfft, axis=1)[:, :out_len]
        hist.append(acc)

    flat = np.concatenate([hist[L][:, S, :] for L in range(degree + 1)], axis=1)
    return flat if scalar_term else flat[:, 1:]
