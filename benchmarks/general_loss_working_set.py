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

"""CPU research solver for L(eta) + alpha * ||beta||_1. L returns a value and its prediction gradient."""

import ctypes as ct
from math import factorial
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

import pysiglib
from benchmarks._prototype_lasso_types import LassoOptions, LassoStats
from benchmarks.lasso_feature_reference import hall_basis


def tensor_index(word, d):
    code = 0
    for a in word:
        code = code * d + a + 1
    return code - 1


class PackedSearch:
    def __init__(self, paths, degree, mode, jobs, storage):
        self.paths, self.degree, self.mode, self.jobs = paths, degree, mode, jobs
        n, length, d = paths.shape
        self.d = d
        self.width = sum(d**k for k in range(1, degree + 1))
        self.tile = max(1, min(n, (1 << 20) // (8 * (4*self.width + 3*d*length))))
        self.words, self.map = None, None
        if mode == 'sig':
            p = self.width
            self.scales = np.repeat([float(factorial(k)) for k in range(1, degree + 1)],
                                    [d**k for k in range(1, degree + 1)])
        else:
            if mode == 'log':
                self.words = pysiglib.lyndon_words(d, degree)
                self.map = [tensor_index(w, d) for w in self.words]
            else:
                self.words, duals = hall_basis(d, degree, local_cache=True)
                offsets, indices, values = [0], [], []
                for w in self.words:
                    for term, coefficient in duals[w].items():
                        indices.append(tensor_index(term, d))
                        values.append(coefficient)
                    offsets.append(len(values))
                self.map = csr_matrix((values, indices, offsets), shape=(len(self.words), self.width))
            self.scales = np.array([float(factorial(len(w))) for w in self.words])
            p = len(self.words)
        self.compact = storage in ('compact','refine','half')
        self.pilot = storage in ('refine','half')
        self.stats = LassoStats()
        self.pivot = storage == 'pivot'
        if storage in ('buffer','pivot'):
            self.cache_ids = np.empty(0, dtype=int)
            self.cache_X = np.empty((0,n))
            self.cap = min(p, max(64, min(4*int(np.ceil(np.sqrt(p))), (2 << 20)//(8*n))))
            self.anchor = self.norm = None
            return
        dtype = np.float16 if storage == 'half' else np.float32 if self.compact else np.float64
        self.X = np.empty((n, p), dtype=dtype)
        self.peak_value = np.zeros(p) if self.compact else None
        self.roundoff = np.zeros(p) if self.compact else None
        for start, block in self.tiles():
            dest = self.X[start:start+len(block)]
            dest[:] = block
            if self.compact:
                np.maximum(self.peak_value, np.max(np.abs(block), axis=0), out=self.peak_value)
                np.maximum(self.roundoff, np.abs(block-dest).max(axis=0), out=self.roundoff)
        if not np.isfinite(self.X).all():
            raise FloatingPointError('Nonfinite feature dictionary')

    def tiles(self):
        for start in range(0, len(self.paths), self.tile):
            path = self.paths[start:start+self.tile]
            v = abs(np.diff(path, axis=1)).max(axis=2).sum(axis=1)
            normed = (path-path[:, :1]) / np.where(v > 0, v, 1.)[:, None, None]
            if self.degree == 1:
                yield start, normed[:,-1]
                continue
            if self.d == 1 and self.mode == 'sig':
                yield start, normed[:,-1]**np.arange(1,self.degree+1)
                continue
            if self.mode == 'sig':
                block = pysiglib.sig(normed, self.degree, n_jobs=self.jobs)
            else:
                tensor = pysiglib.log_sig(normed, self.degree, method=0, n_jobs=self.jobs)
                block = tensor[:, self.map] if self.mode == 'log' else (self.map @ tensor.T).T
            block *= self.scales
            yield start, block

    def word(self, index):
        if self.words is not None:
            return self.words[index]
        word = []
        code = index + 1
        while code:
            code, a = divmod(code-1, self.d)
            word.append(a)
        return tuple(reversed(word))

    def query(self, grad, threshold, active, capacity):
        g = grad - grad.mean()
        if hasattr(self, 'cache_ids'):
            return self.buffered(g, threshold, active, capacity)
        if self.compact:
            # Bound quantization and the float32 dot product. Recheck candidates in double.
            eps, tiny = np.finfo(np.float32).eps, np.finfo(np.float32).tiny
            factor = (2*len(g)+8)*eps
            g32 = g.astype(np.float32)
            if self.X.dtype == np.float16:
                acc = np.zeros(self.X.shape[1],dtype=np.float32)
                for i in range(0,len(g),self.tile):
                    acc += g32[i:i+self.tile] @ self.X[i:i+self.tile].astype(np.float32)
                s = np.abs(acc).astype(np.float64)
            else:
                s = np.abs(g32 @ self.X).astype(np.float64)
            scale = np.linalg.norm(g, 1)
            if factor >= .5 or not np.all(np.isfinite(s)):
                s.fill(np.inf)
            elif not self.pilot:
                s += (self.roundoff + (self.peak_value+self.roundoff)*(factor/(1-factor)))*scale
                s += tiny*len(g)*(1+scale+self.peak_value)
        else:
            s = abs(g @ self.X)
        if active:
            s[list(active)] = -np.inf
        candidates = np.flatnonzero(s > threshold)
        while candidates.size:
            if candidates.size > capacity:
                batch = candidates[np.argpartition(-s[candidates], capacity-1)[:capacity]]
            else:
                batch = candidates
            batch = np.sort(batch)
            if self.compact:
                cols = self.X[:,batch].T.astype(np.float64) if self.pilot else self.exact_columns(batch)
                keep = abs(cols @ g) > threshold
                if not keep.any():
                    candidates = np.setdiff1d(candidates, batch, assume_unique=True)
                    continue
                return list(batch[keep]), cols[keep]
            return list(batch), self.X[:, batch].T.astype(np.float64,copy=False)
        return [], np.empty((0, len(g)))

    def exact_columns(self, ids):
        cols = np.empty((len(ids),len(self.paths)))
        if len(ids):
            for start, block in self.tiles():
                cols[:, start:start+len(block)] = block[:,ids].T
        return cols

    def buffered(self, g, threshold, active, capacity):
        while True:
            scores = abs(self.cache_X @ g)
            if active:
                scores[np.isin(self.cache_ids,active)] = 0.
            ready = np.flatnonzero(scores > threshold)
            if ready.size:
                if ready.size > capacity:
                    ready = ready[np.argpartition(-scores[ready],capacity-1)[:capacity]]
                return list(self.cache_ids[ready]), self.cache_X[ready]
            if (self.anchor is not None and self.excluded.issubset(active)
                    and np.linalg.norm(g-self.anchor) < self.radius):
                return [], np.empty((0,len(g)))
            if self.pivot and self.anchor is not None:
                norm2 = self.anchor @ self.anchor
                t = (g @ self.anchor)/norm2 if norm2 else 0.
                delta = np.linalg.norm(g-t*self.anchor)
                safety = 64*len(g)*np.finfo(float).eps*(1+np.linalg.norm(g)+abs(t)*np.linalg.norm(self.anchor))
                upper = abs(t*self.anchor_scores) + self.norm*(delta+safety)
                upper[self.cache_ids] = 0.
                if active:
                    upper[list(active)] = 0.
                if upper.max(initial=0.) < threshold:
                    return [], np.empty((0,len(g)))
            corr = np.zeros(len(self.scales))
            squares = np.zeros_like(corr) if self.norm is None else None
            for start, block in self.tiles():
                corr += g[start:start+len(block)] @ block
                if squares is not None:
                    squares += np.einsum('ij,ij->j',block,block)
            if squares is not None:
                self.norm = np.sqrt(squares)*(1+1e-12)
            if self.pivot:
                self.anchor_scores = corr.copy()
            if active:
                corr[list(active)] = 0.
            if abs(corr).max() <= threshold:
                return [], np.empty((0,len(g)))
            # Distance in gradient space to the nearest omitted KKT boundary.
            with np.errstate(over='ignore'):
                distance = (threshold-abs(corr))/np.maximum(self.norm,np.finfo(float).tiny)
            if active:
                distance[list(active)] = np.inf
            ranked = np.argsort(distance,kind='stable')
            self.cache_ids = np.sort(ranked[:self.cap])
            self.radius = (distance[ranked[self.cap]] if self.cap < len(ranked) else np.inf)
            self.radius -= 64*len(g)*np.finfo(float).eps*(1+np.linalg.norm(g))
            self.anchor, self.excluded = g.copy(),set(active)
            self.cache_X = np.empty((len(self.cache_ids),len(g)))
            for start, block in self.tiles():
                self.cache_X[:, start:start+len(block)] = block[:,self.cache_ids].T
            # Both scans can differ by a few ulps at the numerical threshold.
            if not np.any(abs(self.cache_X @ g) > threshold):
                raise ArithmeticError('Buffered scan disagrees with retained column correlations')

    def close(self):
        pass


class TreeSearch:
    def __init__(self, paths, degree, mode, jobs, setting, cache):
        with os.add_dll_directory(str(Path('out/build/windows-cpu/siglib/cpsig/Release').resolve())):
            self.lib = ct.CDLL(str(Path('output/unified_general/bridge.dll').resolve()))
        self.lib.lookahead_open.argtypes = [ct.POINTER(LassoOptions), ct.c_void_p, ct.c_uint32]
        self.lib.lookahead_open.restype = ct.c_void_p
        self.lib.working_set_query.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_double, ct.c_void_p, ct.c_void_p,
                                              ct.c_size_t, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_size_t]
        self.lib.working_set_query.restype = ct.c_int64
        self.lib.lookahead_close.argtypes = [ct.c_void_p, ct.POINTER(LassoStats)]
        self.lib.lookahead_close.restype = None
        self.degree, self.n = degree, len(paths)
        self.stats = LassoStats()
        options = LassoOptions(*paths.shape, degree, 1, 3000, 64, cache, 4 << 30, 1e-8, jobs, 0, 1, 0, setting, 1)
        self.handle = self.lib.lookahead_open(ct.byref(options), paths.ctypes.data, ('sig','log','hall').index(mode))
        if not self.handle:
            raise RuntimeError('Native tree preparation failed')

    def query(self, grad, threshold, active, capacity):
        words = np.empty((len(active), self.degree), dtype=np.uint64)
        lengths = np.empty(len(active), dtype=np.uint64)
        for i, w in enumerate(active):
            words[i, :len(w)], lengths[i] = w, len(w)
        out = np.empty((capacity, self.degree), dtype=np.uint64)
        size = np.empty(capacity, dtype=np.uint64)
        cols = np.empty((capacity, self.n))
        # The native search divides by n. grad is the gradient of the scalar loss.
        g = np.ascontiguousarray(grad * self.n)
        k = self.lib.working_set_query(self.handle, g.ctypes.data, threshold, words.ctypes.data,
                                       lengths.ctypes.data, len(active), out.ctypes.data, size.ctypes.data,
                                       cols.ctypes.data, capacity)
        if k < 0:
            raise ArithmeticError('Native tree search could not certify the query')
        return [tuple(w[:length]) for w, length in zip(out[:k], size[:k])], cols[:k]

    @staticmethod
    def word(key):
        return key

    def close(self):
        self.lib.lookahead_close(self.handle, ct.byref(self.stats))


def restricted(loss, X, b, beta, lam, tol, feature_tol):
    # Scale variables, including the corresponding penalties, for the inner solve.
    n = X.shape[1]
    scale = np.sqrt(n)/np.linalg.norm(X, axis=1) if len(X) else np.empty(0)
    Z = X * scale[:, None]
    penalty = lam*scale
    k = len(scale)

    def evaluate(u):
        value, grad = loss(u[0] + (u[1:1+k]-u[1+k:]) @ Z)
        corr = Z @ grad
        return value + penalty @ (u[1:1+k]+u[1+k:]), np.r_[grad.sum(), corr+penalty, -corr+penalty]

    origin = np.r_[b, np.maximum(beta, 0)/scale, np.maximum(-beta, 0)/scale]
    bounds = [(None,None)] + [(0., None)]*(2*k)
    for _ in range(3):
        opt = minimize(evaluate, origin, method='L-BFGS-B', jac=True, bounds=bounds,
                       options=dict(ftol=0., gtol=min(1e-9, tol/10), maxiter=2000, maxls=60))
        b = opt.x[0]
        beta = (opt.x[1:1+k] - opt.x[1+k:])*scale
        value, grad = loss(b + beta @ X)
        corr = X @ grad
        error = np.where(beta != 0, abs(corr + lam*np.sign(beta)), np.maximum(abs(corr)-lam, 0.))
        if np.isfinite(value) and abs(grad.sum()) <= tol and error.max(initial=0.) <= feature_tol:
            return b, beta, grad
        origin = np.r_[b, np.maximum(beta, 0)/scale, np.maximum(-beta, 0)/scale]
    raise ArithmeticError('Restricted solve failed its KKT check')


def general_loss_fit(paths, degree, mode, loss, alpha, *, jobs=1, tol=1e-7):
    """Fit L(b + X beta) + alpha * ||beta||_1 on CPU paths of shape (n, length, d).

    loss(eta) returns a scalar value and its length-n NumPy gradient, including any
    averaging in L. The result contains words, coefficients and the intercept on
    the original normalized features. mode is 'sig', 'log' or 'hall'.
    """
    return _fit(paths, degree, mode, loss, alpha, jobs, tol, 'auto', 'adaptive', 16 << 20)


def _fit(paths, degree, mode, loss, alpha, jobs, tol, engine, admission, cache):
    paths = np.ascontiguousarray(paths, dtype=np.float64)
    if paths.ndim != 3 or min(paths.shape) < 1 or degree not in range(1, 171) or mode not in ('sig','log','hall'):
        raise ValueError('Invalid path, degree or dictionary')
    if not np.isfinite(paths).all():
        raise ValueError('Expected finite paths')
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError('Expected a positive finite penalty and tolerance')
    n, _, d = paths.shape
    if d == 1 and mode != 'sig':
        degree = 1
    if engine == 'auto':
        width = sum(d**k for k in range(1,degree+1))
        if width*8 > 1 << 20 or (mode == 'hall' and degree > 5):
            engine = 'tree'
        else:
            counts = [0]
            for k in range(1,degree+1):
                counts.append((d**k-sum(j*counts[j] for j in range(1,k) if k%j == 0))//k)
            columns = width if mode == 'sig' else sum(counts)
            engine = 'packed' if 8*n*columns <= 256 << 10 else 'pivot'
        if degree >= 4 and engine != 'packed':
            engine = ('second-' if 8*n*(d+d*d) <= 2 << 20 else 'first-')+engine
    selected_engine = engine
    stage = 0
    if engine.startswith(('first-', 'second-')):
        prefix,engine = engine.split('-',1)
        stage = min(degree-1, 1 if prefix == 'first' else 2)

    def full_search():
        if engine in ('packed', 'compact', 'buffer','pivot','refine','half'):
            return PackedSearch(paths, degree, mode, jobs, engine)
        setting = {'plain': 0, 'grouped': 1, 'signed': 2, 'both': 3,
                   'tree': 0 if mode == 'hall' else (3 if mode == 'sig' and degree >= 4 else 2)}[engine]
        return TreeSearch(paths, degree, mode, jobs, setting, cache)

    search = PackedSearch(paths, stage, mode, jobs, 'packed') if stage else full_search()
    keys, X, means, beta = [], np.empty((0, n)), np.empty(0), np.empty(0)
    scans = admitted = peak = 0
    # An absolute tolerance alone can be a material fraction of a small penalty.
    feature_tol = min(tol, 1e-4*alpha)
    b = 0.
    next_capacity = 16
    try:
        for _ in range(2000):
            b, beta, grad = restricted(loss, X, b, beta, alpha, tol/2, feature_tol/2)
            keep = beta != 0
            if not keep.all():
                keys = [key for key, use in zip(keys, keep) if use]
                X, means, beta = X[keep], means[keep], beta[keep]
            if admission == 'adaptive':
                cap = next_capacity
            elif admission == 'global':
                cap = len(search.scales)
            elif admission in ('lookahead','ties64','angle64'):
                cap = 64
            else:
                cap = 16 if admission in ('top16','ties','angle') else min(64, max(4,len(keys)))
            ids, cols = search.query(grad, alpha+feature_tol, keys, cap)
            scans += 1
            if not ids and stage:
                if engine not in ('packed','compact','buffer','pivot','refine','half'):
                    keys = [search.word(j) for j in keys]
                search.close()
                search = None
                search = full_search()
                stage = 0
                ids, cols = search.query(grad, alpha+feature_tol, keys, cap)
                scans += 1
            if not ids and getattr(search,'pilot',False):
                search.pilot = False
                X = search.exact_columns(keys)
                new_means = X.mean(axis=1)
                b += beta @ (new_means-means)
                means = new_means
                X -= means[:,None]
                continue
            if not ids:
                break
            centers = cols.mean(axis=1)
            cols -= centers[:, None]
            if admission in ('lookahead', 'global'):
                corr = -cols @ grad
                sign = np.sign(corr)
                norm = np.einsum('ij,ij->i', cols, cols)
                if admission == 'global' and len(ids) > 64:
                    gain = (abs(corr)-alpha)**2/np.maximum(norm,np.finfo(float).tiny)
                    ix = np.argsort(-gain,kind='stable')[:64]
                    corr, sign, norm, centers, cols = corr[ix], sign[ix], norm[ix], centers[ix], cols[ix]
                    ids = [ids[j] for j in ix]
                chosen = []
                for _ in range(min(16,len(ids))):
                    v = np.maximum(sign*corr-alpha, 0.)
                    v[chosen] = 0.
                    j = np.argmax(v**2/np.maximum(norm, np.finfo(float).tiny))
                    if v[j] <= feature_tol:
                        break
                    chosen.append(j)
                    corr -= sign[j]*v[j]*(cols @ cols[j])/norm[j]
                if not chosen:
                    raise ArithmeticError('Candidate search disagrees with column correlations')
            elif admission in ('adaptive','ties','ties64','angle','angle64'):
                score = abs(cols @ grad)
                order = np.argsort(-score,kind='stable')
                if admission.startswith('ties') or admission == 'adaptive':
                    gap = np.diff(-score[order])
                    distinct = order[np.r_[True, gap > 1e-10*score[order[0]]]]
                    count = len(distinct)
                    next_capacity = min(64,(16*len(ids)+count-1)//count) if len(ids)==cap else 16
                    chosen = distinct[:16]
                else:
                    gram = cols @ cols.T
                    norms = np.sqrt(gram.diagonal())
                    similar = abs(gram) >= .995*norms[:,None]*norms[None,:]
                    available = np.ones(len(ids),dtype=bool)
                    chosen = []
                    for j in order:
                        if available[j]:
                            chosen.append(j)
                            available &= ~similar[j]
                            if len(chosen) == 16:
                                break
            else:
                chosen = np.arange(len(ids))
            keys.extend(ids[j] for j in chosen)
            X = np.concatenate((X, cols[chosen]))
            means = np.r_[means, centers[chosen]]
            beta = np.r_[beta, np.zeros(len(chosen))]
            admitted += len(chosen)
            peak = max(peak,len(keys))
        else:
            raise ArithmeticError('Working set iteration limit reached')
        return dict(intercept=b-beta@means, coef=beta, words=[search.word(j) for j in keys],
                    scans=scans, admitted=admitted, peak_set=peak, engine=selected_engine, stats=search.stats)
    finally:
        if search is not None:
            search.close()
