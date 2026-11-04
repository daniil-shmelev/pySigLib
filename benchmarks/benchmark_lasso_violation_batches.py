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

"""Compare candidate selection policies on degree-three signature Lasso.

Run with python benchmarks/benchmark_lasso_violation_batches.py from the repo root.
Use --case correlated for correlated Gaussian paths or --case arabic for the
saved training split. Each row describes one timed fit after a warm-up.
Timings exclude construction of the design and envelopes and the full KKT audit.
"""

import argparse
import csv
import itertools
import math
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits


POLICIES = ["top1", "top4", "top16", "top128", "0.03", "0.1", "all",
            "support", "gain16", "gainmass", "lookahead", "hybrid", "0.25", "0.5",
            "effective", "diverse", "feedback", "drift", "agreement", "interference"]


def design(dx):
    n, _, channels = dx.shape
    variation = abs(dx).max(axis=2).sum(axis=1)
    dx = dx / np.where(variation > 0, variation, 1)[:, None, None]
    speed = abs(dx).max(axis=2)
    remaining = speed[:, ::-1].cumsum(axis=1)[:, ::-1]
    s1 = np.zeros((n, channels))
    s2 = np.zeros((n, channels, channels))
    s3 = np.zeros((n, channels, channels, channels))
    for step in dx.transpose(1, 0, 2):
        square = np.einsum("ni,nj->nij", step, step)
        s3 += (np.einsum("nij,nk->nijk", s2, step)
               + np.einsum("ni,njk->nijk", s1, square) / 2
               + np.einsum("ni,njk->nijk", step, square) / 6)
        s2 += np.einsum("ni,nj->nij", s1, step) + square / 2
        s1 += step
    X = np.column_stack((s1, 2*s2.reshape(n, -1), 6*s3.reshape(n, -1)))
    X -= X.mean(axis=0)

    # The degree-one prefix is affine on a segment. Interpolate the
    # absolute endpoint values for a polynomial upper bound on its modulus.
    pos = dx.cumsum(axis=1)
    envelopes = []
    for degree in (1, 2):
        layer = []
        for word in itertools.product(range(channels), repeat=degree):
            moments = []
            for q in range(4-degree):
                integral = np.zeros(n)
                for t in (0.5-0.5/math.sqrt(3), 0.5+0.5/math.sqrt(3)):
                    end = pos[:, :, word[0]]
                    start = end - dx[:, :, word[0]]
                    prefix = 1 if degree == 1 else (1-t)*abs(start) + t*abs(end)
                    integral += .5*(prefix*abs(dx[:, :, word[-1]])*(remaining-t*speed)**q).sum(axis=1)
                moments.append(integral*math.factorial(degree+q)/math.factorial(q))
            layer.append(moments)
        envelopes.append(np.array(layer))
    return X, envelopes


def data(seed, channels, case):
    rng = np.random.default_rng(seed)
    n = 240
    dx = rng.normal(size=(n, 12, channels))
    if case == "correlated":
        dx[:, :, 1::2] = .8*dx[:, :, :channels-1:2] + .6*dx[:, :, 1::2]
    if case == "anisotropic":
        dx[:, :, 3:] *= .2
    X, envelopes = design(dx)
    signal_cols = X[:, [0, channels+channels+2, channels+channels**2+2*channels**2+channels]]
    signal = (signal_cols/signal_cols.std(axis=0)) @ np.array([.9, -.7, .45])
    binary = rng.binomial(1, expit(signal))
    counts = rng.poisson(np.exp(.45*signal))
    continuous = signal + .5*rng.standard_t(3, n)
    return X, envelopes, binary, continuous, counts


def select_new_policy(policy, take, ids, corr, excess, X, r, active, lam, state):
    order = take[np.argsort(-excess[take], kind="stable")]
    strongest = order[:1]
    if policy == "effective":
        v = excess[take]
        return order[:math.ceil(v.sum()**2 / (v @ v))]
    if policy == "feedback":
        budget = state.get("budget", 8)
        if "previous" in state:
            used = np.isin(state["previous"], active).mean()
            if used >= .75:
                budget = min(128, 2*budget)
            elif used < .25:
                budget = max(2, budget//2)
        result = order[:budget]
        state.update(previous=ids[result], budget=budget)
        return result
    if policy == "drift":
        previous = state.get("r")
        state["r"] = r
        if previous is None:
            return order[:16]
        margin = abs((r-previous) @ X[:, ids[take]])/len(r)
        return np.union1d(strongest, take[excess[take] > margin])
    if policy == "agreement":
        if "split" not in state:
            split = np.random.default_rng(19).permutation(len(r)) < len(r)//2
            state["split"] = np.where(split, 1., -1.)
        margin = abs((r*state["split"]) @ X[:, ids[take]])/len(r)
        return np.union1d(strongest, take[excess[take] > margin])

    shortlist = order[:64]
    Z = X[:, ids[shortlist]]
    norm2 = np.einsum("ij,ij->j", Z, Z)/len(r)
    if policy == "diverse":
        gram = (Z.T @ Z)/len(r)
        available = np.ones(len(shortlist), dtype=bool)
        kept = []
        for j in range(len(shortlist)):
            if available[j]:
                kept.append(j)
                available &= gram[j]**2 <= .7**2*norm2[j]*norm2
                if len(kept) == 16:
                    break
        return shortlist[kept]

    # Fit a quadratic along a simultaneous soft-thresholded direction.
    sign = np.sign(corr[shortlist])
    v = excess[shortlist]
    delta = sign*v/norm2
    move = Z @ delta
    energy = (move @ move)/len(r)
    if energy <= 0:
        return strongest
    step = (v @ abs(delta))/energy
    peer_effect = (move @ Z)/len(r) - norm2*delta
    return np.union1d(strongest, shortlist[v > step*sign*peer_effect])


def query(X, envelopes, r, eligible, lam, tol, policy, active, norm2, state):
    n = len(r)
    d = len(envelopes[0])
    weights = abs(r-r.mean())/n
    first = (envelopes[0] @ weights).max(axis=1) >= lam-1e-12
    second = first.repeat(d) & ((envelopes[1] @ weights).max(axis=1) >= lam-1e-12)
    ids = np.flatnonzero(eligible & np.r_[first, second, second.repeat(d)])
    return select_candidates(X, r, ids, lam, tol, policy, active, norm2, state)


def select_candidates(X, r, ids, lam, tol, policy, active, norm2, state):
    n = len(r)
    corr = r @ X[:, ids]/n
    excess = abs(corr)-lam
    take = np.flatnonzero(excess > tol)
    if take.size:
        if policy in ("effective", "diverse", "feedback", "drift", "agreement", "interference"):
            take = select_new_policy(policy, take, ids, corr, excess, X, r, active, lam, state)
        elif policy.startswith("top"):
            take = take[np.argsort(-excess[take], kind="stable")[:int(policy[3:])]]
        elif policy in ("support", "gain16", "gainmass", "lookahead", "hybrid"):
            budget = max(4, 2*len(active))
            if policy in ("support", "hybrid"):
                if policy == "hybrid":
                    take = take[excess[take] >= .03*excess[take].max()]
                take = take[np.argsort(-excess[take], kind="stable")[:budget]]
            else:
                missing = ids[take[np.isnan(norm2[ids[take]])]]
                columns = X[:, missing]
                norm2[missing] = np.einsum("ij,ij->j", columns, columns)/n
                gain = excess[take]**2/norm2[ids[take]]
                order = np.argsort(-gain, kind="stable")
                if policy == "gain16":
                    take = take[order[:16]]
                elif policy == "gainmass":
                    gains = gain[order].cumsum()
                    size = gains.searchsorted(.9*gains[-1]) + 1
                    take = take[order[:size]]
                else:
                    shortlist = take[order[:64]]
                    Z = X[:, ids[shortlist]]
                    sign = np.sign(corr[shortlist])
                    projected = corr[shortlist].copy()
                    curvature = norm2[ids[shortlist]]
                    selected = []
                    for _ in range(min(16, len(shortlist))):
                        remaining = np.maximum(sign*projected-lam, 0)
                        remaining[selected] = 0
                        j = np.argmax(remaining**2/curvature)
                        if remaining[j] <= tol:
                            break
                        selected.append(j)
                        projected -= (sign[j]*remaining[j]/curvature[j])*(Z.T @ Z[:, j])/n
                    take = shortlist[selected]
        elif policy != "all":
            take = take[excess[take] >= float(policy)*excess[take].max()]
        take.sort()  # Give identical candidate sets to the solver in the same order.
    return ids[take], np.sign(corr[take])


def intercept_only(y, kind):
    if kind == "logistic":
        return logit(y.mean())
    if kind == "poisson":
        return np.log(y.mean())
    if kind == "huber":
        return brentq(lambda b: np.clip(b-y, -1., 1.).mean(), y.min()-1, y.max()+1)
    return y.mean()


def residual(y, eta, kind):
    if kind == "logistic":
        return y-expit(eta)
    if kind == "poisson":
        return y-np.exp(eta)
    if kind == "huber":
        return np.clip(y-eta, -1., 1.)
    return y-eta


def fit(X, envelopes, y, kind, lam, policy, tol, search=query):
    n, p = X.shape
    beta = np.zeros(p)
    b = intercept_only(y, kind)
    pool = np.zeros(p, dtype=bool)
    norm2 = np.full(p, np.nan)
    state = {}
    scans, admitted, peak = 0, 0, 0
    for _ in range(2000):
        eta = b+X@beta
        r = residual(y, eta, kind)
        active = np.flatnonzero(beta)
        if kind == "squared":
            pending = np.flatnonzero(pool & (beta == 0))
            corr = r @ X[:, pending]/n
            if pending.size and abs(corr).max() > lam+tol:
                j = abs(corr).argmax()
                ids, signs = pending[[j]], np.sign(corr[[j]])
            else:
                scans += 1
                ids, _ = search(X, envelopes, r, ~pool, lam, tol, policy, active, norm2, state)
                if not ids.size:
                    break
                pool[ids] = True
                admitted += len(ids)
                peak = max(peak, len(ids))
                continue
        else:
            scans += 1
            ids, signs = search(X, envelopes, r, beta == 0, lam, tol, policy, active, norm2, state)
            if not ids.size:
                break
            admitted += len(ids)
            peak = max(peak, len(ids))
        indices = np.r_[active, ids]
        signs = np.r_[np.sign(beta[active]), signs]
        if kind == "squared":
            # Exact signed quadratic steps on the current face.
            while indices.size:
                Z = X[:, indices]
                old = beta[indices]
                new = np.linalg.solve(Z.T@Z, Z.T@(y-b)-n*lam*signs)
                hit = (signs*new < 0)
                if not hit.any():
                    beta[indices] = new
                    break
                step = np.min(old[hit]/(old[hit]-new[hit]))
                current = old+step*(new-old)
                keep = abs(current) > 1e-12
                beta[indices] = np.where(keep, current, 0)
                indices, signs = indices[keep], signs[keep]
        else:
            Z = X[:, indices]*signs

            def objective(u):
                eta = u[0]+Z@u[1:]
                if kind == "logistic":
                    deriv = expit(eta)-y
                    terms = np.logaddexp(0, eta)-y*eta
                elif kind == "poisson":
                    mu = np.exp(eta)
                    deriv, terms = mu-y, mu-y*eta
                else:
                    dev = eta-y
                    deriv = np.clip(dev, -1., 1.)
                    terms = deriv*(dev-.5*deriv)
                value = terms.mean()+lam*u[1:].sum()
                return value, np.r_[deriv.mean(), Z.T@deriv/n+lam]

            bounds = [(None, None)]+[(0, None)]*len(indices)
            options = {"ftol": 0., "gtol": 1e-9, "maxiter": 2500, "maxls": 60}
            opt = minimize(objective, np.r_[b, abs(beta[indices])], jac=True,
                           method="L-BFGS-B", bounds=bounds, options=options)
            u = opt.x
            if signed_kkt(u, opt.jac) > tol:
                scale = np.r_[1., np.sqrt(n)/np.linalg.norm(Z, axis=0)]

                def scaled_objective(v):
                    value, gradient = objective(scale*v)
                    return value, scale*gradient

                opt = minimize(scaled_objective, u/scale, jac=True, method="L-BFGS-B",
                               bounds=bounds, options=options)
                u = scale*opt.x
                if signed_kkt(u, opt.jac/scale) > tol:
                    raise ArithmeticError("Signed subproblem failed its KKT check")
            b = u[0]
            beta[indices] = u[1:]*signs
    else:
        raise RuntimeError("Outer iteration limit reached")
    return b, beta, scans, admitted, peak


def signed_kkt(u, grad):
    error = np.where(u[1:] > 0, abs(grad[1:]), np.maximum(0., -grad[1:]))
    return max(abs(grad[0]), error.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("synthetic", "correlated", "anisotropic", "arabic"), default="synthetic")
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--digits", nargs="+", type=int, default=[0, 4, 8])
    parser.add_argument("--ratios", nargs="+", type=float, default=[.12, .03])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7201, 7202, 7203])
    parser.add_argument("--losses", nargs="+", choices=("squared", "logistic", "poisson", "huber"),
                        default=["squared", "logistic"])
    args = parser.parse_args()
    if args.channels < 3 or args.repeats < 1:
        parser.error("At least three channels and one timed repetition are required.")
    if any(digit not in range(10) for digit in args.digits):
        parser.error("Arabic digit indices must be between 0 and 9.")
    if not all(np.isfinite(ratio) and ratio > 0 for ratio in args.ratios):
        parser.error("Penalty ratios must be finite and positive.")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["case","channels","seed_or_digit","loss","ratio","policy","rep","ms","scans","admitted","max_batch","nnz","kkt","prediction_difference","support_difference"])
    policies = args.policies
    with threadpool_limits(limits=1):
        if args.case == "arabic":
            with np.load("sig_branch_and_bound_experiments/spoken_arabic_digit/results/normalized/integral_comparison/prepared.npz") as saved:
                paths = saved["train"]
                responses = saved["response"]
            X, envelopes = design(np.diff(paths.astype(np.float64), axis=1))
        for seed in args.digits if args.case == "arabic" else args.seeds:
            if args.case == "arabic":
                binary = continuous = counts = (responses.argmax(axis=1) == seed).astype(float)
            else:
                X, envelopes, binary, continuous, counts = data(seed, args.channels, args.case)
            for kind in args.losses:
                y = {"squared": continuous, "huber": continuous, "poisson": counts, "logistic": binary}[kind]
                for ratio in args.ratios:
                    r0 = residual(y, intercept_only(y, kind), kind)
                    lam = ratio*abs(X.T@r0/len(y)).max()
                    reference = None
                    for rep in range(args.repeats+1):
                        order = np.random.default_rng([seed, rep]).permutation(policies) if rep else policies
                        for policy in order:
                            start = perf_counter()
                            b, beta, scans, admitted, peak = fit(X, envelopes, y, kind, lam, policy, 1e-7)
                            ms = 1000*(perf_counter()-start)
                            eta = b+X@beta
                            r = residual(y, eta, kind)
                            corr = r @ X/len(y)
                            kkt = max(abs(r.mean()), np.where(beta == 0, np.maximum(abs(corr)-lam, 0),
                                                              abs(corr-lam*np.sign(beta))).max())
                            if not np.isfinite(kkt) or kkt > 1e-7:
                                raise ArithmeticError((seed, kind, ratio, policy, kkt))
                            if reference is None:
                                reference = eta
                                reference_support = beta != 0
                            if rep:
                                writer.writerow([args.case,len(envelopes[0]),seed,kind,ratio,policy,rep,ms,scans,admitted,
                                                 peak,np.count_nonzero(beta),kkt,max(abs(eta-reference)),
                                                 np.count_nonzero(reference_support != (beta != 0))])


if __name__ == "__main__":
    main()
