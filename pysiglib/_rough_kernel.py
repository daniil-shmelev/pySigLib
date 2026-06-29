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
"""Finite state-space (sum-of-exponentials) approximation of the fractional
Volterra kernel

    K_beta(t) = t ** (beta - 1) / Gamma(beta),   beta in (1/2, 1).

The kernel is approximated by ``sum_r w_r exp(-x_r t)`` with real positive
nodes ``x_r`` and weights ``w_r``, so that the resulting diagonal finite
state-space realization ``1^T exp(-Lambda t) b ~= K_beta(t)`` can be fed to the
exact Volterra signature recursion. The relation to the rough-kernel notation
is ``H = beta - 1/2``, so ``K_H(t) = t ** (H - 1/2) / Gamma(H + 1/2)``.

The BL2 / "european" positive-Hurst quadrature logic is adapted from the public
implementation

    https://github.com/SimonBreneis/approximations_to_fractional_stochastic_volterra_equations/

specifically the positive-Hurst european/BL2 branch of
``RoughKernel.quadrature_rule``.

``scipy`` is required for the node optimization and is imported lazily by the
caller; it is not a hard dependency of pysiglib.
"""

import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.special import gamma, gammainc


def bl2_quadrature_rule(beta, R, T):
    """Return BL2/european exponential nodes and weights for ``K_beta``.

    :param beta: Fractional exponent in ``(1/2, 1)``.
    :param R: Number of exponential factors (state-space dimension).
    :param T: Approximation horizon.
    :return: ``(nodes, weights)``, both real arrays of shape ``(R,)`` sorted by
        increasing node.
    """
    _validate_beta_R_T(beta=beta, R=R, T=T)
    H = float(beta) - 0.5
    nodes, weights = _european_rule(H=H, R=int(R), T=float(T))
    weights[np.logical_and(nodes < 1.0, np.abs(weights) > 100.0)] = 0.0
    return _sort_rule(nodes, weights)


def _european_rule(*, H, R, T):
    """Positive-Hurst european/BL2 rule."""
    H = float(H)
    R = int(R)
    T = float(T)

    last_nodes: np.ndarray

    def optimizing_func(R_, tol_, bound_):
        if R_ == 1:
            nod = np.array([1.0 / T], dtype=np.float64)
        elif len(last_nodes) == R_:
            nod = last_nodes
        else:
            nod = np.empty(R_, dtype=np.float64)
            nod[:-1] = last_nodes
            nod[-1] = float(bound_)

        nod = nod / 1.03 ** np.fmin(np.arange(1, R_ + 1) ** 2, 100)
        return _optimize_error_l2(H=H, R=R_, T=T, tol=tol_, bound=bound_, init_nodes=nod)

    _, nodes, weights = optimizing_func(R_=1, tol_=1e-6, bound_=None)
    if R == 1:
        return nodes, weights

    L_step = 1.15
    bound = float(np.amax(nodes) / L_step)
    current_R = 1
    last_nodes = nodes

    while current_R < R:
        increase_R = 0
        L_step = 1.15

        while increase_R < 2:
            bound = bound * L_step
            error_, nodes, weights = optimizing_func(
                R_=current_R + 1, tol_=1e-7 / current_R, bound_=bound)

            p = np.argsort(nodes)
            nodes = nodes[p]
            weights = weights[p]

            if (
                np.amin(nodes[1:] / nodes[:-1]) < 1.4
                or np.abs(np.amin(weights)) < 1e-2
                or np.abs(np.amin(weights[1:] / weights[:-1])) < 0.4
            ):
                increase_R = 0
                L_step = 1.15
            elif error_ < optimizing_func(
                R_=current_R, tol_=1e-7 / current_R, bound_=bound)[0]:
                increase_R += 1
                if L_step > 1.06:
                    L_step = 1.05
                    bound = bound / 1.15
            else:
                increase_R = 0
                L_step = 1.15

        current_R += 1
        last_nodes = nodes

    if R >= 4:
        return nodes, weights

    if R == 2:
        L_4 = bound * 2.0
        L_5 = bound * 3.0
        L_6 = bound * 4.0
    else:
        L_4 = bound
        L_5 = bound * 1.25
        L_6 = bound * 1.5

    error_4, nodes_4, weights_4 = optimizing_func(R_=R, tol_=1e-8, bound_=L_4)
    error_5, nodes_5, weights_5 = optimizing_func(R_=R, tol_=1e-8, bound_=L_5)
    error_6, nodes_6, weights_6 = optimizing_func(R_=R, tol_=1e-8, bound_=L_6)

    if error_4 <= error_5 and error_4 <= error_6:
        return nodes_4, weights_4
    if error_5 <= error_6:
        return nodes_5, weights_5
    return nodes_6, weights_6


def _optimize_error_l2(*, H, R, T, tol=1e-8, bound=None, init_nodes):
    """Optimize the L2 error over nodes, using optimal weights."""
    H = float(H)
    R = int(R)
    T = float(T)

    if bound is None:
        bound = 1e100

    nodes = np.asarray(init_nodes, dtype=np.float64)
    if nodes.shape != (R,):
        raise ValueError("init_nodes must have shape (" + str(R) + ",), got " + str(nodes.shape) + ".")

    lower_bound = 1.0 / (10.0 * R * T) * ((0.5 - H) / 0.4) ** 2
    nodes = np.fmin(np.fmax(nodes, lower_bound), bound)

    bounds = ((np.log(lower_bound), np.log(bound)),) * R
    original_error, original_weights = _error_l2_optimal_weights(
        H=H, T=T, nodes=nodes, output="error")
    original_nodes = nodes.copy()

    def func(x):
        err, grad, _ = _error_l2_optimal_weights(H=H, T=T, nodes=np.exp(x), output="gradient")
        return err, np.exp(x) * grad

    res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, jac=True)

    nodes = np.exp(res.x)
    err, weights = _error_l2_optimal_weights(H=H, T=T, nodes=nodes, output="error")

    if err > 2.0 * np.fmax(original_error, 1e-9):
        return (
            np.array([np.sqrt(np.fmax(original_error, 0.0))]),
            original_nodes,
            original_weights,
        )

    return np.array([np.sqrt(np.fmax(err, 0.0))]), nodes, weights


def _error_l2_optimal_weights(*, H, T, nodes, output="error"):
    """L2 error and optimal weights for fixed nodes (positive-Hurst scalar T)."""
    H = float(H)
    T = float(T)
    nodes = np.asarray(nodes, dtype=np.float64)

    if len(nodes) == 1:
        node = np.fmax(1e-4, nodes[0])
        gamma_1 = gamma(H + 0.5)

        nT = node * T
        gamma_ints = gammainc(H + 0.5, nT)
        exp_node_matrix = _exp_underflow(2.0 * nT)
        exp_node_vec = _exp_underflow(nT)

        A = (1.0 - exp_node_matrix) / (2.0 * node)
        b = -2.0 * gamma_ints / node ** (H + 0.5)
        c = T ** (2.0 * H) / (2.0 * H * gamma_1 ** 2)

        v = b / A
        err = c - 0.25 * b * v
        opt_weight = np.array([-0.5 * v])

        if output in {"error", "err"}:
            return err, opt_weight

        A_grad = (-1.0 + (1.0 + 2.0 * nT) * exp_node_matrix) / (4.0 * node ** 2)
        b_grad = (
            -2.0
            * (nT ** (H + 0.5) * exp_node_vec / gamma_1 - (H + 0.5) * gamma_ints)
            / node ** (H + 1.5)
        )
        grad = 0.5 * (A_grad * v - b_grad) * v

        if output in {"gradient", "grad"}:
            return err, grad, opt_weight

        raise NotImplementedError("Unsupported output=" + repr(output) + ".")

    nodes = _regularize_nodes(nodes)

    node_matrix = nodes[:, None] + nodes[None, :]
    gamma_1 = gamma(H + 0.5)

    nT = nodes * T
    nmT = node_matrix * T
    gamma_ints = gammainc(H + 0.5, nT)
    exp_node_matrix = _exp_underflow(nmT)

    A = (1.0 - exp_node_matrix) / node_matrix
    b = -2.0 * gamma_ints / nodes ** (H + 0.5)
    c = T ** (2.0 * H) / (2.0 * H * gamma_1 ** 2)

    try:
        v = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        v = np.linalg.lstsq(A, b, rcond=None)[0]

    if np.amax(v) > 0.0:
        v = lsq_linear(A, b).x

    err = 0.25 * v @ A @ v - 0.5 * np.dot(b, v) + c
    opt_weights = -0.5 * v

    if output in {"error", "err"}:
        return err, opt_weights

    exp_node_vec = _exp_underflow(nT)
    A_grad = (-1.0 + (1.0 + nmT) * exp_node_matrix) / node_matrix ** 2
    b_grad = (
        -2.0
        * (nT ** (H + 0.5) * exp_node_vec / gamma_1 - (H + 0.5) * gamma_ints)
        / nodes ** (H + 1.5)
    )
    grad = 0.5 * v * (A_grad @ v) - 0.5 * b_grad * v

    if output in {"gradient", "grad"}:
        return err, grad, opt_weights

    raise NotImplementedError("Unsupported output=" + repr(output) + ".")


def _regularize_nodes(nodes):
    """Sort-copy regularization used by the L2 optimizer."""
    nodes = np.asarray(nodes, dtype=np.float64)

    perm = np.argsort(nodes)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)

    sorted_nodes = nodes[perm].copy()
    sorted_nodes[0] = np.fmax(1e-4, sorted_nodes[0])

    for i in range(len(sorted_nodes) - 1):
        if 1.01 * sorted_nodes[i] > sorted_nodes[i + 1]:
            sorted_nodes[i + 1] = sorted_nodes[i] * 1.01

    return sorted_nodes[inv]


def _exp_underflow(x):
    """Compute exp(-x) with large-x underflow protection."""
    if isinstance(x, np.ndarray):
        if x.dtype == int:
            x = x.astype(float)
        eps = np.finfo(x.dtype).tiny
    else:
        if isinstance(x, int):
            x = float(x)
        eps = np.finfo(x.__class__).tiny

    log_eps = -np.log(eps) / 2.0
    result = np.exp(-np.fmin(x, log_eps))
    return np.where(x > log_eps, 0.0, result)


def _sort_rule(nodes, weights):
    """Sort nodes and weights jointly by increasing node."""
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if nodes.ndim != 1:
        raise ValueError("nodes must be one-dimensional, got " + str(nodes.shape) + ".")
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional, got " + str(weights.shape) + ".")
    if nodes.shape != weights.shape:
        raise ValueError("nodes and weights must have matching shapes.")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must be finite.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite.")
    if np.any(nodes < 0.0):
        raise ValueError("nodes must be non-negative.")

    perm = np.argsort(nodes)
    return nodes[perm], weights[perm]


def _validate_beta_R_T(*, beta, R, T):
    if not (0.5 < float(beta) < 1.0):
        raise ValueError(
            "beta must lie in (1/2, 1) for the positive-Hurst BL2 rule; got beta=" + str(beta) + ".")
    if int(R) <= 0:
        raise ValueError("R must be positive, got " + str(R) + ".")
    if float(T) <= 0.0:
        raise ValueError("T must be positive, got " + str(T) + ".")
