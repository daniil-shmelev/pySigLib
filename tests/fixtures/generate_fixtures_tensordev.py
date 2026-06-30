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

"""Generate tensordev-based Volterra signature fixtures.

Requires: tensordev, jax.

Run:
    python tests/fixtures/generate_fixtures_tensordev.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import tensordev
from tensordev import FSSK
from tensordev.sss import StateSpaceSignature, fssk_vsig
from tensordev.sss.rough_approx import _bl2_quadrature_rule, fractional_fssk
from tensordev.volterra import vsig as conv_vsig
from tensordev.volterra.kernel import FractionalKernel, GammaKernel


FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(FIXTURE_DIR, "volterra_signature_tensordev.npz")


def _reference(path, degree, lambda_diag, A, b, dt, quad_order, tau_dt=0.):
    kernel = StateSpaceSignature.from_matrix(
        Lambda=jnp.diag(jnp.asarray(lambda_diag)),
        A=jnp.asarray(A),
        b=jnp.asarray(b),
        trunc=degree,
        quad_order=quad_order,
    )
    levels = kernel.vsig(jnp.asarray(path), dt=dt, tau_dt=tau_dt)
    return np.asarray(tensordev.tensor_to_flat(levels))


def _fractional_reference(path, degree, beta, R, A, dt, T, quad_order, tau_dt=0.):
    kernel = fractional_fssk(
        beta=beta, R=R, A=jnp.asarray(A), T=T, coef_quad_order=quad_order)
    levels = fssk_vsig(
        jnp.asarray(path), kernel=kernel, dt=dt, trunc=degree, tau_dt=tau_dt)
    return np.asarray(tensordev.tensor_to_flat(levels))


def _conv_fractional_reference(path, degree, beta, A, dt, order=0, dyadic_order=0):
    kernel = FractionalKernel(beta=jnp.atleast_1d(jnp.asarray(beta)), A=jnp.asarray(A))
    scheme = "quadratic" if (order == 0 and dyadic_order == 0) else "fft"
    levels = conv_vsig(jnp.asarray(path), kernel=kernel, trunc=degree, dt=dt,
                       scheme=scheme, order=order, dyadic_order=dyadic_order)
    return np.asarray(tensordev.tensor_to_flat(levels))


def _conv_gamma_reference(path, degree, beta, scale, rate, quad_order, A, dt,
                          order=0, dyadic_order=0):
    kernel = GammaKernel(beta=jnp.asarray(beta), A=jnp.asarray(A),
                         scale=jnp.asarray(scale), rate=jnp.asarray(rate),
                         quad_order=int(quad_order))
    scheme = "quadratic" if (order == 0 and dyadic_order == 0) else "fft"
    levels = conv_vsig(jnp.asarray(path), kernel=kernel, trunc=degree, dt=dt,
                       scheme=scheme, order=order, dyadic_order=dyadic_order)
    return np.asarray(tensordev.tensor_to_flat(levels))


jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng(42)
scalar_path = rng.normal(size=(2, 5, 2)).astype(np.float64)
scalar_degree = np.array(3, dtype=np.int64)
scalar_dt = np.array(0.125, dtype=np.float64)
scalar_quad_order = np.array(32, dtype=np.int64)
scalar_lambdas = np.array([0.4, 1.3], dtype=np.float64)
scalar_b = np.array([[0.7, 0.2]], dtype=np.float64)
scalar_A = np.eye(2, dtype=np.float64).reshape(1, 2, 2)
scalar_expected = _reference(
    scalar_path,
    int(scalar_degree),
    scalar_lambdas,
    scalar_A,
    scalar_b,
    float(scalar_dt),
    int(scalar_quad_order),
)

matrix_path = np.array(
    [
        [0., 0.1, -0.2],
        [0.4, -0.1, 0.0],
        [0.5, 0.2, 0.3],
        [0.2, 0.5, 0.1],
    ],
    dtype=np.float64,
)
matrix_degree = np.array(3, dtype=np.int64)
matrix_dt = np.array(0.2, dtype=np.float64)
matrix_tau_dt = np.array(0.1, dtype=np.float64)
matrix_quad_order = np.array(32, dtype=np.int64)
matrix_lambdas = np.array([0.25, 1.1], dtype=np.float64)
matrix_b = np.array([[0.8, -0.1], [0.3, 0.4]], dtype=np.float64)
matrix_A = np.array(
    [
        [[1.0, 0.2, -0.1], [0.0, 0.5, 0.3]],
        [[-0.4, 0.1, 0.6], [0.7, -0.2, 0.0]],
    ],
    dtype=np.float64,
)
matrix_expected = _reference(
    matrix_path,
    int(matrix_degree),
    matrix_lambdas,
    matrix_A,
    matrix_b,
    float(matrix_dt),
    int(matrix_quad_order),
    tau_dt=float(matrix_tau_dt),
)

fractional_path = np.array(
    [
        [0., 0.1, -0.2],
        [0.4, -0.1, 0.0],
        [0.5, 0.2, 0.3],
        [0.2, 0.5, 0.1],
        [0.3, -0.2, 0.4],
    ],
    dtype=np.float64,
)
fractional_degree = np.array(3, dtype=np.int64)
fractional_dt = np.array(0.1, dtype=np.float64)
fractional_tau_dt = np.array(0.05, dtype=np.float64)
fractional_beta = np.array(0.7, dtype=np.float64)
fractional_R = np.array(4, dtype=np.int64)
fractional_T = np.array(1.0, dtype=np.float64)
fractional_quad_order = np.array(32, dtype=np.int64)
fractional_A = np.array(
    [[[1.0, 0.2, -0.1], [0.0, 0.5, 0.3]]],
    dtype=np.float64,
)
fractional_nodes, fractional_weights = _bl2_quadrature_rule(
    beta=float(fractional_beta), R=int(fractional_R), T=float(fractional_T))
fractional_expected = _fractional_reference(
    fractional_path,
    int(fractional_degree),
    float(fractional_beta),
    int(fractional_R),
    fractional_A,
    float(fractional_dt),
    float(fractional_T),
    int(fractional_quad_order),
    tau_dt=float(fractional_tau_dt),
)

# Oscillatory (complex-spectrum) dense Lambda: real 2x2 block [[a, w], [-w, a]]
# has eigenvalues a +- i w, exercising the complex state recursion.
osc_path = np.array(
    [
        [0., 0.1, -0.2],
        [0.4, -0.1, 0.0],
        [0.5, 0.2, 0.3],
        [0.2, 0.5, 0.1],
    ],
    dtype=np.float64,
)
osc_degree = np.array(3, dtype=np.int64)
osc_dt = np.array(0.1, dtype=np.float64)
osc_tau_dt = np.array(0.05, dtype=np.float64)
osc_quad_order = np.array(32, dtype=np.int64)
osc_Lambda = np.array([[0.6, 1.3], [-1.3, 0.6]], dtype=np.float64)
osc_b = np.array([[0.8, -0.1], [0.3, 0.4]], dtype=np.float64)
osc_A = np.array(
    [
        [[1.0, 0.2, -0.1], [0.0, 0.5, 0.3]],
        [[-0.4, 0.1, 0.6], [0.7, -0.2, 0.0]],
    ],
    dtype=np.float64,
)
osc_kernel = StateSpaceSignature.from_matrix(
    Lambda=jnp.asarray(osc_Lambda),
    A=jnp.asarray(osc_A),
    b=jnp.asarray(osc_b),
    trunc=int(osc_degree),
    quad_order=int(osc_quad_order),
)
osc_expected = np.asarray(tensordev.tensor_to_flat(
    osc_kernel.vsig(jnp.asarray(osc_path), dt=float(osc_dt), tau_dt=float(osc_tau_dt))))

# Prony / Jordan kernel: real poles (size 1) + an oscillatory pair (size 1),
# state vectors built from Prony coefficients.
prony_path = np.array(
    [
        [0., 0.1, -0.2],
        [0.4, -0.1, 0.0],
        [0.5, 0.2, 0.3],
        [0.2, 0.5, 0.1],
    ],
    dtype=np.float64,
)
prony_degree = np.array(3, dtype=np.int64)
prony_dt = np.array(0.1, dtype=np.float64)
prony_tau_dt = np.array(0.05, dtype=np.float64)
prony_quad_order = np.array(32, dtype=np.int64)
prony_real_rates = np.array([0.4, 0.9], dtype=np.float64)
prony_real_sizes = np.array([1, 1], dtype=np.int64)
prony_osc_decays = np.array([0.6], dtype=np.float64)
prony_osc_freqs = np.array([1.3], dtype=np.float64)
prony_osc_sizes = np.array([1], dtype=np.int64)
prony_alpha = np.array([[0.7, -0.3], [0.2, 0.5]], dtype=np.float64)
prony_beta = np.array([[0.4], [-0.2]], dtype=np.float64)
prony_delta = np.array([[0.1], [0.3]], dtype=np.float64)
prony_A = np.array(
    [
        [[1.0, 0.2, -0.1], [0.0, 0.5, 0.3]],
        [[-0.4, 0.1, 0.6], [0.7, -0.2, 0.0]],
    ],
    dtype=np.float64,
)
prony_kernel = FSSK.from_prony(
    A=jnp.asarray(prony_A),
    real_rates=jnp.asarray(prony_real_rates), real_sizes=tuple(int(s) for s in prony_real_sizes),
    osc_decays=jnp.asarray(prony_osc_decays), osc_freqs=jnp.asarray(prony_osc_freqs),
    osc_sizes=tuple(int(s) for s in prony_osc_sizes),
    alpha=jnp.asarray(prony_alpha), beta=jnp.asarray(prony_beta), delta=jnp.asarray(prony_delta),
    quad_order=int(prony_quad_order))
prony_b = np.asarray(prony_kernel.b)
prony_expected = np.asarray(tensordev.tensor_to_flat(
    fssk_vsig(jnp.asarray(prony_path), kernel=prony_kernel, dt=float(prony_dt),
              trunc=int(prony_degree), tau_dt=float(prony_tau_dt))))

# Defective (size>1) Jordan block: a repeated pole producing t^k e^(-lambda t)
# kernel terms, which need the general matrix recursion.
jordan_path = np.array(
    [
        [0., 0.1, -0.2],
        [0.4, -0.1, 0.0],
        [0.5, 0.2, 0.3],
        [0.2, 0.5, 0.1],
    ],
    dtype=np.float64,
)
jordan_degree = np.array(3, dtype=np.int64)
jordan_dt = np.array(0.1, dtype=np.float64)
jordan_tau_dt = np.array(0.05, dtype=np.float64)
jordan_quad_order = np.array(32, dtype=np.int64)
jordan_real_rates = np.array([0.5], dtype=np.float64)
jordan_real_sizes = np.array([2], dtype=np.int64)
jordan_b = np.array([[0.8, 0.3], [-0.2, 0.5]], dtype=np.float64)
jordan_A = np.array(
    [
        [[1.0, 0.2, -0.1], [0.0, 0.5, 0.3]],
        [[-0.4, 0.1, 0.6], [0.7, -0.2, 0.0]],
    ],
    dtype=np.float64,
)
jordan_kernel = FSSK.from_jordan(
    A=jnp.asarray(jordan_A), b=jnp.asarray(jordan_b),
    real_rates=jnp.asarray(jordan_real_rates),
    real_sizes=tuple(int(s) for s in jordan_real_sizes),
    quad_order=int(jordan_quad_order))
jordan_expected = np.asarray(tensordev.tensor_to_flat(
    fssk_vsig(jnp.asarray(jordan_path), kernel=jordan_kernel, dt=float(jordan_dt),
              trunc=int(jordan_degree), tau_dt=float(jordan_tau_dt))))

# General convolution scheme (tensordev "quadratic", order 0), q == 1.
conv_path = np.array(
    [
        [0., 0.1, -0.2],
        [0.4, -0.1, 0.0],
        [0.5, 0.2, 0.3],
        [0.2, 0.5, 0.1],
        [0.3, -0.2, 0.4],
        [0.1, 0.0, -0.1],
    ],
    dtype=np.float64,
)
conv_degree = np.array(3, dtype=np.int64)
conv_dt = np.array(0.1, dtype=np.float64)
conv_A = np.array(
    [[[1.0, 0.2, -0.1], [0.0, 0.5, 0.3]]],
    dtype=np.float64,
)

conv_frac_beta = np.array(0.7, dtype=np.float64)
conv_frac_expected = _conv_fractional_reference(
    conv_path, int(conv_degree), float(conv_frac_beta), conv_A, float(conv_dt))

# Multivariate (q=2) fractional convolution kernel: two components with
# distinct fractional exponents, exercising the shuffle-algebra evalVtE.
conv_frac2_beta = np.array([0.6, 0.9], dtype=np.float64)
conv_frac2_A = np.array(
    [
        [[1.0, 0.2, -0.1], [0.0, 0.5, 0.3]],
        [[-0.4, 0.1, 0.6], [0.7, -0.2, 0.0]],
    ],
    dtype=np.float64,
)
conv_frac2_expected = _conv_fractional_reference(
    conv_path, int(conv_degree), conv_frac2_beta, conv_frac2_A, float(conv_dt))

# Higher-order quadrature (order 1/2) and dyadic refinement (FFT scheme).
conv_frac_o1 = _conv_fractional_reference(
    conv_path, int(conv_degree), float(conv_frac_beta), conv_A, float(conv_dt), order=1)
conv_frac_o2 = _conv_fractional_reference(
    conv_path, int(conv_degree), float(conv_frac_beta), conv_A, float(conv_dt), order=2)
conv_frac_d2 = _conv_fractional_reference(
    conv_path, int(conv_degree), float(conv_frac_beta), conv_A, float(conv_dt), dyadic_order=2)
conv_frac2_o2 = _conv_fractional_reference(
    conv_path, int(conv_degree), conv_frac2_beta, conv_frac2_A, float(conv_dt), order=2)
conv_frac2_o1_d1 = _conv_fractional_reference(
    conv_path, int(conv_degree), conv_frac2_beta, conv_frac2_A, float(conv_dt), order=1, dyadic_order=1)

conv_gamma_beta = np.array(0.8, dtype=np.float64)
conv_gamma_scale = np.array(1.3, dtype=np.float64)
conv_gamma_rate = np.array(0.5, dtype=np.float64)
conv_gamma_quad_order = np.array(48, dtype=np.int64)
conv_gamma_expected = _conv_gamma_reference(
    conv_path, int(conv_degree), float(conv_gamma_beta), float(conv_gamma_scale),
    float(conv_gamma_rate), int(conv_gamma_quad_order), conv_A, float(conv_dt))
conv_gamma_o2 = _conv_gamma_reference(
    conv_path, int(conv_degree), float(conv_gamma_beta), float(conv_gamma_scale),
    float(conv_gamma_rate), int(conv_gamma_quad_order), conv_A, float(conv_dt), order=2)

np.savez_compressed(
    OUT_PATH,
    conv_path=conv_path,
    conv_degree=conv_degree,
    conv_dt=conv_dt,
    conv_A=conv_A,
    conv_frac_beta=conv_frac_beta,
    conv_frac_expected=conv_frac_expected,
    conv_frac2_beta=conv_frac2_beta,
    conv_frac2_A=conv_frac2_A,
    conv_frac2_expected=conv_frac2_expected,
    conv_frac_o1=conv_frac_o1,
    conv_frac_o2=conv_frac_o2,
    conv_frac_d2=conv_frac_d2,
    conv_frac2_o2=conv_frac2_o2,
    conv_frac2_o1_d1=conv_frac2_o1_d1,
    conv_gamma_o2=conv_gamma_o2,
    conv_gamma_beta=conv_gamma_beta,
    conv_gamma_scale=conv_gamma_scale,
    conv_gamma_rate=conv_gamma_rate,
    conv_gamma_quad_order=conv_gamma_quad_order,
    conv_gamma_expected=conv_gamma_expected,
    jordan_path=jordan_path,
    jordan_degree=jordan_degree,
    jordan_dt=jordan_dt,
    jordan_tau_dt=jordan_tau_dt,
    jordan_quad_order=jordan_quad_order,
    jordan_real_rates=jordan_real_rates,
    jordan_real_sizes=jordan_real_sizes,
    jordan_b=jordan_b,
    jordan_A=jordan_A,
    jordan_expected=jordan_expected,
    prony_path=prony_path,
    prony_degree=prony_degree,
    prony_dt=prony_dt,
    prony_tau_dt=prony_tau_dt,
    prony_quad_order=prony_quad_order,
    prony_real_rates=prony_real_rates,
    prony_real_sizes=prony_real_sizes,
    prony_osc_decays=prony_osc_decays,
    prony_osc_freqs=prony_osc_freqs,
    prony_osc_sizes=prony_osc_sizes,
    prony_alpha=prony_alpha,
    prony_beta=prony_beta,
    prony_delta=prony_delta,
    prony_A=prony_A,
    prony_b=prony_b,
    prony_expected=prony_expected,
    osc_path=osc_path,
    osc_degree=osc_degree,
    osc_dt=osc_dt,
    osc_tau_dt=osc_tau_dt,
    osc_quad_order=osc_quad_order,
    osc_Lambda=osc_Lambda,
    osc_b=osc_b,
    osc_A=osc_A,
    osc_expected=osc_expected,
    fractional_path=fractional_path,
    fractional_degree=fractional_degree,
    fractional_dt=fractional_dt,
    fractional_tau_dt=fractional_tau_dt,
    fractional_beta=fractional_beta,
    fractional_R=fractional_R,
    fractional_T=fractional_T,
    fractional_quad_order=fractional_quad_order,
    fractional_A=fractional_A,
    fractional_nodes=fractional_nodes,
    fractional_weights=fractional_weights,
    fractional_expected=fractional_expected,
    scalar_path=scalar_path,
    scalar_degree=scalar_degree,
    scalar_dt=scalar_dt,
    scalar_quad_order=scalar_quad_order,
    scalar_lambdas=scalar_lambdas,
    scalar_b=scalar_b,
    scalar_A=scalar_A,
    scalar_expected=scalar_expected,
    matrix_path=matrix_path,
    matrix_degree=matrix_degree,
    matrix_dt=matrix_dt,
    matrix_tau_dt=matrix_tau_dt,
    matrix_quad_order=matrix_quad_order,
    matrix_lambdas=matrix_lambdas,
    matrix_b=matrix_b,
    matrix_A=matrix_A,
    matrix_expected=matrix_expected,
)

size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"[tensordev] wrote {OUT_PATH} ({size_kb:.1f} KB)")
