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
from tensordev.sss import StateSpaceSignature


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

np.savez_compressed(
    OUT_PATH,
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
