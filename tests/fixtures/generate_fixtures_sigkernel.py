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

"""Generate sigkernel-based fixtures.

Requires: sigkernel, torch, pysiglib, numpy<2

Setup:

    pip install sigkernel "numpy<2"
    pip install -e .

Run:  python tests/fixtures/generate_fixtures_sigkernel.py
"""

import os
import numpy as np
import torch
import sigkernel
import pysiglib

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(FIXTURE_DIR, "reference_data.npz")

torch.manual_seed(99999)  # different seed from iisig script to avoid collision

KERN_DIM = 3

data = {}

# =========================================================================
# Input paths for kernel tests (self-contained, not shared with iisig)
# =========================================================================

kern_X = torch.rand(4, 10, KERN_DIM, dtype=torch.double) / 100
kern_Y = torch.rand(4, 10, KERN_DIM, dtype=torch.double) / 100
kern_X2 = torch.rand(4, 20, KERN_DIM, dtype=torch.double) / 100
kern_Y2 = torch.rand(4, 5, KERN_DIM, dtype=torch.double) / 100
gram_Y = torch.rand(6, 10, KERN_DIM, dtype=torch.double) / 100

data["kern_X"] = kern_X.numpy()
data["kern_Y"] = kern_Y.numpy()
data["kern_X2"] = kern_X2.numpy()
data["kern_Y2"] = kern_Y2.numpy()
data["gram_Y"] = gram_Y.numpy()

# =========================================================================
# Signature kernel (linear, scaled linear, RBF) x dyadic_order
# =========================================================================

for do in range(3):
    sk_lin = sigkernel.SigKernel(sigkernel.LinearKernel(), do)
    data[f"kernel_linear__do{do}"] = sk_lin.compute_kernel(kern_X, kern_Y, 100).numpy()

    sk_scaled = sigkernel.SigKernel(sigkernel.LinearKernel(0.5), do)
    data[f"kernel_scaled_linear__do{do}"] = sk_scaled.compute_kernel(kern_X, kern_Y, 100).numpy()

    sk_rbf = sigkernel.SigKernel(sigkernel.RBFKernel(0.5), do)
    data[f"kernel_rbf__do{do}"] = sk_rbf.compute_kernel(kern_X, kern_Y, 100).numpy()

    # Non-square lengths
    data[f"kernel_nonsq_long_short__do{do}"] = sk_lin.compute_kernel(kern_X2, kern_Y2, 100).numpy()
    data[f"kernel_nonsq_short_long__do{do}"] = sk_lin.compute_kernel(kern_Y2, kern_X2, 100).numpy()

    # Lead-lag kernel (use pysiglib's transform for consistency)
    kern_X_ll = torch.tensor(np.array(pysiglib.transform_path(kern_X.numpy(), lead_lag=True)))
    kern_Y_ll = torch.tensor(np.array(pysiglib.transform_path(kern_Y.numpy(), lead_lag=True)))
    data[f"kernel_lead_lag__do{do}"] = sk_lin.compute_kernel(kern_X_ll, kern_Y_ll, 100).numpy()

    # Gram matrix
    sk_gram = sigkernel.SigKernel(sigkernel.LinearKernel(), do)
    data[f"kernel_gram__do{do}"] = sk_gram.compute_Gram(kern_X, gram_Y, False, 100).numpy()

    # Gram with lead-lag
    gram_Y_ll = torch.tensor(np.array(pysiglib.transform_path(gram_Y.numpy(), lead_lag=True)))
    data[f"kernel_gram_ll__do{do}"] = sk_gram.compute_Gram(kern_X_ll, gram_Y_ll, False, 100).numpy()

# =========================================================================
# ESS and MMD
# =========================================================================

ess_X = torch.rand(4, 10, KERN_DIM, dtype=torch.double) / 100
ess_Y = torch.rand(4, 10, KERN_DIM, dtype=torch.double) / 100
data["ess_X"] = ess_X.numpy()
data["ess_Y"] = ess_Y.numpy()

for do in range(3):
    sk_ess_lin = sigkernel.SigKernel(sigkernel.LinearKernel(), do)
    data[f"ess_linear__do{do}"] = np.array(float(
        sk_ess_lin.compute_expected_scoring_rule(ess_X, ess_Y, 100)
    ))
    data[f"mmd_linear__do{do}"] = np.array(float(
        sk_ess_lin.compute_mmd(ess_X, ess_Y, 100)
    ))

    sk_ess_rbf = sigkernel.SigKernel(sigkernel.RBFKernel(2.), do)
    data[f"ess_rbf__do{do}"] = np.array(float(
        sk_ess_rbf.compute_expected_scoring_rule(ess_X, ess_Y, 100)
    ))
    data[f"mmd_rbf__do{do}"] = np.array(float(
        sk_ess_rbf.compute_mmd(ess_X, ess_Y, 100)
    ))

    # Non-square ESS/MMD
    ess_X2 = torch.rand(4, 10, KERN_DIM, dtype=torch.double)
    ess_Y2 = torch.rand(4, 20, KERN_DIM, dtype=torch.double)
    data[f"ess_nonsq_10_20__do{do}_X"] = ess_X2.numpy()
    data[f"ess_nonsq_10_20__do{do}_Y"] = ess_Y2.numpy()
    data[f"ess_nonsq_10_20__do{do}"] = np.array(float(
        sk_ess_lin.compute_expected_scoring_rule(ess_X2, ess_Y2, 100)
    ))
    data[f"mmd_nonsq_10_20__do{do}"] = np.array(float(
        sk_ess_lin.compute_mmd(ess_X2, ess_Y2, 100)
    ))

    ess_X3 = torch.rand(4, 20, KERN_DIM, dtype=torch.double)
    ess_Y3 = torch.rand(4, 10, KERN_DIM, dtype=torch.double)
    data[f"ess_nonsq_20_10__do{do}_X"] = ess_X3.numpy()
    data[f"ess_nonsq_20_10__do{do}_Y"] = ess_Y3.numpy()
    data[f"ess_nonsq_20_10__do{do}"] = np.array(float(
        sk_ess_lin.compute_expected_scoring_rule(ess_X3, ess_Y3, 100)
    ))
    data[f"mmd_nonsq_20_10__do{do}"] = np.array(float(
        sk_ess_lin.compute_mmd(ess_X3, ess_Y3, 100)
    ))

# =========================================================================
# Kernel full grid
# =========================================================================

def _transform(X_np, time_aug=False, lead_lag=False):
    """Apply pysiglib's transform_path and return as torch double tensor."""
    return torch.tensor(np.array(pysiglib.transform_path(X_np, time_aug=time_aug, lead_lag=lead_lag)))

def _compute_full_grid(X, Y, sk):
    """Compute full prefix kernel grid: result[b, i, j] = kernel(X[b, :i+1], Y[b, :j+1])."""
    batch, len1, _ = X.shape
    len2 = Y.shape[1]
    result = np.ones((batch, len1, len2))
    for b in range(batch):
        for i in range(1, len1):
            for j in range(1, len2):
                result[b, i, j] = float(sk.compute_kernel(
                    X[b:b+1, :i+1], Y[b:b+1, :j+1], 100
                ))
    return result

for len1, len2 in [(10, 10), (10, 5), (5, 10)]:
    gX = torch.rand(2, len1, KERN_DIM, dtype=torch.double) / 100
    gY = torch.rand(2, len2, KERN_DIM, dtype=torch.double) / 100
    sk_grid = sigkernel.SigKernel(sigkernel.LinearKernel(), 0)
    data[f"kernel_grid__{len1}x{len2}__X"] = gX.numpy()
    data[f"kernel_grid__{len1}x{len2}__Y"] = gY.numpy()
    data[f"kernel_grid__{len1}x{len2}__expected"] = _compute_full_grid(gX, gY, sk_grid)

# Full grid with time_aug
gX_ta = torch.rand(2, 5, KERN_DIM, dtype=torch.double) / 100
gY_ta = torch.rand(2, 10, KERN_DIM, dtype=torch.double) / 100
gX_ta_t = _transform(gX_ta.numpy(), time_aug=True)
gY_ta_t = _transform(gY_ta.numpy(), time_aug=True)
sk_grid_ta = sigkernel.SigKernel(sigkernel.LinearKernel(), 0)
data["kernel_grid_ta__X"] = gX_ta.numpy()
data["kernel_grid_ta__Y"] = gY_ta.numpy()
data["kernel_grid_ta__expected"] = _compute_full_grid(gX_ta_t, gY_ta_t, sk_grid_ta)

# Full grid with lead_lag
gX_ll = torch.rand(2, 5, KERN_DIM, dtype=torch.double) / 100
gY_ll = torch.rand(2, 10, KERN_DIM, dtype=torch.double) / 100
gX_ll_t = _transform(gX_ll.numpy(), lead_lag=True)
gY_ll_t = _transform(gY_ll.numpy(), lead_lag=True)
sk_grid_ll = sigkernel.SigKernel(sigkernel.LinearKernel(), 0)
data["kernel_grid_ll__X"] = gX_ll.numpy()
data["kernel_grid_ll__Y"] = gY_ll.numpy()
data["kernel_grid_ll__expected"] = _compute_full_grid(gX_ll_t, gY_ll_t, sk_grid_ll)

# Full grid with time_aug + lead_lag
gX_ta_ll = torch.rand(2, 5, KERN_DIM, dtype=torch.double) / 100
gY_ta_ll = torch.rand(2, 10, KERN_DIM, dtype=torch.double) / 100
gX_ta_ll_t = _transform(gX_ta_ll.numpy(), time_aug=True, lead_lag=True)
gY_ta_ll_t = _transform(gY_ta_ll.numpy(), time_aug=True, lead_lag=True)
sk_grid_ta_ll = sigkernel.SigKernel(sigkernel.LinearKernel(), 0)
data["kernel_grid_ta_ll__X"] = gX_ta_ll.numpy()
data["kernel_grid_ta_ll__Y"] = gY_ta_ll.numpy()
data["kernel_grid_ta_ll__expected"] = _compute_full_grid(gX_ta_ll_t, gY_ta_ll_t, sk_grid_ta_ll)

# =========================================================================
# Save (merge with existing)
# =========================================================================

existing = {}
if os.path.exists(OUT_PATH):
    existing = dict(np.load(OUT_PATH, allow_pickle=False))
existing.update(data)
np.savez_compressed(OUT_PATH, **existing)
size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"[sigkernel] {len(data)} arrays -> {OUT_PATH} ({size_kb:.1f} KB)")
