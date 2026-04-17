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

"""Generate iisignature-based fixtures.

Requires: iisignature, pysiglib, numpy<2

Setup:

    pip install iisignature "numpy<2"
    pip install -e .

Run:  python tests/fixtures/generate_fixtures_iisig.py
"""

import os
import numpy as np
import iisignature
import pysiglib

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(FIXTURE_DIR, "reference_data.npz")

np.random.seed(12345)

DIM = 3
DEGS = range(1, 6)

data = {}

# =========================================================================
# Shared input paths - also used by signatory script (must run first)
# =========================================================================

path = np.random.uniform(size=(10, 10, DIM))
path_dim2 = np.random.uniform(size=(10, 10, 2))

data["path"] = path
data["path_dim2"] = path_dim2

# Pre-compute transformed paths
path_ta = np.array(pysiglib.transform_path(path, time_aug=True))
path_ll = np.array(pysiglib.transform_path(path_dim2, lead_lag=True))
path_ta_ll = np.array(pysiglib.transform_path(path_dim2, time_aug=True, lead_lag=True))
path_batch_ll = np.array(pysiglib.transform_path(path, lead_lag=True))

dim_ta = path_ta.shape[-1]
dim_batch_ll = path_batch_ll.shape[-1]

# =========================================================================
# Signature forward
# =========================================================================

for d in DEGS:
    data[f"sig__d{d}"] = iisignature.sig(path, d)
    data[f"sig_ta__d{d}"] = iisignature.sig(path_ta, d)
    data[f"sig_ll__d{d}"] = iisignature.sig(path_ll, d)
    data[f"sig_ta_ll__d{d}"] = iisignature.sig(path_ta_ll, d)

# =========================================================================
# Signature backprop
# =========================================================================

for d in DEGS:
    sl = pysiglib.sig_length(DIM, d)
    derivs = np.random.uniform(size=(path.shape[0], sl))
    data[f"sig_bp_derivs__d{d}"] = derivs
    data[f"sig_bp_expected__d{d}"] = iisignature.sigbackprop(
        derivs[:, 1:], path, d
    )

# =========================================================================
# Log-signature forward expanded
# =========================================================================

for d in DEGS:
    s = iisignature.prepare(DIM, d, "x")
    data[f"logsig_exp__d{d}"] = iisignature.logsig(path, s, "x")

    s_ta = iisignature.prepare(dim_ta, d, "x")
    data[f"logsig_exp_ta__d{d}"] = iisignature.logsig(path_ta, s_ta, "x")

    s_ll = iisignature.prepare(dim_batch_ll, d, "x")
    if d <= 4:
        data[f"logsig_exp_ll__d{d}"] = iisignature.logsig(path_batch_ll[:2], s_ll, "x")

# =========================================================================
# Log-signature forward Lyndon basis
# =========================================================================

for d in DEGS:
    s = iisignature.prepare(DIM, d, "s")
    data[f"logsig_basis__d{d}"] = iisignature.logsig(path, s, "s")

# =========================================================================
# Log-signature backprop expanded
# =========================================================================

for d in DEGS:
    s = iisignature.prepare(DIM, d, "x")
    ls = iisignature.logsig(path, s, "x")
    derivs = np.random.uniform(size=ls.shape)
    data[f"logsig_bp_exp_derivs__d{d}"] = derivs
    data[f"logsig_bp_exp_expected__d{d}"] = iisignature.logsigbackprop(
        derivs, path, s, "x"
    )

# Log-signature backprop expanded time_aug
for d in DEGS:
    s_ta = iisignature.prepare(dim_ta, d, "x")
    ls_ta = iisignature.logsig(path_ta, s_ta, "x")
    derivs_ta = np.random.uniform(size=ls_ta.shape)
    data[f"logsig_bp_exp_ta_derivs__d{d}"] = derivs_ta
    bp = iisignature.logsigbackprop(derivs_ta, path_ta, s_ta, "x")
    data[f"logsig_bp_exp_ta_expected__d{d}"] = bp[:, :, :-1]

# =========================================================================
# Log-signature backprop Lyndon basis
# =========================================================================

for d in DEGS:
    s = iisignature.prepare(DIM, d, "s")
    ls = iisignature.logsig(path, s, "s")
    derivs = np.random.uniform(size=ls.shape)
    data[f"logsig_bp_basis_derivs__d{d}"] = derivs
    data[f"logsig_bp_basis_expected__d{d}"] = iisignature.logsigbackprop(
        derivs, path, s, "s"
    )

# =========================================================================
# Sig combine backprop
# =========================================================================

for d in DEGS:
    sl = pysiglib.sig_length(DIM, d)
    sig1 = np.random.uniform(size=(10, sl))
    sig2 = np.random.uniform(size=(10, sl))
    comb_derivs = np.random.uniform(size=(10, sl))
    d1, d2 = iisignature.sigcombinebackprop(
        comb_derivs[:, 1:], sig1[:, 1:], sig2[:, 1:], DIM, d
    )
    data[f"sig_comb_bp_sig1__d{d}"] = sig1
    data[f"sig_comb_bp_sig2__d{d}"] = sig2
    data[f"sig_comb_bp_derivs__d{d}"] = comb_derivs
    data[f"sig_comb_bp_d1__d{d}"] = d1
    data[f"sig_comb_bp_d2__d{d}"] = d2

# =========================================================================
# Save (merge with existing)
# =========================================================================

existing = {}
if os.path.exists(OUT_PATH):
    existing = dict(np.load(OUT_PATH, allow_pickle=False))
existing.update(data)
np.savez_compressed(OUT_PATH, **existing)
size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"[iisig] {len(data)} arrays -> {OUT_PATH} ({size_kb:.1f} KB)")
