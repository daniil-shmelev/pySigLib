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

"""Generate signatory-based fixtures (Lyndon words).

Requires: signatory==1.2.6.1.9.0, torch==1.9.0, pysiglib
Must run AFTER generate_fixtures_iisig.py (loads shared paths from it).

Signatory only works with Python 3.9 + torch 1.9. Create a dedicated venv:

    py -3.9 -m venv .venv-signatory
    .venv-signatory/Scripts/pip install --upgrade pip
    .venv-signatory/Scripts/pip install "torch==1.9.0+cpu" -f https://download.pytorch.org/whl/torch_stable.html
    .venv-signatory/Scripts/pip install signatory==1.2.6.1.9.0 "numpy<2"
    .venv-signatory/Scripts/pip install -e .

Then run:
    .venv-signatory/Scripts/python tests/fixtures/generate_fixtures_signatory.py
"""

import os
import numpy as np
import torch
import signatory
import pysiglib

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(FIXTURE_DIR, "reference_data.npz")

if not os.path.exists(OUT_PATH):
    raise RuntimeError("Run generate_fixtures_iisig.py first to create shared paths")

existing = dict(np.load(OUT_PATH, allow_pickle=False))
path = existing["path"]  # (2, 10, 3) - shared with iisig script

torch.manual_seed(54321)

DEGS = range(1, 6)

data = {}

# =========================================================================
# Log-signature forward Lyndon words
# =========================================================================

for d in DEGS:
    X = torch.from_numpy(path)
    ls = signatory.logsignature(X, d, mode="words")
    data[f"logsig_words__d{d}"] = ls.detach().numpy()

# =========================================================================
# Log-signature backprop Lyndon words
# =========================================================================

for d in DEGS:
    X = torch.from_numpy(path).requires_grad_(True)
    ls = signatory.logsignature(X, d, mode="words")
    derivs = torch.rand(ls.shape, dtype=torch.double)
    ls.backward(derivs)
    data[f"logsig_bp_words_derivs__d{d}"] = derivs.numpy()
    data[f"logsig_bp_words_expected__d{d}"] = X.grad.numpy()

# =========================================================================
# sig_to_log_sig backprop
# =========================================================================

for d in DEGS:
    sl = pysiglib.sig_length(1, d)
    # signatory expects signature WITHOUT leading scalar term
    X_full = torch.rand(1, sl, dtype=torch.double)
    X_no_scalar = X_full[:, 1:].clone().detach().requires_grad_(True)
    ls = signatory.signature_to_logsignature(X_no_scalar, 1, d, mode="expand")
    ls[0].backward(torch.ones(ls.shape[-1], dtype=torch.double))
    data[f"sig_to_logsig_bp_input__d{d}"] = X_full.numpy()  # full sig including leading 1
    data[f"sig_to_logsig_bp_expected__d{d}"] = X_no_scalar.grad.numpy()  # grad w.r.t. non-scalar part

# =========================================================================
# Save (merge with existing)
# =========================================================================

existing.update(data)
np.savez_compressed(OUT_PATH, **existing)
size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"[signatory] {len(data)} arrays -> {OUT_PATH} ({size_kb:.1f} KB)")
