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

"""Regenerate the Stochastax branched-log fixture.

Install Stochastax first.

The fixture is restricted to degree 2 because Stochastax currently raises
``NotImplementedError`` for GL/MKW products in degree 3 and above.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra


HERE = Path(__file__).resolve().parent
OUT = HERE / "branched_log_sig_stochastax_degree2.json"


def main() -> None:
    dimension = 2
    degree = 2
    path = np.array([[0.0, 0.0], [0.3, -0.7], [1.1, 0.2]], dtype=np.float64)

    # Recursive pySigLib order at degree 2:
    # 1, (0), (1), ((0),0), ((1),0), ((0),1), ((1),1).
    signature = np.array([
        1.0,
        1.1,
        0.19999999999999996,
        0.605,
        0.5249999999999999,
        -0.30499999999999994,
        0.020000000000000018,
    ], dtype=np.float64)

    algebra = GLHopfAlgebra.build(dimension, degree)
    levels = [jnp.asarray(signature[1:3]), jnp.asarray(signature[3:7])]
    log_levels = algebra.log(levels)
    log_signature = np.concatenate([[0.0], *[np.asarray(x) for x in log_levels]])

    data = {
        "source": "Stochastax stochastax.hopf_algebras.hopf_algebras.HopfAlgebra.log",
        "note": "Degree 2 fixture because Stochastax GL/MKW products currently implement only degree 2.",
        "dimension": dimension,
        "degree": degree,
        "path": path.tolist(),
        "recursive_trees": [None, [0], [1], [[0], 0], [[1], 0], [[0], 1], [[1], 1]],
        "signature_scalar": signature.tolist(),
        "log_signature_scalar": log_signature.tolist(),
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
