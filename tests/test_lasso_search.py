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

import numpy as np
import pytest

import pysiglib


@pytest.mark.parametrize("fit", [pysiglib.sig_lasso, pysiglib.log_sig_lasso])
def test_equal_scores_on_independent_columns(fit):
    dimension = 20
    basis = np.eye(dimension)
    endpoints = np.concatenate((basis, -basis))
    paths = np.stack((np.zeros_like(endpoints), endpoints), axis=1)
    y = endpoints.sum(axis=1)
    alpha = 0.5 / dimension

    with pytest.warns(RuntimeWarning, match="global KKT check"):
        first = fit(paths, y, 1, alpha, max_iter=1)
    assert not first["converged"]
    assert first["words"] == [(j,) for j in range(16)]

    model = fit(paths, y, 1, alpha)
    assert model["converged"]
    assert model["words"] == [(j,) for j in range(dimension)]
    np.testing.assert_allclose(model["coef"], 0.5)
