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

import platform

import numpy as np
import pytest

import pysiglib


@pytest.mark.skipif(platform.machine().lower() not in ("amd64", "x86_64", "i386", "i686"), reason="x86 subnormal mode")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["horner", "ps"])
@pytest.mark.parametrize("batch,jobs,points,degree", [(1, 1, 2, 2), (17, 4, 2, 2), (3, 4, 17, 4)])
def test_cpu_signature_flush_subnormals_is_scoped(monkeypatch, dtype, method, batch, jobs, points, degree):
    monkeypatch.setenv("PYSIGLIB_CPU_FLUSH_SUBNORMALS", "0")
    increment = dtype(np.sqrt(np.finfo(dtype).tiny) / (4 * points))
    path = np.zeros((batch, points, 2), dtype=dtype)
    path[:] = (np.arange(points, dtype=dtype) * increment).reshape(1, -1, 1)
    options = dict(method=method, n_jobs=jobs)
    signature = pysiglib.sig(path, degree, **options)
    seeds = np.zeros_like(signature)
    seeds[:, 2:] = increment
    gradient = pysiglib.sig_backprop(path, signature, seeds, degree, **options)
    assert np.any(signature[:, 2:] != 0)
    assert np.any(gradient != 0)
    monkeypatch.setenv("PYSIGLIB_CPU_FLUSH_SUBNORMALS", "1")
    fast_signature = pysiglib.sig(path, degree, **options)
    fast_gradient = pysiglib.sig_backprop(path, fast_signature, seeds, degree, **options)
    np.testing.assert_array_equal(fast_signature[:, :2], signature[:, :2])
    np.testing.assert_array_equal(fast_signature[:, 2:], 0)
    np.testing.assert_array_equal(fast_gradient, 0)
    assert increment * increment != 0
    monkeypatch.setenv("PYSIGLIB_CPU_FLUSH_SUBNORMALS", "0")
    np.testing.assert_array_equal(pysiglib.sig(path, degree, **options), signature)
    np.testing.assert_array_equal(pysiglib.sig_backprop(path, signature, seeds, degree, **options), gradient)


@pytest.mark.parametrize("value", ["2", "-1", "true"])
def test_cpu_signature_rejects_invalid_subnormal_option(monkeypatch, value):
    monkeypatch.setenv("PYSIGLIB_CPU_FLUSH_SUBNORMALS", value)
    with pytest.raises(Exception, match="PYSIGLIB_CPU_FLUSH_SUBNORMALS"):
        pysiglib.sig(np.zeros((2, 2)), 2)
