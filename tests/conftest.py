# Copyright 2025 Daniil Shmelev
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

import os
import functools
import numpy as np
import torch
import pytest
import pysiglib


def check_close(a, b, atol=None, single_atol=None, double_atol=None):
    """Compare arrays/tensors element-wise within tolerance.

    If atol is given, uses it directly.
    Otherwise auto-selects based on dtype:
      float32 -> single_atol (default 1e-4)
      float64 -> double_atol (default 1e-10)
    """
    a_ = a.detach().cpu().numpy() if hasattr(a, 'cpu') else np.asarray(a)
    b_ = b.detach().cpu().numpy() if hasattr(b, 'cpu') else np.asarray(b)
    if atol is None:
        s = single_atol if single_atol is not None else 1e-4
        d = double_atol if double_atol is not None else 1e-10
        # Use loose tolerance if either operand is float32
        atol = s if (a_.dtype == np.float32 or b_.dtype == np.float32) else d
    max_diff = np.max(np.abs(a_ - b_))
    assert not max_diff > atol, f"Max diff: {max_diff}"


skip_no_cuda = pytest.mark.skipif(
    not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()),
    reason="CUDA not available or disabled"
)

DEVICES = ["cpu"] + (["cuda"] if pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available() else [])


def assert_device(tensor, device):
    """Assert a tensor is on the expected device."""
    assert tensor.device.type == device, f"Expected device '{device}', got '{tensor.device.type}'"


_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@functools.lru_cache(maxsize=None)
def load_fixtures(filename):
    """Load pre-computed reference data from a fixture file (cached)."""
    path = os.path.join(_FIXTURE_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Fixture file not found: {path}\n"
            f"Run the scripts in tests/fixtures/ to generate it."
        )
    return dict(np.load(path, allow_pickle=False))


@pytest.fixture(autouse=True)
def _skip_on_cuda_oom():
    """Convert CUDA out-of-memory errors into test skips."""
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            pytest.skip(f"Insufficient GPU memory: {e}")
        raise
