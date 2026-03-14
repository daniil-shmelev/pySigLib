# Block signatory import if its native extension is broken.
# signatory's _impl.pyd may trigger a Windows fatal exception
# (STATUS_ENTRYPOINT_NOT_FOUND) when compiled against an incompatible
# PyTorch version.  The SEH exception can corrupt the CUDA driver
# context, causing unrelated numba-based tests to crash later.
#
# Pre-setting sys.modules["signatory"] = None makes subsequent
# `import signatory` raise ModuleNotFoundError without loading any DLLs.
# The try/except blocks in the test files handle this gracefully.

import subprocess, sys

def _signatory_is_importable():
    """Check in a subprocess so a crash can't affect this process."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import signatory"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False

if not _signatory_is_importable():
    sys.modules["signatory"] = None

# =========================================================================
# Shared test utilities
# =========================================================================

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
    a_ = np.array(a.cpu() if hasattr(a, 'cpu') else a)
    b_ = np.array(b.cpu() if hasattr(b, 'cpu') else b)
    if atol is None:
        s = single_atol if single_atol is not None else 1e-4
        d = double_atol if double_atol is not None else 1e-10
        atol = s if a_.dtype == np.float32 else d
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


def get_true_sig_coefs(multi_indices, X, *args, **kwargs):
    dim = X.shape[-1]
    sig = pysiglib.signature(X, *args, **kwargs)
    res = []
    for idx in multi_indices:
        flat_idx = 0
        for i in idx:
            flat_idx *= dim
            flat_idx += i + 1
        res.append(sig[..., flat_idx])
    return np.array(res).T
