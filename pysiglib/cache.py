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

from pathlib import Path

from .error_codes import err_msg
from .load_siglib import CPSIG, CUSIG, BUILT_WITH_CUDA
from .param_checks import check_type


def set_cache_dir(
        dir : str
):
    """
    Sets the directory used by pySigLib disk caches when ``use_disk=True``.
    If the cache directory is not explicitly
    set by a call to this function, a default directory will be used:

    - Windows: ``%LOCALAPPDATA%``
    - Linux: ``~/.cache``
    - Mac: ``~/Library/Caches``

    This function is not thread safe.

    :param dir: Path to cache directory
    :type dir: str

    Example usage:
    ----------------

    .. code-block::

        import pysiglib

        # Set cache dir to a folder "my_cache_dir" in the current working directory
        pysiglib.set_cache_dir("./my_cache_dir")

        pysiglib.prepare_log_sig(5, 3, lead_lag=True, method=2, use_disk=True)

        X = torch.rand((32,100,5))
        X_log_sig = pysiglib.log_sig(X, 3, lead_lag=True, method=2)

    """
    check_type(dir, "dir", str)
    p = Path(dir)
    if not p.exists():
        raise ValueError(f"Path does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Path is not a directory: {p}")

    err_code = CPSIG.set_cache_dir(dir.encode("utf-8"))
    if err_code:
        raise Exception("Error in pysiglib.set_cache_dir: " + err_msg(err_code))

    if BUILT_WITH_CUDA:
        err_code = CUSIG.set_cache_dir_cuda(dir.encode("utf-8"))
        if err_code:
            raise Exception("Error in pysiglib.set_cache_dir (CUDA): " + err_msg(err_code))


def clear_cache(
        *,
        use_disk : bool = False,
        device : str = "both"
):
    """
    Clears all pySigLib native caches on the requested devices. On CPU this
    includes log-signature, BCH, branched-signature, and prepared Volterra
    signature caches. On CUDA this includes log-signature, BCH,
    branched-signature, branched-log-signature, and workspace caches.

    :param use_disk: If ``False``, will clear the cache from memory only.
        If ``True``, will also clear the shared disk cache directory.
        See additionally the documentation for
        ``pysiglib.set_cache_dir``.
    :type use_disk: bool
    :param device: Which device caches to clear. Must be one of ``"cpu"``, ``"cuda"``,
        or ``"both"`` (default).
    :type device: str

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        pysiglib.prepare_log_sig(dimension=5, degree=4, method=2, use_disk=True)

        path = torch.rand((10, 100, 5))
        log_sig = pysiglib.log_sig(path, 4, n_jobs = -1)
        print(log_sig)

        pysiglib.clear_cache() # Clear native caches from memory but keep disk caches

    """
    if device not in ("cpu", "cuda", "both"):
        raise ValueError("device must be 'cpu', 'cuda', or 'both'")

    if device in ("cpu", "both"):
        err_code = CPSIG.clear_cache(use_disk)
        if err_code:
            raise Exception("Error in pysiglib.clear_cache: " + err_msg(err_code))

    if BUILT_WITH_CUDA and device in ("cuda", "both"):
        err_code = CUSIG.clear_cache_cuda(use_disk)
        if err_code:
            raise Exception("Error in pysiglib.clear_cache (CUDA): " + err_msg(err_code))
