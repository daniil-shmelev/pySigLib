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

from __future__ import annotations

CPP_ERR_MSG = {
    1 : "Failed to allocate memory",
    2: "Invalid argument",
    3: "Out of range",
    4: "Filesystem error",
    5: "Could not find prepared cache. Please call the corresponding preparation function with the correct parameters.",
    6: "Directory does not exist",
    7: "Failed to get default cache directory. Please ensure default directory exists or provide one explicitly using pysiglib.set_cache_dir",
    8: "Unexpected internal error. Cache directory was not set correctly.",
    9: "Tried to read an invalid cache file. Cache may have been corrupted.",
    10: "Runtime error",
    11: "Unknown exception"
}


def _native_error_message(device):
    from ..load_siglib import BUILT_WITH_CUDA, CPSIG, CUSIG

    device_type = getattr(device, "type", device)
    library = CUSIG if device_type == "cuda" and BUILT_WITH_CUDA else CPSIG
    getter_name = (
        "cusig_last_error_message" if library is CUSIG
        else "cpsig_last_error_message"
    )
    getter = getattr(library, getter_name, None)
    if getter is None:
        return ""
    message = getter()
    return message.decode("utf-8", errors="replace") if message else ""


def err_msg(err_code, device="cpu"):
    if err_code < 100000:
        message = CPP_ERR_MSG[err_code] + " (" + str(err_code) + ")"
    elif err_code == 100500:
        message = "CUDA error: named symbol not found (500). pysiglib: This error may suggest your GPU's compute capability is currently not supported by pysiglib. Please contact the developer."
    else:
        message = "CUDA error (" + str(err_code - 100000) + ")"
    detail = _native_error_message(device)
    return message + (": " + detail if detail else "")
