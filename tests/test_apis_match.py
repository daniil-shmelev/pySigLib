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

import inspect
import pytest

import pysiglib as base_api
import pysiglib.torch_api as torch_api

# Backprop functions excluded from torch_api
EXCLUDED_SUFFIXES = ("_backprop",)

# Functions not yet added to torch_api (TODO: add torch autograd wrappers)
TEMPORARILY_EXCLUDED = {"log_sig_combine"}

def get_public_functions(module):
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if (
            not name.startswith("_")
            and not name.endswith(EXCLUDED_SUFFIXES)
            and name not in TEMPORARILY_EXCLUDED
        )
    }


BASE_API_FUNCS = get_public_functions(base_api)
TORCH_API_FUNCS = get_public_functions(torch_api)

def test_same_function_names():
    """Check both APIs expose the same public functions."""
    assert BASE_API_FUNCS.keys() == TORCH_API_FUNCS.keys(), (
        f"Function sets differ:\n"
        f"Only in base_api: {BASE_API_FUNCS.keys() - TORCH_API_FUNCS.keys()}\n"
        f"Only in torch_api: {TORCH_API_FUNCS.keys() - BASE_API_FUNCS.keys()}"
    )


def test_torch_api_has_no_backprop_functions():
    """torch_api must not expose backprop functions."""
    bad = [
        name for name, _ in inspect.getmembers(torch_api, inspect.isfunction)
        if name.endswith("_backprop")
    ]
    assert not bad, f"torch_api should not expose backprop functions: {bad}"


@pytest.mark.parametrize("name", sorted(BASE_API_FUNCS.keys()))
def test_function_signature_and_docstring_match(name):
    """Each function matches signature and docstring."""
    assert name in TORCH_API_FUNCS, f"{name} missing from torch_api"

    sig_base_api = inspect.signature(BASE_API_FUNCS[name])
    sig_torch_api = inspect.signature(TORCH_API_FUNCS[name])

    assert sig_base_api == sig_torch_api, (
        f"Signature mismatch for '{name}':\n"
        f"base_api: {sig_base_api}\n"
        f"torch_api: {sig_torch_api}\n"
    )

    doc_base_api = (inspect.getdoc(BASE_API_FUNCS[name]) or "").strip()
    doc_torch_api = (inspect.getdoc(TORCH_API_FUNCS[name]) or "").strip()
    assert doc_base_api == doc_torch_api, f"Docstring mismatch for '{name}'"

