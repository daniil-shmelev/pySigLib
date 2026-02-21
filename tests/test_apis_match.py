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

# Args which appear in base api but not torch_api
ALLOWED_MISSING_ARGS_IN_TORCH_API = {
    "sig_kernel": {"return_grid"},
    "sig_kernel_gram": {"return_grid"}
}

# Args which appear in torch_api but not base api
ALLOWED_MISSING_ARGS_IN_BASE_API = {
    "sig_kernel_gram": {"save_kernel"}
}

def get_public_functions(module):
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if (
            not name.startswith("_")
            and not name.endswith(EXCLUDED_SUFFIXES)
        )
    }


def normalize_signature(sig, remove_params=frozenset()):
    """Return signature with selected parameters removed."""
    params = [
        p for name, p in sig.parameters.items()
        if name not in remove_params
    ]
    return sig.replace(parameters=params)

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

    remove_base_api = ALLOWED_MISSING_ARGS_IN_TORCH_API.get(name, set())
    rename_torch_api = ALLOWED_MISSING_ARGS_IN_BASE_API.get(name, set())

    sig_base_api_norm = normalize_signature(sig_base_api, remove_base_api)
    sig_torch_api_norm = normalize_signature(sig_torch_api, rename_torch_api)

    assert sig_base_api_norm == sig_torch_api_norm, (
        f"Signature mismatch for '{name}':\n"
        f"base_api: {sig_base_api}\n"
        f"torch_api: {sig_torch_api}\n"
        f"Allowed missing in base_api: {rename_torch_api}"
        f"Allowed missing in torch_api: {remove_base_api}"
    )

    doc_base_api = (inspect.getdoc(BASE_API_FUNCS[name]) or "").strip()
    doc_torch_api = (inspect.getdoc(TORCH_API_FUNCS[name]) or "").strip()
    assert doc_base_api == doc_torch_api, f"Docstring mismatch for '{name}'"


def test_expected_missing_args_really_missing():
    """Ensure declared missing args are actually absent."""
    for name, missing in ALLOWED_MISSING_ARGS_IN_TORCH_API.items():
        sig_torch_api = inspect.signature(TORCH_API_FUNCS[name])
        for arg in missing:
            assert arg not in sig_torch_api.parameters, (
                f"{name}: '{arg}' unexpectedly present in torch_api"
            )
