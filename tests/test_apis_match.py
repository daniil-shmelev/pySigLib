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

# Backprop functions are intentionally not exposed in the derivable APIs.
EXCLUDED_SUFFIXES = ("_backprop",)


def get_public_members(module):
    """Return {name: obj} for public top-level functions and classes."""
    return {
        name: obj
        for name, obj in inspect.getmembers(module)
        if (
            not name.startswith("_")
            and not name.endswith(EXCLUDED_SUFFIXES)
            and (inspect.isfunction(obj) or inspect.isclass(obj))
        )
    }


def _params_fingerprint(obj):
    """Name / kind / default tuple for each public parameter.

    Annotations are ignored to side-step noise from ``from __future__ import
    annotations`` (runtime annotations become strings rather than actual type
    objects). Leading-underscore parameters are filtered out so that private
    override hooks on base classes don't leak into the parity check.
    """
    sig = inspect.signature(obj)
    return [
        (p.name, p.kind, p.default)
        for p in sig.parameters.values()
        if not p.name.startswith("_")
    ]


BASE_API_MEMBERS = get_public_members(base_api)
TORCH_API_MEMBERS = get_public_members(torch_api)

try:
    import pysiglib.jax_api as jax_api
    JAX_AVAILABLE = True
    JAX_API_MEMBERS = get_public_members(jax_api)
except ImportError:
    JAX_AVAILABLE = False
    JAX_API_MEMBERS = {}


# ---------------------------------------------------------------------------
# Membership tests
# ---------------------------------------------------------------------------

def test_same_member_names_torch():
    """base_api and torch_api must expose the same public functions and classes."""
    assert BASE_API_MEMBERS.keys() == TORCH_API_MEMBERS.keys(), (
        f"Member sets differ:\n"
        f"Only in base_api: {sorted(BASE_API_MEMBERS.keys() - TORCH_API_MEMBERS.keys())}\n"
        f"Only in torch_api: {sorted(TORCH_API_MEMBERS.keys() - BASE_API_MEMBERS.keys())}"
    )


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_same_member_names_jax():
    """base_api and jax_api must expose the same public functions and classes."""
    assert BASE_API_MEMBERS.keys() == JAX_API_MEMBERS.keys(), (
        f"Member sets differ:\n"
        f"Only in base_api: {sorted(BASE_API_MEMBERS.keys() - JAX_API_MEMBERS.keys())}\n"
        f"Only in jax_api: {sorted(JAX_API_MEMBERS.keys() - BASE_API_MEMBERS.keys())}"
    )


def test_torch_api_has_no_backprop_functions():
    """torch_api must not expose backprop functions."""
    bad = [
        name for name, _ in inspect.getmembers(torch_api, inspect.isfunction)
        if name.endswith("_backprop")
    ]
    assert not bad, f"torch_api should not expose backprop functions: {bad}"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_jax_api_has_no_backprop_functions():
    """jax_api must not expose backprop functions."""
    bad = [
        name for name, _ in inspect.getmembers(jax_api, inspect.isfunction)
        if name.endswith("_backprop")
    ]
    assert not bad, f"jax_api should not expose backprop functions: {bad}"


# ---------------------------------------------------------------------------
# Signature + docstring parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(BASE_API_MEMBERS.keys()))
def test_signature_match_torch(name):
    """Each base_api member has a torch_api counterpart with matching signature."""
    assert name in TORCH_API_MEMBERS, f"{name} missing from torch_api"

    base_params = _params_fingerprint(BASE_API_MEMBERS[name])
    torch_params = _params_fingerprint(TORCH_API_MEMBERS[name])
    assert base_params == torch_params, (
        f"Signature mismatch for '{name}':\n"
        f"base_api:  {base_params}\n"
        f"torch_api: {torch_params}"
    )



@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
@pytest.mark.parametrize("name", sorted(BASE_API_MEMBERS.keys()))
def test_signature_match_jax(name):
    """Each base_api member has a jax_api counterpart with matching signature."""
    assert name in JAX_API_MEMBERS, f"{name} missing from jax_api"

    if name == "StaticKernel":
        return  # JAX kernels use a callable protocol and JAX autodiff.
    base_params = _params_fingerprint(BASE_API_MEMBERS[name])
    jax_params = _params_fingerprint(JAX_API_MEMBERS[name])
    assert base_params == jax_params, (
        f"Signature mismatch for '{name}':\n"
        f"base_api: {base_params}\n"
        f"jax_api:  {jax_params}"
    )
