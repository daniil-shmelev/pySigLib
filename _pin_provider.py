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
"""scikit-build-core dynamic-metadata provider for the base pysiglib wheel.

Returns ``optional-dependencies`` with the ``cuda`` extra's
``pysiglib-cuda==<version>`` pin derived from ``pysiglib/_version.py`` (the
single source of truth). The ``jax`` extra is unaffected by version bumps so
its constraint is hard-coded here.

PEP 621 forbids expressions in ``[project.optional-dependencies]``, hence the
build backend's ``[tool.scikit-build.metadata.optional-dependencies]``
indirection.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# This file lives at the repo root next to pyproject.toml; pysiglib/_version.py
# is one level down.
_VERSION_FILE = Path(__file__).resolve().parent / "pysiglib" / "_version.py"


def _read_pysiglib_version() -> str:
    text = _VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"(?P<v>[^"]+)"', text, re.MULTILINE)
    if match is None:
        raise RuntimeError(
            "Could not parse __version__ from " + str(_VERSION_FILE)
        )
    return match.group("v")


def dynamic_metadata(field: str, _settings: dict[str, Any] | None) -> dict[str, list[str]]:
    if field != "optional-dependencies":
        raise ValueError(
            "only 'optional-dependencies' is supported, got " + repr(field)
        )
    version = _read_pysiglib_version()
    return {
        "cuda": ["pysiglib-cuda==" + version],
        # jax>=0.9.1 is the first release whose XLA FFI framework version is 0.3,
        # matching what our FFI handlers are compiled against. Older jax versions
        # will fail at handler registration with an API-version-mismatch error.
        # array-api-compat 1.13 recognizes current JAX array classes.
        "jax": ["jax>=0.9.1", "array-api-compat>=1.13"],
        "torch": ["torch"],
    }


def get_requires_for_dynamic_metadata(_settings: dict[str, Any] | None) -> list[str]:
    return []
