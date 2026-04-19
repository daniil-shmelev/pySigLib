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
"""scikit-build-core dynamic-metadata provider.

Returns ``dependencies = ["pysiglib==<version>"]`` for pysiglib-cuda's wheel
metadata, where ``<version>`` is read from ``pysiglib/_version.py`` (the
single source of truth). This keeps the cusig <-> cpsig ABI lockstep without
a literal version string in pyproject.toml.

PEP 621 forbids expressions in ``[project] dependencies``, hence the build
backend's ``[tool.scikit-build.metadata.dependencies]`` indirection.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# plugins/cuda/_pin_provider.py -> repo root is two levels up
_VERSION_FILE = Path(__file__).resolve().parent.parent.parent / "pysiglib" / "_version.py"


def _read_pysiglib_version() -> str:
    text = _VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"(?P<v>[^"]+)"', text, re.MULTILINE)
    if match is None:
        raise RuntimeError(
            "Could not parse __version__ from " + str(_VERSION_FILE)
        )
    return match.group("v")


def dynamic_metadata(field: str, _settings: dict[str, Any] | None) -> list[str]:
    if field != "dependencies":
        raise ValueError("only 'dependencies' is supported, got " + repr(field))
    return ["pysiglib==" + _read_pysiglib_version()]


def get_requires_for_dynamic_metadata(_settings: dict[str, Any] | None) -> list[str]:
    return []
