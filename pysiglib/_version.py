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

# Single source of truth for pysiglib's version. Read by:
#   - pysiglib/__init__.py (runtime __version__)
#   - pyproject.toml [tool.scikit-build.metadata.version] (wheel metadata)
#   - plugins/cuda/pyproject.toml (plugin wheel metadata; same version for ABI lockstep)
#   - docs/conf.py (Sphinx release tag)
#   - plugins/cuda/pysiglib_cuda/__init__.py (runtime ABI guard)
__version__ = "3.0.3rc0"
