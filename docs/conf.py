# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys

DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_DIR.parent))

subprocess.run(["doxygen", "Doxyfile"], cwd=DOCS_DIR, check=True)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Load the version string directly from _version.py so we don't import the
# pysiglib package here (which would require the CMake-built _config.py).
_vspec = importlib.util.spec_from_file_location(
    "_pysiglib_version",
    os.path.join(os.path.dirname(__file__), "..", "pysiglib", "_version.py"),
)
_vmod = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(_vmod)
release = _vmod.__version__

project = "pysiglib"
copyright = "2026, Daniil Shmelev"
author = "Daniil Shmelev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "breathe",  # For C++
]

autodoc_typehints = "none"
nitpicky = True
nitpick_ignore = [("cpp:identifier", "uint64_t")]
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/2.14/", None),
}
intersphinx_timeout = 20

# Read the Docs has no compiled cpsig.so or _config.py. Mock load_siglib so
# autodoc can introspect the Python wrappers without trying to load native code.
autodoc_mock_imports = ["pysiglib.load_siglib", "pysiglib._config"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Syntax highlighting ------------------------------------------------------
pygments_style = "friendly"
pygments_dark_style = "monokai"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_theme_options = {
    "logo": {
        "image_light": "_static/logo_light.svg",
        "image_dark": "_static/logo_dark.svg",
    },
    "repository_url": "https://github.com/daniil-shmelev/pySigLib",
    "use_repository_button": True,
    "use_download_button": False,
    "show_toc_level": 2,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo_light.svg"
html_favicon = ""
add_module_names = False

html_title = "Documentation"
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://pysiglib.readthedocs.io/en/stable/",
)

rst_epilog = (
    """
|

Citation
------------
If you found this library useful in your research, please consider citing the paper:

.. code-block:: text

   @article{shmelev2025pysiglib,
     title={pySigLib-Fast Signature-Based Computations on CPU and GPU},
     author={Shmelev, Daniil and Salvi, Cristopher},
     journal={arXiv preprint arXiv:2509.10613},
     year={2025}
   }

.. |release| replace:: %s
"""
    % release
)

# -- C++ options -------------------------------------------------

breathe_projects = {"siglib": str(DOCS_DIR / "_build/doxygen/xml")}
breathe_default_project = "siglib"


def public_class_signature(app, what, name, obj, options, signature, return_annotation):
    if what == "class" and name.startswith("pysiglib."):
        parameters = inspect.signature(obj).parameters.values()
        if any(parameter.name.startswith("_") for parameter in parameters):
            public = [parameter.replace(annotation=inspect.Parameter.empty) for parameter in parameters if not parameter.name.startswith("_")]
            return str(inspect.Signature(public)), return_annotation
    return None


def setup(app):
    app.connect("autodoc-process-signature", public_class_signature)
