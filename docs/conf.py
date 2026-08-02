# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import importlib.util
import subprocess


def run_doxygen():
    """Run doxygen to generate XML before Sphinx builds."""
    if not os.path.exists("_build/doxygen/xml"):
        os.makedirs("_build/doxygen/xml")
    subprocess.call("doxygen Doxyfile", shell=True)


run_doxygen()

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
    "sphinx_copybutton",
    "sphinx_design",
    "breathe",  # For C++
]

autodoc_typehints = "none"

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

breathe_projects = {"siglib": "_build/doxygen/xml"}
breathe_default_project = "siglib"
