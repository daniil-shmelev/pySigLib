# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import subprocess

def run_doxygen():
    """Run doxygen to generate XML before Sphinx builds."""
    if not os.path.exists("_build/doxygen/xml"):
        os.makedirs("_build/doxygen/xml")
    subprocess.call("doxygen Doxyfile", shell=True)

run_doxygen()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pysiglib'
copyright = '2025, Daniil Shmelev'
author = 'Daniil Shmelev'
release = '1.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'breathe' #For C++
]

autodoc_typehints = "none"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'

html_theme_options = {
    "logo": {
          "image_light": "_static/logo_light.png",
          "image_dark": "_static/logo_dark.png",
       },
    "repository_url": "https://github.com/daniil-shmelev/pySigLib",
    "use_repository_button": True,
}

html_static_path = ['_static']
html_logo = "_static/logo_light.png"
add_module_names = False

html_title = "Documentation"

rst_epilog = """
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
""" % release

# -- C++ options -------------------------------------------------

breathe_projects = {
    "siglib": "_build/doxygen/xml"
}
breathe_default_project = "siglib"