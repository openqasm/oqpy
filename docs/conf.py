import os
import sys

from oqpy import __version__

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "OQpy"
copyright = "2022, OQpy Contributors <oqpy-contributors@amazon.com>"
author = "OQpy Contributors <oqpy-contributors@amazon.com>"
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "myst_parser",
]

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
