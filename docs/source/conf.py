# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.join('../../piva')))
# sys.path.insert(0, os.path.abspath(os.path.join('../../')))
# sys.path.insert(0, os.path.abspath('_ext/'))


# -- Project information -----------------------------------------------------

project = 'piva'
copyright = '2025, Wojtek Pudelko'
author = 'Wojtek Pudelko'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'nbsphinx',
    'myst_nb',
    'sphinx.ext.mathjax',  # Optional, for LaTeX rendering
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = '../img/logo.png'
html_theme_options = {
    "logo_only": True,
}


# Load cutsom stylesheet
def setup(app):
    app.add_css_file('custom.css')


# Intersphinx config
intersphinx_mapping = {
    'pyqtgraph': 
    ('https://pyqtgraph.readthedocs.io/en/latest/', None),
    'python':
    ('https://docs.python.org/3', None),
    'numpy':
    ('https://numpy.org/doc/stable/', None),
    'matplotlib':
    ('https://matplotlib.org/', None),
    'data-slicer':
    ('https://data-slicer.readthedocs.io/en/latest/', None),
}

# -- Options for autodoc ----------------------------------------------------

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"
autodoc_member_order = 'bysource'

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"


# including jupyter notebook
nbsphinx_allow_errors = True  # Allow notebooks with errors to render
nbsphinx_execute = 'never'  # Always execute notebooks during the build
nb_execution_mode = 'off'
nb_render_text_lexer = 'python'  # Syntax highlighting for code cells
# nb_ipywidgets_js = True          # Support for Jupyter widgets (optional)

