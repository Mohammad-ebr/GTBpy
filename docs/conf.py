# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'GTBpy Documentation'
copyright = '2024, Mohammad Ebrahimi'
author = 'Mohammad Ebrahimi'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# New
autodoc_member_order = 'bysource'

# -- Custom Skip Function ----------------------------------------------------

def skip_member(app, what, name, obj, skip, options):
    # Skip if Sphinx is already skipping it
    if skip:
        return True

    # Skip if the member has no docstring
    docstring = getattr(obj, '__doc__', None)
    if not docstring:
        return True  # Skip this member

    # Do not skip the member
    return False

def setup(app):
    app.connect('autodoc-skip-member', skip_member)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
