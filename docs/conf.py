# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata
from pdf_gen import __version__, __name__ as __package_name__

# Get package metadata.
_metadata = importlib.metadata.metadata(__package_name__)

# --- Project information ---
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = __package_name__
author = _metadata.get("Author", _metadata.get("Author-email", "Ettore"))
copyright = f"2026-%Y, {author}"

version = __version__
release = version


# --- General configuration ---
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # Supporto per docstring in stile NumPy
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": True,
}

todo_include_todos = True

# Options for syntax highlighting.
pygments_style = "default"
pygments_dark_style = "default"

# Options for internationalization.
language = "en"

# Options for markup.
rst_prolog = f"""
.. |Python| replace:: `Python <https://www.python.org/>`__
.. |Sphinx| replace:: `Sphinx <https://www.sphinx-doc.org/en/master/>`__
.. |numpy| replace:: `NumPy <https://numpy.org/>`__
.. |scipy| replace:: `SciPy <https://scipy.org/>`__
.. |matplotlib| replace:: `Matplotlib <https://matplotlib.org/>`__
.. |GitHub| replace:: `GitHub <https://github.com/>`__
"""

# Options for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Options for templating.
templates_path = ["_templates"]


# --- Options for HTML output ---
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"

html_theme_options = {
    "awesome_external_links": True,
}

# Decommenta queste righe se aggiungi un logo e una favicon nella cartella _static/
# html_logo = "_static/logo_small.png"
# html_favicon = "_static/favicon.ico"

html_permalinks_icon = "<span>#</span>"
html_static_path = ["_static"]