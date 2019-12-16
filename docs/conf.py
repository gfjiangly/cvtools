# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# import mock
#
# sys.path.insert(0, os.path.abspath('..'))
#
# MOCK_MODULES = [
#     'numpy', 'opencv-python', 'matplotlib', 'matplotlib.pyplot', 'pillow',
#     'tqdm', 'mmcv', 'shapely', 'terminaltables'
# ]
#
# for mod_name in MOCK_MODULES:
#     sys.modules[mod_name] = mock.Mock()

version_file = '../cvtools/version.py'
with open(version_file, 'r') as f:
    exec(compile(f.read(), version_file, 'exec'))
__version__ = locals()['__version__']

# -- Project information -----------------------------------------------------

project = 'cvtools'
copyright = '2019, jiang.g.f'
author = 'jiang.g.f'

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'recommonmark',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'zh_CN'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# 在2.0版更改: 默认值从'contents'更改为 'index'. 但ReadThedocs的Sphinx的版本较低
master_doc = 'index'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


def run_apidoc(_):
    from sphinx.ext.apidoc import main
    parentFolder = os.path.join(os.path.dirname(__file__), '..')
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(parentFolder)
    module = os.path.join(parentFolder, 'cvtools')
    output_path = os.path.join(cur_dir, 'api')
    main(['-e', '-f', '-o', output_path, module])


def setup(app):
    # overrides for wide tables in RTD theme
    app.add_stylesheet('theme_overrides.css')
    # trigger the run_apidoc
    app.connect('builder-inited', run_apidoc)
