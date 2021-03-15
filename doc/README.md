# Documentation

## Rendering the documentation
Rendering requires the installation of `sphinx` and `sphinx-rtd-theme`:
```
pip install sphinx sphinx-rtd-theme
```

The docs can then be rendered using the make file.
* Render html: `make html`

To clear the docs: `make clean`

## Adding to the documentation

### Adding to the algorithm documentation
Algorithm documentation is auto generated based on the docstring in `recpack.algorithms.__ini__.py`.

Documentation of each algorithm is rendered based on its docstrings.

To add a new algorithm to the docs, you need to add it to one of the `autosummary` entries in the docstring.
To generate the page run `sphinx-autogen -o source/generated/ source/recpack.algorithms.rst -t source/_templates/` This will generate the documentation pages in the source/generated directory.

### Adding to existing modules
The documentation for existing modules is written in the docstring of each module.
These docstrings are written in reStructuredText. Typical modules contain:
* overview of important members, using `autosummary`
* Example(s) some snippets of code that explain usage of the module
* End with a header under which the detailed api of the members of the module will be rendered.


Example for autosummary, will render a table with the 4 classes and their 1 line summary
```
.. currentmodule:: recpack.data.datasets

.. autosummary::

    Dataset
    CiteULike
    MovieLens25M
    RecsysChallenge2015
```

To add a class to a module, make sure it has the necessary docstrings, and add it to the autosummary list.

### Creating a new module
To create a new module you have to create a `.rst` file, with as filename the import path.
eg. `recpack.data` is the rst file for the data module.

In this rst file you can write rst to document the class, but we suggest strongly to use the module docstring for almost all use cases. That way docs are kept even closer to the code.
As such most rst files will contain a title and an `automodule` entry.

eg.

```
recpack.data
====================

.. automodule:: recpack.data
    :members:
```
