# Documentation

## Rendering the documentation
Rendering requires the installation the extra documentation dependendies:
```
pip install -e '.[doc]'
```

The docs can then be rendered using the make file.
* go to doc folder : `cd doc`
* Render html: `make html`

To clear the docs: `make clean`

## Adding to the documentation

### Adding to the algorithm documentation
Algorithm documentation is auto generated based on the docstring in `recpack.algorithms.__init__.py`.

Documentation of each algorithm is rendered based on its docstrings.

To add a new algorithm to the docs, you need to add it to one of the `autosummary` entries in the docstring.

### Adding to existing modules
The documentation for existing modules is written in the docstring of each module.
These docstrings are written in reStructuredText. Typical modules contain:
* An explanation of the module
* Example(s): some snippets of code that explain usage of the module
* The links to the classes and functions in this module. Using the `autosummary` keyword


Example for autosummary, will render a table with the 4 classes and their 1 line summary
```
.. currentmodule:: recpack.data.datasets

.. autosummary::
    :toctree: generated/

    Dataset
    CiteULike
    MovieLens25M
    RecsysChallenge2015
```

To add a class to a module, make sure it has the necessary docstrings, and add it to the autosummary list.

### Creating a new module
To create a new module you have to create a `.rst` file, with as filename the import path.
eg. `recpack.data` is the rst file for the data module.

In this rst file you can write rst to document the class, but we suggest strongly to use the module docstring for almost all use cases. That way docs are kept close to the code.
As such most rst files will only contain a title and an `automodule` entry.

eg.

```
recpack.data
====================

.. automodule:: recpack.data
    :members:
```
