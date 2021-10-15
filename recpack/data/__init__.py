"""The data module contains functionality for representing data, and loading datasets.

.. currentmodule:: recpack.data

Data
------

.. autosummary::
    :toctree: generated/

    matrix.InteractionMatrix
    matrix.to_binary
    matrix.to_csr_matrix

Example
~~~~~~~~~

An InteractionMatrix object can be constructed from a pandas DataFrame
with a row for each interaction.
The ``item`` and ``user`` values will be indices in the resulting matrix.
The following example constructs a 4x4 matrix, with 4 nonzero values::

    import pandas as pd

    from recpack.data.matrix import InteractionMatrix
    data = {
        "user": [3, 2, 1, 1],
        "item": [1, 1, 2, 3],
        "timestamp": [1613736000, 1613736300, 1613736600, 1613736900]
    }
    df = pd.DataFrame.from_dict(data)
    demo_data = InteractionMatrix(df, "item", "user", timestamp_ix="timestamp")


Datasets
----------

Recpack provides a selection of datasets ready for use.

.. autosummary::
    :toctree: generated/

    datasets.Dataset
    datasets.DummyDataset
    datasets.CiteULike
    datasets.MovieLens25M
    datasets.RecsysChallenge2015

Example
~~~~~~~~~

Loading a dataset only takes a couple of lines.
If the file specified does not exist, the dataset is downloaded and written into this file.
Subsequent loading of the dataset then happens from this file. ::

    from recpack.data.datasets import MovieLens25M

    # Folder needs to exist, file will be downloaded if not present
    # This can take a while
    ml_loader = MovieLens25M('datasets/ml-25m.csv')
    data = ml_loader.load_interaction_matrix()

Each dataset has its own default preprocessing steps, documented in the classes respectively.
To use custom preprocessing a couple more lines should be added to the example. ::

    from recpack.data.datasets import MovieLens25M
    from recpack.preprocessing.filters import MinRating, MinUsersPerItem, MinItemsPerUser

    ml_loader = MovieLens25M('datasets/ml-25m.csv', preprocess_default=False)
    # Only consider ratings 4 or higher as interactions
    ml_loader.add_filter(MinRating(
        4,
        ml_loader.RATING_IX,
    ))
    # Keep users with at least 5 interactions
    ml_loader.add_filter(MinItemsPerUser(
        5,
        ml_loader.ITEM_IX,
        ml_loader.USER_IX,
    ))
    # Keep items with at least 30 interactions
    ml_loader.add_filter(MinUsersPerItem(
        30,
        ml_loader.ITEM_IX,
        ml_loader.USER_IX,
    ))

    data = ml_loader.load_interaction_matrix()
"""
