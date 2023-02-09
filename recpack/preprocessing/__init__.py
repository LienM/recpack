# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""
.. currentmodule:: recpack.preprocessing

In collaborative filtering it is customary to transform the data into a user-item interaction matrix.
To do so efficiently, :class:`preprocessors.DataFramePreprocessor` transforms
the user and item identifiers into matrix indices.
Secondly, it is common to apply some filtering to your raw dataset.
For this purpose RecPack provides a set of preimplemented Filters.


Preprocessors
-----------------------

.. currentmodule:: recpack.preprocessing.preprocessors

The preprocessor provides all functionality to bundle preprocessing in  one step.
This makes it less prone to error, when applying the same  processing to different input data.
It also makes initialisation more declarative, rather than having to chain outputs yourself.

.. autosummary::
    :toctree: generated/

    DataFramePreprocessor

Filters
---------

.. currentmodule:: recpack.preprocessing.filters

Preprocessing is a fundamental part of any experiment.
The raw data needs to be cleaned up, to make an optimally useful dataset.

.. autosummary::
    :toctree: generated/

    Filter
    MinUsersPerItem
    MinItemsPerUser
    MaxItemsPerUser
    NMostPopular
    NMostRecent
    Deduplicate
    MinRating

Filters can be applied manually, simply pass the DataFrame to be processed to the apply function.::

    import pandas as pd
    from recpack.preprocessing.filters import NMostPopular

    data = {
        "user": [3, 3, 2, 1, 1],
        "item": [1, 2, 1, 2, 3],
        "timestamp": [1613736000, 1613736005, 1613736300, 1613736600, 1613736900]
    }
    df = pd.DataFrame.from_dict(data)
    # parameters are N, and the item_ix
    f = NMostPopular(2, "item")
    # processed_df will contain rows of items 1 and 2
    processed_df = f.apply(df)

The preferred way to use filters though is through
the :class:`recpack.preprocessing.preprocessors.DataFramePreprocessor`.
That way all preprocessing happens in a more controlled way, leaving less room for error.::

    import pandas as pd
    from recpack.preprocessing.preprocessors import DataFramePreprocessor
    from recpack.preprocessing.filters import Deduplicate

    data = {
        "user": [3, 3, 2, 1, 1],
        "item": [1, 1, 1, 2, 3],
        "timestamp": [1613736000, 1613736005, 1613736300, 1613736600, 1613736900]
    }
    df = pd.DataFrame.from_dict(data)

    df_pp = DataFramePreprocessor("item", "user", "timestamp")
    df_pp.add_filter(
        Deduplicate("item", "user", "timestamp")
    )
    # Output will be an InteractionMatrix of shape (3,3)
    # With all interactions except the second (3, 1) interaction.
    im = df_pp.process(df)
"""
