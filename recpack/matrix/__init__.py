# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""The matrix module contains the InteractionMatrix class to represent data within the Recpack framework.

.. currentmodule:: recpack.matrix

.. autosummary::
    :toctree: generated/

    InteractionMatrix

Example
~~~~~~~~~

An InteractionMatrix object can be constructed from a pandas DataFrame
with a row for each interaction.
The ``item`` and ``user`` values will be indices in the resulting matrix.
The following example constructs a 4x4 matrix, with 4 nonzero values::

    import pandas as pd

    from recpack.matrix import InteractionMatrix
    data = {
        "user": [3, 2, 1, 1],
        "item": [1, 1, 2, 3],
        "timestamp": [1613736000, 1613736300, 1613736600, 1613736900]
    }
    df = pd.DataFrame.from_dict(data)
    demo_data = InteractionMatrix(df, "item", "user", timestamp_ix="timestamp")

"""

from recpack.matrix.interaction_matrix import InteractionMatrix
from recpack.matrix.util import Matrix, to_binary, to_csr_matrix
