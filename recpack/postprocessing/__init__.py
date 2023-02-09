# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""Module with classes for postprocessing recommendations

Postprocessor
-----------------------

.. currentmodule:: recpack.postprocessing.postprocessors

The post processor provides all functionality to bundle postprocessing in  one step.
This makes it less prone to error, when applying the same processing to different recommendation outputs.
It also makes initialisation more declarative, rather than having to chain outputs yourself.

.. autosummary::
    :toctree: generated/

    Postprocessor

Filters
-----------------------

.. currentmodule:: recpack.postprocessing.filters

Filters provide functionality to retain only a part of the generated
recommendations, setting the rest to 0.

.. autosummary::
    :toctree: generated/

    PostFilter
    ExcludeItems
    SelectItems


Filters can be applied manually, simply pass the csr_matrix to be processed to the apply function.::

    import numpy as np
    from scipy.sparse import csr_matrix

    from recpack.postprocessing.filters import ExcludeItems

    # Generate random recommendations
    AMOUNT_OF_USERS = 20
    AMOUNT_OF_ITEMS = 5
    recommendations = csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS)))

    # Remove all recommendations for items 1 and 3
    items_to_remove = [1, 3]
    filter = ExcludeItems(items_to_remove)
    processed = filter.apply(recommendations)

The preferred way to use filters though is through the :class:`recpack.postprocessing.postprocessors.Postprocessor`.
That way all postprocessing happens in a more controlled way, leaving less room for error.::

    import numpy as np
    from scipy.sparse import csr_matrix

    from recpack.postprocessing.filters import ExcludeItems, RemovePreviousInteractions
    from recpack.postprocessing.postprocessors import Postprocessor

    # Generate random recommendations
    AMOUNT_OF_USERS = 20
    AMOUNT_OF_ITEMS = 5
    recommendations_1 = csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS)))
    recommendations_2 = csr_matrix(np.random.random_sample(size=(AMOUNT_OF_USERS, AMOUNT_OF_ITEMS)))

    # Construct processor which removes items 1 and 3.
    processor = Postprocessor()
    processor.add_filter(ExcludeItems([1, 3]))

    # Apply processing step to both recommendation matrices.
    processed_1, processed_2 = process.process_many(recommendations_1, recommendations_2)

"""
