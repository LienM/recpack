# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from recpack.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from recpack.algorithms.time_aware_item_knn.base import TARSItemKNNCoocDistance
from recpack.matrix import InteractionMatrix
from recpack.util import get_top_K_values


class TARSItemKNNXia(TARSItemKNNCoocDistance):
    """Time Decaying Nearest Neighbours model.

    First described in 'Dynamic Item-Based Recommendation Algorithm with Time Decay'
    Chaolun Xia, Xiaohong Jiang, Sen Liu, Zhaobo Luo, Zhang Yu,
    2010 Sixth International Conference on Natural Computation (ICNC 2010).

    For each item the K most similar items are computed during fit.
    Decay function parameter decides how to compute the similarity between two items.

    .. math::
        \\text{sim}(i,j) = \\sum\\limits_{u=1}^{|U|} R_{u,i} \\cdot R_{u,j} \\cdot \\theta(|T_{u,i} - T_{u,j}|)

    Supported options are: ``"concave"``, ``"convex"`` and ``"linear"``.

    - Concave decay function between item i and j is computed as:
    .. math::
        \\theta(x) = \\alpha^{x} \\text{for} \\alpha \\in  [0, 1]

    - Convex decay function between item i and j is computed as:
    .. math::
        \\theta(x) = 1 - \\beta^{t-x} \\text{for} \\beta \\in  (0, 1)

    - Linear decay function between item i and j is computed as:
    .. math::
        \\theta(x) = 1 - \\frac{x}{t} \\cdot \\gamma \\text{for} \\gamma \\in  [0, 1]

    With ``t`` the absolute time interval between the interactions on both items.

    **Example of use**::

        import numpy as np
        from recpack.matrix import InteractionMatrix
        from recpack.algorithms.experimental import TimeDecayingNearestNeighbour

        USER_IX = InteractionMatrix.USER_IX
        ITEM_IX = InteractionMatrix.ITEM_IX
        TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX

        data = {
            TIMESTAMP_IX: [1, 1, 1, 2, 3, 4],
            ITEM_IX: [0, 0, 1, 2, 1, 2],
            USER_IX: [0, 1, 2, 2, 1, 0]
        }
        df = pd.DataFrame.from_dict(data)

        X = InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)

        # We'll only keep the closest neighbour for each item.
        # Default uses concave decay function with coefficient equal to 0.5
        algo = TimeDecayingNearestNeighbour(K=1)
        # Fit algorithm
        algo.fit(X)

        # We can inspect the fitted model
        print(algo.similarity_matrix_.nnz)
        # 3

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200.
    :type K: int, optional
    :param fit_decay: How strongly the decay function should influence the scores,
        make sure to pick a value in the correct interval
        for the selected decay function.
        Defaults to 0.5.
    :type fit_decay: float, optional
    :param decay_function: The decay function that needs to
        be applied on the item similarity scores.
        Defaults to `"concave"`.
    :type decay_function: str, optional
    :param decay_interval: Defines the basic time interval unit in seconds.
        Defaults to 24*3600.
    :typ decay_interval: int, optional
    """

    SUPPORTED_DECAY_FUNCTIONS = ["concave", "convex", "linear"]
    """The supported Decay function options."""

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 0.5,
        decay_function: str = "convex",
        decay_interval: int = 24 * 3600,
    ):
        if decay_function not in self.SUPPORTED_DECAY_FUNCTIONS:
            raise ValueError(f"decay_function {decay_function} not supported")

        super().__init__(
            K=K,
            fit_decay=fit_decay,
            predict_decay=0,
            decay_interval=decay_interval,
            similarity="cooc",
            decay_function=decay_function,
        )
