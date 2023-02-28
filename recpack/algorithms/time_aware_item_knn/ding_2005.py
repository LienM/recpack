# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN


class TARSItemKNNDing(TARSItemKNN):
    """Time aware algorithm applying a decay on the user history when predicting.

    Algorithm as presented in Ding, Yi, and Xue Li. "Time weight collaborative filtering."
    Computation of the similarity matrix is the same as normal ItemKNN.
    When predicting however the user's older interactions are given less weight in the final prediction score.

    .. math::

        \\text{sim}(u, i) = \\sum\\limits_{j \\in X_u} e^{-\\alpha \\cdot \\delta t_{u,j}} \\cdot \\text{sim}(i, j)

    Where :math:`\\alpha` is the predict_decay parameter.

    :param K: Amount of neighbours to keep. Defaults to 200.
    :type K: int, Optional
    :param predict_decay: Defines the decay scaling used for decay during prediction.
        Defaults to (1/3600).
    :type predict_decay: float, Optional
    :param similarity: Which similarity measure to use. Defaults to `"cosine"`.
        ``["cosine", "conditional_probability"]`` are supported.
    :type similarity: str, Optional
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability"]

    def __init__(self, K: int = 200, predict_decay: float = 1 / 3600, similarity: str = "cosine"):
        super().__init__(K=K, fit_decay=0, predict_decay=predict_decay, similarity=similarity)
