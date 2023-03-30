# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
from recpack.algorithms.time_aware_item_knn.base import TARSItemKNNCoocDistance


class TARSItemKNNXia(TARSItemKNNCoocDistance):
    """Time aware variant of ItemKNN that considers the time between two interactions
    when computing similarity between two items.

    First described in
    C. Xia, X. Jiang, Sen Liu, Zhaobo Luo and Zhang Yu,
    "Dynamic item-based recommendation algorithm with time decay,"
    2010 Sixth International Conference on Natural Computation,
    Yantai, 2010, pp. 242-247, doi: 10.1109/ICNC.2010.5582899.

    For each item the K most similar items are computed during fit.
    The ``decay_function`` parameter decides how to compute the similarity between two items.

    .. math::
        \\text{sim}(i,j) = \\sum\\limits_{u \\in U} R_{u,i} \\cdot R_{u,j} \\cdot \\theta(|T_{u,i} - T_{u,j}|)

    Supported options are: ``"concave"``, ``"convex"`` and ``"linear"``.

    - Concave decay function between item i and j is computed as:

      .. math::

        \\theta(x) = \\alpha^{x}, \\alpha \\in  [0, 1]

    - Convex decay function between item i and j is computed as:

      .. math::

        \\theta(x) = 1 - \\beta^{t-x}, \\beta \\in  (0, 1)

    - Linear decay function between item i and j is computed as:

      .. math::

        \\theta(x) = 1 - \\frac{x}{t} \\cdot \\gamma, \\gamma \\in  [0, 1]

    Where :math:`t` is the time between the interactions with both items.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200.
    :type K: int, optional
    :param fit_decay: Defines the decay scaling used for decay during model fitting.
        Defaults to 0.5.
    :type fit_decay: float, optional
    :param decay_function: The decay function that needs to
        be applied on the item similarity scores.
        Defaults to ``"convex"``.
    :type decay_function: str, optional
    :param decay_interval: Size of a single time unit in seconds.
        Allows more finegrained parameters for large scale datasets where events are collected over months of data.
        Defaults to 1 (second).
    :type decay_interval: int, optional
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
            decay_interval=decay_interval,
            similarity="cooc",
            decay_function=decay_function,
        )
