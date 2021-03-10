"""Algorithm implementations in the recpack repository.

.. currentmodule:: recpack.algorithms


.. rubric:: Simple baselines

.. autosummary::
    :toctree:

    Popularity
    Random

.. rubric:: Item Similarity algorithms

.. autosummary::
    :toctree:

    EASE
    EASE_XY
    SLIM
    ItemKNN
    NMFItemToItem
    SVDItemToItem

.. rubric:: User and Item Embedding algorithms

.. autosummary::
    :toctree:

    NMF
    SVD
    WeightedMatrixFactorization
    BPRMF
    RecVAE
    MultVAE


.. rubric:: Abstract base classes

TODO: add a page with base classes, and info on creating your own algorithms

.. autosummary::
    :toctree:

    Algorithm
    base.ItemSimilarityMatrixAlgorithm
    base.TopKItemSimilarityMatrixAlgorithm
    base.TorchMLAlgorithm

"""

from recpack.algorithms.base import Algorithm
from recpack.algorithms.baseline import Popularity, Random
from recpack.algorithms.factorization import (
    NMF,
    SVD,
    NMFItemToItem,
    SVDItemToItem,
)
from recpack.algorithms.ease import EASE, EASE_XY
from recpack.algorithms.slim import SLIM

from recpack.algorithms.nearest_neighbour import ItemKNN
from recpack.algorithms.BPRMF import BPRMF

# from recpack.algorithms.metric_learning.cml import CML

from recpack.algorithms.mult_vae import MultVAE
from recpack.algorithms.rec_vae import RecVAE
from recpack.algorithms.wmf import WeightedMatrixFactorization
