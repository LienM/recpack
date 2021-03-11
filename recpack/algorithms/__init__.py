"""

The algorithm module in recpack contains a collection of implemented algorithms,
as well as utilities such as loss functions, and
samplers used by these algorithms
and helpful for use in creating new algorithms.

.. currentmodule:: recpack.algorithms

.. contents:: Table of Contents
    :depth: 2

Simple baselines
------------------

These are simple baseline methods, that allow quick testing of pipelines,
and provide a trivial baseline to compare other algorithms against.

.. autosummary::
    :toctree:

    Popularity
    Random

Item Similarity algorithms
----------------------------

These algorithms all compute item to item similirities,
and at prediction time the user is represented by the items
they have interacted with.

.. autosummary::
    :toctree:

    EASE
    EASE_XY
    SLIM
    ItemKNN
    NMFItemToItem
    SVDItemToItem

User and Item Embedding algorithms
------------------------------------

This class of algorithms fits both item and user embeddings,
prediction happens by calculating distance between user and item embeddings.

.. autosummary::
    :toctree:

    NMF
    SVD
    WeightedMatrixFactorization
    BPRMF
    RecVAE
    MultVAE


Abstract base classes
-----------------------

To provide structure to the classes of algorithms we provide several base classes,
which can be used to create new algorithms.

.. autosummary::
    :toctree:

    base.Algorithm
    base.ItemSimilarityMatrixAlgorithm
    base.TopKItemSimilarityMatrixAlgorithm
    base.FactorizationAlgorithm
    base.TorchMLAlgorithm

Stopping Criterion
--------------------

When creating an algorithm that learns a model iteratively,
we need a way to decide which is the best model, and when to stop.
The Stopping Criterion module provides this functionality.

.. autosummary::
    :toctree:

    stopping_criterion.StoppingCriterion
    stopping_criterion.EarlyStoppingException

Loss Functions
----------------

Iterative methods often use a loss function to improve a model,
in this module we already provide several preimplemented loss functions.

To use these loss functions in a StoppingCriterion,
we also provide metric wrappers around the raw loss functions.

.. autosummary::
    :toctree:

    loss_functions.covariance_loss
    loss_functions.warp_loss
    loss_functions.warp_loss_metric
    loss_functions.bpr_loss
    loss_functions.bpr_loss_metric

Samplers
----------

In multiple algorithms sampling plays an important role,
as such we provide some preimplemented sampling methods.

.. autosummary::
    :toctree:

    samplers.bootstrap_sample_pairs
    samplers.warp_sample_pairs


Utility Functions
-------------------

The ``util`` module contains several functions used in multiple algorithms, 
that can help simplify certain tasks in creating a new algorithm

.. autosummary::
    :toctree:

    util.normalize
    util.get_batches
    util.sample
    util.naive_sparse2tensor
    util.naive_tensor2sparse
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
