# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""

The algorithms module in recpack contains a wide array of state-of-the-art
collaborative filtering algorithms.
Also included are some baseline algorithms, as well as several reusable building blocks
such as commonly used loss functions and sampling methods.

.. currentmodule:: recpack.algorithms

**Example of use**::

    from scipy.sparse import csr_matrix
    from recpack.algorithms import Random

    X = csr_matrix(np.array([[1, 0, 1], [1, 1, 0], [1, 1, 0]]))

    # Set hyper-parameter values
    algo = Random(K=3)

    # Fit algorithm
    algo.fit(X)

    # Get random recos for each nonzero user
    predictions = algo.predict(X)

    # Predictions is a csr matrix, inspecting the scores with
    predictions.toarray()


Baselines
------------------

In recpack, baseline algorithms are algorithms that are not personalized.
Use these baselines if you wish to quickly test a pipeline,
or for comparison in experiments.

.. autosummary::
    :toctree: generated/

    Popularity
    Random

Item Similarity Algorithms
----------------------------

Item similarity algorithms exploit relationships between items to make recommendations.
At prediction time, the user is represented by the items
they have interacted with.

.. autosummary::
    :toctree: generated/

    SLIM
    ItemKNN
    ItemPNN
    NMFItemToItem
    SVDItemToItem
    Prod2Vec
    Prod2VecClustered


Hybrid Similarity Algorithms
----------------------------

Hybrid similarity algorithms use a combination of user and item similarities
to generate recommendations.

.. autosummary::
    :toctree: generated/

    KUNN


Factorization Algorithms
------------------------------------

Factorization algorithms factorize the interaction matrix into
a user embeddings (U) and item embeddings (V) matrix, that can be
user to reconstruct the original interaction matrix R = UV^T.

.. autosummary::
    :toctree: generated/

    NMF
    SVD
    WeightedMatrixFactorization
    BPRMF


Autoencoder Algorithms
------------------------

Autoencoder algorithms aim to learn a function f, such that X = f(X).
More information on autoencoders can be found on `Wikipedia <https://en.wikipedia.org/wiki/Autoencoder>`_

.. autosummary::
    :toctree: generated/

    RecVAE
    MultVAE
    EASE


Session-Based Algorithms
-------------------------

.. autosummary::
    :toctree: generated/

    GRU4RecNegSampling
    GRU4RecCrossEntropy
    STAN
    SequentialRules

Time Aware Algorithms
----------------------

.. autosummary::
    :toctree: generated/

    TARSItemKNN
    TARSItemKNNDing
    TARSItemKNNLee
    TARSItemKNNLiu
    TARSItemKNNLiu2012
    TARSItemKNNVaz
    
    TARSItemKNNCoocDistance
    TARSItemKNNHermann
    TARSItemKNNXia

.. _algorithm-base-classes:

Abstract Base Classes
-----------------------

Recpack algorithm implementations inherit from one of these base classes.
These base classes provide the basic building blocks to easily create new algorithm
implementations that can be used within the recpack evaluation framework.

For more information on how to create your own recpack algorithm, see :ref:`guides-algorithms`.

.. autosummary::
    :toctree: generated/
    :template: autosummary/base_algorithm_class.rst

    Algorithm
    ItemSimilarityMatrixAlgorithm
    TopKItemSimilarityMatrixAlgorithm
    FactorizationAlgorithm
    TorchMLAlgorithm

Stopping Criterion
--------------------
.. currentmodule:: recpack.algorithms.stopping_criterion

When creating an algorithm that learns a model iteratively,
we need a way to decide which is the best model, and when to stop.
The Stopping Criterion module provides this functionality.

.. autosummary::
    :toctree: generated/

    StoppingCriterion
    EarlyStoppingException

Loss Functions
----------------

.. currentmodule:: recpack.algorithms.loss_functions

Recommendation models learned iteratively by means of gradient descent (or ascent)
require a loss function.
in this module you will find some of the most common loss functions that can be used
with any TorchMLAlgorithm.

To use these loss functions in a StoppingCriterion,
we also provide metric wrappers around the raw loss functions.

.. autosummary::
    :toctree: generated/

    covariance_loss
    warp_loss
    warp_loss_wrapper
    bpr_loss
    bpr_loss_wrapper
    vae_loss
    bpr_max_loss
    top1_loss
    top1_max_loss

Samplers
----------

.. currentmodule:: recpack.algorithms.samplers

In multiple recommendation algorithms (e.g. BPRMF) sampling methods play
an important role.
As such recpack contains a number of commonly used sampling methods.

.. autosummary::
    :toctree: generated/

    PositiveNegativeSampler
    BootstrapSampler
    WarpSampler
    SequenceMiniBatchSampler
    SequenceMiniBatchPositivesTargetsNegativesSampler


Utility Functions
-------------------

.. currentmodule:: recpack.algorithms.util

The ``util`` module contains a number of utility functions
used across algorithms.
Use these to simplify certain tasks (such as batching) when creating a new algorithm.


.. autosummary::
    :toctree: generated/

    get_batches
    sample_rows
    naive_sparse2tensor
    naive_tensor2sparse
"""


from recpack.algorithms.base import (
    Algorithm,
    TopKItemSimilarityMatrixAlgorithm,
    TorchMLAlgorithm,
    ItemSimilarityMatrixAlgorithm,
    FactorizationAlgorithm,
)
from recpack.algorithms.baseline import Popularity, Random
from recpack.algorithms.factorization import (
    NMF,
    SVD,
)
from recpack.algorithms.factorization_item_similarity import (
    NMFItemToItem,
    SVDItemToItem,
)
from recpack.algorithms.ease import EASE
from recpack.algorithms.slim import SLIM

from recpack.algorithms.nearest_neighbour import ItemKNN, ItemPNN
from recpack.algorithms.kunn import KUNN
from recpack.algorithms.bprmf import BPRMF

# from recpack.algorithms.metric_learning.cml import CML

from recpack.algorithms.mult_vae import MultVAE
from recpack.algorithms.rec_vae import RecVAE
from recpack.algorithms.wmf import WeightedMatrixFactorization

from recpack.algorithms.p2v import Prod2Vec
from recpack.algorithms.p2v_clustered import Prod2VecClustered
from recpack.algorithms.gru4rec import (
    GRU4Rec,
    GRU4RecNegSampling,
    GRU4RecCrossEntropy,
)

from recpack.algorithms.stan import STAN
from recpack.algorithms.time_aware_item_knn import (
    TARSItemKNN,
    TARSItemKNNCoocDistance,
    TARSItemKNNDing,
    TARSItemKNNHermann,
    TARSItemKNNLee,
    TARSItemKNNLiu,
    TARSItemKNNLiu2012,
    TARSItemKNNXia,
    TARSItemKNNVaz,
)

from recpack.algorithms.sequential_rules import SequentialRules
