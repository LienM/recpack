from recpack.algorithms.base import Algorithm
from recpack.algorithms.baseline import Popularity, Random
from recpack.algorithms.factorization import (
    NMF,
    SVD,
    NMFItemToItem,
    SVDItemToItem,
)
from recpack.algorithms.linear import (
    EASE,
    SLIM,
)

from recpack.algorithms.nearest_neighbour import ItemKNN
from recpack.algorithms.BPRMF import BPRMF

# from recpack.algorithms.metric_learning.cml import CML

from recpack.algorithms.vae.mult_vae import MultVAE
from recpack.algorithms.vae.rec_vae import RecVAE
from recpack.algorithms.wmf import WeightedMatrixFactorization
