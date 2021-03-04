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
