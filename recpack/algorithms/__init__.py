from recpack.algorithms.base import Algorithm
from recpack.algorithms.similarity.linear import (
    EASE,
    SLIM,
)
from recpack.algorithms.baseline import Popularity, Random
from recpack.algorithms.similarity.nearest_neighbour import ItemKNN
from recpack.algorithms.similarity.factorization import (
    NMF,
    SVD,
)

from recpack.algorithms.vae.mult_vae import MultVAE
from recpack.algorithms.vae.rec_vae import RecVAE


class AlgorithmRegistry:

    # TODO Resolve based on imports
    ALGORITHMS = {
        "ease": EASE,
        "random": Random,
        "popularity": Popularity,
        "itemKNN": ItemKNN,
        "NMF": NMF,
        "SVD": SVD,
        "slim": SLIM,
        "mult_vae": MultVAE,
        "rec_vae": RecVAE
    }

    @classmethod
    def get(cls, algorithm_name):
        return cls.ALGORITHMS[algorithm_name]

    @classmethod
    def register(cls, algorithm_name, algorithm: Algorithm):
        assert algorithm_name not in cls.ALGORITHMS

        cls.ALGORITHMS[algorithm_name] = algorithm

    @classmethod
    def list(cls):
        return cls.ALGORITHMS


algorithm_registry = AlgorithmRegistry()
