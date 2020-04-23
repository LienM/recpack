from recpack.algorithms.algorithm_base import Algorithm
from recpack.algorithms.user_item_interactions_algorithms.linear_algorithms import (
    EASE,
    SLIM,
)
from recpack.algorithms.true_baseline_algorithms import Popularity, Random
from recpack.algorithms.user_item_interactions_algorithms.nn_algorithms import ItemKNN
from recpack.algorithms.user_item_interactions_algorithms.matrix_factorization_algorithms import (
    NMF,
    SVD,
)


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
