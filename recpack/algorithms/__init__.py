from .linear_algorithms import EASE, SLIM
from .true_baseline_algorithms import Popularity, Random
from .nn_algorithms import ItemKNN
from .matrix_factorization_algorithms import NMF


ALGORITHMS = {
    'ease': EASE,
    'random': Random,
    'popularity': Popularity,
    'itemKNN': ItemKNN,
    'NMF': NMF,
    'slim': SLIM,
}


def get_algorithm(algorithm_name):
    return ALGORITHMS[algorithm_name]
