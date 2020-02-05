from .linear_algorithms import EASE, SLIM
from .true_baseline_algorithms import Popularity, Random
from .nn_algorithms import ItemKNN


ALGORITHMS = {
    'ease': EASE,
    'random': Random,
    'popularity': Popularity,
    'itemKNN': ItemKNN,
    'slim': SLIM
}


def get_algorithm(algorithm_name):
    return ALGORITHMS[algorithm_name]
