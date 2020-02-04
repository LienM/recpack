from .linear_algorithms import EASE
from .true_baseline_algorithms import Popularity, Random


ALGORITHMS = {
    'ease': EASE,
    'random': Random,
    'popularity': Popularity
}


def get_algorithm(algorithm_name):
    return ALGORITHMS[algorithm_name]
