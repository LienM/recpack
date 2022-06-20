import numpy as np

# TODO: docstrings!


def exponential_decay(age_arr: np.array, decay: float):
    return np.exp(-decay * age_arr)


def convex_decay(age_arr: np.array, decay: float):
    return np.power(decay, age_arr)


def concave_decay(age_arr: np.array, decay: float):
    return 1 - np.power(decay, 1 - age_arr)


def log_decay(age_arr: np.array, decay: float):
    """Computes a log based decay function.

    :param age_arr: array of values with the ages of events
    :type age_arr: np.array
    :param decay: decay parameter, in interval [0, 1]
    :type decay: float
    """
    m = age_arr.max()
    return (np.log(((decay - 1) * (1 - age_arr / m)) + 1) / np.log(decay)) + 1


def linear_decay(age_arr: np.array, decay: float):
    m = age_arr.max()
    return 1 - (age_arr / m) * decay
