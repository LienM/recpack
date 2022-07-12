import numpy as np

# TODO: docstrings!


def exponential_decay(age_arr: np.array, decay: float):
    """Applies a exponential based decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        e^{-\\alpha * x}

    where alpha is the decay parameter.

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the [0, 1] interval.
    :type decay: float
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """
    if not (0 <= decay <= 1):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: [0, 1].")
    return np.exp(-decay * age_arr)


def convex_decay(age_arr: np.array, decay: float):
    """Applies a convex decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        \\alpha^{x}

    where alpha is the decay parameter.

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the ]0, 1] interval.
    :type decay: float
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """
    if not (0 < decay <= 1):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: ]0, 1].")

    return np.power(decay, age_arr)


def concave_decay(age_arr: np.array, decay: float):
    """Applies a concave decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        1 - \\alpha^{1-\\frac{x}{max_{x \\in X_u}(x)}}

    where alpha is the decay parameter.

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the [0, 1[ interval.
    :type decay: float
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """
    if not (0 <= decay < 1):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: [0, 1[.")

    m = age_arr.max()
    return 1 - np.power(decay, 1 - (age_arr / m))


def log_decay(age_arr: np.array, decay: float):
    """Applies a log decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        log_\\alpha ((\\alpha-1)(1-\\frac{x}{max_{x \\in X_u}(x)}) + 1) + 1

    where alpha is the decay parameter.

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the ]1, inf[ interval.
    :type decay: float
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """
    if not (1 < decay):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: ]1, inf[.")

    m = age_arr.max()
    return (np.log(((decay - 1) * (1 - age_arr / m)) + 1) / np.log(decay)) + 1


def linear_decay(age_arr: np.array, decay: float):
    """Applies a linear decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        1 - (\\frac{x}{max_{x \\in X_u}(x)}) \\alpha

    where alpha is the decay parameter.

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the [0, 1] interval.
    :type decay: float
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """
    if not (0 <= decay <= 1):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: [0, 1].")

    m = age_arr.max()
    return 1 - (age_arr / m) * decay


def linear_decay_steeper(age_arr: np.array, decay: float):
    """Applies a linear decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        1 - (\\frac{x}{max_{x \\in X_u}(x)}) \\alpha

    where alpha is the decay parameter. If the decayed value is below 0, it is set to 0

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the [0, +inf[ interval.
    :type decay: float
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """

    if not (0 <= decay):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: [0, +inf[.")

    m = age_arr.max()
    results = 1 - (age_arr / m) * decay
    results[results < 0] = 0
    return results
