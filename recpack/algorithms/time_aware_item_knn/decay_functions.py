import numpy as np

# TODO: docstrings!


def exponential_decay(age_arr: np.array, decay: float, max_age: float = None):
    """Applies a exponential based decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        e^{-\\alpha * x}

    where alpha is the decay parameter.

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the [0, 1] interval.
    :type decay: float
    :param max_age: The max age normalisation parameter (Unused in this decay)
    :type max_age: float, optional
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """
    if not (0 <= decay <= 1):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: [0, 1].")
    return np.exp(-decay * age_arr)


def convex_decay(age_arr: np.array, decay: float, max_age: float = None):
    """Applies a convex decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        \\alpha^{x}

    where alpha is the decay parameter.

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the ]0, 1] interval.
    :type decay: float
    :param max_age: The max age normalisation parameter (Unused in this decay)
    :type max_age: float, optional
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """
    if not (0 < decay <= 1):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: ]0, 1].")

    return np.power(decay, age_arr)


def concave_decay(age_arr: np.array, decay: float, max_age: float = None):
    """Applies a concave decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        1 - \\alpha^{1-\\frac{x}{max_{x \\in X_u}(x)}}

    where alpha is the decay parameter.

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the [0, 1[ interval.
    :type decay: float
    :param max_age: The max age normalisation parameter
    :type max_age: float, optional
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """
    if not (0 <= decay < 1):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: [0, 1[.")

    max_age = max_age if max_age is not None else age_arr.max()
    return 1 - np.power(decay, 1 - (age_arr / max_age))


def log_decay(age_arr: np.array, decay: float, max_age: float = None):
    """Applies a log decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        log_\\alpha ((\\alpha-1)(1-\\frac{x}{max_{x \\in X_u}(x)}) + 1)

    where alpha is the decay parameter.

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the ]1, inf[ interval.
    :type decay: float
    :param max_age: The max age normalisation parameter
    :type max_age: float, optional
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """
    if not (1 < decay):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: ]1, inf[.")

    max_age = max_age if max_age is not None else age_arr.max()
    return np.log(((decay - 1) * (1 - age_arr / max_age)) + 1) / np.log(decay)


def linear_decay(age_arr: np.array, decay: float, max_age: float = None):
    """Applies a linear decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        1 - (\\frac{x}{max_{x \\in X_u}(x)}) \\alpha

    where alpha is the decay parameter.

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the [0, 1] interval.
    :type decay: float
    :param max_age: The max age normalisation parameter
    :type max_age: float, optional
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """
    if not (0 <= decay <= 1):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: [0, 1].")

    max_age = max_age if max_age is not None else age_arr.max()
    return 1 - (age_arr / max_age) * decay


def linear_decay_steeper(age_arr: np.array, decay: float, max_age: float = None):
    """Applies a linear decay function.

    For each value x in the ``age_arr`` the decayed value is computed as

    .. math::

        1 - (\\frac{x}{max_{x \\in X_u}(x)}) \\alpha

    where alpha is the decay parameter. If the decayed value is below 0, it is set to 0.

    :param age_array: array of age of events which will be decayed.
    :type age_array: np.array
    :param decay: The decay parameter, should be in the [0, +inf[ interval.
    :type decay: float
    :param max_age: The max age normalisation parameter
    :type max_age: float, optional
    :returns: The decayed input time array. Larger values in the original array will have lower values.
    :rtype: np.array
    """

    if not (0 <= decay):
        raise ValueError(f"decay parameter = {decay} is not in the supported range: [0, +inf[.")

    max_age = max_age if max_age is not None else age_arr.max()
    results = 1 - (age_arr / max_age) * decay
    results[results < 0] = 0
    return results


def inverse_decay(age_arr: np.array, decay: float, max_age: float = None):
    """Invert the scores.

    Decay parameter only added for interface unity.

    .. math::

        \\frac{1}{x}

    """
    results = age_arr.astype(float).copy()
    results[results > 0] = 1 / results[results > 0]
    results[results == 0] = 1
    return results
