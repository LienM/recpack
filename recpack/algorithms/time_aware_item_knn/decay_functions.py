from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from recpack.algorithms.util import to_binary


class DecayFunction:
    def __init__(self, decay: float):
        self.decay = decay

    def __call__(self, age_array: ArrayLike, max_age: Optional[float] = None) -> ArrayLike:
        """Apply the decay.

        :param age_array: Array of event ages to decay.
        :type age_array: ArrayLike
        :param max_age: Maximum age of an event.
        :type max_age: float, optional
        :returns: Array of event ages to which decays have been applied.
        :rtype: ArrayLike
        """
        raise NotImplementedError()


class ExponentialDecay(DecayFunction):
    """Applies exponential decay.

    For each value x in ``age_array`` the decayed value is computed as

    .. math::

        e^{-\\alpha * x}

    where alpha is the decay parameter.

    :param decay: Exponential decay parameter, should be in the [0, 1] interval.
    :type decay: float
    """

    def __init__(self, decay: float):
        if not (0 <= decay <= 1):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: [0, 1].")
        self.decay = decay

    def __call__(self, age_array: ArrayLike, max_age: Optional[float] = None) -> ArrayLike:
        """Apply the decay function.

        :param age_array: array of age of events which will be decayed.
        :type age_array: np.array
        :param max_age: The max age normalisation parameter (Unused in this decay)
        :type max_age: float, optional
        :returns: The decayed input time array. Larger values in the original array will have lower values.
        :rtype: np.array
        """

        return np.exp(-self.decay * age_array)


class ConvexDecay(DecayFunction):
    """Applies a convex decay function.

    For each value x in the ``age_array`` the decayed value is computed as

    .. math::

        \\alpha^{x}

    where alpha is the decay parameter.
    :param decay: The decay parameter, should be in the ]0, 1] interval.
    :type decay: float
    """

    def __init__(self, decay: float):
        if not (0 < decay <= 1):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: ]0, 1].")
        self.decay = decay

    def __call__(self, age_array: np.array, max_age: Optional[float] = None):
        """Apply the decay function.

        :param age_array: array of age of events which will be decayed.
        :type age_array: np.array
        :param max_age: The max age normalisation parameter (Unused in this decay)
        :type max_age: float, optional
        :returns: The decayed input time array. Larger values in the original array will have lower values.
        :rtype: np.array
        """

        return np.power(self.decay, age_array)


class ConcaveDecay(DecayFunction):
    """Applies a concave decay function.

    For each value x in the ``age_array`` the decayed value is computed as

    .. math::

        1 - \\alpha^{1-\\frac{x}{max_{x \\in X_u}(x)}}

    where alpha is the decay parameter.
    :param decay: The decay parameter, should be in the [0, 1[ interval.
    :type decay: float
    """

    def __init__(self, decay: float):
        if not (0 <= decay < 1):
            raise ValueError(f"decay parameter = {decay} is not in the supported range: [0, 1[.")
        self.decay = decay

    def __call__(self, age_array: np.array, max_age: Optional[float] = None):
        """Apply the decay function.

        :param age_array: array of age of events which will be decayed.
        :type age_array: np.array
        :param max_age: The max age normalisation parameter (Unused in this decay)
        :type max_age: float, optional
        :returns: The decayed input time array. Larger values in the original array will have lower values.
        :rtype: np.array
        """
        # TODO Based on your formula, 
        # I think max_age can just be taken from age_array here as it is user-specific.
        # Unless the formula is wrong
        max_age = max_age if max_age is not None else age_array.max()
        return 1 - np.power(self.decay, 1 - (age_array / max_age))


class LogDecay(DecayFunction):
    """Applies a logarithmic decay function.

    For each value x in the ``age_array`` the decayed value is computed as

    .. math::

        log_\\alpha ((\\alpha-1)(1-\\frac{x}{max_{x \\in X_u}(x)}) + 1)

    where alpha is the decay parameter.
    :param decay: The decay parameter, should be in the range ]1, inf[
    :type decay: float
    """

    def __init__(self, decay: float):
        if not (1 < decay):
            raise ValueError(f"decay parameter = {decay} is not in the supported range: ]1, inf[.")
        self.decay = decay

    def __call__(self, age_array: np.array, max_age: Optional[float] = None):
        """Apply the decay function.

        :param age_array: array of age of events which will be decayed.
        :type age_array: np.array
        :param max_age: The max age normalisation parameter (Unused in this decay)
        :type max_age: float, optional
        :returns: The decayed input time array. Larger values in the original array will have lower values.
        :rtype: np.array
        """

        max_age = max_age if max_age is not None else age_array.max()
        return np.log(((self.decay - 1) * (1 - age_array / max_age)) + 1) / np.log(self.decay)


class LinearDecay(DecayFunction):
    """Applies a linear decay function.

    For each value x in the ``age_array`` the decayed value is computed as

    .. math::

        1 - (\\frac{x}{max_{x \\in X_u}(x)}) \\alpha

    where alpha is the decay parameter. If the decayed value is below 0, it is set to 0.

    :param decay: The decay parameter, should be in the [0, inf[ interval.
    :type decay: float
    """

    def __init__(self, decay: float):
        if not (0 <= decay):
            raise ValueError(f"decay parameter = {decay} is not in the supported range: [0, +inf[.")
        self.decay = decay

    def __call__(self, age_array: np.array, max_age: Optional[float] = None):
        """Apply the decay function.

        :param age_array: array of age of events which will be decayed.
        :type age_array: np.array
        :param max_age: The max age normalisation parameter (Unused in this decay)
        :type max_age: float, optional
        :returns: The decayed input time array. Larger values in the original array will have lower values.
        :rtype: np.array
        """

        max_age = max_age if max_age is not None else age_array.max()
        results = 1 - (age_array / max_age) * self.decay
        results[results < 0] = 0
        return results


class InverseDecay(DecayFunction):
    """Invert the scores.

    Decay parameter only added for interface unity.

    .. math::

        \\frac{1}{x}

    :param decay: The decay parameter, willl not be used during computation.
    :type decay: float
    """

    def __init__(self, decay: float):
        self.decay = decay

    def __call__(self, age_array: np.array, max_age: Optional[float] = None):
        """Apply the decay function.

        :param age_array: array of age of events which will be decayed.
        :type age_array: np.array
        :param max_age: The max age normalisation parameter (Unused in this decay)
        :type max_age: float, optional
        :returns: The decayed input time array. Larger values in the original array will have lower values.
        :rtype: np.array
        """

        results = age_array.astype(float).copy()
        results[results > 0] = 1 / results[results > 0]
        results[results == 0] = 1
        return results


class NoDecay(DecayFunction):
    """Turns the array into a binary array."""

    def __init__(self, decay: float):
        self.decay = decay

    def __call__(self, age_array: np.array, max_age: Optional[float] = None):
        """Apply the decay function.

        :param age_array: array of age of events which will be decayed.
        :type age_array: np.array
        :param max_age: The max age normalisation parameter (Unused in this decay)
        :type max_age: float, optional
        :returns: The decayed input time array. Larger values in the original array will have lower values.
        :rtype: np.array
        """
        return to_binary(age_array)
