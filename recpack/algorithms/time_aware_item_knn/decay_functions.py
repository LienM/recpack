# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
from numpy.typing import ArrayLike


class DecayFunction:
    def __call__(self, time_distances: ArrayLike) -> ArrayLike:
        """Apply the decay.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: Array of event ages to which decays have been applied.
        :rtype: ArrayLike
        """
        raise NotImplementedError()


class ExponentialDecay(DecayFunction):
    """Applies exponential decay.

    For each value x in ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = e^{-\\alpha * x}

    where alpha is the decay parameter.

    :param decay: Exponential decay parameter, should be in the [0, 1] interval.
    :type decay: float
    """

    @classmethod
    def validate_decay(cls, decay: float):
        """Verify if the decay parameter is in the right range for this decay function."""
        if not (0 <= decay <= 1):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: [0, 1].")

    def __init__(self, decay: float):
        self.validate_decay(decay)
        self.decay = decay

    def __call__(self, time_distances: ArrayLike) -> ArrayLike:
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed time array.
        :rtype: ArrayLike
        """

        return np.exp(-self.decay * time_distances)


class ConvexDecay(DecayFunction):
    """Applies a convex decay function.

    For each value x in the ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = \\alpha^{x}

    where :math:`alpha` is the decay parameter.

    :param decay: The decay parameter, should be in the ]0, 1] interval.
    :type decay: float
    """

    @classmethod
    def validate_decay(cls, decay: float):
        """Verify if the decay parameter is in the right range for this decay function."""
        if not (0 < decay <= 1):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: ]0, 1].")

    def __init__(self, decay: float):
        self.validate_decay(decay)
        self.decay = decay

    def __call__(self, time_distances: ArrayLike):
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed time array.
        :rtype: ArrayLike
        """

        return np.power(self.decay, time_distances)


class ConcaveDecay(DecayFunction):
    """Applies a concave decay function.

    For each value x in the ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = 1 - \\alpha^{1-\\frac{x}{N}}

    where :math:`alpha` is the decay parameter and :math:`N` is the ``max_distance`` parameter.

    :param decay: The decay parameter, should be in the [0, 1[ interval.
    :type decay: float
    :param max_distance: Normalizing parameter, to put distances in the [0, 1].
    :type max_distance: float
    """

    @classmethod
    def validate_decay(cls, decay: float):
        """Verify if the decay parameter is in the right range for this decay function."""
        if not (0 < decay <= 1):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: ]0, 1].")

    def __init__(self, decay: float, max_distance: float):
        self.validate_decay(decay)
        self.decay = decay
        self.max_distance = max_distance

    def __call__(self, time_distances: ArrayLike):
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed array.
        :rtype: ArrayLike
        """
        if (time_distances > self.max_distance).any():
            raise ValueError("At least one of the distances is bigger than the specified max_distance.")
        return 1 - np.power(self.decay, 1 - (time_distances / self.max_distance))


class LogDecay(DecayFunction):
    """Applies a logarithmic decay function.

    For each value x in the ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = log_\\alpha ((\\alpha-1)(1-\\frac{x}{N}) + 1)

    where :math:`alpha` is the decay parameter and :math:`N` is the ``max_distance`` parameter.

    :param decay: The decay parameter, should be in the range ]1, inf[
    :type decay: float
    :param max_distance: Normalizing parameter, to put distances in the [0, 1].
    :type max_distance: float
    """

    @classmethod
    def validate_decay(cls, decay: float):
        """Verify if the decay parameter is in the right range for this decay function."""
        if not (1 < decay):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: ]1, inf[.")

    def __init__(self, decay: float, max_distance: float):
        self.validate_decay(decay)
        self.decay = decay
        self.max_distance = max_distance

    def __call__(self, time_distances: ArrayLike):
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed time array.
        :rtype: ArrayLike
        """
        if (time_distances > self.max_distance).any():
            raise ValueError("At least one of the distances is bigger than the specified max_distance.")
        return np.log(((self.decay - 1) * (1 - time_distances / self.max_distance)) + 1) / np.log(self.decay)


class LinearDecay(DecayFunction):
    """Applies a linear decay function.

    For each value x in the ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = \\max(1 - (\\frac{x}{N}) \\alpha, 0)

    where :math:`alpha` is the decay parameter and :math:`N` is the ``max_distance`` parameter.

    :param decay: The decay parameter, should be in the [0, inf[ interval.
    :type decay: float
    :param max_distance: Normalizing parameter, to put distances in the [0, 1].
    :type max_distance: float
    """

    @classmethod
    def validate_decay(cls, decay: float):
        if not (0 <= decay):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: [0, +inf[.")

    def __init__(self, decay: float, max_distance: float):
        self.validate_decay(decay)
        self.decay = decay
        self.max_distance = max_distance

    def __call__(self, time_distances: ArrayLike):
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed time array.
        :rtype: ArrayLike
        """
        if (time_distances > self.max_distance).any():
            raise ValueError("At least one of the distances is bigger than the specified max_distance.")
        results = 1 - (time_distances / self.max_distance) * self.decay
        results[results < 0] = 0
        return results


class InverseDecay(DecayFunction):
    """Invert the scores.

    Decay parameter only added for interface unity.
    For each value x in the ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = \\frac{1}{x}
    """

    def __call__(self, time_distances: ArrayLike):
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed time array.
        :rtype: ArrayLike
        """

        results = time_distances.astype(float).copy()
        results[results > 0] = 1 / results[results > 0]
        results[results == 0] = 1
        return results


class NoDecay(ExponentialDecay):
    """Turns the array into a binary array."""

    def __init__(self):
        super().__init__(0)
