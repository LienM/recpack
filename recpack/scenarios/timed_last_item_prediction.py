# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from typing import Optional
import numpy as np

from recpack.matrix import InteractionMatrix
from recpack.scenarios import Scenario
from recpack.scenarios.splitters import MostRecentSplitter, UserInteractionTimeSplitter, TimestampSplitter


class TimedLastItemPrediction(Scenario):
    """Predict users’ last interaction, given information about historical interactions.

    Scenario frequently used in evaluation of sequential recommendation algorithms.

    .. warning::

        The scenario can only be used when the dataset has timestamp information,
        because the order of interactions is needed to correctly split the data.

    The scenario splits the data such that the last interaction of a user is the target for prediction,
    while the earlier ones are used for training and as history.
    Only users with an interaction after ``t`` (or ``t_validation``) are considered for evaluation.

    - :attr:`full_training_data` contains all events from all users before ``t``
    - :attr:`validation_training_data` contains all events from all users before ``t_validation``

    - :attr:`validation_data_out` contains the most recent interaction of
      a user in the interval ``[t_validation, min(t, t_validation + t_delta_out)[``.
    - :attr:`validaton_data_in` contains the earlier interactions of the validation users.
      The ``n_most_recent_in`` are used per user.

    - :attr:`test_data_out` contains the most recent interaction of
      a user in the interval ``[t, t + delta_out]``.
    - :attr:`test_data_in` contains the earlier interactions of the test users.
      The ``n_most_recent_in`` are used per user.


    **Example**

    As an example, splitting following data with ``t = 4``, ``t_validation = 2`` and ``validation = True``::

        time    0   1   2   3   4   5
        Alice   X   X
        Bob         X   X   X   X
        Carol       X   X   X


    would yield full_training_data::

        time    0   1   2   3   4   5
        Alice   X   X
        Bob         X   X   X
        Carol       X   X   X

    validation_training_data::

        time    0   1   2   3   4   5
        Alice   X   X
        Bob         X
        Carol       X

    validation_data_in::

        time    0   1   2   3   4   5
        Bob         X   X
        Carol       X   X

    validation_data_out::

        time    0   1   2   3   4   5
        Bob                 X
        Carol               X

    test_data_in::

        time    0   1   2   3   4   5
        Bob         X   X   X

    test_data_out::

        time    0   1   2   3   4   5
        Bob                     X

    :param t: Timestamp for splitting full training data and test data.
        Simulates a training moment in real time evaluation.
    :param t_validation: timestamp for splitting validation training and evaluation data.
        Only required if validation is True.
    :param n_most_recent_in: The number of interactions to use as history.
        Most recent interactions are taken per user.
        Defaults to max integer value.
    :type n_most_recent_in: int, optional
    :param delta_out: Seconds past t. Upper bound on the timestamp
        of interactions in the target datasets return value.
        Defaults to np.iinfo(np.int32).max (infinity).
    :type delta_out: int, optional
    :param validation: Assign a portion of the full training dataset
        to validation data if True, else split without validation data into
        only a training and test dataset. Defaults to False.
    :type validation: boolean, optional
    :param seed: Seed for randomisation parts of the scenario.
        This scenario is deterministic, so changing seed should not matter.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional
    """

    def __init__(
        self,
        t: float,
        t_validation: float = None,
        n_most_recent_in: Optional[int] = np.iinfo(np.int32).max,
        delta_out: int = np.iinfo(np.int32).max,
        validation: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(validation=validation, seed=seed)
        self.t = t
        self.t_validation = t_validation
        self.n_most_recent_in = n_most_recent_in
        self.delta_out = delta_out
        if self.validation and not self.t_validation:
            raise Exception("t_validation should be provided when using validation split.")

        # Used to select test users (users with an interaction after t)
        self.user_selector_test = UserInteractionTimeSplitter(t)

        self.splitter_full_training_data = TimestampSplitter(t)
        if self.validation:
            assert self.t_validation < self.t
            # Used to select validation users
            self.user_selector_val = UserInteractionTimeSplitter(t_validation)
            self.splitter_validation_training = TimestampSplitter(t_validation)

        self.most_recent_splitter = MostRecentSplitter(1)

        if n_most_recent_in == 0:
            raise ValueError("Using n_most_recent_in = 0 is not supported.")

        self.history_splitter = MostRecentSplitter(n_most_recent_in)

    def _split(self, data: InteractionMatrix):

        self._full_train_X, _ = self.splitter_full_training_data.split(data)

        # Selects data from users with an interaction in the [t, t+delta_out[ interval
        _, te_data = self.user_selector_test.split(data.timestamps_lt(self.t + self.delta_out))

        # Split off the last item for each of the test users as test_data_out
        # and add all prior interactions to history.
        full_test_user_history, self._test_data_out = self.most_recent_splitter.split(te_data)

        # Retain only the n_most_recent_in last interactions of a user as history
        _, self._test_data_in = self.history_splitter.split(full_test_user_history)

        if self.validation:
            self._validation_train_X, _ = self.splitter_validation_training.split(self._full_train_X)

            # Selects data from users with an interaction in the [t_val, min(t_val+delta_out, t)[ interval
            _, val_data = self.user_selector_val.split(
                self._full_train_X.timestamps_lt(self.t_validation + self.delta_out)
            )

            # Split off the last item for each of the validation users as validation_data_out
            # and add all prior interactions to history.
            (
                full_val_user_history,
                self._validation_data_out,
            ) = self.most_recent_splitter.split(val_data)

            # cut the history to the requested size.
            _, self._validation_data_in = self.history_splitter.split(full_val_user_history)
