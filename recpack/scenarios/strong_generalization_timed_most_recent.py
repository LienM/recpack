# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.matrix import InteractionMatrix
from recpack.scenarios import Scenario
from recpack.scenarios.splitters import MostRecentSplitter, UserInteractionTimeSplitter


class StrongGeneralizationTimedMostRecent(Scenario):
    """Predict the next interaction(s) for previously unseen users.

    - :attr:`full_training_data` contains events from all users
      whose most recent interaction was before ``t``

    Test data contains all users whose most recent interactions was after ``t``:

    - :attr:`test_data_out` contains the ``n_most_recent_out`` most recent interactions of
      a user whose most recent interactions was after ``t``.
    - :attr:`test_data_in` contains all earlier interactions of the test users.

    If validation data is requested, validation evaluation users are those training users whose most recent interaction
    occurred after ``t_validation``.

    - :attr:`validation_training_data` contains users whose most
      recent interaction happened before ``t_validation``.

    - :attr:`validation_data_out` contains the ``n_most_recent_out`` most recent interactions of
      a user whose most recent interactions was in the interval ``[t_validation, t[``.
    - :attr:`validaton_data_in` contains all earlier interactions of the
      validation_evaluation users.

    .. warning::

        The scenario can only be used when the dataset has timestamp information.

    **Example**

    As an example, splitting following data with ``t = 4``, ``t_validation = 2``,
    ``n_most_recent_out = 1`` and ``validation = True``::

        time    0   1   2   3   4   5
        Alice   X   X
        Bob         X   X   X   X
        Carol       X   X   X


    would yield full_training_data::

        time    0   1   2   3   4   5
        Alice   X   X
        Carol       X   X   X

    validation_training_data::

        time    0   1   2   3   4   5
        Alice   X   X

    validation_data_in::

        time    0   1   2   3   4   5
        Carol       X

    validation_data_out::

        time    0   1   2   3   4   5
        Carol           X   X

    test_data_in::

        time    0   1   2   3   4   5
        Bob         X   X

    test_data_out::

        time    0   1   2   3   4   5
        Bob                 X   X

    :param t: Users whose last action has ``time >= t`` are placed in the test set,
        all other users are placed in the training or validation sets.
    :param t_validation: Users whose last action has ``time >= t_validation`` and
        ``time < t`` are put in the validation set. Users whose
        last action has ``time < t_validation`` are put in train.
        Only required if validation is True.
    :param n_most_recent_out: The number of user actions to consider as target.
        Defaults to 1.
    :param validation: Assign a portion of the full training dataset
        to validation data if True,
        else split without validation data into only a training and test dataset.
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
        n_most_recent_out: int = 1,
        validation: bool = False,
        seed=None,
    ):
        super().__init__(validation=validation, seed=seed)
        self.t = t
        self.t_validation = t_validation
        self.n_most_recent_out = n_most_recent_out

        if self.n_most_recent_out <= 0:
            raise ValueError("n_most_recent_out should be a strictly positive integer.")

        if self.validation and not self.t_validation:
            raise Exception("t_validation should be provided when using validation split.")

        self.user_splitter_test = UserInteractionTimeSplitter(t)
        if self.validation:
            assert self.t_validation < self.t
            self.user_splitter_val = UserInteractionTimeSplitter(t_validation)
        self.most_recent_splitter = MostRecentSplitter(self.n_most_recent_out)

    def _split(self, data: InteractionMatrix):
        """Splits users into non-overlapping training,
            validation and test user sets based on the time of
            their most recent interaction.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        self._full_train_X, te_data = self.user_splitter_test.split(data)

        (
            self._test_data_in,
            self._test_data_out,
        ) = self.most_recent_splitter.split(te_data)

        if self.validation:
            self._validation_train_X, val_data = self.user_splitter_val.split(self._full_train_X)
            (
                self._validation_data_in,
                self._validation_data_out,
            ) = self.most_recent_splitter.split(val_data)
