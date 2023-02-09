# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np

from recpack.matrix import InteractionMatrix
from recpack.scenarios import Scenario
from recpack.scenarios.splitters import MostRecentSplitter


class LastItemPrediction(Scenario):
    """Predict a user's next interaction.

    Scenario frequently used in evaluation of sequential recommendation algorithms.

    .. warning::

        The scenario can only be used when the dataset has timestamp information,
        because the order of interactions is needed to correctly split the data.

    The scenario splits the data such that the last interaction of a user is the target for prediction,
    while the earlier ones are used for training and as history.

    - :attr:`full_training_data` contains all but the most recent
      interaction for each of the users.
    - :attr:`test_data_in`: contains the ``n_most_recent_in`` interactions before
      the last interaction of each user.
    - :attr:`test_data_out` contains the last interaction of all users.
    - :attr:`validation_training_data` contains all but the most recent interaction of
      each user in the full training dataset.
    - :attr:`validation_data_in` contains the ``n_most_recent_in`` interactions before
      the last interaction of each user in the full training dataset.
    - :attr:`validation_data_out` contains the most recent interaction
      of each user in the full training dataset

    **Example**

    As an example, we split this data with ``validation = True`` and ``n_most_recent_in = 1``::

        time    0   1   2   3   4   5
        Alice   X   X   X
        Bob         X   X   X   X

    would yield full_training_data::

        time    0   1   2   3   4   5
        Alice   X   X
        Bob         X   X   X

    validation_training_data::

        time    0   1   2   3   4   5
        Alice   X
        Bob         X   X

    validation_data_in::

        time    0   1   2   3   4   5
        Alice   X
        Bob             X

    validation_data_out::

        time    0   1   2   3   4   5
        Alice       X
        Bob                 X

    test_data_in::

        time    0   1   2   3   4   5
        Alice       X
        Bob                 X

    test_data_out::

        time    0   1   2   3   4   5
        Alice           X
        Bob                     X


    :param validation: construct validation datasets if True,
        else split without validation data into only a full training and test dataset.
    :type validation: boolean, optional
    :param seed: Seed for randomisation parts of the scenario.
        This scenario is deterministic, so changing seed does not matter.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional
    :param n_most_recent_in: How much of the historic events to use as input history.
        Defaults to the maximal integer value.
    :type n_most_recent_in: int, optional
    """

    def __init__(self, validation=False, seed=None, n_most_recent_in=np.iinfo(np.int32).max):
        super().__init__(validation=validation, seed=seed)
        self.most_recent_splitter = MostRecentSplitter(1)
        self.n_most_recent_in = n_most_recent_in

        if n_most_recent_in == 0:
            raise ValueError("Using n_most_recent_in = 0 is not supported.")

        self.history_splitter = MostRecentSplitter(n_most_recent_in)

    def _split(self, data: InteractionMatrix):
        """Splits data so that a user's second to last interaction is used to predict their last interaction.
            Trains on all other interactions.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        (
            self._full_train_X,
            self._test_data_out,
        ) = self.most_recent_splitter.split(data)
        # Retain only the n_most_recent_in last interactions of a user as history
        _, self._test_data_in = self.history_splitter.split(self._full_train_X)

        if self.validation:
            # Select the 1 event to use as val_out per user
            (
                self._validation_train_X,
                self._validation_data_out,
            ) = self.most_recent_splitter.split(self._full_train_X)

            # Retain only the n_most_recent_in last interactions ofo a user as history.
            _, self._validation_data_in = self.history_splitter.split(self._validation_train_X)
