# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.matrix import InteractionMatrix
from recpack.scenarios import Scenario
from recpack.scenarios.splitters import FractionInteractionSplitter, StrongGeneralizationSplitter


class StrongGeneralization(Scenario):
    """Predict (randomly) held-out interactions of previously unseen users.

    During splitting each user is randomly assigned to one of three groups of users:
    training, validation and testing.

    The full training data contains ``frac_users_train`` of the users.
    If validation data is requested validation_training data contains
    ``frac_users_train`` * 0.8 of the users,
    the remaining 20% are assigned to validation data.

    Test and validation users' interactions are split into a `data_in` (fold-in) and
    `data_out` (held-out) set.
    ``frac_interactions_in`` of interactions are assigned to fold-in, the remainder
    to the held-out set.

    Strong generalization is considered more robust, harder and more realistic than
    :class:`recpack.splitters.scenarios.WeakGeneralization`

    **Example**

    As an example, we split this interaction matrix with ``frac_users_train = 5/6``,
    ``frac_interactions_in = 0.75`` and ``validation = True``::

                0   1   2   3   4   5
        Alice   X   X
        Bob         X   X   X   X
        Carol   X   X       X       X
        Dave    X       X   X
        Erin    X   X   X
        Frank   X   X   X   X

    would yield full_training_data::

                0   1   2   3   4   5
        Alice   X   X
        Carol   X   X       X       X
        Dave    X       X   X
        Erin    X   X   X
        Frank   X   X   X   X

    validation_training_data::

                0   1   2   3   4   5
        Alice   X   X
        Carol   X   X       X       X
        Erin    X   X   X
        Frank   X   X   X   X

    validation_data_in::

                0   1   2   3   4   5
        Dave    X           X

    validation_data_out::

                0   1   2   3   4   5
        Dave            X

    test_data_in::

                0   1   2   3   4   5
        Bob         X       X   X

    test_data_out::

                0   1   2   3   4   5
        Bob             X

    :param frac_users_train: Fraction of users assigned to the full training dataset.
        Between 0 and 1. Defaults to 0.8.
    :type frac_users_train: float, optional
    :param frac_interactions_in: Fraction of the users'
        interactions to be used as fold-in set (user history). Between 0 and 1.
        Defaults to 0.8.
    :type frac_interactions_in: float, optional
    :param validation: Assign a portion of the full training dataset
        to validation datasets if True,
        else split without validation data into only a training and test dataset.
    :type validation: boolean, optional
    :param seed: The seed to use for the random components of the splitter.
        If None, a random seed will be used. Defaults to None
    :type seed: int, optional
    """

    def __init__(
        self,
        frac_users_train: float = 0.8,
        frac_interactions_in: float = 0.8,
        validation: bool = False,
        seed: int = None,
    ):
        super().__init__(validation=validation, seed=seed)
        self.frac_users_train = frac_users_train
        self.frac_interactions_in = frac_interactions_in

        self.strong_gen = StrongGeneralizationSplitter(frac_users_train, seed=self.seed)
        self.interaction_split = FractionInteractionSplitter(frac_interactions_in, seed=self.seed)

    def _split(self, data: InteractionMatrix) -> None:
        """Splits your data so that a user can only be in one of
            training, validation or test set.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        """
        self._full_train_X, test_data = self.strong_gen.split(data)

        if self.validation:
            (
                self._validation_train_X,
                validation_data,
            ) = self.validation_splitter.split(self._full_train_X)

            (
                self._validation_data_in,
                self._validation_data_out,
            ) = self.interaction_split.split(validation_data)

        self._test_data_in, self._test_data_out = self.interaction_split.split(test_data)
