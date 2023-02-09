# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Union
from warnings import warn

from recpack.scenarios.splitters import StrongGeneralizationSplitter
from recpack.matrix import InteractionMatrix


class Scenario(ABC):
    """Base class for defining an evaluation scenario.

    A scenario is a set of steps that splits data into training,
    validation and test datasets.
    The test dataset is made up of two components:
    a fold-in set of interactions that is used to predict another held-out
    set of interactions.
    The creation of the validation dataset, from the full training dataset,
    should follow the same splitting strategy as
    the one used to create training and test datasets from the full dataset.

    :param validation: Create validation datasets when True,
        else split into training and test datasets.
    :type validation: boolean, optional
    :param seed: Seed for randomisation parts of the scenario.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional
    """

    def __init__(self, validation=False, seed=None):
        if seed is None:
            # Set seed if it was not set before.
            seed = np.random.get_state()[1][0]
        self.seed = seed
        self.validation = validation
        if validation:
            self.validation_splitter = StrongGeneralizationSplitter(0.8, seed=self.seed)

    @abstractmethod
    def _split(self, data_m: InteractionMatrix) -> None:
        """Abstract method to be implemented by the scenarios.

        Splits the data and assigns to :attr:`train_X`,
        :attr:`test_data_in, :attr:`test_data_out`, :attr:`validation_data_in`
        and :attr:`validation_data_out`

        :param data_m: Interaction matrix to be split.
        :type data_m: InteractionMatrix
        """

    def split(self, data_m: InteractionMatrix) -> None:
        """Splits ``data_m`` according to the scenario.

        After splitting properties :attr:`training_data`,
        :attr:`validation_data` and :attr:`test_data` can be used
        to retrieve the splitted data.

        :param data_m: Interaction matrix that should be split.
        :type data: InteractionMatrix
        """
        self._split(data_m)

        self._check_split()

    @property
    def full_training_data(self) -> InteractionMatrix:
        """The full training dataset, which should be used for a final training
        after hyper parameter optimisation.

        :return: Interaction Matrix of training interactions.
        :rtype: InteractionMatrix
        """
        if not hasattr(self, "_full_train_X"):
            raise KeyError("Split before trying to access the full_training_data property.")

        return self._full_train_X

    @property
    def validation_training_data(self) -> InteractionMatrix:
        """The training data to be used during validation."""
        if not self.validation:
            raise KeyError("This scenario was created without validation data.")

        if not hasattr(self, "_validation_train_X"):
            raise KeyError("Split before trying to access the validation_training_data property.")

        return self._validation_train_X

    @property
    def validation_data(
        self,
    ) -> Union[Tuple[InteractionMatrix, InteractionMatrix], None]:
        """The validation dataset. Consist of a fold-in and hold-out set of interactions.

        Data is processed such that both matrices contain the exact same users.
        Users that were present in only one of the matrices
        and not in the other are removed.

        :return: Validation data matrices as
            InteractionMatrix in, InteractionMatrix out.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        if not self.validation:
            raise KeyError("This scenario was created without validation_data.")

        if not hasattr(self, "_validation_data_in"):
            raise KeyError("Split before trying to access the validation_data property.")

        # make sure users match both.
        in_users = self._validation_data_in.active_users
        out_users = self._validation_data_out.active_users

        matching = list(in_users.intersection(out_users))
        return (
            self._validation_data_in.users_in(matching),
            self._validation_data_out.users_in(matching),
        )

    @property
    def validation_data_in(self):
        """Fold-in part of the validation dataset"""
        return self.validation_data[0]

    @property
    def validation_data_out(self):
        """Held-out part of the validation dataset"""
        return self.validation_data[1]

    @property
    def test_data_in(self):
        """Fold-in part of the test dataset"""
        return self.test_data[0]

    @property
    def test_data_out(self):
        """Held-out part of the test dataset"""
        return self.test_data[1]

    @property
    def test_data(self) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """The test dataset. Consist of a fold-in and hold-out set of interactions.

        Data is processed such that both matrices contain the exact same users.
        Users that were present in only one of the matrices
        and not in the other are removed.

        :return: Test data matrices as InteractionMatrix in, InteractionMatrix out.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        # make sure users match.
        in_users = self._test_data_in.active_users
        out_users = self._test_data_out.active_users

        matching = list(in_users.intersection(out_users))
        return (
            self._test_data_in.users_in(matching),
            self._test_data_out.users_in(matching),
        )

    def _check_split(self):
        """Checks that the splits have been done properly.

        Makes sure all expected attributes are set.
        """
        assert hasattr(self, "_full_train_X") and self._full_train_X is not None
        if self.validation:
            assert hasattr(self, "_validation_train_X") and self._validation_train_X is not None
            assert hasattr(self, "_validation_data_in") and self._validation_data_in is not None
            assert hasattr(self, "_validation_data_out") and self._validation_data_out is not None

        assert hasattr(self, "_test_data_in") and self._test_data_in is not None
        assert hasattr(self, "_test_data_out") and self._test_data_out is not None

        self._check_size()

    def _check_size(self):
        """
        Warns user if any of the sets is unusually small or empty
        """
        n_train = self._full_train_X.num_interactions
        n_test_in = self._test_data_in.num_interactions
        n_test_out = self._test_data_out.num_interactions
        n_test = n_test_in + n_test_out
        n_total = n_train + n_test

        if self.validation:
            n_val_in = self._validation_data_in.num_interactions
            n_val_out = self._validation_data_out.num_interactions
            n_val_train = self._validation_train_X.num_interactions
            n_val = n_val_in + n_val_out
            n_total += n_val

        def check(name, count, total, threshold):
            if (count + 1e-9) / (total + 1e-9) < threshold:
                warn(f"{name} resulting from {type(self).__name__} is unusually small.")

        check("Training set", n_train, n_total, 0.05)
        check("Test set", n_test, n_total, 0.01)
        check("Test in set", n_test_in, n_test, 0.05)
        check("Test out set", n_test_out, n_test, 0.01)
        if self.validation:
            check("Validation set", n_val, n_total, 0.01)
            check("Validation train set", n_val_train, n_train, 0.05)
            check("Validation in set", n_val_in, n_val, 0.05)
            check("Validation out set", n_val_out, n_val, 0.01)
