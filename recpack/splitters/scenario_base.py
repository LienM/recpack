from abc import ABC, abstractmethod
from typing import Tuple, Union
from warnings import warn

import recpack.splitters.splitter_base as splitter_base
from recpack.data.matrix import InteractionMatrix


class Scenario(ABC):
    def __init__(self, validation=False):
        """
        Base class for defining an evaluation "Scenario",
        i.e. a set of steps that splits data into training,
        validation and test sets and creates the required data
        folds for evaluation where a fold of the user's history is
        used to predict another fold.
        """
        self.validation = validation
        if validation:
            self.validation_splitter = splitter_base.StrongGeneralizationSplitter(0.8)

    @abstractmethod
    def split(self, data_m: InteractionMatrix):
        """
        Method to be implemented in all classes that inherit
        from Scenario. Used to create the required data objects.
        Returns them as follows:
        train, test_in, test_out
        or when separate labels for training are provided by the scenario:
        train_X, train_y, test_in, test_out

        :param data_m: Interaction matrix
        :type data: InteractionMatrix
        """
        pass

    @property
    def training_data(self) -> Union[Tuple[InteractionMatrix, InteractionMatrix], InteractionMatrix]:
        return (
            (self.train_X, self.train_y) if hasattr(self, "train_y") else self.train_X
        )

    @property
    def validation_data(self) -> Union[Tuple[InteractionMatrix, InteractionMatrix], None]:
        """
        Returns validation data.

        :return: Validation data matrices as InteractionMatrix in, InteractionMatrix out.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        # TODO: make sure the users in and out match (Here? or elsewhere?)
        if not hasattr(self, "_validation_data_in"):
            return None

        # make sure users match both.
        in_users = self._validation_data_in.active_users
        out_users = self._validation_data_out.active_users

        matching = list(in_users.intersection(out_users))
        return (
            self._validation_data_in.users_in(matching),
            self._validation_data_out.users_in(matching),
        )

    @property
    def test_data(self) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """
        Returns test data.

        :return: Test data matrices as InteractionMatrix in, InteractionMatrix out.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        # make sure users match.
        in_users = self.test_data_in.active_users
        out_users = self.test_data_out.active_users

        matching = list(in_users.intersection(out_users))
        return (
            self.test_data_in.users_in(matching),
            self.test_data_out.users_in(matching),
        )

    def validate(self):
        # TODO Test presency of train_y
        assert hasattr(self, "train_X") and self.train_X is not None
        if self.validation:
            assert (
                hasattr(self, "_validation_data_in")
                and self._validation_data_in is not None
            )
            assert (
                hasattr(self, "_validation_data_out")
                and self._validation_data_out is not None
            )

        assert hasattr(self, "test_data_in") and self.test_data_in is not None
        assert hasattr(self, "test_data_out") and self.test_data_out is not None

        self._check_size()

    def _check_size(self):
        """
        Warns user if any of the sets is unusually small or empty
        """
        n_train = self.train_X.num_interactions
        n_test_in = self.test_data_in.num_interactions
        n_test_out = self.test_data_out.num_interactions
        n_test = n_test_in + n_test_out
        n_total = n_train + n_test

        if self.validation:
            n_val_in = self._validation_data_in.num_interactions
            n_val_out = self._validation_data_out.num_interactions
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
            check("Validation in set", n_val_in, n_val, 0.05)
            check("Validation out set", n_val_out, n_val, 0.01)

