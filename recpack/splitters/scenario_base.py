from abc import ABC, abstractmethod
from typing import Tuple, Union

import recpack.splitters.splitter_base as splitter_base
from recpack.data.data_matrix import DataM


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
    def split(self, data_m: DataM):
        """
        Method to be implemented in all classes that inherit
        from Scenario. Used to create the required data objects.
        Returns them as follows:
        train, test_in, test_out
        or when separate labels for training are provided by the scenario:
        train_X, train_y, test_in, test_out

        :param data_m: Interaction matrix
        :type data: DataM
        """
        pass

    @property
    def training_data(self) -> Tuple[DataM, DataM]:
        return (
            (self.train_X, self.train_y) if hasattr(self, "train_y") else self.train_X
        )

    @property
    def validation_data(self) -> Union[Tuple[DataM, DataM], None]:
        """
        Returns validation data.

        :return: Validation data matrices as DataM in, DataM out.
        :rtype: Tuple[DataM, DataM]
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
    def test_data(self) -> Tuple[DataM, DataM]:
        """
        Returns test data.

        :return: Test data matrices as DataM in, DataM out.
        :rtype: Tuple[DataM, DataM]
        """
        # make sure users match.
        in_users = self.test_data_in.active_users
        out_users = self.test_data_out.active_users

        matching = list(in_users.intersection(out_users))
        self.test_data_in.users_in(matching, inplace=True)
        self.test_data_out.users_in(matching, inplace=True)

        return (self.test_data_in, self.test_data_out)

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
