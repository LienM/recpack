from abc import ABC, abstractmethod
from typing import Tuple, Union

import recpack.splitters.splitter_base as splitter_base
from recpack.data_matrix import DataM


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
            self.validation_splitter = splitter_base.StrongGeneralizationSplitter(0.5)

    @abstractmethod
    def split(self, *data_ms: DataM):
        """
        Method to be implemented in all classes that inherit
        from Scenario. Used to create the required data objects.
        Returns them as follows:
        train, test_in, test_out
        or when separate labels for training are provided by the scenario:
        train_X, train_y, test_in, test_out

        :param data_ms: List of datasets
        :type data: List[DataM]
        """
        pass

    @property
    def training_data(self) -> Tuple[DataM, DataM]:
        return (
            (self.train_X, self.train_y)
            if hasattr(self, "train_y")
            else self.train_X
        )

    @property
    def validation_data(self) -> Union[Tuple[DataM, DataM], None]:
        """
        Returns validation data.

        :return: Validation data matrices as DataM in, DataM out.
        :rtype: Tuple[DataM, DataM]
        """
        return (self.validation_data_in, self.validation_data_out) if self.validation_data_in else None

    @property
    def test_data(self) -> Tuple[DataM, DataM]:
        """
        Returns test data.

        :return: Test data matrices as DataM in, DataM out.
        :rtype: Tuple[DataM, DataM]
        """
        return (self.test_data_in, self.test_data_out)

    def validate(self):
        # TODO Test presency of train_y
        assert hasattr(self, "train_X") and self.train_X is not None
        if self.validation:
            assert hasattr(self, "validation_data_in") and self.validation_data_in is not None
            assert hasattr(self, "validation_data_out") and self.validation_data_out is not None

        assert hasattr(self, "test_data_in") and self.test_data_in is not None
        assert hasattr(self, "test_data_out") and self.test_data_out is not None
