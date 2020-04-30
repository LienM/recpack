from abc import ABC, abstractmethod
from typing import Tuple

import recpack.splitters.splitter_base as splitter_base
from recpack.splitters.splitter_base import FoldIterator
from recpack.data_matrix import DataM


class Scenario(ABC):

    def __init__(self):
        """
        Base class for defining an evaluation "Scenario",
        i.e. a set of steps that splits data into training,
        validation and test sets and creates the required data
        folds for evaluation where a fold of the user's history is
        used to predict another fold.
        """
        self.training_data = None
        self.validation_data_in = None
        self.validation_data_out = None
        self.test_data_in = None
        self.test_data_out = None

        self.validation_splitter = splitter_base.StrongGeneralizationSplitter(0.5)

    @abstractmethod
    def split(self, *data_ms: DataM):
        """
        Method to be implemented in all classes that inherit
        from Scenario. Used to create the required data objects.
        Assumes no more than two data matrices will be used in any
        given Scenario.

        :param data: First data object.
        :type data: DataM
        :param data_2: Second data object.
        :type data_2: DataM
        """
        pass

    @property
    def validation_data(self) -> Tuple[DataM, DataM]:
        """
        Returns validation data.

        :return: Validation data matrices as DataM in, DataM out.
        :rtype: Tuple[DataM, DataM]
        """
        return (self.validation_data_in, self.validation_data_out)

    @property
    def test_data(self) -> Tuple[DataM, DataM]:
        """
        Returns test data.

        :return: Test data matrices as DataM in, DataM out.
        :rtype: Tuple[DataM, DataM]
        """
        return (self.test_data_in, self.test_data_out)

    @property
    def test_iterator(self) -> FoldIterator:
        # TODO Fix/Make sure this works
        return FoldIterator(self.test_data_in, self.test_data_out)
