from abc import ABC, abstractmethod
from typing import Tuple

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
