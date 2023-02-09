# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""Filter classes for handling postprocessing of Algorithm outputs."""

from typing import List
from abc import ABC, abstractmethod

from scipy.sparse import csr_matrix
import numpy as np
from numpy.typing import ArrayLike


class PostFilter(ABC):
    """Abstract baseclass for postprocessing filter implementations

    A filter needs to implement an :meth:`apply` method,
    which takes as input a csr_matrix, and returns a processed csr_matrix.
    """

    def apply_all(self, *csr_matrices: csr_matrix) -> List[csr_matrix]:
        """Apply the filter to each of the matrices.

        :param csr_matrices: The matrices to apply the filter to.
        :type csr_matrices: csr_matrix
        :return: The list of processed csr_matrices
        :rtype: List[csr_matrix]
        """
        if len(csr_matrices) == 0:
            return []

        first = csr_matrices[0].shape
        if not all(first == x.shape for x in csr_matrices):
            raise ValueError("Not all csr_matrices are the same shape.")

        return [self.apply(csr) for csr in csr_matrices]

    @abstractmethod
    def apply(self, X_pred: csr_matrix) -> csr_matrix:
        """Process the predictions, and return the processed matrix.

        :param X_pred: csr_matrix to filter
        :type X_pred: csr_matrix
        :return: The processed csr_matrix.
        :rtype: csr_matrix
        """
        raise NotImplementedError("Subclasses should implement this method!")

    def __str__(self):
        attrs = self.__dict__
        return f"{self.__class__.__name__}({', '.join((f'{k}={v}' for k, v in attrs.items()))})"


class ExcludeItems(PostFilter):
    """Remove the recommendations of specified items.
    https://numpy.org/doc/stable/reference/typing.html#numpy.typing.ArrayLike

    :param items: ArrayLike with the identifiers for items to be removed
    :type items: ArrayLike
    """

    def __init__(self, items: ArrayLike):
        self.items = items

    def apply(self, X_pred: csr_matrix) -> csr_matrix:
        amount_of_items = X_pred.shape[1]

        if len(self.items) == 0 or np.amax(self.items) > amount_of_items:
            raise ValueError(f"{X_pred.shape} and {self.items.shape}")

        mask = np.ones((amount_of_items,), dtype=bool)
        mask[self.items] = False

        return X_pred.multiply(mask).tocsr()


class SelectItems(PostFilter):
    """Select only the recommendations of specified items.
    https://numpy.org/doc/stable/reference/typing.html#numpy.typing.ArrayLike

    :param items: ArrayLike with the identifiers for items to be selected
    :type items: ArrayLike
    """

    def __init__(self, items: ArrayLike):
        self.items = items

    def apply(self, X_pred: csr_matrix) -> csr_matrix:
        amount_of_items = X_pred.shape[1]

        if len(self.items) == 0 or np.amax(self.items) > amount_of_items:
            raise ValueError(f"{X_pred.shape} and {self.items.shape}")

        mask = np.zeros((amount_of_items,), dtype=bool)
        mask[self.items] = True

        return X_pred.multiply(mask).tocsr()
