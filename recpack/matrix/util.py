# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from typing import Any, Tuple, Union

from scipy.sparse import csr_matrix

from recpack.matrix.interaction_matrix import InteractionMatrix
from recpack.util import to_binary

# Conversion and validation of the various matrix data types supported by recpack.

# In this module the Matrix type is defined, as the union of the InteractionMatrix object,
# and csr_matrix, the typically used sparse represenation.

# This allows you to use the classes that support Matrix as parameter type
# to be used without the use of the InteractionMatrix object.
Matrix = Union[InteractionMatrix, csr_matrix]

_supported_types = Matrix.__args__  # type: ignore


def to_csr_matrix(
    X: Union[Matrix, Tuple[Matrix, ...]], binary: bool = False
) -> Union[csr_matrix, Tuple[csr_matrix, ...]]:
    """Convert a matrix-like object to a scipy csr_matrix.

    :param X: Matrix-like object or tuple of objects to convert.
    :type X: csr_matrix
    :param binary: If true, ensure matrix is binary by setting non-zero values to 1.
    :type binary: bool, optional
    :raises: UnsupportedTypeError
    :return: Matrices as csr_matrix.
    :rtype: Union[csr_matrix, Tuple[csr_matrix, ...]]
    """
    if isinstance(X, (tuple, list)):
        return type(X)(to_csr_matrix(x, binary=binary) for x in X)
    if isinstance(X, csr_matrix):
        res = X
    elif isinstance(X, InteractionMatrix):
        res = X.values
    else:
        raise UnsupportedTypeError(X)
    return to_binary(res) if binary else res


def _is_supported(t: Any) -> bool:
    """Returns whether a given matrix type is supported by recpack.

    :param t: The type of the object.
    :type t: Any
    :return: True if supported, else False.
    :rtype: bool
    """
    if not isinstance(t, type):
        t = type(t)
    return issubclass(t, _supported_types)


class UnsupportedTypeError(Exception):
    """Raised when a matrix of type not supported by recpack is received.

    :param X: The matrix object received
    :type X: Any
    """

    def __init__(self, X: Any):
        assert not _is_supported(X)
        super().__init__(
            "Recpack only supports matrix types {}. Received {}.".format(
                ", ".join(t.__name__ for t in _supported_types), type(X).__name__
            )
        )
