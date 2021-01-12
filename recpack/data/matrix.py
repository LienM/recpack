"""
Conversion and validation of the various matrix data types supported by recpack.

In this module the Matrix type is defined, as the union of the DataM object,
and csr_matrix, the typically used sparse represenation.

This allows you to use the classes that support Matrix as parameter type
to be used without the use of the DataM object.
"""
import pandas as pd

from recpack.data.data_matrix import DataM, USER_IX, ITEM_IX, VALUE_IX, TIMESTAMP_IX
from scipy.sparse import csr_matrix
from typing import Union, Any, Optional, Tuple
from warnings import warn


Matrix = Union[DataM, csr_matrix]
_supported_types = Matrix.__args__  # type: ignore


def to_csr_matrix(
    X: Union[Matrix, Tuple[Matrix, ...]], binary: bool = None
) -> csr_matrix:
    """
    Convert a matrix-like object to a scipy csr_matrix.

    :param X: Matrix-like object or tuple of objects to convert
    :param binary: Ensure matrix is binary, sets non-zero values to 1 if not
    :raises: UnsupportedTypeError
    """
    if isinstance(X, (tuple, list)):
        return type(X)(to_csr_matrix(x, binary=binary) for x in X)
    if isinstance(X, csr_matrix):
        X = X
    elif isinstance(X, DataM):
        X = X.values
    else:
        raise UnsupportedTypeError(X)
    return _to_binary(X) if binary else X


def to_datam(X: Union[Matrix, Tuple[Matrix, ...]], timestamps: bool = None) -> DataM:
    """
    Converts any matrix to recpack's DataM.

    :param X: Matrix-like object or tuple of objects to convert
    :param timestamps: If True, an error is raised if timestamps are unavailable
    :raises: UnsupportedTypeError, InvalidConversionError
    """
    if isinstance(X, (tuple, list)):
        return type(X)(to_datam(x, timestamps=timestamps) for x in X)
    if timestamps and not _get_timestamps(X):
        raise InvalidConversionError("Source matrix has no timestamp information.")
    if isinstance(X, csr_matrix):
        df = _matrix_to_df(X)
        return DataM(df, shape=X.shape)
    if isinstance(X, DataM):
        return X
    raise UnsupportedTypeError(X)


def to_same_type(X: Union[Matrix, Tuple[Matrix, ...]], Y: Matrix) -> Matrix:
    """
    Converts any matrix to the same type as a target matrix.

    :param X: Matrix-like object or tuple of objects to convert
    :param Y: Matrix-like object with the desired type
    :raises: UnsupportedTypeError, InvalidConversionError
    """
    if isinstance(X, (tuple, list)):
        return type(X)(to_same_type(x, Y) for x in X)
    f_conv = {csr_matrix: to_csr_matrix, DataM: to_datam}.get(type(Y))
    if not f_conv:
        raise UnsupportedTypeError(Y)
    return f_conv(X)


def _to_binary(X: csr_matrix) -> csr_matrix:
    """
    Checks if a matrix is binary, sets all non-zero values to 1 if not.
    """
    X_binary = X.astype(bool).astype(X.dtype)
    is_binary = (X_binary != X).nnz == 0
    if not is_binary:
        warn("Expected a binary matrix. Setting all non-zero values to 1.")
        return X_binary
    return X


def _matrix_to_df(X: csr_matrix) -> pd.DataFrame:
    """
    Converts a user-item matrix to the dataframe format used by DataM.
    """
    uids, iids = X.nonzero()
    data_nz = X.data[X.data != 0]  # Might contain explicit zeros
    return pd.DataFrame({USER_IX: uids, ITEM_IX: iids, VALUE_IX: data_nz})


def _get_timestamps(X: Matrix) -> Optional[pd.Series]:
    """
    Returns timestamp information as a pandas Series if available, None if not.
    """
    try:
        return X.timestamps
    except:
        return None


def _is_supported(t: Any) -> bool:
    """
    Returns whether a given matrix type is supported by recpack.
    """
    if not isinstance(t, type):
        t = type(t)
    return issubclass(t, _supported_types)


class UnsupportedTypeError(Exception):
    """
    Raised when a matrix of type not supported by recpack is received.

    :param X: The matrix object received
    """

    def __init__(self, X: Any):
        assert not _is_supported(X)
        super().__init__(
            "Recpack only supports matrix types {}. Received {}.".format(
                ", ".join(t.__name__ for t in _supported_types), type(X).__name__
            )
        )


class InvalidConversionError(Exception):
    """
    Raised when converting between two incompatible matrix formats.

    :param message: Reason the conversion failed
    """

    def __init__(self, message: str):
        super().__init__(message)
