"""
Conversions between the various matrix data types supported by recpack.
"""
import pandas as pd

from recpack.data.data_matrix import DataM, USER_IX, ITEM_IX, VALUE_IX, TIMESTAMP_IX
from scipy.sparse import csr_matrix
from typing import Union, Any, Optional, Tuple


Matrix = Union[DataM, csr_matrix]
_supported_types = Matrix.__args__  # type: ignore


def to_csr_matrix(X: Union[Matrix, Tuple[Matrix, ...]]) -> csr_matrix:
    """
    Converts any matrix to a scipy csr_matrix.

    :param X: Matrix-like object or tuple of objects to convert
    :raises: UnsupportedTypeError
    """
    if isinstance(X, (tuple, list)):
        return type(X)(to_csr_matrix(x) for x in X)
    if isinstance(X, csr_matrix):
        return X
    if isinstance(X, DataM):
        return X.values
    raise UnsupportedTypeError(X)


def to_datam(X: Union[Matrix, Tuple[Matrix, ...]], timestamps: bool = False) -> DataM:
    """
    Converts any matrix to recpack's DataM.

    :param X: Matrix-like object or tuple of objects to convert
    :param timestamps: Require timestamp information, fail if none is available
    :raises: UnsupportedTypeError, InvalidConversionError
    """
    if isinstance(X, (tuple, list)):
        return type(X)(to_datam(x, timestamps) for x in X)
    if timestamps and not _get_timestamps(X):
        raise InvalidConversionError("Source matrix has no timestamp information.")
    if isinstance(X, csr_matrix):
        df = _matrix_to_df(X)
        return DataM(df, shape=X.shape)
    if isinstance(X, DataM):
        return X
    raise UnsupportedTypeError(X)


def to_same_type(X: Matrix, Y: Matrix) -> Matrix:
    """
    Converts any matrix to the same type as a target matrix.

    :param X: Matrix-like object to convert
    :param Y: Matrix-like object with the desired type
    :raises: UnsupportedTypeError, InvalidConversionError
    """
    f_conv = {csr_matrix: to_csr_matrix, DataM: to_datam}.get(type(Y))
    if not f_conv:
        raise UnsupportedTypeError(Y)
    return f_conv(X)


def _matrix_to_df(X: csr_matrix) -> pd.DataFrame:
    """
    Converts a user-item matrix to the dataframe format used by DataM.
    """
    uids, iids = X.nonzero()
    return pd.DataFrame({USER_IX: uids, ITEM_IX: iids, VALUE_IX: X.data})


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
