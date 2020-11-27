import numpy as np
import pandas as pd
import pytest

from recpack.data.matrix import (
    to_csr_matrix,
    to_datam,
    to_same_type,
    UnsupportedTypeError,
    InvalidConversionError,
    _get_timestamps,
)
from recpack.data.data_matrix import DataM, USER_IX, ITEM_IX, TIMESTAMP_IX, VALUE_IX
from scipy.sparse import csr_matrix


# fmt: off
@pytest.fixture
def m_csr():
    m = [[1, 2, 0, 0], 
         [0, 3, 4, 0], 
         [0, 0, 0, 0]]
    return csr_matrix(m, dtype=np.float)


@pytest.fixture
def m_csr_binary():
    m = [[1, 1, 0, 0], 
         [0, 1, 1, 0], 
         [0, 0, 0, 0]]
    return csr_matrix(m)


@pytest.fixture
def m_datam():
    df = pd.DataFrame(
        {
            USER_IX: [0, 0, 1, 1, 1],
            ITEM_IX: [0, 1, 1, 2, 2],
            VALUE_IX: [1, 2, 3, 2, 2],
            TIMESTAMP_IX: [3, 2, 4, 1, 2],
        }
    )
    return DataM(df, shape=(3, 4))
# fmt: on


def matrix_equal(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, csr_matrix):
        return np.array_equal(a.toarray(), b.toarray())
    if isinstance(a, DataM):
        ts_a = _get_timestamps(a)
        ts_b = _get_timestamps(b)
        return matrix_equal(a.values, b.values) and (
            (ts_a is None and ts_b is None) or (ts_a is not None and ts_a.equals(ts_b))
        )


def test_to_csr_matrix(m_csr, m_datam):
    # csr_matrix -> csr_matrix
    result = to_csr_matrix(m_csr)
    assert result is m_csr
    # DataM -> csr_matrix
    result = to_csr_matrix(m_datam)
    assert matrix_equal(result, m_csr)
    # tuple -> tuple
    result = to_csr_matrix((m_datam, m_datam))
    assert all(matrix_equal(r, m_csr) for r in result)
    # tuple(Matrix, tuple(Matrix, Matrix))
    # Useful for when validation data and train data need to be converted
    result_1, (r_2_1, r_2_2) = to_csr_matrix((m_datam, (m_datam, m_datam)))
    assert matrix_equal(result_1, m_csr)
    assert matrix_equal(r_2_1, m_csr)
    assert matrix_equal(r_2_2, m_csr)
    # unsupported type
    with pytest.raises(UnsupportedTypeError):
        result = to_csr_matrix([1, 2, 3])


def test_to_datam(m_csr, m_datam):
    # DataM -> DataM
    result = to_datam(m_datam)
    assert result is m_datam
    # csr_matrix -> DataM
    result = to_datam(m_csr)
    assert matrix_equal(result.values, m_datam.values)
    # tuple -> tuple
    result = to_datam((m_csr, m_csr))
    assert all(matrix_equal(r.values, m_datam.values) for r in result)
    # unsupported type
    with pytest.raises(UnsupportedTypeError):
        result = to_datam([1, 2, 3])
    # missing timestamp information
    with pytest.raises(InvalidConversionError):
        result = to_datam(m_csr, timestamps=True)


def test_to_same_type(m_csr, m_datam):
    # csr_matrix -> DataM
    result = to_same_type(m_csr, m_datam)
    assert matrix_equal(result.values, m_datam.values)
    # DataM -> csr_matrix
    result = to_same_type(m_datam, m_csr)
    assert matrix_equal(result, m_csr)
    # unsupported type
    with pytest.raises(UnsupportedTypeError):
        result = to_csr_matrix([1, 2, 3])


def test_to_binary(m_csr, m_datam, m_csr_binary):
    # csr_matrix -> csr_matrix
    result = to_csr_matrix(m_csr, binary=True)
    assert matrix_equal(result, m_csr_binary)
    assert result.dtype == m_csr.dtype
    result = to_csr_matrix(m_csr_binary, binary=True)
    assert result is m_csr_binary
    # DataM -> csr_matrix
    result = to_csr_matrix(m_datam, binary=True)
    assert matrix_equal(result, m_csr_binary)
    # tuple -> tuple
    result = to_csr_matrix((m_csr, m_datam), binary=True)
    assert matrix_equal(result[0], m_csr_binary)
    assert matrix_equal(result[1], m_csr_binary)