import recpack.preprocessing.helpers as helpers
import pandas
import pytest
import scipy.sparse
import numpy as np


def test_create_temporal_csr_matrix_from_pandas_df():
    data = {'timestamp': [3, 2, 1, 1], 'item_id': [1, 1, 2, 3], 'user_id': [0, 1, 1, 2]}

    df = pandas.DataFrame.from_dict(data)

    mat = helpers.create_temporal_csr_matrix_from_pandas_df(df, 'item_id', 'user_id', 'timestamp')
    assert (mat.toarray() == np.array([[0, 3, 0, 0], [0, 2, 1, 0], [0, 0, 0, 1]], dtype=np.int32)).all()


def test_create_csr_matrix_from_pandas_df():
    data = {'timestamp': [3, 2, 1, 1], 'item_id': [1, 1, 2, 3], 'user_id': [0, 1, 1, 2]}

    df = pandas.DataFrame.from_dict(data)

    mat = helpers.create_csr_matrix_from_pandas_df(df, 'item_id', 'user_id')
    assert (mat.toarray() == np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.int32)).all()


def test_create_data_M_from_pandas_df():
    data = {'timestamp': [3, 2, 1, 1], 'item_id': [1, 1, 2, 3], 'user_id': [0, 1, 1, 2]}

    df = pandas.DataFrame.from_dict(data)

    d = helpers.create_data_M_from_pandas_df(df, 'item_id', 'user_id', 'timestamp')
    assert d.timestamps is not None
    assert d.values is not None

    assert d.shape == (3, 4)

    d2 = helpers.create_data_M_from_pandas_df(df, 'item_id', 'user_id')
    with pytest.raises(AttributeError):
        d2.timestamps
    assert d2.values is not None
    assert d2.shape == (3, 4)
