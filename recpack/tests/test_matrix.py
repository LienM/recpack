# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from itertools import islice
import warnings

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, mock_open, patch
import yaml

from recpack.matrix.util import (
    UnsupportedTypeError,
)
from recpack.matrix import InteractionMatrix, to_csr_matrix
from scipy.sparse import csr_matrix


USER_IX = InteractionMatrix.USER_IX
ITEM_IX = InteractionMatrix.ITEM_IX
TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX


@pytest.fixture(scope="function")
def df():
    data = {TIMESTAMP_IX: [3, 2, 1, 1], ITEM_IX: [1, 1, 2, 3], USER_IX: [0, 1, 1, 2]}
    df = pd.DataFrame.from_dict(data)

    return df


@pytest.fixture(scope="function")
def df_w_duplicate():
    data = {
        TIMESTAMP_IX: [3, 2, 4, 1, 1],
        ITEM_IX: [1, 1, 1, 2, 3],
        USER_IX: [0, 1, 1, 1, 2],
    }
    df = pd.DataFrame.from_dict(data)

    return df


@pytest.fixture(scope="function")
def interaction_m(df):
    d = InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)

    return d


@pytest.fixture(scope="function")
def interaction_m_w_duplicate(df_w_duplicate):
    d = InteractionMatrix(df_w_duplicate, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)

    return d


def test_init_w_timestamps(interaction_m):
    assert interaction_m.timestamps is not None
    assert interaction_m.values is not None

    assert interaction_m.shape == (3, 4)

    assert (
        interaction_m.values.toarray() == np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.int32)
    ).all()

    assert interaction_m.has_timestamps


def test_init_without_timestamps(df):
    # Drop the column, otherwise it matches the column being checked for.
    # And the class detects timestamps are actually available.
    d2 = InteractionMatrix(df.drop(columns=[TIMESTAMP_IX]), ITEM_IX, USER_IX)
    with pytest.raises(AttributeError):
        d2.timestamps
    assert d2.values is not None
    assert d2.shape == (3, 4)
    assert not d2.has_timestamps


def test_copy(interaction_m):
    im = interaction_m.copy()

    
def test_init_smaller_shapes(df):
    with pytest.raises(ValueError) as e:
        InteractionMatrix(df, ITEM_IX, USER_IX, TIMESTAMP_IX, shape=(1, 4))

    assert e.match("fewer rows than maximal user identifier")

    with pytest.raises(ValueError) as e:
        InteractionMatrix(df, ITEM_IX, USER_IX, TIMESTAMP_IX, shape=(3, 1))

    assert e.match("fewer columns than maximal item identifier")


def test_values_w_dups(interaction_m_w_duplicate):

    assert id(im) != id(interaction_m)


def test_values_w_dups(interaction_m_w_duplicate):
    assert (
        interaction_m_w_duplicate.values.toarray()
        == np.array([[0, 1, 0, 0], [0, 2, 1, 0], [0, 0, 0, 1]], dtype=np.int32)
    ).all()


def test_binary_values_w_dups(interaction_m_w_duplicate):
    binary_values = interaction_m_w_duplicate.binary_values

    assert (binary_values.toarray() == np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.int32)).all()


def test_timestamps_no_dups(interaction_m):
    assert (interaction_m.timestamps.values == np.array([3, 2, 1, 1])).all()


def test_timestamps_w_dups(interaction_m_w_duplicate):
    assert (interaction_m_w_duplicate.timestamps.values == np.array([3, 2, 4, 1, 1])).all()


def test_timestamps_gt_w_dups(interaction_m_w_duplicate):
    filtered_d_w_duplicate = interaction_m_w_duplicate.timestamps_gt(2)

    assert (filtered_d_w_duplicate.timestamps.values == np.array([3, 4])).all()

    assert (
        filtered_d_w_duplicate.values.toarray() == np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    ).all()


def test_timestamps_lt_w_dups(interaction_m_w_duplicate):
    filtered_d_w_duplicate = interaction_m_w_duplicate.timestamps_lt(2)

    # data = {'timestamp': [3, 2, 1, 1, 4], 'item_id': [1, 1, 2, 3, 1], 'user_id': [0, 1, 1, 2, 1]}

    assert (filtered_d_w_duplicate.timestamps.values == np.array([1, 1])).all()
    assert (
        filtered_d_w_duplicate.values.toarray() == np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.int32)
    ).all()


def test_timestamps_gte_w_dups(interaction_m_w_duplicate):
    filtered_d_w_duplicate = interaction_m_w_duplicate.timestamps_gte(2)

    assert (filtered_d_w_duplicate.timestamps.values == np.array([3, 2, 4])).all()

    assert (
        filtered_d_w_duplicate.values.toarray() == np.array([[0, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    ).all()


def test_timestamps_lte_w_dups(interaction_m_w_duplicate):
    filtered_d_w_duplicate = interaction_m_w_duplicate.timestamps_lte(2)

    # data = {'timestamp': [3, 2, 1, 1, 4], 'item_id': [1, 1, 2, 3, 1], 'user_id': [0, 1, 1, 2, 1]}

    assert (filtered_d_w_duplicate.timestamps.values == np.array([2, 1, 1])).all()
    assert (
        filtered_d_w_duplicate.values.toarray() == np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.int32)
    ).all()


def test_indices_in(interaction_m):
    U = [0, 1]
    I = [1, 2]

    filtered_df = interaction_m.indices_in((U, I))

    assert (filtered_df.timestamps.values == np.array([3, 1])).all()
    assert (filtered_df.values.toarray() == np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.int32)).all()


def test_eliminate_timestamps(df):
    d = InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)

    assert d.timestamps is not None

    d2 = d.eliminate_timestamps()
    assert d.timestamps is not None
    with pytest.raises(AttributeError):
        d2.timestamps
    assert InteractionMatrix.TIMESTAMP_IX not in d2._df

    t = d.eliminate_timestamps(inplace=True)
    assert t is None
    with pytest.raises(AttributeError):
        d.timestamps
    assert InteractionMatrix.TIMESTAMP_IX not in d._df


def test_users_in(df):
    d = InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)

    d2 = d.users_in([0, 1])
    assert d2.shape == (3, 4)
    assert len(list(d2.binary_item_history)) == 2

    # user_id 2 is not known to the DataFrame
    d.users_in([2, 3], inplace=True)
    assert len(list(d.binary_item_history)) == 1


def test_items_in(df):
    d = InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)

    d2 = d.items_in([0, 1])
    assert d2.shape == (3, 4)
    assert len(list(d2.binary_item_history)) == 2

    # item_id 0 is not known to the DataFrame
    d.items_in([0], inplace=True)
    assert len(list(d.binary_item_history)) == 0


def test_interactions_in_empty_set(df):
    d = InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)
    with pytest.warns(UserWarning, match="No interaction IDs given, returning empty InteractionMatrix.") as w:
        d2 = d.interactions_in([])
        assert d2.shape == (3, 4)
        assert (d2.values.toarray() == np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)).all()


def test_interactions_in(df):
    d = InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)

    d2 = d.interactions_in([0, 1])
    assert d2.shape == (3, 4)
    assert (d2.values.toarray() == np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int32)).all()

    with pytest.warns(UserWarning, match="IDs \{10\} not present in data") as w:
        # interaction_id 10 is not known to the DataFrame
        d.interactions_in([0, 1, 10], inplace=True)
        assert (d.values.toarray() == np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int32)).all()


def test_get_timestamp_raises(df):
    df_no_timestamps = df.drop(columns=[InteractionMatrix.TIMESTAMP_IX], errors="ignore", inplace=False)

    d = InteractionMatrix(df_no_timestamps, ITEM_IX, USER_IX)

    # No timestamps, will raise exception
    with pytest.raises(AttributeError):
        d.get_timestamp(0)


def test_get_timestamp_raises_keyerror(interaction_m):
    # Unknown interaction id, will raise exception
    with pytest.raises(KeyError):
        interaction_m.get_timestamp(100)


def test_get_timestamp(interaction_m):
    ts = interaction_m.get_timestamp(3)
    assert ts == 1


def test_binary_item_history(interaction_m_w_duplicate):
    histories = interaction_m_w_duplicate.binary_item_history
    expected_histories = {0: [1], 1: [1, 2], 2: [3]}
    for i, hist in histories:
        assert sorted(hist) == expected_histories[i]


def test_interaction_history(interaction_m):
    for uid, user_history in interaction_m.interaction_history:
        hist = list(user_history)

        assert len(hist) == interaction_m.values[uid, :].nnz


def test_sorted_interaction_history_no_timestamps_raises(df):
    df_no_timestamps = df.drop(columns=[InteractionMatrix.TIMESTAMP_IX], errors="ignore", inplace=False)
    d = InteractionMatrix(df_no_timestamps, ITEM_IX, USER_IX)

    with pytest.raises(AttributeError):
        for uid, user_history in d.sorted_interaction_history:
            pass


def test_sorted_interaction_history2(interaction_m_w_duplicate):
    cnt_users = 0

    for uid, user_history in interaction_m_w_duplicate.sorted_interaction_history:
        hist = list(user_history)
        for id1, id2 in zip(hist, islice(hist, 1, None)):
            t1 = interaction_m_w_duplicate.get_timestamp(id1)
            t2 = interaction_m_w_duplicate.get_timestamp(id2)

            assert t1 <= t2

        cnt_users += 1

    assert cnt_users == interaction_m_w_duplicate.num_active_users


def test_sorted_interaction_history(interaction_m):
    for uid, user_history in interaction_m.sorted_interaction_history:
        hist = list(user_history)
        for id1, id2 in zip(hist, islice(hist, 1, None)):
            t1 = interaction_m.get_timestamp(id1)
            t2 = interaction_m.get_timestamp(id2)

            assert t1 <= t2


def test_active_users(interaction_m):
    assert interaction_m.active_users == {0, 1, 2}


def test_active_items(interaction_m):
    assert interaction_m.active_items == {1, 2, 3}


def test_density(interaction_m):
    np.testing.assert_almost_equal(interaction_m.density, 1 / 3)


def test_num_active_users(interaction_m):
    assert interaction_m.num_active_users == 3


def test_num_active_items(interaction_m):
    assert interaction_m.num_active_items == 3


def test_num_interactions(interaction_m):
    assert interaction_m.num_interactions == 4


def test_num_interactions2(interaction_m_w_duplicate):
    assert interaction_m_w_duplicate.num_interactions == 5


def test_from_csr_matrix(data):
    interaction_m = InteractionMatrix.from_csr_matrix(data)

    assert not interaction_m.has_timestamps

    assert (interaction_m.values.toarray() == data.toarray()).all()


# ----- TEST CONVERSIONS


@pytest.fixture
def m_csr():
    m = [[1, 1, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0]]
    return csr_matrix(m, dtype=np.int32)


@pytest.fixture
def m_csr_binary():
    m = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    return csr_matrix(m, dtype=np.int32)


@pytest.fixture
def m_datam():
    df = pd.DataFrame(
        {
            USER_IX: [0, 0, 1, 1, 1],
            ITEM_IX: [0, 1, 1, 2, 2],
            TIMESTAMP_IX: [3, 2, 4, 1, 2],
        }
    )
    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX, shape=(3, 4))


def matrix_equal(a, b):
    if type(a) != type(b):
        return False
    return np.array_equal(a.toarray(), b.toarray())


def test_to_csr_matrix_csr(m_csr):
    # csr_matrix -> csr_matrix
    result = to_csr_matrix(m_csr)
    assert result is m_csr


def test_to_csr_matrix_interaction_matrix(m_csr, m_datam):
    # InteractionMatrix -> csr_matrix
    result = to_csr_matrix(m_datam)
    assert matrix_equal(result, m_csr)


def test_to_csr_matrix_tup_interaction_matrix(m_csr, m_datam):
    # tuple -> tuple
    result = to_csr_matrix((m_datam, m_datam))
    assert all(matrix_equal(r, m_csr) for r in result)


def test_to_csr_matrix_tup_interaction_matrix2(m_csr, m_datam):
    # tuple(Matrix, tuple(Matrix, Matrix))
    # Useful for when validation data and train data need to be converted
    result_1, (r_2_1, r_2_2) = to_csr_matrix((m_datam, (m_datam, m_datam)))
    assert matrix_equal(result_1, m_csr)
    assert matrix_equal(r_2_1, m_csr)
    assert matrix_equal(r_2_2, m_csr)


def test_to_csr_matrix_unsupported_type():
    # unsupported type
    with pytest.raises(UnsupportedTypeError):
        result = to_csr_matrix([1, 2, 3])


def test_to_binary_csr(m_csr, m_datam, m_csr_binary):
    # csr_matrix -> csr_matrix
    result = to_csr_matrix(m_csr, binary=True)
    assert matrix_equal(result, m_csr_binary)
    assert result.dtype == m_csr.dtype
    result = to_csr_matrix(m_csr_binary, binary=True)
    assert matrix_equal(result, m_csr_binary)


def test_to_binary_csr2(m_csr, m_datam, m_csr_binary):
    # InteractionMatrix -> csr_matrix
    result = to_csr_matrix(m_datam, binary=True)
    assert matrix_equal(result, m_csr_binary)


def test_to_binary_csr3(m_csr, m_datam, m_csr_binary):
    # tuple -> tuple
    result = to_csr_matrix((m_csr, m_datam), binary=True)
    assert matrix_equal(result[0], m_csr_binary)
    assert matrix_equal(result[1], m_csr_binary)


def test_save(larger_mat):
    mocker = mock_open()
    open_name = "recpack.matrix.interaction_matrix.open"
    mocker2 = MagicMock()
    with patch("recpack.matrix.interaction_matrix.pd.DataFrame.to_csv", mocker2):
        with patch(open_name, mocker):
            larger_mat.save("test_data")

    mocker.assert_called_once_with("test_data_properties.yaml", "w")

    handle = mocker()
    # handle.call
    assert handle.write.call_count == 1
    handle.write.assert_called_once_with(yaml.safe_dump(larger_mat.properties.to_dict()))

    assert mocker2.call_count == 1
    mocker2.assert_called_once_with("test_data.csv", header=True, index=False)


def test_load(larger_mat):
    mocker2 = MagicMock(return_value=larger_mat._df)
    with patch("recpack.matrix.interaction_matrix.pd.read_csv", mocker2):
        with patch(
            "recpack.matrix.interaction_matrix.open",
            mock_open(read_data=yaml.safe_dump(larger_mat.properties.to_dict())),
        ) as mocker:
            im = InteractionMatrix.load("test_data")

    mocker.assert_called_once_with("test_data_properties.yaml", "r")

    assert mocker2.call_count == 1
    mocker2.assert_called_once_with("test_data.csv")

    assert im.shape == larger_mat.shape
    assert im.active_users == larger_mat.active_users

    # Added to check bug loading shape of InteractionMatrix
    prop = im.properties


def test_add(larger_mat):
    double_mat = larger_mat + larger_mat

    assert double_mat.num_interactions == larger_mat.num_interactions * 2
    assert double_mat.active_users == larger_mat.active_users
    np.testing.assert_array_equal(double_mat.binary_values.toarray(), larger_mat.binary_values.toarray())
    assert (
        double_mat._df[double_mat.INTERACTION_IX].nunique() == 2 * larger_mat._df[larger_mat.INTERACTION_IX].nunique()
    )
    np.testing.assert_array_equal(double_mat._df.columns, larger_mat._df.columns)


def test_add_mismatch(mat, larger_mat):
    with pytest.raises(ValueError):
        mat + larger_mat


def test_im_to_timestamp_matrix(matrix_sessions):
    # fmt: off
    expected_matrix = np.array([
        [0, 1, 2, 0],
        [0, 5, 1, 4],
        [0, 7, 6, 0],
        [0, 7, 6, 1],
        [0, 6, 7, 0]
    ])
    # fmt: on

    np.testing.assert_array_equal(matrix_sessions.last_timestamps_matrix.toarray(), expected_matrix)
