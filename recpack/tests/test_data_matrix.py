from recpack.data.data_matrix import DataM
import pandas as pd
import pytest
import numpy as np


@pytest.fixture(scope="function")
def df():
    data = {"timestamp": [3, 2, 1, 1], "item_id": [1, 1, 2, 3], "user_id": [0, 1, 1, 2]}
    df = pd.DataFrame.from_dict(data)

    return df


@pytest.fixture(scope="function")
def df_w_duplicate():
    data = {
        "timestamp": [3, 2, 4, 1, 1],
        "item_id": [1, 1, 1, 2, 3],
        "user_id": [0, 1, 1, 1, 2],
    }
    df = pd.DataFrame.from_dict(data)

    return df


@pytest.fixture(scope="function")
def df_w_values_and_duplicate():
    data = {
        "timestamp": [3, 2, 4, 1, 1],
        "item_id": [1, 1, 1, 2, 3],
        "user_id": [0, 1, 1, 1, 2],
        "value": [2, 3, 4, 5, 6],
    }
    df = pd.DataFrame.from_dict(data)

    return df


def test_create_data_M_from_pandas_df(df):
    d = DataM.create_from_dataframe(df, "item_id", "user_id", timestamp_ix="timestamp")
    assert d.timestamps is not None
    assert d.values is not None

    assert d.shape == (3, 4)

    d2 = DataM.create_from_dataframe(df, "item_id", "user_id")
    with pytest.raises(AttributeError):
        d2.timestamps
    assert d2.values is not None
    assert d2.shape == (3, 4)


def test_values_no_dups(df):
    d = DataM.create_from_dataframe(df, "item_id", "user_id", timestamp_ix="timestamp")
    assert (
        d.values.toarray()
        == np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.int32)
    ).all()


def test_values_w_dups(df_w_duplicate):
    d_w_duplicate = DataM.create_from_dataframe(
        df_w_duplicate, "item_id", "user_id", timestamp_ix="timestamp"
    )
    assert (
        d_w_duplicate.values.toarray()
        == np.array([[0, 1, 0, 0], [0, 2, 1, 0], [0, 0, 0, 1]], dtype=np.int32)
    ).all()


def test_values_w_values_and_dups(df_w_values_and_duplicate):
    d = DataM.create_from_dataframe(
        df_w_values_and_duplicate,
        "item_id",
        "user_id",
        value_ix="value",
        timestamp_ix="timestamp",
    )
    assert (
        d.values.toarray()
        == np.array([[0, 2, 0, 0], [0, 7, 5, 0], [0, 0, 0, 6]], dtype=np.int32)
    ).all()


def test_binary_values_w_dups(df_w_duplicate):
    d_w_duplicate = DataM.create_from_dataframe(
        df_w_duplicate, "item_id", "user_id", timestamp_ix="timestamp"
    )

    binary_values = d_w_duplicate.binary_values

    assert (
        binary_values.toarray()
        == np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.int32)
    ).all()


def test_timestamps_no_dups(df):
    d = DataM.create_from_dataframe(df, "item_id", "user_id", timestamp_ix="timestamp")

    assert (d.timestamps.values == np.array([3, 2, 1, 1])).all()


def test_timestamps_w_dups(df_w_duplicate):
    d = DataM.create_from_dataframe(
        df_w_duplicate, "item_id", "user_id", timestamp_ix="timestamp"
    )

    assert (d.timestamps.values == np.array([3, 2, 4, 1, 1])).all()


def test_timestamps_gt_w_dups(df_w_duplicate):
    d_w_duplicate = DataM.create_from_dataframe(
        df_w_duplicate, "item_id", "user_id", timestamp_ix="timestamp"
    )

    filtered_d_w_duplicate = d_w_duplicate.timestamps_gt(2)

    assert (filtered_d_w_duplicate.timestamps.values == np.array([3, 4])).all()

    assert (
        filtered_d_w_duplicate.values.toarray()
        == np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    ).all()


def test_timestamps_lt_w_dups(df_w_duplicate):
    d_w_duplicate = DataM.create_from_dataframe(
        df_w_duplicate, "item_id", "user_id", timestamp_ix="timestamp"
    )

    filtered_d_w_duplicate = d_w_duplicate.timestamps_lt(2)

    # data = {'timestamp': [3, 2, 1, 1, 4], 'item_id': [1, 1, 2, 3, 1], 'user_id': [0, 1, 1, 2, 1]}

    assert (filtered_d_w_duplicate.timestamps.values == np.array([1, 1])).all()
    assert (
        filtered_d_w_duplicate.values.toarray()
        == np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.int32)
    ).all()


def test_timestamps_gte_w_dups(df_w_duplicate):
    d_w_duplicate = DataM.create_from_dataframe(
        df_w_duplicate, "item_id", "user_id", timestamp_ix="timestamp"
    )

    filtered_d_w_duplicate = d_w_duplicate.timestamps_gte(2)

    assert (filtered_d_w_duplicate.timestamps.values == np.array([3, 2, 4])).all()

    assert (
        filtered_d_w_duplicate.values.toarray()
        == np.array([[0, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    ).all()


def test_timestamps_lte_w_dups(df_w_duplicate):
    d_w_duplicate = DataM.create_from_dataframe(
        df_w_duplicate, "item_id", "user_id", timestamp_ix="timestamp"
    )

    filtered_d_w_duplicate = d_w_duplicate.timestamps_lte(2)

    # data = {'timestamp': [3, 2, 1, 1, 4], 'item_id': [1, 1, 2, 3, 1], 'user_id': [0, 1, 1, 2, 1]}

    assert (filtered_d_w_duplicate.timestamps.values == np.array([2, 1, 1])).all()
    assert (
        filtered_d_w_duplicate.values.toarray()
        == np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.int32)
    ).all()


def test_timestamps_gt_w_values_and_dups(df_w_values_and_duplicate):
    d_w_duplicate = DataM.create_from_dataframe(
        df_w_values_and_duplicate, "item_id", "user_id", timestamp_ix="timestamp"
    )

    filtered_d_w_duplicate = d_w_duplicate.timestamps_gt(2)

    assert (filtered_d_w_duplicate.timestamps.values == np.array([3, 4])).all()

    assert (
        filtered_d_w_duplicate.values.toarray()
        == np.array([[0, 2, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    ).all()


def test_timestamps_lt_w_values_and_dups(df_w_values_and_duplicate):
    d_w_duplicate = DataM.create_from_dataframe(
        df_w_values_and_duplicate, "item_id", "user_id", timestamp_ix="timestamp"
    )

    filtered_d_w_duplicate = d_w_duplicate.timestamps_lt(2)

    # data = {'timestamp': [3, 2, 1, 1, 4], 'item_id': [1, 1, 2, 3, 1], 'user_id': [0, 1, 1, 2, 1]}

    assert (filtered_d_w_duplicate.timestamps.values == np.array([1, 1])).all()
    assert (
        filtered_d_w_duplicate.values.toarray()
        == np.array([[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 0, 6]], dtype=np.int32)
    ).all()


def test_timestamps_gte_w_values_and_dups(df_w_values_and_duplicate):
    d_w_duplicate = DataM.create_from_dataframe(
        df_w_values_and_duplicate, "item_id", "user_id", timestamp_ix="timestamp"
    )

    filtered_d_w_duplicate = d_w_duplicate.timestamps_gte(2)

    assert (filtered_d_w_duplicate.timestamps.values == np.array([3, 2, 4])).all()

    assert (
        filtered_d_w_duplicate.values.toarray()
        == np.array([[0, 2, 0, 0], [0, 7, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    ).all()


def test_timestamps_lte_w_values_and_dups(df_w_values_and_duplicate):
    d_w_duplicate = DataM.create_from_dataframe(
        df_w_values_and_duplicate, "item_id", "user_id", timestamp_ix="timestamp"
    )

    filtered_d_w_duplicate = d_w_duplicate.timestamps_lte(2)

    # data = {'timestamp': [3, 2, 1, 1, 4], 'item_id': [1, 1, 2, 3, 1], 'user_id': [0, 1, 1, 2, 1]}

    assert (filtered_d_w_duplicate.timestamps.values == np.array([2, 1, 1])).all()
    assert (
        filtered_d_w_duplicate.values.toarray()
        == np.array([[0, 0, 0, 0], [0, 3, 5, 0], [0, 0, 0, 6]], dtype=np.int32)
    ).all()


def test_indices_in(df):
    d = DataM.create_from_dataframe(df, "item_id", "user_id", timestamp_ix="timestamp")

    U = [0, 1]
    I = [1, 2]

    filtered_df = d.indices_in((U, I))

    assert (filtered_df.timestamps.values == np.array([3, 1])).all()
    assert (
        filtered_df.values.toarray()
        == np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.int32)
    ).all()


def test_user_history(df):
    d = DataM.create_from_dataframe(df, "item_id", "user_id", timestamp_ix="timestamp")

    histories = d.user_history
    expected_histories = {0: [1], 1: [1, 2], 2: [3]}
    for i, hist in histories:
        assert hist == expected_histories[i]
