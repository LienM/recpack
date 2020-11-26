from recpack.data.data_matrix import DataM, USER_IX, ITEM_IX, TIMESTAMP_IX, VALUE_IX
from scipy.sparse import csr_matrix
import pandas as pd
import pytest
import numpy as np

num_users = 50
num_items = 100
num_interactions = 5000

min_t = 0
max_t = 100


@pytest.fixture(scope="function")
def data_m():
    np.random.seed(42)

    input_dict = {
        "userId": [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        "movieId": [
            np.random.randint(0, num_items) for _ in range(0, num_interactions)
        ],
        "timestamp": [
            np.random.randint(min_t, max_t) for _ in range(0, num_interactions)
        ],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates(["userId", "movieId"], inplace=True)
    data = DataM.create_from_dataframe(
        df, "movieId", "userId", timestamp_ix="timestamp"
    )
    return data


@pytest.fixture(scope="function")
def data_m_w_dups():
    np.random.seed(42)

    input_dict = {
        "userId": [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        "movieId": [
            np.random.randint(0, num_items) for _ in range(0, num_interactions)
        ],
        "timestamp": [
            np.random.randint(min_t, max_t) for _ in range(0, num_interactions)
        ],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates(["userId", "movieId", "timestamp"], inplace=True)
    data = DataM.create_from_dataframe(
        df, "movieId", "userId", timestamp_ix="timestamp"
    )
    return data


@pytest.fixture(scope="function")
def data_m_w_values():
    np.random.seed(42)

    input_dict = {
        "userId": [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        "movieId": [
            np.random.randint(0, num_items) for _ in range(0, num_interactions)
        ],
        "timestamp": [
            np.random.randint(min_t, max_t) for _ in range(0, num_interactions)
        ],
        "value": [np.random.randint(1, 5) for _ in range(0, num_interactions)],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates(["userId", "movieId"], inplace=True)
    data = DataM.create_from_dataframe(
        df, "movieId", "userId", value_ix="value", timestamp_ix="timestamp"
    )
    return data


@pytest.fixture(scope="function")
def data_m_small():
    """Data matrix for test on validation split with users in train not in validation"""
    input_dict = {
        "userId": [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        "movieId": [0, 1, 2, 0, 1, 2, 3, 3, 5, 6],
        "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates(["userId", "movieId"], inplace=True)
    data = DataM.create_from_dataframe(
        df, "movieId", "userId", timestamp_ix="timestamp"
    )
    return data


@pytest.fixture(scope="function")
def data_m_sessions():
    """Data matrix with sessions of varying time overlap for testing time-based splits"""
    # (user, time) matrix, non-zero entries are item ids
    user_time = csr_matrix(
        [
            #0  1  2  3  4  5  6  7
            [1, 0, 2, 1, 0, 0, 0, 0],  # time: mean 5/3, median 2, min 0, max 3
            [0, 1, 1, 0, 3, 0, 0, 0],  # time: mean 7/3, median 2, min 1, max 4
            [0, 0, 0, 0, 0, 2, 1, 1],  # time: mean 6.0, median 6, min 5, max 6
        ]
    )
    user_ids, timestamps = user_time.nonzero()
    item_ids = user_time.data
    df = pd.DataFrame(
        {
            USER_IX: user_ids,
            ITEM_IX: item_ids,
            TIMESTAMP_IX: timestamps,
        }
    )
    return DataM(df)
