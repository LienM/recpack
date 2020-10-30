from recpack.data.data_matrix import DataM
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
