from recpack.data.matrix import InteractionMatrix
from scipy.sparse import csr_matrix
import pandas as pd
import pytest
import numpy as np

num_users = 50
num_items = 100
num_interactions = 5000

min_t = 0
max_t = 100

USER_IX = InteractionMatrix.USER_IX
ITEM_IX = InteractionMatrix.ITEM_IX
TIMESTAMP_IX = "ts"


@pytest.fixture(scope="function")
def data_m():
    np.random.seed(42)

    input_dict = {
        InteractionMatrix.USER_IX: [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        InteractionMatrix.ITEM_IX: [
            np.random.randint(0, num_items) for _ in range(0, num_interactions)
        ],
        InteractionMatrix.TIMESTAMP_IX: [
            np.random.randint(min_t, max_t) for _ in range(0, num_interactions)
        ],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates([InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX], inplace=True)
    data = InteractionMatrix(
        df, InteractionMatrix.ITEM_IX, InteractionMatrix.USER_IX, timestamp_ix=InteractionMatrix.TIMESTAMP_IX
    )
    return data


@pytest.fixture(scope="function")
def data_m_w_dups():
    np.random.seed(42)

    input_dict = {
        InteractionMatrix.USER_IX: [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        InteractionMatrix.ITEM_IX: [
            np.random.randint(0, num_items) for _ in range(0, num_interactions)
        ],
        InteractionMatrix.TIMESTAMP_IX: [
            np.random.randint(min_t, max_t) for _ in range(0, num_interactions)
        ],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates([InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX, InteractionMatrix.TIMESTAMP_IX], inplace=True)
    data = InteractionMatrix(
        df, InteractionMatrix.ITEM_IX, InteractionMatrix.USER_IX, timestamp_ix=InteractionMatrix.TIMESTAMP_IX
    )
    return data


@pytest.fixture(scope="function")
def data_m_small():
    """Data matrix for test on validation split with users in train not in validation"""
    input_dict = {
        InteractionMatrix.USER_IX: [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        InteractionMatrix.ITEM_IX: [0, 1, 2, 0, 1, 2, 3, 3, 5, 6],
        InteractionMatrix.TIMESTAMP_IX: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates([InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX], inplace=True)
    data = InteractionMatrix(
        df, InteractionMatrix.ITEM_IX, InteractionMatrix.USER_IX, timestamp_ix=InteractionMatrix.TIMESTAMP_IX
    )
    return data


@pytest.fixture(scope="function")
def data_m_sessions():
    """Data matrix with sessions of varying time overlap for testing time-based splits"""
    # (user, time) matrix, non-zero entries are item ids
    user_time = csr_matrix(
        [
            #0  1  2  3  4  5  6  7
            [1, 0, 2, 1, 0, 0, 0, 0],  # time: max 3
            [0, 1, 1, 0, 3, 0, 0, 0],  # time: max 4
            [0, 0, 0, 0, 0, 2, 1, 1],  # time: max 7
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

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)
