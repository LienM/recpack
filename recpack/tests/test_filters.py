import random

import pytest
import pandas as pd

import recpack.preprocessing.filters as filters


# @pytest.fixture(scope="function")
# def dataframe_20_events_5_users_5_items():
#     random.seed(38405)

#     num_users = 5
#     num_items = 5
#     num_events = 20

#     input_dict = {
#         "userId": [random.randint(0, num_users) for _ in range(num_events)],
#         "iid": [random.randint(0, num_items) for _ in range(num_events)],
#         "timestamp": [random.randint(0, 400) for _ in range(num_events)],
#     }

#     df = pd.DataFrame.from_dict(input_dict)

#     return df


def test_min_users_per_item_filter():

    users = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3]
    items = [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1]

    input_dict = {
        "uid": users,
        "iid": items,
        "timestamp": [random.randint(0, 400) for _ in range(len(users))],
    }

    df = pd.DataFrame.from_dict(input_dict)

    myfilter = filters.MinUsersPerItem(3, "uid", "iid", "timestamp")
    filtered_df = myfilter.apply(df)

    filtered_iids = filtered_df["iid"].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 2 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids


def test_min_items_per_user_filter():

    users = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3]
    items = [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1]

    input_dict = {
        "uid": users,
        "iid": items,
        "timestamp": [random.randint(0, 400) for _ in range(len(users))],
    }

    df = pd.DataFrame.from_dict(input_dict)

    myfilter = filters.MinItemsPerUser(3, "uid", "iid", "timestamp")
    filtered_df = myfilter.apply(df)

    filtered_uids = filtered_df["uid"].unique()

    assert 0 in filtered_uids
    assert 1 in filtered_uids
    assert 2 in filtered_uids
    assert 3 not in filtered_uids


def test_nmost_popular():

    users = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3]
    items = [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1]

    input_dict = {
        "uid": users,
        "iid": items,
        "timestamp": [random.randint(0, 400) for _ in range(len(users))],
    }

    df = pd.DataFrame.from_dict(input_dict)

    myfilter = filters.NMostPopular(3, "uid", "iid", "timestamp")
    filtered_df = myfilter.apply(df)

    filtered_iids = filtered_df["iid"].unique()

    assert 0 in filtered_iids
    assert 1 in filtered_iids
    assert 2 in filtered_iids
    assert 3 not in filtered_iids
    assert 4 not in filtered_iids





