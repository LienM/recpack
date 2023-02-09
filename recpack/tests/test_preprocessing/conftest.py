# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import random

import pandas as pd
import pytest

from recpack.matrix import InteractionMatrix


@pytest.fixture(scope="function")
def dataframe():
    users = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 1, 1, 4, 4, 4]
    items = [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 5, 5, 5, 1, 1, 1]

    input_dict = {
        InteractionMatrix.USER_IX: users,
        InteractionMatrix.ITEM_IX: items,
        InteractionMatrix.TIMESTAMP_IX: [random.randint(0, 400) for _ in range(len(users))],
    }

    df = pd.DataFrame.from_dict(input_dict)

    return df


@pytest.fixture(scope="function")
def dataframe_with_fixed_timestamps():
    users = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 1, 1, 4, 4, 4]
    items = [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 5, 5, 5, 1, 1, 1]

    input_dict = {
        InteractionMatrix.USER_IX: users,
        InteractionMatrix.ITEM_IX: items,
        InteractionMatrix.TIMESTAMP_IX: [i for i in range(len(users))],
    }

    df = pd.DataFrame.from_dict(input_dict)

    return df


@pytest.fixture(scope="function")
def dataframe_with_fixed_timestamps_inverted():
    """A DataFrame with timestamps in descending order, allows testing reordering parts."""
    users = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 1, 1, 4, 4, 4]
    items = [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 5, 5, 5, 1, 1, 1]

    input_dict = {
        InteractionMatrix.USER_IX: users,
        InteractionMatrix.ITEM_IX: items,
        InteractionMatrix.TIMESTAMP_IX: [i for i in range(len(users), 0, -1)],
    }

    df = pd.DataFrame.from_dict(input_dict)

    return df


@pytest.fixture(scope="function")
def rating_dataframe():
    users = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 1, 1, 4, 4, 4]
    items = [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 5, 5, 5, 1, 1, 1]
    ratings = [1, 1, 2, 2, 4, 1, 1, 2, 2, 4, 1, 1, 2, 2, 4, 1, 1, 2, 2, 4]

    input_dict = {
        InteractionMatrix.USER_IX: users,
        InteractionMatrix.ITEM_IX: items,
        InteractionMatrix.TIMESTAMP_IX: [random.randint(0, 400) for _ in range(len(users))],
        "rating": ratings,
    }

    df = pd.DataFrame.from_dict(input_dict)

    return df
