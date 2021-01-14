import random

import pandas as pd
import pytest

from recpack.data.matrix import InteractionMatrix


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
