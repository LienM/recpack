# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
from typing import List, Optional

import numpy as np
import pandas as pd

from recpack.datasets.base import Dataset
from recpack.preprocessing.filters import (
    Filter,
    MinItemsPerUser,
    MinUsersPerItem,
)


class DummyDataset(Dataset):
    """Small randomly generated dummy dataset that allows testing of pipelines
    and other components without needing to load a full scale dataset.

    :param path: The path to the data directory. UNUSED because dataset is generated and not read from file.
        Defaults to `data`
    :type path: str, optional
    :param filename: UNUSED because dataset is generated and not read from file.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional
    :param seed: Seed for the random data generation. Defaults to None.
    :type seed: int, optional
    :param num_users: The amount of users to use when generating data, defaults to 100
    :type num_users: int, optional
    :param num_items: The number of items to use when generating data, defaults to 20
    :type num_items: int, optional
    :param num_interactions: The number of interactions to generate, defaults to 500
    :type num_interactions: int, optional
    :param min_t: The minimum timestamp when generating data, defaults to 0
    :type min_t: int, optional
    :param max_t: The maximum timestamp when generating data, defaults to 500
    :type max_t: int, optional
    """

    USER_IX = "user_id"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "item_id"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""

    DEFAULT_FILENAME = "dummy_input.csv"

    def __init__(
        self,
        path: str = "data",
        filename: Optional[str] = None,
        use_default_filters=True,
        seed=None,
        num_users=100,
        num_items=20,
        num_interactions=500,
        min_t=0,
        max_t=500,
    ):
        super().__init__(path, filename, use_default_filters)

        self.seed = seed
        if self.seed is None:
            self.seed = seed = np.random.get_state()[1][0]

        self.num_users = num_users
        self.num_items = num_items
        self.num_interactions = num_interactions
        self.min_t = min_t
        self.max_t = max_t

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the Dummy dataset

        Filters users and items that do not have enough interactions.
        At least 2 users per item and 2 interactions per user.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinUsersPerItem(2, self.ITEM_IX, self.USER_IX),
            MinItemsPerUser(2, self.ITEM_IX, self.USER_IX),
        ]

    def _download_dataset(self):
        # There is no downloading necessary.
        pass

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """
        np.random.seed(self.seed)

        input_dict = {
            self.USER_IX: [np.random.randint(0, self.num_users) for _ in range(0, self.num_interactions)],
            self.ITEM_IX: [np.random.randint(0, self.num_items) for _ in range(0, self.num_interactions)],
            self.TIMESTAMP_IX: [np.random.randint(self.min_t, self.max_t) for _ in range(0, self.num_interactions)],
        }

        df = pd.DataFrame.from_dict(input_dict)
        return df
