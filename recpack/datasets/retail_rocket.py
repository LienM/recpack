# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
import os
from typing import List, Union, Tuple
import zipfile

import numpy as np
import pandas as pd

from recpack.datasets.base import Dataset, _fetch_remote
from recpack.preprocessing.filters import (
    Filter,
    MinItemsPerUser,
    MinUsersPerItem,
)


class RetailRocket(Dataset):
    """Handles data from the Retail Rocket dataset on Kaggle.

    All information and downloads can be found at
    https://www.kaggle.com/retailrocket/ecommerce-dataset.
    Because downloading the data requires a Kaggle account we can't download it here,
    you should download the data manually and provide the path to the downloaded folder.

    Default processing makes sure that:

    - Each remaining item has been interacted with by at least 50 users.
    - Each user has interacted with at least 3 items

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional
    :param event_types: The dataset contains view, addtocart, transaction events.
        You can select a subset of them.
        Defaults to ("view", )
    :type event_types: Union[List[str], Tuple[str]], optional
    """

    USER_IX = "visitorid"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "itemid"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains timestamp."""
    EVENT_TYPE_IX = "event"
    """Name of the column in the DataFrame that contains the event_types"""

    DEFAULT_FILENAME = "events.csv"
    """Default filename that will be used if it is not specified by the user."""

    ALLOWED_EVENT_TYPES = ["view", "addtocart", "transaction"]

    def __init__(
        self,
        path: str = "data",
        filename: str = None,
        use_default_filters=True,
        event_types: Union[List[str], Tuple[str]] = ("view",),
    ):
        super().__init__(path, filename, use_default_filters)

        for event_type in event_types:
            if event_type not in self.ALLOWED_EVENT_TYPES:
                raise ValueError(
                    f"{event_type} is not in the allowed event types. " f"Please use one of {self.ALLOWED_EVENT_TYPES}"
                )

        self.event_types = event_types

    def _download_dataset(self):
        """Downloading this dataset is not supported."""
        raise NotImplementedError(
            "RetailRocket dataset should be downloaded manually, "
            "you can get it at: https://www.kaggle.com/retailrocket/ecommerce-dataset."
        )

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the RetailRocket dataset

        Filters items that do not have enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinUsersPerItem(50, self.ITEM_IX, self.USER_IX, count_duplicates=True),
            MinItemsPerUser(3, self.ITEM_IX, self.USER_IX, count_duplicates=True),
        ]

    @property
    def _columns(self) -> List[str]:
        columns = [self.USER_IX, self.ITEM_IX, self.TIMESTAMP_IX, self.EVENT_TYPE_IX]
        return columns

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()

        df = pd.read_csv(
            self.file_path,
        )

        # Adapt timestamp, this makes it so the timestamp is always seconds since epoch
        # Original data is in milliseconds since epoch,
        # and so should be divided by 1000
        df[self.TIMESTAMP_IX] = df[self.TIMESTAMP_IX].view(int) / 1e3  # pandas datetime -> seconds from epoch

        # Select only the specified event_types
        if self.event_types:
            df = df[df[self.EVENT_TYPE_IX].isin(self.event_types)].copy()

        df = df[self._columns].copy()

        return df
