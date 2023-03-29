# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
import os
from typing import List
import zipfile

import numpy as np
import pandas as pd

from recpack.datasets.base import Dataset, _fetch_remote
from recpack.preprocessing.filters import (
    Filter,
    MinUsersPerItem,
)


class RecsysChallenge2015(Dataset):
    """Handles data from the Recsys Challenge 2015, yoochoose dataset.

    All information and downloads can be found at https://www.kaggle.com/chadgostopp/recsys-challenge-2015.
    Because downloading the data requires a Kaggle account we can't download it here,
    you should download the data manually and provide the path to the `yoochoose-clicks.dat` file.

    Default processing makes sure that:

    - Each remaining  item has been interacted with by at least 5 users.

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional

    """

    USER_IX = "session"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "item_id"
    """Name of the column in the DataFrame with item identifiers"""
    TIMESTAMP_IX = "seconds_since_epoch"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""

    DEFAULT_FILENAME = "yoochoose-clicks.dat"
    """Default filename that will be used if it is not specified by the user."""

    def _download_dataset(self):
        """Downloading this dataset is not supported."""
        raise NotImplementedError(
            "Recsys Challenge dataset should be downloaded manually, you can get it at: https://www.kaggle.com/chadgostopp/recsys-challenge-2015."
        )

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the RecsysChallenge 2015 dataset

        Filters items that do not have enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinUsersPerItem(5, self.ITEM_IX, self.USER_IX, count_duplicates=True),
        ]

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
            names=[self.USER_IX, self.TIMESTAMP_IX, self.ITEM_IX],
            dtype={
                self.USER_IX: np.int64,
                self.TIMESTAMP_IX: str,
                self.ITEM_IX: np.int64,
            },
            parse_dates=[self.TIMESTAMP_IX],
            usecols=[0, 1, 2],
        )

        # Adapt timestamp, this makes it so the timestamp is always seconds since epoch
        df[self.TIMESTAMP_IX] = df[self.TIMESTAMP_IX].astype(int) / 1e9  # pandas datetime -> seconds from epoch

        return df
