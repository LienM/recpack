# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
import os
from typing import List, Optional, Tuple, Union
import zipfile

import pandas as pd

from recpack.datasets.base import Dataset
from recpack.preprocessing.filters import (
    Filter,
    MinItemsPerUser,
    MinUsersPerItem,
)


class Globo(Dataset):
    """Handles data from the "News Portal User Interactions by Globo.com" dataset on Kaggle.

    All information and downloads can be found at
    https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom.
    Because downloading the data requires a Kaggle account we can't download it here.
    You should download the data manually and provide the path to the downloaded zipfile.

    Default processing makes sure that:

    - Each remaining item has been interacted with by at least 10 users
    - Each user has interacted with at least 3 items

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional
    """

    USER_IX = "user_id"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "click_article_id"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "click_timestamp"
    """Name of the column in the DataFrame that contains timestamp."""

    DEFAULT_FILENAME = "archive.zip"
    """Default filename that will be used if it is not specified by the user."""

    def __init__(
        self,
        path: str = "data",
        filename: str = None,
        use_default_filters=True,
    ):
        super().__init__(path, filename, use_default_filters)

    def _download_dataset(self):
        """Downloading this dataset is not supported."""
        raise NotImplementedError(
            "The Globo dataset should be downloaded manually, "
            "you can get it at: https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom."
        )

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the CosmeticsShop dataset

        Filters items that do not have enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinUsersPerItem(10, self.ITEM_IX, self.USER_IX, count_duplicates=True),
            MinItemsPerUser(3, self.ITEM_IX, self.USER_IX, count_duplicates=True),
        ]

    @property
    def _columns(self) -> List[str]:
        columns = [self.USER_IX, self.ITEM_IX, self.TIMESTAMP_IX]

        return columns

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()
        dfs = []
        with zipfile.ZipFile(self.file_path, "r") as zip_ref:
            click_files = [name for name in zip_ref.namelist() if '/clicks_hour' in name]
            for item in click_files:
                with zip_ref.open(item, 'r') as f:
                    df = pd.read_csv(f)
                    if df.shape[0] == 0:
                        # Some files are empty
                        continue
                    else:
                        # Timestamp from miliseconds sinds epoch to seconds since epoch
                        df[self.TIMESTAMP_IX] = df[self.TIMESTAMP_IX].view(int) / 1e3

                        dfs.append(df)

        return pd.concat(dfs)[self._columns].copy()
