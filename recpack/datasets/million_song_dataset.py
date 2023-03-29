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
    MinItemsPerUser,
    MinUsersPerItem,
    MinRating,
)


class MillionSongDataset(Dataset):
    """Handles Taste Profile subset of the Million Song Dataset.

    All information on the dataset can be found at http://millionsongdataset.com/tasteprofile/.
    Uses the `train_triplets.txt` file to generate an interaction matrix.

    Default processing is done as in "Variational autoencoders for collaborative filtering." Liang, Dawen, et al.:

    - Each remaining user has interacted with at least 20 items
    - Each remaining item has been interacted with by at least 200 users

    You can also manually set the preprocessing filters, e.g.,::

        from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
        from recpack.datasets import MillionSongDataset
        d = MillionSongDataset(path='path', use_default_filters=False)
        d.add_filter(MinItemsPerUser(20, d.ITEM_IX, d.USER_IX))
        d.add_filter(MinUsersPerItem(200, d.ITEM_IX, d.USER_IX))

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional

    """

    USER_IX = "userId"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "songId"
    """Name of the column in the DataFrame that contains item identifiers."""
    COUNT_IX = "playCount"
    """Name of the column in the DataFrame that contains how often an item was played."""

    DEFAULT_FILENAME = "msd_train_triplets.tsv"
    """Default filename that will be used if it is not specified by the user."""

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the Movielens 10M dataset

        By default each rating is considered as an interaction.
        Filters users and items with not enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [MinItemsPerUser(20, self.ITEM_IX, self.USER_IX), MinUsersPerItem(200, self.ITEM_IX, self.USER_IX)]

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.file_path`
        """
        REMOTE_FILENAME = "train_triplets.txt.zip"
        DATASETURL = f"http://millionsongdataset.com/sites/default/files/challenge/{REMOTE_FILENAME}"

        # Download the zip into the data directory
        _fetch_remote(DATASETURL, os.path.join(self.path, REMOTE_FILENAME))

        # Extract the ratings file which we will use
        with zipfile.ZipFile(os.path.join(self.path, REMOTE_FILENAME), "r") as zip_ref:
            zip_ref.extract("train_triplets.txt", self.path)

        # Rename the ratings file to the specified filename
        os.rename(os.path.join(self.path, "train_triplets.txt"), self.file_path)

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
            dtype={self.USER_IX: str, self.ITEM_IX: str, self.COUNT_IX: np.int64},
            sep="\t",
            names=[self.USER_IX, self.ITEM_IX, self.COUNT_IX],
        )

        flattened_df = df.loc[df.index.repeat(df[self.COUNT_IX])]

        return flattened_df.drop(columns=self.COUNT_IX).reset_index(drop=True)


TasteProfile = MillionSongDataset
