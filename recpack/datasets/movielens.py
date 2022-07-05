"""Module responsible for the MovieLens25M dataset."""

import numpy as np
import os
import pandas as pd
from typing import List
import zipfile
from recpack.datasets.base import Dataset, _fetch_remote

from recpack.preprocessing.filters import (
    Filter,
    MinItemsPerUser,
    MinUsersPerItem,
    MinRating,
)


class MovieLens25M(Dataset):
    """Handles Movielens 25M dataset.

    All information on the dataset can be found at https://grouplens.org/datasets/movielens/25m/.
    Uses the `ratings.csv` file to generate an interaction matrix.

    Default processing makes sure that:

    - Each rating above or equal to 4 is used as interaction as is done in Liang, Dawen, et al.
      "Variational autoencoders for collaborative filtering." Proceedings of the 2018 world wide web conference. 2018.
    - Each remaining user has interacted with at least 3 items
    - Each remaining  item has been interacted with by at least 5 users

    To use another value as minimal rating to mark interaction as positive,
    you have to manually set the preprocessing filters.::

        from recpack.preprocessing.filters import MinRating, MinItemsPerUser, MinUsersPerItem
        from recpack.datasets import MovieLens25M
        d = MovieLens25M('path/to/file', preprocess_default=False)
        d.add_filter(MinRating(3, d.RATING_IX, 3))
        d.add_filter(MinItemsPerUser(3, d.ITEM_IX, d.USER_IX))
        d.add_filter(MinUsersPerItem(5, d.ITEM_IX, d.USER_IX))

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: str, optional
    :param preprocess_default: Should a default set of filters be initialised? Defaults to True
    :type preprocess_default: bool, optional

    """

    USER_IX = "userId"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "movieId"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
    RATING_IX = "rating"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""

    DEFAULT_FILENAME = "ratings.csv"
    """Default filename that will be used if it is not specified by the user."""

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the Movielens 25M dataset

        By default each rating is considered as an interaction.
        Filters users and items with not enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinRating(4, self.RATING_IX),
            MinItemsPerUser(3, self.ITEM_IX, self.USER_IX),
            MinUsersPerItem(5, self.ITEM_IX, self.USER_IX),
        ]

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.file_path`
        """
        DATASETURL = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"

        # Download the zip into the data directory
        _fetch_remote(DATASETURL, os.path.join(self.path, "ml-25m.zip"))

        # Extract the ratings file which we will use
        with zipfile.ZipFile(os.path.join(self.path, "ml-25m.zip"), "r") as zip_ref:
            zip_ref.extract("ml-25m/ratings.csv", self.path)

        # Rename the ratings file to the specified filename
        os.rename(os.path.join(self.path, "ml-25m/ratings.csv"), self.file_path)

    def load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()
        df = pd.read_csv(
            self.file_path,
            dtype={
                self.USER_IX: np.int64,
                self.TIMESTAMP_IX: np.int64,
                self.ITEM_IX: np.int64,
                self.RATING_IX: np.float64,
            },
        )

        return df
