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


class MovieLensDataset(Dataset):
    """
    Base class for MovieLens Datasets
    """

    USER_IX = "userId"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "movieId"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
    RATING_IX = "rating"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""

    DATASETURL = "http://files.grouplens.org/datasets/movielens"

    REMOTE_ZIPNAME = ""
    """Name of the zip-file on the MovieLens server."""

    REMOTE_FILENAME = "ratings.csv"
    """Name of the file containing user ratings on the MovieLens server."""

    @property
    def DEFAULT_FILENAME(self) -> str:
        """Default filename that will be used if it is not specified by the user."""
        return f"{self.REMOTE_ZIPNAME}_{self.REMOTE_FILENAME}"

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the MovieLens datasets.

        Default processing  is done as in "Variational autoencoders for collaborative filtering." Liang, Dawen, et al.:

        - Ratings above or equal to 4 are interpreted as implicit feedback
        - Each remaining item has been interacted with by at least 5 users

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinRating(4, self.RATING_IX),
            MinUsersPerItem(5, self.ITEM_IX, self.USER_IX),
        ]

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.file_path`
        """
        # Download the zip into the data directory
        _fetch_remote(
            f"{self.DATASETURL}/{self.REMOTE_ZIPNAME}.zip", os.path.join(self.path, f"{self.REMOTE_ZIPNAME}.zip")
        )

        # Extract the ratings file which we will use
        with zipfile.ZipFile(os.path.join(self.path, f"{self.REMOTE_ZIPNAME}.zip"), "r") as zip_ref:
            zip_ref.extract(f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}", self.path)

        # Rename the ratings file to the specified filename
        os.rename(os.path.join(self.path, f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}"), self.file_path)


class MovieLens25M(MovieLensDataset):
    """Handles Movielens 25M dataset.

    All information on the dataset can be found at https://grouplens.org/datasets/movielens/25m/.
    Uses the `ratings.csv` file to generate an interaction matrix.

    Default processing  is done as in "Variational autoencoders for collaborative filtering." Liang, Dawen, et al.:

    - Ratings above or equal to 4 are interpreted as implicit feedback
    - Each remaining item has been interacted with by at least 5 users

    You can also manually set the preprocessing filters, e.g.,::

        from recpack.preprocessing.filters import MinRating, MinItemsPerUser, MinUsersPerItem
        from recpack.datasets import MovieLens25M
        d = MovieLens25M(path='path/to/', use_default_filters=False)
        d.add_filter(MinRating(3, d.RATING_IX, 3))
        d.add_filter(MinItemsPerUser(3, d.ITEM_IX, d.USER_IX))
        d.add_filter(MinUsersPerItem(5, d.ITEM_IX, d.USER_IX))

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional

    """

    REMOTE_FILENAME = "ratings.csv"
    REMOTE_ZIPNAME = "ml-25m"

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
            dtype={
                self.USER_IX: np.int64,
                self.TIMESTAMP_IX: np.int64,
                self.ITEM_IX: np.int64,
                self.RATING_IX: np.float64,
            },
        )

        return df


class MovieLens100K(MovieLensDataset):
    """Handles Movielens 100K dataset.

    All information on the dataset can be found at https://grouplens.org/datasets/movielens/100k/.
    Uses the `u.data` file to generate an interaction matrix.

    Default processing  is done as in "Variational autoencoders for collaborative filtering." Liang, Dawen, et al.:

    - Ratings above or equal to 4 are interpreted as implicit feedback
    - Each remaining item has been interacted with by at least 5 users

    To use another value as minimal rating to mark interaction as positive,
    you have to manually set the preprocessing filters.::

        from recpack.preprocessing.filters import MinRating, MinItemsPerUser, MinUsersPerItem
        from recpack.datasets import MovieLens100K
        d = MovieLens100K(path='path/to/', use_default_filters=False)
        d.add_filter(MinRating(3, d.RATING_IX, 3))
        d.add_filter(MinItemsPerUser(3, d.ITEM_IX, d.USER_IX))
        d.add_filter(MinUsersPerItem(5, d.ITEM_IX, d.USER_IX))

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional

    """

    REMOTE_FILENAME = "u.data"
    REMOTE_ZIPNAME = "ml-100k"

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()
        df = pd.read_table(
            self.file_path,
            dtype={
                self.USER_IX: np.int64,
                self.ITEM_IX: np.int64,
                self.RATING_IX: np.float64,
                self.TIMESTAMP_IX: np.int64,
            },
            sep="\t",
            names=[self.USER_IX, self.ITEM_IX, self.RATING_IX, self.TIMESTAMP_IX],
        )

        return df


class MovieLens1M(MovieLensDataset):
    """Handles Movielens 1M dataset.

    All information on the dataset can be found at https://grouplens.org/datasets/movielens/1m/.
    Uses the `ratings.dat` file to generate an interaction matrix.

    Default processing  is done as in "Variational autoencoders for collaborative filtering." Liang, Dawen, et al.:

    - Ratings above or equal to 4 are interpreted as implicit feedback
    - Each remaining item has been interacted with by at least 5 users

    To use another value as minimal rating to mark interaction as positive,
    you have to manually set the preprocessing filters.::

        from recpack.preprocessing.filters import MinRating, MinItemsPerUser, MinUsersPerItem
        from recpack.datasets import MovieLens1M
        d = MovieLens1M(path='path/to/', use_default_filters=False)
        d.add_filter(MinRating(3, d.RATING_IX, 3))
        d.add_filter(MinItemsPerUser(3, d.ITEM_IX, d.USER_IX))
        d.add_filter(MinUsersPerItem(5, d.ITEM_IX, d.USER_IX))

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional

    """

    REMOTE_FILENAME = "ratings.dat"
    REMOTE_ZIPNAME = "ml-1m"

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()
        df = pd.read_table(
            self.file_path,
            dtype={
                self.USER_IX: np.int64,
                self.ITEM_IX: np.int64,
                self.RATING_IX: np.float64,
                self.TIMESTAMP_IX: np.int64,
            },
            sep="\:\:",
            names=[self.USER_IX, self.ITEM_IX, self.RATING_IX, self.TIMESTAMP_IX],
        )

        return df


class MovieLens10M(MovieLensDataset):
    """Handles Movielens 10M dataset.

    All information on the dataset can be found at https://grouplens.org/datasets/movielens/10m/.
    Uses the `ratings.dat` file to generate an interaction matrix.

    Default processing  is done as in "Variational autoencoders for collaborative filtering." Liang, Dawen, et al.:

    - Ratings above or equal to 4 are interpreted as implicit feedback
    - Each remaining item has been interacted with by at least 5 users

    To use another value as minimal rating to mark interaction as positive,
    you have to manually set the preprocessing filters.::

        from recpack.preprocessing.filters import MinRating, MinItemsPerUser, MinUsersPerItem
        from recpack.datasets import MovieLens10M
        d = MovieLens10M(path='path/to/', use_default_filters=False)
        d.add_filter(MinRating(3, d.RATING_IX, 3))
        d.add_filter(MinItemsPerUser(3, d.ITEM_IX, d.USER_IX))
        d.add_filter(MinUsersPerItem(5, d.ITEM_IX, d.USER_IX))

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional

    """

    REMOTE_FILENAME = "ratings.dat"
    REMOTE_ZIPNAME = "ml-1m"

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.file_path`
        """
        # Download the zip into the data directory
        _fetch_remote(
            f"{self.DATASETURL}/{self.REMOTE_ZIPNAME}zip", os.path.join(self.path, f"{self.REMOTE_ZIPNAME}.zip")
        )

        # Extract the ratings file which we will use
        with zipfile.ZipFile(os.path.join(self.path, f"{self.REMOTE_ZIPNAME}.zip"), "r") as zip_ref:
            zip_ref.extract(f"ml-10M100K/{self.REMOTE_FILENAME}", self.path)

        # Rename the ratings file to the specified filename
        os.rename(os.path.join(self.path, f"ml-10M100K/{self.REMOTE_FILENAME}"), self.file_path)

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.file_path`
        """
        DATASETURL = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"

        # Download the zip into the data directory
        _fetch_remote(DATASETURL, os.path.join(self.path, "ml-10m.zip"))

        # Extract the ratings file which we will use
        with zipfile.ZipFile(os.path.join(self.path, "ml-10m.zip"), "r") as zip_ref:
            zip_ref.extract("ml-10M100K/ratings.dat", self.path)

        # Rename the ratings file to the specified filename
        os.rename(os.path.join(self.path, "ml-10M100K/ratings.dat"), self.file_path)

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()
        df = pd.read_table(
            self.file_path,
            dtype={
                self.USER_IX: np.int64,
                self.ITEM_IX: np.int64,
                self.RATING_IX: np.float64,
                self.TIMESTAMP_IX: np.int64,
            },
            sep="\:\:",
            names=[self.USER_IX, self.ITEM_IX, self.RATING_IX, self.TIMESTAMP_IX],
        )

        return df
