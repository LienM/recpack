# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
import os
from typing import List
import tarfile

import numpy as np
import pandas as pd

from recpack.datasets.base import Dataset, _fetch_remote
from recpack.preprocessing.filters import (
    Filter,
    MinItemsPerUser,
    MinRating,
)


class Netflix(Dataset):
    """Handles the Netflix Prize dataset.

    All information on the dataset can be found at https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data.
    The separate files are processed into a consolidated csv file.


    Default processing follows the preprocessing of MultVAE ('Variational Autoencoders for Collaborative Filtering',
    D. Liang et al. @ KDD2018), and makes sure that:

    - Only ratings 4 or higher are considered as positive
    - Each remaining user has interacted with at least 5 items

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the csv ratings file.
        If None, the :attr:`DEFAULT_FILENAME` will be used.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised?
        Defaults to True
    :type use_default_filters: bool, optional

    """

    USER_IX = "user_id"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "item_id"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
    RATING_IX = "rating"
    """Name of the column in the DataFrame that contains the rating."""

    DEFAULT_FILENAME = "netflix.csv"
    """Default filename that will be used if it is not specified by the user."""

    DATASET_URL = "https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz"
    """URL to fetch the dataset from."""

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the Netflix Prize dataset

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinRating(4, self.RATING_IX),
            MinItemsPerUser(5, self.ITEM_IX, self.USER_IX),
        ]

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.file_path`
        """

        zipfile_name = "nf_prize_dataset.tar.gz"
        zipped_folder = "download"
        files_zipfile_name = "training_set.tar"  # inside the main tar there is a files tar.

        zipfile_full_path = os.path.join(self.path, zipfile_name)

        # Download the zip into the data directory
        _fetch_remote(self.DATASET_URL, zipfile_full_path)

        tar = tarfile.open(zipfile_full_path)
        dataset_compressed = tar.extractfile(f"{zipped_folder}/{files_zipfile_name}")
        dataset_tar = tarfile.open(fileobj=dataset_compressed)

        dfs = []
        # For each file in the directory:
        for f in dataset_tar.getmembers():
            if not f.isfile():
                continue
            # open the file from the tarfile.
            g = dataset_tar.extractfile(f.name)
            # first line is <item_id>:
            first_line = g.readline().decode("utf-8")
            # Rest is comma-eparated with columns user, rating, date
            df = pd.read_csv(
                g, names=[self.USER_IX, self.RATING_IX, "date"], parse_dates=["date"], infer_datetime_format=True
            )
            df[self.ITEM_IX] = first_line[:-2]
            # Translate the date to a timestamp
            df[self.TIMESTAMP_IX] = (df["date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

            dfs.append(df[[self.USER_IX, self.ITEM_IX, self.TIMESTAMP_IX, self.RATING_IX]])
        df = pd.concat(dfs, ignore_index=True)

        df.dropna(inplace=True)
        df.to_csv(os.path.join(self.path, self.filename), header=True, index=False)

        # delete the zipfile
        os.remove(zipfile_full_path)

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
            dtype={self.USER_IX: str, self.ITEM_IX: str, self.TIMESTAMP_IX: np.int64, self.RATING_IX: int},
        )

        return df
