# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import json
import os
import tarfile
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from recpack.datasets.base import Dataset, _fetch_remote
from recpack.preprocessing.filters import (
    Filter,
    MinItemsPerUser,
    MinUsersPerItem,
)


class AdressaOneWeek(Dataset):
    """Handles the 1 week dataset of adressa.

    All information on the dataset can be found at https://reclab.idi.ntnu.no/dataset/.
    Uses the `ratings.csv` file to generate an InteractionMatrix.

    Default processing makes sure that:

    - Each remaining user has interacted with at least 3 items
    - Each remaining item has been interacted with by at least 5 users

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default
        will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised?
        Defaults to True
    :type use_default_filters: bool, optional

    """

    USER_IX = "userId"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "id"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "time"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""

    DEFAULT_FILENAME = "adressa_one_week.csv"
    """Default filename that will be used if it is not specified by the user."""

    DATASET_URL = "https://reclab.idi.ntnu.no/dataset/one_week.tar.gz"
    """URL to fetch the dataset from."""

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the Adressa dataset

        Filters users and items with not enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinItemsPerUser(3, self.ITEM_IX, self.USER_IX),
            MinUsersPerItem(5, self.ITEM_IX, self.USER_IX),
        ]

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.file_path`
        """

        zipfile_name = "one_week.tar.gz"
        zipfile_full_path = os.path.join(self.path, zipfile_name)

        # Download the zip into the data directory
        _fetch_remote(self.DATASET_URL, zipfile_full_path)

        # Subfiles that should be present in the tarfile
        fs = [
            "20170101",
            "20170102",
            "20170103",
            "20170104",
            "20170105",
            "20170106",
            "20170107",
        ]

        tar = tarfile.open(zipfile_full_path)
        dfs = []
        # For each file in the directory:
        for ix, f in enumerate(fs):
            # open the file from the tarfile.
            g = tar.extractfile(f"one_week/{f}")
            dfs.append(pd.DataFrame.from_records([
                {
                    self.USER_IX: x[self.USER_IX],
                    self.ITEM_IX: x[self.ITEM_IX],
                    self.TIMESTAMP_IX: x[self.TIMESTAMP_IX]
                }
                for x in [json.loads(line)for line in tqdm(g.readlines(), desc=f"loading {f} file {ix+1}/{len(fs)}")]
                if self.USER_IX in x and self.ITEM_IX in x and self.TIMESTAMP_IX in x
            ]))

        df = pd.concat(dfs)

        df.dropna(inplace=True)
        df.to_csv(os.path.join(self.path, self.filename), header=True, index=False)

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
            dtype={
                self.USER_IX: str,
                self.TIMESTAMP_IX: np.int64,
                self.ITEM_IX: str,
            },
        )

        return df
