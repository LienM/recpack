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
)


class ThirtyMusicSessions(Dataset):
    """A collection of listening and playlists data retrieved from Internet radio stations through Last.fm API.

    Dataset presented in Turrin, Roberto, et al. "30Music Listening and Playlists Dataset." RecSys Posters. 2015.
    For info and download link see https://recsys.deib.polimi.it/datasets/.

    .. warning::

        RecPack currently does not support downloading and parsing the raw files of the dataset.
        We expect a CSV file with the user-item-timestamp information instead.

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional
    """

    USER_IX = "sid"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "tid"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "position"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the 30MusicSessions dataset

        Filters users and items that do not have enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinItemsPerUser(5, self.ITEM_IX, self.USER_IX),
            MinUsersPerItem(5, self.ITEM_IX, self.USER_IX),
        ]

    def _download_dataset(self):
        # TODO Implement Download
        # TODO parse idomaar files?
        pass

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """
        # self.fetch_dataset()

        df = pd.read_csv(self.file_path)
        df.drop(columns=["numtracks", "playtime", "uid"], inplace=True)
        df = df.astype({self.TIMESTAMP_IX: "int32"})
        return df
