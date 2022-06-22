"""Module responsible for the ThirtyMusicSessions dataset."""

import pandas as pd
from typing import List
from recpack.datasets.base import Dataset

from recpack.preprocessing.filters import (
    Filter,
    MinItemsPerUser,
    MinUsersPerItem,
)


class ThirtyMusicSessions(Dataset):
    # TODO Write documentation

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

    def load_dataframe(self) -> pd.DataFrame:
        """Load the data from file, and return as a Pandas DataFrame.

        Downloads the data file if it is not yet present.
        The output will contain a dataframe with a user_id and item_id column.
        Each interaction is stored in a separate row.

        :return: The interactions as a dataframe, with a row for each interaction.
        :rtype: pandas.DataFrame
        """
        # self.fetch_dataset()

        df = pd.read_csv(self.file_path)
        df.drop(columns=["numtracks", "playtime", "uid"], inplace=True)
        df = df.astype({self.TIMESTAMP_IX: "int32"})
        return df
