"""Module responsible for the CiteULike dataset."""

import numpy as np
import pandas as pd
from typing import List
from recpack.datasets.base import Dataset, _fetch_remote

from recpack.preprocessing.filters import (
    Filter,
    MinItemsPerUser,
    MinUsersPerItem,
)


class CiteULike(Dataset):
    """Dataset class for the CiteULike dataset.

    Full information  on the dataset can be found at https://github.com/js05212/citeulike-a.
    Uses the `users.dat` file from the dataset to construct an implicit feedback interaction matrix.

    Default processing makes sure that:

    - Each remaining user has interacted with at least 3 items
    - Each remaining  item has been interacted with by at least 5 users

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: str, optional
    :param preprocess_default: Should a default set of filters be initialised? Defaults to True
    :type preprocess_default: bool, optional
    """

    TIMESTAMP_IX = None
    """The dataset has no notion of time, so there is no timestamp column present in the DataFrame."""

    DEFAULT_FILENAME = "users.dat"
    """Default filename that will be used if it is not specified by the user."""

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the CiteULike dataset

        Filters users and items that do not have enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinItemsPerUser(3, self.ITEM_IX, self.USER_IX),
            MinUsersPerItem(5, self.ITEM_IX, self.USER_IX),
        ]

    def _download_dataset(self):
        """Download the users.dat file from the github repository.
        The file is saved at the specified `self.file_path`
        """
        DATASETURL = "https://raw.githubusercontent.com/js05212/citeulike-a/master/users.dat"
        _fetch_remote(DATASETURL, self.file_path)

    def load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """

        # This will download the dataset if necessary
        self.fetch_dataset()

        u_i_pairs = []
        with open(self.file_path, "r") as f:

            for user, line in enumerate(f.readlines()):
                item_cnt = line.strip("\n").split(" ")[0]  # First element is a count
                items = line.strip("\n").split(" ")[1:]
                assert len(items) == int(item_cnt)

                for item in items:
                    # Make sure the identifiers are correct.
                    assert item.isdecimal()
                    u_i_pairs.append((user, int(item)))

        # Rename columns to default ones ?
        df = pd.DataFrame(
            u_i_pairs,
            columns=[self.USER_IX, self.ITEM_IX],
            dtype=np.int64,
        )
        return df
