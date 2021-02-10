import numpy as np
import os
import pandas as pd
from urllib.request import urlretrieve
import zipfile

from recpack.preprocessing.filters import Filter, MinItemsPerUser, MinUsersPerItem
from recpack.data.matrix import InteractionMatrix, to_binary
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.util import to_tuple


def _fetch_remote(url, filename):
    # TODO: checksum?
    urlretrieve(url, filename)
    return filename


# TODO: Timestamp column rename to seconds_since_epoch?
# That would make clear what the data is, not sure though
class Dataset:

    """
    - Dataset downloading
    - Dataset local load (if downloaded)
    - Default preprocessing
    - Add custom / specific Filters
        + Create min_rating filter
    - Load dataframe
    - Load InteractionMatrix
    """

    USER_IX = "user_id"
    ITEM_IX = "item_id"
    TIMESTAMP_IX = "timestamp"

    def __init__(self, filename, preprocess_default=True):
        self.filename = filename
        self.preprocessor = DataFramePreprocessor(
            self.ITEM_IX, self.USER_IX, self.TIMESTAMP_IX, dedupe=False
        )
        if preprocess_default:
            for f in self._default_filters:
                self.add_filter(f)

    @property
    def _default_filters(self):
        # Return a list of filters to be added to the preprocessor
        # if default is enabled.

        return [
            MinItemsPerUser(3, self.ITEM_IX, self.USER_IX),
            MinUsersPerItem(5, self.ITEM_IX, self.USER_IX),
        ]

    def add_filter(self, _filter: Filter):
        self.preprocessor.add_filter(_filter)

    def fetch_dataset(self, force=False):
        """Check if dataset is present, if not download

        if force => Overwrite
        """
        if not os.path.exists(self.filename) or force:
            self._download_dataset()

    def _download_dataset(self):
        raise NotImplementedError("Should still be implemented")

    def load_dataframe(self):
        raise NotImplementedError("Needs to be implemented")

    def load_interaction_matrix(self):
        df = self.load_dataframe()

        return self.preprocessor.process(df)


class CiteULike(Dataset):
    USER_IX = "user"
    ITEM_IX = "item"
    TIMESTAMP_IX = None

    def _download_dataset(self):
        DATASETURL = (
            "https://raw.githubusercontent.com/js05212/citeulike-a/master/users.dat"
        )
        _fetch_remote(DATASETURL, self.filename)

    def load_dataframe(self):
        # This will download the dataset if necessary
        self.fetch_dataset()

        u_i_pairs = []
        with open(self.filename, "r") as f:

            for user, line in enumerate(f.readlines()):
                item_cnt = line.strip("\n").split(" ")[0]  # First element is a count
                items = line.strip("\n").split(" ")[1:]
                assert len(items) == int(item_cnt)

                for item in items:
                    assert item.isdecimal()  # Make sure the identifiers are correct.
                    u_i_pairs.append((user, int(item)))

        # Rename columns to default ones ?
        df = pd.DataFrame(u_i_pairs, columns=[self.USER_IX, self.ITEM_IX])
        return df


class MovieLens20M(Dataset):
    USER_IX = "userId"
    ITEM_IX = "movieId"
    TIMESTAMP_IX = "timestamp"
    RATING_IX = "rating"

    def _download_dataset(self):
        DATASETURL = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"

        # Get the director part of the file specified
        dir_f = os.path.dirname(self.filename)

        # Download the zip into the directory
        _fetch_remote(DATASETURL, os.path.join(dir_f, "ml-25m.zip"))

        # Extract the ratings file which we will use
        with zipfile.ZipFile(os.path.join(dir_f, "ml-25m.zip"), "r") as zip_ref:
            zip_ref.extract("ml-25m/ratings.csv", dir_f)

        # Rename the ratings file to the specified filename
        os.rename(os.path.join(dir_f, "ml-25m/ratings.csv"), self.filename)

    def load_dataframe(self):
        self.fetch_dataset()
        df = pd.read_csv(self.filename)

        return df


class RecsysChallenge2015(Dataset):
    USER_IX = "session"
    ITEM_IX = "item"
    TIMESTAMP_IX = "timestamp"

    def _download_dataset(self):
        raise NotImplementedError(
            "Recsys Challenge dataset should be downloaded manually, you can get it at: https://www.kaggle.com/chadgostopp/recsys-challenge-2015."
        )

    # TODO: The original function had a parameter nrows,
    # which allowed to just parse the first X rows.
    # Do we want to add that functionality, feels a bit unneeded

    @property
    def _default_filters(self):
        # Return a list of filters to be added to the preprocessor
        # if default is enabled.
        # TODO: this had a different preprocessing in the function compared to the others.
        # Do we try to unify this, and make all defaults the same,
        # or was there a particular reason for this one?
        return [
            MinItemsPerUser(3, self.USER_IX, self.ITEM_IX, count_duplicates=True),
            MinUsersPerItem(5, self.USER_IX, self.ITEM_IX, count_duplicates=True),
            MinItemsPerUser(3, self.USER_IX, self.ITEM_IX, count_duplicates=True),
        ]

    def load_dataframe(self):
        self.fetch_dataset()

        df = pd.read_csv(
            self.filename,
            names=[self.USER_IX, self.TIMESTAMP_IX, self.ITEM_IX],
            dtype={
                self.USER_IX: np.int64,
                self.TIMESTAMP_IX: np.str,
                self.ITEM_IX: np.int64,
            },
            parse_dates=[self.TIMESTAMP_IX],
            usecols=[0, 1, 2],
            # nrows=nrows,
        )

        # Adapt timestamp, this makes it so the timestamp is always seconds since epoch
        df[self.TIMESTAMP_IX] = (
            df[self.TIMESTAMP_IX].astype(int) / 1e9
        )  # pandas datetime -> seconds from epoch

        return df
