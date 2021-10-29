"""Module responsible for handling datasets."""

import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import List
from urllib.request import urlretrieve
import zipfile

from recpack.preprocessing.filters import (
    Filter,
    MinItemsPerUser,
    MinUsersPerItem,
    MinRating,
)
from recpack.data.matrix import InteractionMatrix, to_binary
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.util import to_tuple


def _fetch_remote(url: str, filename: str) -> str:
    """Fetch data from remote url and save locally

    :param url: url to fetch data from
    :type url: str
    :param filename: Path to save file to
    :type filename: str
    :return: The filename where data was saved
    :rtype: str
    """
    urlretrieve(url, filename)
    return filename


class Dataset:
    """Represents a collaborative filtering dataset,
    containing users who interacted in some way with a set of items.

    Every Dataset has a set of preprocessing defaults,
    i.e. filters that are commonly applied to the dataset before use in recommendation algorithms.
    These can be disabled and a different set of filters can be applied.

    A Dataset is transformed into an InteractionMatrix

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: Optional[str]
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: Optional[str]
    :param preprocess_default: Should a default set of filters be initialised? Defaults to True
    :type preprocess_default: bool, optional
    """

    USER_IX = "user_id"
    """name of the column in the loaded DataFrame with user identifiers"""
    ITEM_IX = "item_id"
    """name of the column in the loaded DataFrame with item identifiers"""
    TIMESTAMP_IX = "seconds_since_epoch"
    """name of the column in the loaded DataFrame with timestamp, in seconds since epoch"""

    DEFAULT_FILENAME = None
    """Default filename that will be used if it is not specified by the user."""

    def __init__(
        self, path: str = "data", filename: str = None, preprocess_default=True
    ):
        self.filename = filename
        if not self.filename:
            if self.DEFAULT_FILENAME:
                self.filename = self.DEFAULT_FILENAME
            else:
                raise ValueError("No filename specified, and no default known.")

        self.path = path
        self.preprocessor = DataFramePreprocessor(
            self.ITEM_IX, self.USER_IX, self.TIMESTAMP_IX
        )
        if preprocess_default:
            for f in self._default_filters:
                self.add_filter(f)

        self._ensure_path_exists()

    @property
    def file_path(self):
        """The fully classified path to the file that should be loaded from/saved to."""
        # TODO: correctness check?
        return os.path.join(self.path, self.filename)

    def _ensure_path_exists(self):
        """Constructs directory if path is not present on disk."""
        p = Path(self.path)
        p.mkdir(exist_ok=True)

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters to apply to a dataframe.

        Should be defined for a dataset if there is default preprocessing.

        :return: A list of filters
        :rtype: List[Filter]
        """
        return []

    def add_filter(self, _filter: Filter, index=None):
        """Add a filter to be used when calling load_interaction_matrix().

        If the index is specified, the filter is inserted at the specified index.
        Otherwise it is appended.

        :param _filter: Filter to be applied to the loaded DataFrame
                    processing to interaction matrix.
        :type _filter: Filter
        :param index: The index to insert the filter at,
            None will append the filter. Defaults to None
        :type index: int

        """
        self.preprocessor.add_filter(_filter, index=index)

    def fetch_dataset(self, force=False):
        """Check if dataset is present, if not download

        :param force: If True, dataset will be downloaded,
                even if the file already exists.
                Defaults to False.
        :type force: bool, optional
        """
        if not os.path.exists(self.file_path) or force:
            self._download_dataset()

    def _download_dataset(self):
        raise NotImplementedError("Should still be implemented")

    def load_dataframe(self) -> pd.DataFrame:
        """Load the DataFrame from file, and return it as a pandas DataFrame.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """
        raise NotImplementedError("Needs to be implemented")

    def load_interaction_matrix(self) -> InteractionMatrix:
        """Loads data into an InteractionMatrix object.

        Data is loaded into a DataFrame using the `load_dataframe` function.
        Resulting DataFrame is parsed into an `InteractionMatrix` object.
        During parsing the filters are applied in order.

        :return: The resulting InteractionMatrix
        :rtype: InteractionMatrix
        """
        df = self.load_dataframe()

        return self.preprocessor.process(df)


class DummyDataset(Dataset):
    """Small randomly generated dummy dataset that allows testing of pipelines
    and other components without needing to load a full scale dataset.

    :param path: The path to the data directory. UNUSED because dataset is generated and not read from file.
        Defaults to `data`
    :type path: Optional[str]
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        UNUSED because dataset is generated and not read from file.
    :type filename: Optional[str]
    :param preprocess_default: Should a default set of filters be initialised? Defaults to True
    :type preprocess_default: bool, optional
    :param seed: Seed for the random data generation. Defaults to None.
    :type seed: int, optional
    :param num_users: The amount of users to use when generating data, defaults to 100
    :type num_users: int, optional
    :param num_items: The number of items to use when generating data, defaults to 20
    :type num_items: int, optional
    :param num_interactions: The number of interactions to generate, defaults to 500
    :type num_interactions: int, optional
    :param min_t: The minimum timestamp when generating data, defaults to 0
    :type min_t: int, optional
    :param max_t: The maximum timestamp when generating data, defaults to 500
    :type max_t: int, optional
    """

    USER_IX = "user_id"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "item_id"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""

    DEFAULT_FILENAME = "dummy_input.csv"

    def __init__(
        self,
        path: str = "data",
        filename: str = None,
        preprocess_default=True,
        seed=None,
        num_users=100,
        num_items=20,
        num_interactions=500,
        min_t=0,
        max_t=500,
    ):
        super().__init__(path, filename, preprocess_default)

        self.seed = seed
        if self.seed is None:
            self.seed = seed = np.random.get_state()[1][0]

        self.num_users = num_users
        self.num_items = num_users
        self.num_interactions = num_interactions
        self.min_t = min_t
        self.max_t = max_t

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the Dummy dataset

        Filters users and items that do not have enough interactions.
        At least 2 users per item and 2 interactions per user.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinUsersPerItem(2, self.ITEM_IX, self.USER_IX),
            MinItemsPerUser(2, self.ITEM_IX, self.USER_IX),
        ]

    def _download_dataset(self):
        # There is no downloading necessary.
        pass

    def load_dataframe(self) -> pd.DataFrame:
        """Load the data from file, and return as a Pandas DataFrame.

        Downloads the data file if it is not yet present.
        The output will contain a dataframe with a user_id and item_id column.
        Each interaction is stored in a separate row.

        :return: The interactions as a dataframe, with a row for each interaction.
        :rtype: pandas.DataFrame
        """
        np.random.seed(self.seed)

        input_dict = {
            self.USER_IX: [
                np.random.randint(0, self.num_users)
                for _ in range(0, self.num_interactions)
            ],
            self.ITEM_IX: [
                np.random.randint(0, self.num_items)
                for _ in range(0, self.num_interactions)
            ],
            self.TIMESTAMP_IX: [
                np.random.randint(self.min_t, self.max_t)
                for _ in range(0, self.num_interactions)
            ],
        }

        df = pd.DataFrame.from_dict(input_dict)
        return df


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


class CiteULike(Dataset):
    """Dataset class for the CiteULike dataset.

    Full information  on the dataset can be found at https://github.com/js05212/citeulike-a.
    Uses the `users.dat` file from the dataset to construct an implicit feedback interaction matrix.

    Default processing makes sure that:

    - Each remaining user has interacted with at least 3 items
    - Each remaining  item has been interacted with by at least 5 users

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: Optional[str]
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: Optional[str]
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
        """Download thee users.dat file from the github repository.
        The file is saved at the specified `self.file_path`
        """
        DATASETURL = (
            "https://raw.githubusercontent.com/js05212/citeulike-a/master/users.dat"
        )
        _fetch_remote(DATASETURL, self.file_path)

    def load_dataframe(self) -> pd.DataFrame:
        """Load the data from file, and return as a Pandas DataFrame.

        Downloads the data file if it is not yet present.
        The output will contain a DataFrame with a ``user_id`` and ``item_id`` column.
        Each interaction is stored in a separate row.

        :return: The interactions as a DataFrame, with a row for each interaction.
        :rtype: pandas.DataFrame
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


class MovieLens25M(Dataset):
    """Handles Movielens 25M dataset.

    All information on the dataset can be found at https://grouplens.org/datasets/movielens/25m/.
    Uses the `ratings.csv` file to generate an interaction matrix.

    Default processing makes sure that:

    - Each rating above or equal to 1 is used as interaction
    - Each remaining user has interacted with at least 3 items
    - Each remaining  item has been interacted with by at least 5 users

    To use another value as minimal rating to mark interaction as positive,
    you have to manually set the preprocessing filters.::

        from recpack.preprocessing.filters import MinRating, MinItemsPerUser, MinUsersPerItem
        from recpack.data.datasets import MovieLens25M
        d = MovieLens25M('path/to/file', preprocess_default=False)
        d.add_filter(MinRating(3, d.RATING_IX, 3))
        d.add_filter(MinItemsPerUser(3, d.ITEM_IX, d.USER_IX))
        d.add_filter(MinUsersPerItem(5, d.ITEM_IX, d.USER_IX))

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: Optional[str]
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: Optional[str]
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
            MinRating(1, self.RATING_IX),
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
        """Load the data from file, and return as a Pandas DataFrame.

        Downloads the data file if it is not yet present.
        The output will contain the data of the CSV file as a Pandas DataFrame.

        :return: The interactions as a Pandas DataFrame, with a row for each interaction.
        :rtype: pandas.DataFrame
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


class RecsysChallenge2015(Dataset):
    """Handles data from the Recsys Challenge 2015, yoochoose dataset.

    All information and downloads can be found at https://www.kaggle.com/chadgostopp/recsys-challenge-2015.
    Because downloading the data requires a Kaggle account we can't download it here,
    you should download the data manually and provide the path to the `yoochoose-clicks.dat` file.

    Default processing makes sure that:

    - Each remaining  item has been interacted with by at least 5 users.

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: Optional[str]
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: Optional[str]
    :param preprocess_default: Should a default set of filters be initialised? Defaults to True
    :type preprocess_default: bool, optional

    """

    USER_IX = "session"
    """Name of the column in the DataFrame that contains user identifiers."""

    DEFAULT_FILENAME = "yoochoose-clicks.dat"
    """Default filename that will be used if it is not specified by the user."""

    def _download_dataset(self):
        """Downloading this dataset is not supported."""
        raise NotImplementedError(
            "Recsys Challenge dataset should be downloaded manually, you can get it at: https://www.kaggle.com/chadgostopp/recsys-challenge-2015."
        )

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the RecsysChallenge 2015 dataset

        Filters items that do not have enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinUsersPerItem(5, self.USER_IX, self.ITEM_IX, count_duplicates=True),
        ]

    def load_dataframe(self) -> pd.DataFrame:
        """Load the data from file, and return a DataFrame.

        The output will contain a DataFrame with a ``session``, ``item_id`` and ``seconds_since_epoch`` column.
        Each interaction is stored in a separate row.

        :return: The interactions as a DataFrame, with a row for each interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()

        df = pd.read_csv(
            self.file_path,
            names=[self.USER_IX, self.TIMESTAMP_IX, self.ITEM_IX],
            dtype={
                self.USER_IX: np.int64,
                self.TIMESTAMP_IX: str,
                self.ITEM_IX: np.int64,
            },
            parse_dates=[self.TIMESTAMP_IX],
            usecols=[0, 1, 2],
        )

        # Adapt timestamp, this makes it so the timestamp is always seconds since epoch
        df[self.TIMESTAMP_IX] = (
            df[self.TIMESTAMP_IX].astype(int) / 1e9
        )  # pandas datetime -> seconds from epoch

        return df


class CosmeticsShopDataset(Dataset):
    """Handles data from the eCommerce Events History in Cosmetics Shop dataset on Kaggle.

    All information and downloads can be found at
    https://www.kaggle.com/mkechinov/ecommerce-events-history-in-cosmetics-shop.
    Because downloading the data requires a Kaggle account we can't download it here,
    you should download the data manually and provide the path to one of the monthly
    files, or a combined file with all events.
    If a monthly file is given, then only those events will be used. In order to use
    the full dataset you need to combine the files.

    Default processing makes sure that:

    - Each remaining item has been interacted with by at least 50 users.
    - Each user has interacted with at least 3 items

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: Optional[str]
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: Optional[str]
    :param preprocess_default: Should a default set of filters be initialised? Defaults to True
    :type preprocess_default: bool, optional
    :param additional_columns_to_load: Extra columns to load during dataframe creation
    :type additional_columns_to_load: List[str], optional
    :param event_types: The dataset contains view, cart, remove_from_cart, purchase events.
        You can select a subset of them.
        Defaults to ["view"]
    :type event_types: List[str], optional
    """

    USER_IX = "user_id"
    """Name of the column in the DataFrame that contains user identifiers."""

    ITEM_IX = "product_id"
    """Name of the column in the DataFrame that contains item identifiers."""

    TIMESTAMP_IX = "event_time"
    """Name of the column in the DataFrame that contains timestamp."""

    EVENT_TYPE_IX = "event_type"
    """Name of the column in the DataFrame that contains the event_types"""

    DEFAULT_FILENAME = "2019-Dec.csv"
    """Default filename that will be used if it is not specified by the user."""

    ALLOWED_EVENT_TYPES = ["view", "cart", "remove_from_cart", "purchase"]

    def __init__(
        self,
        path: str = "data",
        filename: str = None,
        preprocess_default=True,
        additional_columns_to_load: List[str] = [],
        event_types: List[str] = ["view"],
    ):
        super().__init__(path, filename, preprocess_default)
        self.additional_columns_to_load = additional_columns_to_load

        for event_type in event_types:
            if event_type not in self.ALLOWED_EVENT_TYPES:
                raise ValueError(
                    f"{event_type} is not in the allowed event types. "
                    f"Please use one of {self.ALLOWED_EVENT_TYPES}"
                )

        self.event_types = event_types

    def _download_dataset(self):
        """Downloading this dataset is not supported."""
        raise NotImplementedError(
            "CosmeticsShop dataset should be downloaded manually, "
            "you can get it at: https://www.kaggle.com/mkechinov/ecommerce-events-history-in-cosmetics-shop."
        )

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the CosmeticsShop dataset

        Filters items that do not have enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinUsersPerItem(50, self.USER_IX, self.ITEM_IX, count_duplicates=True),
            MinItemsPerUser(3, self.USER_IX, self.ITEM_IX, count_duplicates=True),
        ]

    @property
    def _columns(self) -> List[str]:
        columns = [self.USER_IX, self.ITEM_IX, self.TIMESTAMP_IX, self.EVENT_TYPE_IX]
        if self.additional_columns_to_load:
            columns = columns + self.additional_columns_to_load

        return columns

    def load_dataframe(self) -> pd.DataFrame:
        """Load the data from file, and return a DataFrame.

        The output will contain a DataFrame.
        Each interaction is stored in a separate row.

        :return: The interactions as a DataFrame, with a row for each interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()

        df = pd.read_csv(
            self.file_path,
            parse_dates=[self.TIMESTAMP_IX],
        )

        # Adapt timestamp, this makes it so the timestamp is always seconds since epoch
        df[self.TIMESTAMP_IX] = (
            df[self.TIMESTAMP_IX].view(int) / 1e9
        )  # pandas datetime -> seconds from epoch

        # Select only the specified event_types
        if self.event_types:
            df = df[df[self.EVENT_TYPE_IX].isin(self.event_types)].copy()

        df = df[self._columns].copy()

        return df


class RetailRocketDataset(Dataset):
    """Handles data from the Retail Rocket dataset on Kaggle.

    All information and downloads can be found at
    https://www.kaggle.com/retailrocket/ecommerce-dataset.
    Because downloading the data requires a Kaggle account we can't download it here,
    you should download the data manually and provide the path to the downloaded folder.

    Default processing makes sure that:

    - Each remaining item has been interacted with by at least 50 users.
    - Each user has interacted with at least 3 items

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: Optional[str]
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: Optional[str]
    :param preprocess_default: Should a default set of filters be initialised? Defaults to True
    :type preprocess_default: bool, optional
    :param event_types: The dataset contains view, addtocart, transaction events.
        You can select a subset of them.
        Defaults to ["view"]
    :type event_types: List[str], optional
    """

    USER_IX = "visitorid"
    """Name of the column in the DataFrame that contains user identifiers."""

    ITEM_IX = "itemid"
    """Name of the column in the DataFrame that contains item identifiers."""

    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains timestamp."""

    EVENT_TYPE_IX = "event"
    """Name of the column in the DataFrame that contains the event_types"""

    DEFAULT_FILENAME = "events.csv"
    """Default filename that will be used if it is not specified by the user."""

    ALLOWED_EVENT_TYPES = ["view", "addtocart", "transaction"]

    def __init__(
        self,
        path: str = "data",
        filename: str = None,
        preprocess_default=True,
        event_types: List[str] = ["view"],
    ):
        super().__init__(path, filename, preprocess_default)

        for event_type in event_types:
            if event_type not in self.ALLOWED_EVENT_TYPES:
                raise ValueError(
                    f"{event_type} is not in the allowed event types. "
                    f"Please use one of {self.ALLOWED_EVENT_TYPES}"
                )

        self.event_types = event_types

    def _download_dataset(self):
        """Downloading this dataset is not supported."""
        raise NotImplementedError(
            "RetailRocket dataset should be downloaded manually, "
            "you can get it at: https://www.kaggle.com/retailrocket/ecommerce-dataset."
        )

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the RetailRocket dataset

        Filters items that do not have enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinUsersPerItem(50, self.USER_IX, self.ITEM_IX, count_duplicates=True),
            MinItemsPerUser(3, self.USER_IX, self.ITEM_IX, count_duplicates=True),
        ]

    @property
    def _columns(self) -> List[str]:
        columns = [self.USER_IX, self.ITEM_IX, self.TIMESTAMP_IX, self.EVENT_TYPE_IX]
        return columns

    def load_dataframe(self) -> pd.DataFrame:
        """Load the data from file, and return a DataFrame.

        The output will contain a DataFrame.
        Each interaction is stored in a separate row.

        :return: The interactions as a DataFrame, with a row for each interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()

        df = pd.read_csv(
            self.file_path,
        )

        # Adapt timestamp, this makes it so the timestamp is always seconds since epoch
        # It's in seconds since epoch, and so should be divided by 1000
        df[self.TIMESTAMP_IX] = (
            df[self.TIMESTAMP_IX].view(int) / 1e3
        )  # pandas datetime -> seconds from epoch

        # Select only the specified event_types
        if self.event_types:
            df = df[df[self.EVENT_TYPE_IX].isin(self.event_types)].copy()

        df = df[self._columns].copy()

        return df
