import numpy as np
import pandas as pd

from recpack.data.data_source import DataSource
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
from recpack.data.data_matrix import DataM
from recpack.preprocessing.preprocessors import DataFramePreprocessor


class RSC2015(DataSource):
    """
    RecSys Challenge 2015 clicks dataset.
    """

    user_id = "sid"
    item_id = "iid"
    timestamp_id = "time"

    def load_df(self, path: str, nrows: int = None) -> pd.DataFrame:
        """
        Loads the dataset as a pandas dataframe, unfiltered.
        """
        df = pd.read_csv(
            path,
            names=["sid", "time", "iid"],
            dtype={"sid": np.int64, "time": np.str, "iid": np.int64},
            parse_dates=["time"],
            usecols=[0, 1, 2],
            nrows=nrows,
        )
        df["time"] = (
            df["time"].astype(int) / 1e9
        )  # pandas datetime -> seconds from epoch
        return df

    def preprocess(self, path: str, min_iu: int = 2, min_ui: int = 5) -> DataM:
        """
        Loads the dataset as a DataM. By default, users with fewer than 2 clicks 
        and items with fewer than 5 clicks are removed.
        """
        df = self.load_df(path)

        filter_users = MinItemsPerUser(
            min_iu,
            item_id=self.item_id,
            user_id=self.user_id,
            timestamp_id=self.timestamp_id,
            count_duplicates=True,
        )
        filter_items = MinUsersPerItem(
            min_ui,
            item_id=self.item_id,
            user_id=self.user_id,
            timestamp_id=self.timestamp_id,
            count_duplicates=True,
        )
        preprocessor = self.preprocessor
        preprocessor.add_filter(filter_users)
        preprocessor.add_filter(filter_items)
        preprocessor.add_filter(filter_users)

        return preprocessor.process(df)[0]

    @property
    def data_name(self):
        return "rsc2015"

    @property
    def preprocessor(self):
        preprocessor = DataFramePreprocessor(
            self.item_id, self.user_id, timestamp_id=self.timestamp_id, dedupe=False
        )
        return preprocessor
