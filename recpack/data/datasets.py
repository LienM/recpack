import numpy as np
import pandas as pd


from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
from recpack.data.matrix import InteractionMatrix
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.util import to_tuple

# TODO Refactor so that it downloads the datasets and names the fields consistently


class Dataset(object):
    user_id = "user"
    item_id = "item"
    value_id = None
    timestamp_id = None

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        raise NotImplementedError("Need to override name")

    # def get_params(self):
    #     TODO This method is broken.
    #     params = super().get_params() if hasattr(super(), "get_params") else dict()
    #     params["data_source"] = self.name
    #     return params

    def load_df(self):
        raise NotImplementedError("Need to override `load_df` or `preprocess`")

    @property
    def preprocessor(self):
        preprocessor = DataFramePreprocessor(
            self.item_id, self.user_id, self.value_id, self.timestamp_id, dedupe=True
        )
        return preprocessor

    def preprocess(self):
        """
        Return a dataset of type InteractionMatrix
        """
        df = to_tuple(self.load_df())
        data_m = self.preprocessor.process(df)
        return data_m

    @property
    def item_id_mapping(self):
        return self.preprocessor.item_id_mapping

    @property
    def user_id_mapping(self):
        return self.preprocessor.user_id_mapping


class CiteULike(Dataset):
    @property
    def name(self):
        return "citeulike"

    user_id = "user_id"
    item_id = "item_id"

    @classmethod
    def load_df(cls, data_file):
        # TODO Download data
        u_i_pairs = []
        with open(data_file, "r") as f:
            for user, line in enumerate(f.readlines()):
                items = line.strip("\n").split(" ")[1:]  # First element is a count
                item_cnt = line.strip("\n").split(" ")[0]
                assert len(items) == int(item_cnt)
                for item in items:
                    assert item.isdecimal()  # Make sure the identifiers are correct.
                    u_i_pairs.append((user, int(item)))

        return pd.DataFrame(u_i_pairs, columns=[cls.user_id, cls.item_id])

    def preprocess(self, data_file, min_iu=5):
        df = self.load_df(data_file)
        preprocessor = self.preprocessor
        preprocessor.add_filter(MinItemsPerUser(min_iu, self.user_id, self.item_id))

        view_data = preprocessor.process(df)
        return view_data


class ML20MDataset(Dataset):
    @property
    def name(self):
        return "ML"

    user_id = "userId"
    item_id = "movieId"
    value_id = "rating"

    def load_df(self, path=None):
        df = pd.read_csv(path)
        return df

    def preprocess(self, path, min_rating=4, min_iu=5):
        df = self.load_df(path=path)
        preferences = df[df[self.value_id] >= min_rating]
        preprocessor = self.preprocessor
        preprocessor.add_filter(MinItemsPerUser(min_iu, self.user_id, self.item_id))

        view_data, = preprocessor.process(preferences)
        return InteractionMatrix(view_data.binary_values)


class RSC2015(Dataset):
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

    def preprocess(
        self, path: str, min_iu: int = 2, min_ui: int = 5, nrows: int = None
    ) -> InteractionMatrix:
        """
        Loads the dataset as a InteractionMatrix. By default, users with fewer than 2 clicks
        and items with fewer than 5 clicks are removed.
        """
        df = self.load_df(path, nrows=nrows)

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

        return preprocessor.process(df)

    @property
    def name(self):
        return "rsc2015"

    @property
    def preprocessor(self):
        preprocessor = DataFramePreprocessor(
            self.item_id, self.user_id, timestamp_id=self.timestamp_id, dedupe=False
        )
        return preprocessor
