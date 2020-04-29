from typing import List

import pandas as pd
import scipy.sparse

import recpack.preprocessing.helpers as helpers
from recpack.data_matrix import DataM
from recpack.preprocessing.filters import Filter


class DataFramePreprocessor:
    def __init__(self, item_id, user_id, timestamp_id=None, dedupe=False):
        """
        Class to preprocess a Pandas Dataframe and turn it into a DataM object.
        All ID mappings are stored, so that processing of multiple DataFrames will lead to consistent mapped identifiers.

        :param item_id: Column name of the Item ID column
        :type item_id: str
        :param user_id: Column name of the User ID column
        :type user_id: str
        :param timestamp_id: Column name of the timestamp column, defaults to None
        :type timestamp_id: str, optional
        :param dedupe: Deduplicate events, such that (user_id, item_id) pairs are unique, defaults to False
        :type dedupe: bool, optional
        """
        self.item_id_mapping = {}
        self.user_id_mapping = {}
        self.item_id = item_id
        self.user_id = user_id
        self.timestamp_id = timestamp_id
        self.dedupe = dedupe
        self.filters = []

    def add_filter(self, _filter: Filter):
        """
        Add a preprocessing filter to be applied before transforming to a DataM object.
        Filters are applied in order, different orderings can lead to different results!

        :param _filter: The filter to be applied
        :type _filter: Filter
        """
        self.filters.append(_filter)

    def map_users(self, df):
        if not self.user_id_mapping:
            raise RuntimeError(
                "User ID Mapping should be fit before attempting to map users"
            )

        return df[self.user_id].map(lambda x: self.user_id_mapping.get(x))

    def map_items(self, df):
        if not self.item_id_mapping:
            raise RuntimeError(
                "Item ID Mapping should be fit before attempting to map items"
            )

        return df[self.item_id].map(lambda x: self.item_id_mapping.get(x))

    @property
    def shape(self):
        return (
            max(self.user_id_mapping.values()) + 1,
            max(self.item_id_mapping.values()) + 1,
        )

    def process(self, *args: pd.DataFrame) -> List[scipy.sparse.csr_matrix]:
        """
        Process all DataFrames passed as arguments.
        If your pipeline requires more than one DataFrame,
        pass all of them to a single call of process to guarantee
        that their dimensions will match.

        :return: A list of sparse matrices in the order they were passed as arguments. 
        :rtype: List[scipy.sparse.csr_matrix]
        """
        for df in args:
            if self.dedupe:
                df.drop_duplicates(
                    [self.user_id, self.item_id], keep="first", inplace=True
                )

            self.update_id_mappings(df)

        cleaned_item_id = "iid"
        cleaned_user_id = "uid"

        data_ms = []

        for df in args:
            df.loc[:, cleaned_item_id] = self.map_items(df)
            df.loc[:, cleaned_user_id] = self.map_users(df)

            # Convert input data into internal data objects
            data_m = DataM.create_from_dataframe(
                df,
                cleaned_item_id,
                cleaned_user_id,
                self.timestamp_id,
                shape=self.shape,
            )

            data_ms.append(data_m)

        return data_ms

    def update_id_mappings(self, df: pd.DataFrame):
        """
        Update the id mapping so we can combine multiple files
        """

        # Convert user and item ids into a continuous sequence to make
        # training faster and use much less memory.
        item_ids = list(df[self.item_id].unique())
        user_ids = list(df[self.user_id].unique())

        self.user_id_mapping = helpers.rescale_id_space(
            user_ids, id_mapping=self.user_id_mapping
        )
        self.item_id_mapping = helpers.rescale_id_space(
            item_ids, id_mapping=self.item_id_mapping
        )

    def apply_item_id_mapping(self, df: pd.DataFrame):
        cleaned_item_id = "iid"
        df.loc[:, cleaned_item_id] = self.map_items(df)
