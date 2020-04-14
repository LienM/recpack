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
        self.filter.append(_filter)

    def process(self, df: pd.DataFrame) -> scipy.sparse.csr_matrix:
        if self.dedupe:
            df = df.drop_duplicates([self.user_id, self.item_id], keep="first")

        self.update_id_mappings(df)

        cleaned_item_id = 'iid'
        cleaned_user_id = 'uid'

        df.loc[:, cleaned_item_id] = df[self.item_id].map(
            lambda x: self.item_id_mapping[x]
        )
        df.loc[:, cleaned_user_id] = df[self.user_id].map(
            lambda x: self.user_id_mapping[x]
        )

        # Convert input data into internal data objects
        data = DataM.create_from_dataframe(
            df,
            cleaned_item_id, cleaned_user_id, self.timestamp_id,
            shape=(
                max(self.user_id_mapping.values()) + 1,
                max(self.item_id_mapping.values()) + 1
            )
        )

        return data

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
