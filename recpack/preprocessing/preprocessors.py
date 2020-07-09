from typing import List

import pandas as pd
import scipy.sparse
import numpy as np

import recpack.preprocessing.helpers as helpers
from recpack.data_matrix import DataM
from recpack.preprocessing.filters import Filter
from recpack.utils import logger

from tqdm.auto import tqdm
tqdm.pandas()


class DataFramePreprocessor:
    def __init__(self, item_id, user_id, value_id=None, timestamp_id=None, dedupe=False):
        """
        Class to preprocess a Pandas Dataframe and turn it into a DataM object.
        All ID mappings are stored, so that processing of multiple DataFrames will lead to consistent mapped identifiers.

        :param item_id: Column name of the Item ID column
        :type item_id: str
        :param user_id: Column name of the User ID column
        :type user_id: str
        :param value_id: Column name of the value column, defaults to None
        :type value_id: str, optional
        :param timestamp_id: Column name of the timestamp column, defaults to None
        :type timestamp_id: str, optional
        :param dedupe: Deduplicate events, such that (user_id, item_id) pairs are unique, defaults to False
        :type dedupe: bool, optional
        """
        self.item_id_mapping = dict()
        self.user_id_mapping = dict()
        self.item_id = item_id
        self.user_id = user_id
        self.value_id = value_id
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
        logger.debug("Map users")
        if not self.user_id_mapping:
            raise RuntimeError(
                "User ID Mapping should be fit before attempting to map users"
            )

        user_id_mapping_array = np.arange(0, max(self.user_id_mapping.keys()) + 1)
        user_id_mapping_array[list(self.user_id_mapping.keys())] = list(self.user_id_mapping.values())
        res = user_id_mapping_array[df[self.user_id]]
        logger.debug("Done")
        return res
        # return df[self.user_id].progress_map(lambda x: self.user_id_mapping.get(x))

    def map_items(self, df):
        logger.debug("Map items")
        if not self.item_id_mapping:
            raise RuntimeError(
                "Item ID Mapping should be fit before attempting to map items"
            )

        item_id_mapping_array = np.arange(0, max(self.item_id_mapping.keys()) + 1)
        item_id_mapping_array[list(self.item_id_mapping.keys())] = list(self.item_id_mapping.values())
        res = item_id_mapping_array[df[self.item_id]]
        logger.debug("Done")
        return res
        # res2 = df[self.item_id].progress_map(lambda x: self.item_id_mapping.get(x))

    @property
    def shape(self):
        return (
            max(self.user_id_mapping.values()) + 1,
            max(self.item_id_mapping.values()) + 1,
        )

    def process(self, *dfs: pd.DataFrame) -> List[DataM]:
        """
        Process all DataFrames passed as arguments.
        If your pipeline requires more than one DataFrame,
        pass all of them to a single call of process to guarantee
        that their dimensions will match.

        :return: A list of sparse matrices in the order they were passed as arguments. 
        :rtype: List[scipy.sparse.csr_matrix]
        """
        dfs = list(dfs)
        for index, df in enumerate(dfs):
            logger.debug(f"Processing df {index}")
            logger.debug(f"\tinteractions before preprocess: {len(df.index)}")
            logger.debug(f"\titems before preprocess: {df[self.item_id].nunique()}")
            logger.debug(f"\tusers before preprocess: {df[self.user_id].nunique()}")
            if self.dedupe:
                df.drop_duplicates(
                    [self.user_id, self.item_id], keep="first", inplace=True
                )
                logger.debug(f"\tinteractions after dedupe: {len(df.index)}")
                logger.debug(f"\titems after dedupe: {df[self.item_id].nunique()}")
                logger.debug(f"\tusers after dedupe: {df[self.user_id].nunique()}")

        for filter in self.filters:
            logger.debug(f"applying filter: {filter}")
            dfs = filter.apply_all(*dfs)
            for index, df in enumerate(dfs):
                logger.debug(f"df {index}")
                logger.debug(f"\tinteractions after filter: {len(df.index)}")
                logger.debug(f"\titems after filter: {df[self.item_id].nunique()}")
                logger.debug(f"\tusers after filter: {df[self.user_id].nunique()}")

        for index, df in enumerate(dfs):
            self.update_id_mappings(df)

        cleaned_item_id = "iid"
        cleaned_user_id = "uid"

        data_ms = []

        for df in dfs:
            df.loc[:, cleaned_item_id] = self.map_items(df)
            df.loc[:, cleaned_user_id] = self.map_users(df)

            # Convert input data into internal data objects
            data_m = DataM.create_from_dataframe(
                df,
                cleaned_item_id,
                cleaned_user_id,
                self.value_id,
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
