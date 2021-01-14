import logging
from typing import List

import pandas as pd
import numpy as np

import recpack.preprocessing.util as util
from recpack.data.matrix import InteractionMatrix
from recpack.preprocessing.filters import Filter

from tqdm.auto import tqdm

tqdm.pandas()


logger = logging.getLogger("recpack")


class DataFramePreprocessor:

    ITEM_IX = "iid"
    USER_IX = "uid"

    def __init__(
        self, item_id, user_id, timestamp_id=None, dedupe=False
    ):
        # CHECK: I removed value_id here, but I could also have done some magic to process the cases where we can anyway.
        # I think this is better though, what do you think?
        """
        Class to preprocess a Pandas Dataframe and turn it into a InteractionMatrix object.
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
        self.item_id_mapping = dict()
        self.user_id_mapping = dict()
        self.item_id = item_id
        self.user_id = user_id
        self.timestamp_id = timestamp_id
        self.dedupe = dedupe
        self.filters = []

    def add_filter(self, _filter: Filter):
        """
        Add a preprocessing filter to be applied before transforming to a InteractionMatrix object.
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
        return df[self.user_id].progress_map(lambda x: self.user_id_mapping.get(x))

    def map_items(self, df):
        logger.debug("Map items")
        if not self.item_id_mapping:
            raise RuntimeError(
                "Item ID Mapping should be fit before attempting to map items"
            )
        return df[self.item_id].progress_map(lambda x: self.item_id_mapping.get(x))

    @property
    def shape(self):
        return (
            max(self.user_id_mapping.values()) + 1,
            max(self.item_id_mapping.values()) + 1,
        )

    def process(self, df: pd.DataFrame) -> InteractionMatrix:
        """
        Process a single DataFrame to a InteractionMatrix object.

        IMPORTANT: If you have multiple DataFrames, use process_many.
        This ensures consistent InteractionMatrix shapes and user/item ID mappings.

        :param df: DataFrame containing user-item interaction pairs.
        :type df: pd.DataFrame
        :return: InteractionMatrix-object containing the DataFrame data.
        :rtype: InteractionMatrix
        """
        return self.process_many(df)[0]

    def process_many(self, *dfs: pd.DataFrame) -> List[InteractionMatrix]:
        """
        Process all DataFrames passed as arguments.
        If your pipeline requires more than one DataFrame,
        pass all of them to a single call of process to guarantee
        that their dimensions will match.

        :return: A list of InteractionMatrix objects in the order the pandas DataFrames were passed in.
        :rtype: List[InteractionMatrix]
        """
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

        data_ms = []

        for df in dfs:
            df.loc[:, DataFramePreprocessor.ITEM_IX] = self.map_items(df)
            df.loc[:, DataFramePreprocessor.USER_IX] = self.map_users(df)

            # Convert input data into internal data objects
            data_m = InteractionMatrix(
                df,
                DataFramePreprocessor.ITEM_IX,
                DataFramePreprocessor.USER_IX,
                timestamp_ix=self.timestamp_id,
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

        self.user_id_mapping = util.rescale_id_space(
            user_ids, id_mapping=self.user_id_mapping
        )
        self.item_id_mapping = util.rescale_id_space(
            item_ids, id_mapping=self.item_id_mapping
        )

    def apply_item_id_mapping(self, df: pd.DataFrame):
        df.loc[:, DataFramePreprocessor.ITEM_IX] = self.map_items(df)
