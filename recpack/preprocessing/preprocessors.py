"""Preprocessors turn data into Recpack internal representations."""

import logging
from typing import List

import pandas as pd

import recpack.preprocessing.util as util
from recpack.data.matrix import InteractionMatrix
from recpack.preprocessing.filters import Filter

from tqdm.auto import tqdm

tqdm.pandas()


logger = logging.getLogger("recpack")


class DataFramePreprocessor:
    """Class to preprocess a Pandas Dataframe and turn it into a InteractionMatrix object.

    Preprocessing has three steps

    - Apply filters to the input data
    - Map user and item identifiers to a consecutive id space.
      Making them usable as indices in a matrix.
    - Construct an InteractionMatrix object

    In order to apply filters they should be added using :meth:`add_filter`.
    The filters and the two other steps get applied during :meth:`process`
    and :meth:`process_many`.

    All ID mappings are stored, so that processing of multiple DataFrames
    will lead to consistent mapped identifiers.

    Example
    ~~~~~~~~~

    This example processes a pandas DataFrame.
    Using filters to

    - Remove duplicates
    - Make sure all users have at least 3 interactions

    ::

        import random
        import pandas as pd
        from recpack.preprocessing.filters import Deduplicate, MinItemsPerUser
        from recpack.preprocessing.preprocessors import DataFramePreprocessor

        # Generate random data
        data = {
            "user": [random.randint(1, 250) for i in range(1000)],
            "item": [random.randint(1, 250) for i in range(1000)],
            "timestamp": [1613736000 + random.randint(1, 3600) for i in range(1000)]
        }
        df = pd.DataFrame.from_dict(data)

        # Construct the processor and add filters
        df_pp = DataFramePreprocessor("item", "user", "timestamp")
        df_pp.add_filter(
            Deduplicate("item", "user", "timestamp")
        )
        df_pp.add_filter(
            MinItemsPerUser(3, "item", "user")
        )

        # apply preprocessing
        im = df_pp.process(df)


    :param item_ix: Column name of the Item ID column
    :type item_ix: str
    :param user_ix: Column name of the User ID column
    :type user_ix: str
    :param timestamp_ix: Column name of the timestamp column.
        If None, no timestamps will be loaded, defaults to None
    :type timestamp_ix: str, optional
    """

    def __init__(self, item_ix, user_ix, timestamp_ix=None):
        self.item_id_mapping = dict()
        self.user_id_mapping = dict()

        self.item_ix = item_ix
        self.user_ix = user_ix
        self.timestamp_ix = timestamp_ix

        self.filters = []

    def add_filter(self, _filter: Filter, index: int = None):
        """Add a preprocessing filter to be applied
        before transforming to a InteractionMatrix object.

        Filters are applied in order, different orderings can lead to different results!

        If the index is specified, the filter is inserted at the specified index.
        Otherwise it is appended.

        :param _filter: The filter to be applied
        :type _filter: Filter
        :param index: The index to insert the filter at,
            None will append the filter. Defaults to None
        :type index: int
        """
        if index is None:
            self.filters.append(_filter)
        else:
            self.filters.insert(index, _filter)

    def _map_users(self, df):
        logger.debug("Map users")
        if not self.user_id_mapping:
            raise RuntimeError(
                "User ID Mapping should be fit before attempting to map users"
            )
        return df[self.user_ix].progress_map(lambda x: self.user_id_mapping.get(x))

    def _map_items(self, df):
        logger.debug("Map items")
        if not self.item_id_mapping:
            raise RuntimeError(
                "Item ID Mapping should be fit before attempting to map items"
            )
        return df[self.item_ix].progress_map(lambda x: self.item_id_mapping.get(x))

    @property
    def shape(self):
        """Shape of the data processed, as `|U| x |I|`"""
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

        :param dfs: Dataframes to process
        :type dfs: pd.DataFrame
        :return: A list of InteractionMatrix objects in the order
            the pandas DataFrames were passed in.
        :rtype: List[InteractionMatrix]
        """

        for index, df in enumerate(dfs):
            logger.debug(f"Processing df {index}")
            logger.debug(f"\tinteractions before preprocess: {len(df.index)}")
            logger.debug(f"\titems before preprocess: {df[self.item_ix].nunique()}")
            logger.debug(f"\tusers before preprocess: {df[self.user_ix].nunique()}")

        for filter in self.filters:
            logger.debug(f"applying filter: {filter}")
            dfs = filter.apply_all(*dfs)
            for index, df in enumerate(dfs):
                logger.debug(f"df {index}")
                logger.debug(f"\tinteractions after filter: {len(df.index)}")
                logger.debug(f"\titems after filter: {df[self.item_ix].nunique()}")
                logger.debug(f"\tusers after filter: {df[self.user_ix].nunique()}")

        for index, df in enumerate(dfs):
            self._update_id_mappings(df)

        interaction_ms = []

        for df in dfs:
            df.loc[:, InteractionMatrix.ITEM_IX] = self._map_items(df)
            df.loc[:, InteractionMatrix.USER_IX] = self._map_users(df)

            # Convert input data into internal data objects
            interaction_m = InteractionMatrix(
                df,
                InteractionMatrix.ITEM_IX,
                InteractionMatrix.USER_IX,
                timestamp_ix=self.timestamp_ix,
                shape=self.shape,
            )

            interaction_ms.append(interaction_m)

        return interaction_ms

    def _update_id_mappings(self, df: pd.DataFrame):
        """
        Update the id mapping so we can combine multiple files
        """

        # Convert user and item ids into a continuous sequence to make
        # training faster and use much less memory.
        item_ids = list(df[self.item_ix].unique())
        user_ids = list(df[self.user_ix].unique())

        self.user_id_mapping = util.rescale_id_space(
            user_ids, id_mapping=self.user_id_mapping
        )
        self.item_id_mapping = util.rescale_id_space(
            item_ids, id_mapping=self.item_id_mapping
        )


class SessionDataFramePreprocessor(DataFramePreprocessor):
    # Not completed yet!
    SESSION_IX = "session_id"

    def __init__(
        self,
        item_ix,
        user_ix,
        timestamp_ix,
        maximal_allowed_gap=2,
    ):
        super().__init__(item_ix, self.SESSION_IX, timestamp_ix)
        self.raw_user_ix = user_ix
        self.maximal_allowed_gap = maximal_allowed_gap

    def process_many(self, *dfs: pd.DataFrame):
        session_dfs = []
        for df in dfs:
            session_dfs.append(self.cut_into_sessions(df))
        return super().process_many(*session_dfs)

    def cut_into_sessions(self, df) -> pd.DataFrame:
        # make a copy to avoid editing in place
        df = df.copy()

        if (
            self.raw_user_ix not in df
            or self.item_ix not in df
            or self.timestamp_ix not in df
        ):
            raise ValueError("One of the element doesn't exist!")

        df = df[[self.raw_user_ix, self.item_ix, self.timestamp_ix]]
        # Sort the dataframe inplace
        df.sort_values([self.raw_user_ix, self.timestamp_ix], inplace=True)
        # By shifting one ahead, we give each row info about the last timestamp and user
        df["last_user"] = df[self.raw_user_ix].shift()
        df["last_timestamp"] = df[self.timestamp_ix].shift()

        # A new session starts if a new user is detected, or if the gap with the previous event is too big
        df["start_of_session"] = (
            df["last_user"].isna()
            | (df[self.raw_user_ix] != df["last_user"])
            | (df[self.timestamp_ix] - df["last_timestamp"] > self.maximal_allowed_gap)
        )

        # Cumsum on a boolean series makes sure that each value is equal to the number of sessions started.
        # this gives values from 1 to ...
        # by reducing by 1 again, we start counting at session 0.
        df[self.SESSION_IX] = df["start_of_session"].cumsum() - 1

        return df[[self.SESSION_IX, self.item_ix, self.timestamp_ix]]
