# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

"""Preprocessors turn data into RecPack internal representations."""
import logging
from typing import List, Optional

import pandas as pd

import recpack.preprocessing.util as util
from recpack.matrix import InteractionMatrix
from recpack.preprocessing.filters import Filter

from tqdm.auto import tqdm

tqdm.pandas()


logger = logging.getLogger("recpack")


class DataFramePreprocessor:
    """Class to preprocess a Pandas Dataframe and turn it into a InteractionMatrix object.

    Preprocessing has three steps

    - Apply filters to the input data
    - Map user and item identifiers to a consecutive id space.
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
        self._item_id_mapping = dict()
        self._user_id_mapping = dict()

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
        :param index: Index at which to insert the filter. Follows the list.insert behaviour,
            None (and values larger than maximal index) will append (default behaviour),
            0 will prepend,
            -1 will insert the item at the second to last position.
        :type index: int, optional
        """
        if index is None:
            self.filters.append(_filter)
        else:
            self.filters.insert(index, _filter)

    def _map_users(self, df):
        logger.debug("Map users")
        if not self._user_id_mapping:
            raise RuntimeError("User ID Mapping should be fit before attempting to map users")
        return df[self.user_ix].progress_map(lambda x: self._user_id_mapping.get(x))

    def _map_items(self, df):
        logger.debug("Map items")
        if not self._item_id_mapping:
            raise RuntimeError("Item ID Mapping should be fit before attempting to map items")
        return df[self.item_ix].progress_map(lambda x: self._item_id_mapping.get(x))

    @property
    def shape(self):
        """Shape of the data processed, as `|U| x |I|`"""
        return (
            max(self._user_id_mapping.values()) + 1,
            max(self._item_id_mapping.values()) + 1,
        )

    def process(self, df: pd.DataFrame) -> InteractionMatrix:
        """Process a single DataFrame to a InteractionMatrix object.

        IMPORTANT: If you have multiple DataFrames, use process_many.
        This ensures consistent InteractionMatrix shapes and user/item ID mappings.

        :param df: DataFrame containing user-item interaction pairs.
        :type df: pd.DataFrame
        :return: InteractionMatrix-object containing the DataFrame data.
        :rtype: InteractionMatrix
        """
        return self.process_many(df)[0]

    def process_many(self, *dfs: pd.DataFrame) -> List[InteractionMatrix]:
        """Process all DataFrames passed as arguments.

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
            df = df.copy()
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

        self._user_id_mapping = util.rescale_id_space(user_ids, id_mapping=self._user_id_mapping)
        self._item_id_mapping = util.rescale_id_space(item_ids, id_mapping=self._item_id_mapping)

    @property
    def item_id_mapping(self) -> pd.DataFrame:
        """Pandas DataFrame containing mapping from original item IDs to internal (consecutive) item IDs as columns."""
        return pd.DataFrame.from_records(
            list(self._item_id_mapping.items()), columns=[self.item_ix, InteractionMatrix.ITEM_IX]
        )

    @property
    def user_id_mapping(self) -> pd.DataFrame:
        """Pandas DataFrame containing mapping from original user IDs to internal (consecutive) user IDs as columns."""
        return pd.DataFrame.from_records(
            list(self._user_id_mapping.items()), columns=[self.user_ix, InteractionMatrix.USER_IX]
        )


class SessionDataFramePreprocessor(DataFramePreprocessor):
    """Class to preprocess a Pandas Dataframe and turn it into a InteractionMatrix object.
    User interaction histories are split into sessions.

    Preprocessing has four steps

    - Cut user histories into sessions based on the maximal allowed gap in seconds
    - Apply filters to the input data
    - Map session and item identifiers to a consecutive id space.
    - Construct an InteractionMatrix object (where each user will represent a session)

    In order to apply filters they should be added using :meth:`add_filter`.
    The filters and the two other steps get applied during :meth:`process`
    and :meth:`process_many`.

    All ID mappings are stored, so that processing of multiple DataFrames
    will lead to consistently mapped identifiers.

    .. note::
        When processing multiple DataFrames all DataFrames are processed together.
        This is because events in one can bridge a gap between events in another DataFrame.

    Example
    ~~~~~~~~~

    This example processes a pandas DataFrame.
    Using filters to

    - Remove duplicates
    - Make sure all sessions contain at least 3 interactions

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
        df_pp = SessionDataFramePreprocessor("item", "user", "timestamp")
        df_pp.add_filter(
            Deduplicate("item", "user", "timestamp")
        )
        # This will now function as a min items per session filter.
        df_pp.add_filter(
            MinItemsPerUser(3, "item", "user")
        )

        # Apply preprocessing
        im = df_pp.process(df)


    :param item_ix: Column name of the Item ID column
    :type item_ix: str
    :param user_ix: Column name of the User ID column
    :type user_ix: str
    :param timestamp_ix: Column name of the timestamp column.
    :type timestamp_ix: str
    :param max_seconds_idle: If there are more than `max_seconds_idle`
        between consecutive events, a new session is created.
        Defaults to 30 * 60 (30 minutes)
    :type max_seconds_idle: int, optional
    """

    SESSION_IX = "session_id"

    def __init__(
        self,
        item_ix: str,
        user_ix: str,
        timestamp_ix: str,
        max_seconds_idle: Optional[int] = 30 * 60,
    ):
        super().__init__(item_ix, self.SESSION_IX, timestamp_ix)
        self.raw_user_ix = user_ix
        self.max_seconds_idle = max_seconds_idle

    def process_many(self, *dfs: pd.DataFrame):
        """Process all DataFrames passed as arguments.

        :param dfs: Dataframes to process
        :type dfs: pd.DataFrame
        :return: A list of InteractionMatrix objects in the order
            the pandas DataFrames were passed in.
        :rtype: List[InteractionMatrix]
        """

        # In order to make sure that session_ids are correct, we will concatenate the DataFrames,
        # split this DataFrame into sessions, and then split into the original DataFrames again.
        num_dfs = len(dfs)
        # Concatenate DataFrames and add a unique index value
        # for every original DataFrame using `keys`
        full_df = pd.concat(dfs, keys=range(0, num_dfs))

        # Check if all required columns are present
        missing_cols = {self.raw_user_ix, self.item_ix, self.timestamp_ix}.difference(full_df.columns)

        if missing_cols:
            raise KeyError(
                f"The SessionDataFrameProcessor is missing columns "
                f"{missing_cols} in one or more of the DataFrames to process."
            )

        full_df = full_df[[self.raw_user_ix, self.item_ix, self.timestamp_ix]]
        # Sort the DataFrame
        full_df = full_df.sort_values([self.raw_user_ix, self.timestamp_ix])
        # Shift users and timestamps by one, so that every row contains
        # the current and previous user and timestamp.
        full_df[["previous_user", "previous_timestamp"]] = full_df[[self.raw_user_ix, self.timestamp_ix]].shift(
            periods=1, axis=0
        )

        # Check if any of the conditions that trigger the start of a new session is met.
        # Transform boolean values to integer values.
        full_df["start_of_session"] = (
            full_df["previous_user"].isna()
            | (full_df[self.raw_user_ix] != full_df["previous_user"])
            | (full_df[self.timestamp_ix] - full_df["previous_timestamp"] > self.max_seconds_idle)
        ).astype(int)

        # Apply cumsum to the "start_of_session" column so that all
        # rows within the same session are assigned the same cumsum value.
        full_df[self.SESSION_IX] = full_df["start_of_session"].cumsum()

        session_df = full_df[[self.SESSION_IX, self.item_ix, self.timestamp_ix]].sort_index()

        # Separate original DataFrames by grouping on the DataFrame index
        # And passing these groups to the super()-call
        grouped = session_df.groupby(level=0)
        return super().process_many(*[grouped.get_group(i) for i in grouped.groups.keys()])
