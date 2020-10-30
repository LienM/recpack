import logging
from typing import List, Set, Tuple, Union

import pandas as pd
import numpy as np

import scipy.sparse

from recpack.util import groupby2

logger = logging.getLogger("recpack")


# Default field names
ITEM_IX = "iid"
USER_IX = "uid"
VALUE_IX = "value"
TIMESTAMP_IX = "ts"


class DataM:
    def __init__(
        self,
        df,
        item_ix=ITEM_IX,
        user_ix=USER_IX,
        value_ix=VALUE_IX,
        timestamp_ix=TIMESTAMP_IX,
        shape=None,
    ):
        col_mapper = {
            item_ix: ITEM_IX,
            user_ix: USER_IX,
        }
        if timestamp_ix:
            col_mapper[timestamp_ix] = TIMESTAMP_IX
        if value_ix:
            col_mapper[value_ix] = VALUE_IX
        self._df = df.rename(columns=col_mapper)
        self.total_users = self._df[USER_IX].max() + 1 if shape is None else shape[0]
        self.total_items = self._df[ITEM_IX].max() + 1 if shape is None else shape[1]

    @property
    def values(self) -> scipy.sparse.csr_matrix:

        if VALUE_IX in self._df:
            values = self._df[VALUE_IX]
        else:
            values = np.ones(self._df.shape[0])
        indices = self._df[[USER_IX, ITEM_IX]].values
        indices = indices[:, 0], indices[:, 1]

        matrix = scipy.sparse.csr_matrix(
            (values, indices), shape=self.shape, dtype=values.dtype
        )
        return matrix

    @property
    def timestamps(self) -> pd.Series:
        if TIMESTAMP_IX not in self._df:
            raise AttributeError(
                "No timestamp column, so timestamps could not be retrieved"
            )
        index = pd.MultiIndex.from_frame(self._df[[USER_IX, ITEM_IX]])
        return self._df[[TIMESTAMP_IX]].set_index(index)[TIMESTAMP_IX]

    def eliminate_timestamps(self):
        if TIMESTAMP_IX in self._df:
            self._df.drop(columns=[TIMESTAMP_IX], inplace=True, errors="ignore")

    @property
    def indices(self) -> Tuple[List[int], List[int]]:
        return self.values.nonzero()

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.total_users, self.total_items)

    def _apply_mask(self, mask, inplace=False):
        c_df = self._df[mask]
        if inplace:
            self._df = c_df
        else:
            return DataM(c_df, shape=self.shape)

    def timestamps_gt(self, timestamp, inplace=False):
        logger.debug("Performing t > timestamp")

        mask = self._df[TIMESTAMP_IX] > timestamp
        return self._apply_mask(mask, inplace=inplace)

    def timestamps_lt(self, timestamp, inplace=False):
        logger.debug("Performing t < timestamp")

        mask = self._df[TIMESTAMP_IX] < timestamp
        return self._apply_mask(mask, inplace=inplace)

    def timestamps_gte(self, timestamp, inplace=False):
        logger.debug("Performing t => timestamp")

        mask = self._df[TIMESTAMP_IX] >= timestamp
        return self._apply_mask(mask, inplace=inplace)

    def timestamps_lte(self, timestamp, inplace=False):
        logger.debug("Performing t <= timestamp")

        mask = self._df[TIMESTAMP_IX] <= timestamp
        return self._apply_mask(mask, inplace=inplace)

    def users_in(self, U: Union[Set[int], List[int]], inplace=False):
        logger.debug("Performing users_in comparison")

        mask = self._df[USER_IX].isin(U)

        return self._apply_mask(mask, inplace=inplace)

    def indices_in(self, u_i_lists, inplace=False):
        logger.debug("Performing indices_in comparison")

        # Data is temporarily duplicated across a MultiIndex and the [USER_IX, ITEM_IX] columns for fast multi-indexing.
        # This index can be dropped safely, as the data is still there in the original columns. 
        index = pd.MultiIndex.from_frame(self._df[[USER_IX, ITEM_IX]])
        tuples = list(zip(*u_i_lists))
        c_df = self._df.set_index(index)
        c_df = c_df.loc[tuples]
        c_df.reset_index(drop=True, inplace=True)

        if not inplace:
            m = DataM(c_df, shape=self.shape)
            return m
        else:
            self._df = c_df

    @property
    def user_history(self):
        return groupby2(*self.indices)

    @property
    def active_users(self):
        U, _ = self.indices
        return set(U)

    @property
    def active_user_count(self):
        U, _ = self.indices
        return len(set(U))

    @property
    def binary_values(self):
        values = self.values
        values[values > 0] = 1
        return values

    def copy(self):
        return DataM(self._df.copy(), shape=self.shape)

    @classmethod
    def create_from_dataframe(
        cls,
        df: pd.DataFrame,
        item_ix: str,
        user_ix: str,
        value_ix: str = None,
        timestamp_ix=None,
        shape=None,
    ):
        return DataM(df, item_ix, user_ix, value_ix, timestamp_ix, shape)
