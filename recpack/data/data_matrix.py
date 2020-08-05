import logging
from typing import List, Tuple

import pandas as pd
import numpy as np

import scipy.sparse

from recpack.util import groupby2, df_to_sparse


logger = logging.getLogger("recpack")


class DataM:

    item_id = "iid"
    user_id = "uid"
    timestamp_id = "ts"

    def __init__(self, values, timestamps=None):
        self._values = values
        self._timestamps = timestamps

    @property
    def values(self) -> scipy.sparse.csr_matrix:
        return self._values  # (user_1, item_1) -> 2

    @property
    def timestamps(self) -> pd.Series:
        if self._timestamps is None:
            raise AttributeError("timestamps is None, and should not be used")
        return self._timestamps  # (user_1, item_1) -> {1000, 1002}

    def eliminate_timestamps(self):
        self._timestamps = None

    @property
    def indices(self) -> Tuple[List[int], List[int]]:
        return self._values.nonzero()

    @property
    def shape(self) -> Tuple[int, int]:
        if self._values is not None:
            return self._values.shape
        else:
            return None

    def _timestamp_comparator(self, func, inplace=False):

        c_timestamps = self._timestamps[func()]

        c_values = df_to_sparse(
            c_timestamps.reset_index(),
            self.item_id,
            self.user_id,
            shape=self._values.shape,
        )

        logger.debug("Timestamp comparison done")

        if not inplace:
            return DataM(c_values, c_timestamps)
        else:
            self._timestamps = c_timestamps
            self._values = c_values

    def timestamps_gt(self, timestamp, inplace=False):
        logger.debug("Performing t > timestamp")

        def func():
            return self._timestamps > timestamp

        return self._timestamp_comparator(func, inplace=inplace)

    def timestamps_lt(self, timestamp, inplace=False):
        logger.debug("Performing t < timestamp")

        def func():
            return self._timestamps < timestamp

        return self._timestamp_comparator(func, inplace=inplace)

    def timestamps_gte(self, timestamp, inplace=False):
        logger.debug("Performing t => timestamp")

        def func():
            return self._timestamps >= timestamp

        return self._timestamp_comparator(func, inplace=inplace)

    def timestamps_lte(self, timestamp, inplace=False):
        logger.debug("Performing t <= timestamp")

        def func():
            return self._timestamps <= timestamp

        return self._timestamp_comparator(func, inplace=inplace)

    def users_in(self, U, inplace=False):
        logger.debug("Performing users_in comparison")

        mask_values = np.ones(len(U))
        I = np.zeros(len(U))

        mask = scipy.sparse.csr_matrix(
            (mask_values, (U, I)), shape=(self._values.shape[0], 1), dtype=np.int32
        )

        c_values = self._values.multiply(mask)
        c_values.eliminate_zeros()

        logger.debug("Users_in comparison done")

        if self._timestamps is None:
            c_timestamps = None
        else:
            u_i_pairs = zip(*c_values.nonzero())
            c_timestamps = self._timestamps.loc[u_i_pairs]

        if not inplace:
            return DataM(c_values, c_timestamps)
        else:
            self._timestamps = c_timestamps
            self._values = c_values

    def indices_in(self, u_i_lists, inplace=False):
        logger.debug("Performing indices_in comparison")

        U, I = u_i_lists

        mask_values = np.ones(len(U))

        mask = scipy.sparse.csr_matrix(
            (mask_values, (U, I)), shape=self._values.shape, dtype=np.int32
        )

        c_values = self._values.multiply(mask)
        c_values.eliminate_zeros()

        logger.debug("Indices_in comparison done")

        if self._timestamps is None:
            c_timestamps = None
        else:
            u_i_pairs = zip(*(u_i_lists))
            c_timestamps = self._timestamps.loc[u_i_pairs]

        if not inplace:
            return DataM(c_values, c_timestamps)
        else:
            self._timestamps = c_timestamps
            self._values = c_values

    @property
    def user_history(self):
        return groupby2(*self.indices)

    @property
    def active_user_count(self):
        U, _ = self.indices
        return len(set(U))

    @property
    def binary_values(self):
        indices = self.indices
        # (user_1, item_1) -> 1

        values = np.ones(len(indices[0]))
        return scipy.sparse.csr_matrix(
            (values, indices), shape=self.shape, dtype=np.int32
        )

    def copy(self):
        c_values = self._values.copy()
        c_timestamps = None if self._timestamps is None else self._timestamps.copy()

        return DataM(c_values, c_timestamps)

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

        sparse_matrix = df_to_sparse(df, item_ix, user_ix, value_ix, shape)

        if timestamp_ix:
            df = df.rename(
                columns={
                    item_ix: cls.item_id,
                    user_ix: cls.user_id,
                    timestamp_ix: cls.timestamp_id,
                }
            )
            timestamps = df.set_index([cls.user_id, cls.item_id])[
                cls.timestamp_id
            ].sort_index()
        else:
            timestamps = None

        return DataM(sparse_matrix, timestamps)


class ShapeError(Exception):
    pass
