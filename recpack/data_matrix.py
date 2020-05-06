from typing import List, Tuple
import pandas as pd
import numpy as np

import scipy.sparse

from recpack.utils import get_logger, groupby2


class DataM:

    item_id = "iid"
    user_id = "uid"
    timestamp_id = "ts"

    def __init__(self, values, timestamps=None):
        self._values = values
        self._timestamps = timestamps
        self.logger = get_logger()

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

        c_values = self.__create_values(
            c_timestamps.reset_index(), self.item_id, self.user_id, self._values.shape
        )

        self.logger.debug("Timestamp comparison done")

        if not inplace:
            return DataM(c_values, c_timestamps)
        else:
            self._timestamps = c_timestamps
            self._values = c_values

    def timestamps_gt(self, timestamp, inplace=False):
        self.logger.debug("Performing t > timestamp")

        def func():
            return self._timestamps > timestamp

        return self._timestamp_comparator(func, inplace=inplace)

    def timestamps_lt(self, timestamp, inplace=False):
        self.logger.debug("Performing t < timestamp")

        def func():
            return self._timestamps < timestamp

        return self._timestamp_comparator(func, inplace=inplace)

    def timestamps_gte(self, timestamp, inplace=False):
        self.logger.debug("Performing t => timestamp")

        def func():
            return self._timestamps >= timestamp

        return self._timestamp_comparator(func, inplace=inplace)

    def timestamps_lte(self, timestamp, inplace=False):
        self.logger.debug("Performing t <= timestamp")

        def func():
            return self._timestamps <= timestamp

        return self._timestamp_comparator(func, inplace=inplace)

    def users_in(self, U, inplace=False):
        self.logger.debug("Performing users_in comparison")

        mask_values = np.ones(len(U))
        I = np.zeros(len(U))

        mask = scipy.sparse.csr_matrix(
            (mask_values, (U, I)), shape=(self._values.shape[0], 1), dtype=np.int32
        )

        c_values = self._values.multiply(mask)
        c_values.eliminate_zeros()

        self.logger.debug("Users_in comparison done")

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
        self.logger.debug("Performing indices_in comparison")

        U, I = u_i_lists

        mask_values = np.ones(len(U))

        mask = scipy.sparse.csr_matrix(
            (mask_values, (U, I)), shape=self._values.shape, dtype=np.int32
        )

        c_values = self._values.multiply(mask)
        c_values.eliminate_zeros()

        self.logger.debug("Indices_in comparison done")

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
        cols, _ = self.indices
        return len(set(cols))

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
        cls, df: pd.DataFrame, item_ix: str, user_ix: str, timestamp_ix=None, shape=None
    ):

        sparse_matrix = DataM.__create_values(df, item_ix, user_ix, shape)

        if timestamp_ix:
            df = df.rename(
                columns={
                    item_ix: cls.item_id,
                    user_ix: cls.user_id,
                    timestamp_ix: cls.timestamp_id,
                }
            )
            timestamps = df.set_index([cls.user_id, cls.item_id])[cls.timestamp_id].sort_index()
        else:
            timestamps = None

        return DataM(sparse_matrix, timestamps)

    @classmethod
    def __create_values(cls, df, item_ix, user_ix, shape):
        num_entries = df.shape[0]
        # Scipy sums up the entries when an index-pair occurs more than once,
        # resulting in the actual counts being stored. Neat!
        values = np.ones(num_entries)

        indices = list(zip(*df.loc[:, [user_ix, item_ix]].values))

        if indices == []:
            indices = [[], []]  # Empty zip does not evaluate right

        if shape is None:
            shape = df[user_ix].max() + 1, df[item_ix].max() + 1
        sparse_matrix = scipy.sparse.csr_matrix(
            (values, indices), shape=shape, dtype=np.int32
        )

        return sparse_matrix


class ShapeError(Exception):
    pass
