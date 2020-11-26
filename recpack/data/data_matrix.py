import logging
from typing import List, Set, Tuple, Union, Callable

import pandas as pd
import numpy as np
import scipy.sparse
import operator

from recpack.util import groupby2

logger = logging.getLogger("recpack")


# Default field names
ITEM_IX = "iid"
USER_IX = "uid"
VALUE_IX = "value"
TIMESTAMP_IX = "ts"


class DataM:
    """
    Stores information about interactions between users and items.

    :param df: Dataframe containing user-item interactions. Must contain at least 
               item ids and user ids.
    :param item_ix: Item ids column name
    :param user_ix: User ids column name
    :param value_ix: Interaction values column name
    :param timestamp_ix: Interaction timestamps column name
    :param shape: The desired shape of the matrix, i.e. the number of users and items.
                  If no shape is specified, the number of users will be equal to the 
                  maximum user id plus one, the number of items to the maximum item 
                  id plus one.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        item_ix: str = ITEM_IX,
        user_ix: str = USER_IX,
        value_ix: str = VALUE_IX,
        timestamp_ix: str = TIMESTAMP_IX,
        shape: Tuple[int, int] = None,
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
        """
        All user-item interactions as a sparse matrix of size (users, items).

        Each entry is the sum of interaction values for that user and item. If no 
        interaction values are known, the entry is the total number of interactions 
        between that user and item.

        If there are no interactions between a user and item, the entry is 0.
        """
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
    def dataframe(self) -> pd.DataFrame:
        """
        All interactions as a pandas DataFrame.
        
        The item ids, user ids, interaction values and timestamps are stored in columns 
        `ITEM_IX`, `USER_IX`, `VALUE_IX` and `TIMESTAMP_IX` respectively.
        """
        return self._df.copy()

    @property
    def timestamps(self) -> pd.Series:
        """Timestamps of interactions as a pandas Series, indexed by user and item id."""
        if TIMESTAMP_IX not in self._df:
            raise AttributeError(
                "No timestamp column, so timestamps could not be retrieved"
            )
        index = pd.MultiIndex.from_frame(self._df[[USER_IX, ITEM_IX]])
        return self._df[[TIMESTAMP_IX]].set_index(index)[TIMESTAMP_IX]

    def eliminate_timestamps(self) -> None:
        """Removes all timestamp information, inplace."""
        if TIMESTAMP_IX in self._df:
            self._df.drop(columns=[TIMESTAMP_IX], inplace=True, errors="ignore")

    @property
    def indices(self) -> Tuple[List[int], List[int]]:
        """All user-item combinations that have at least one interaction."""
        return self.values.nonzero()

    @property
    def shape(self) -> Tuple[int, int]:
        """The shape of the matrix, i.e. the number of users and items."""
        return (self.total_users, self.total_items)

    def _apply_mask(self, mask, inplace=False):
        c_df = self._df[mask]
        if inplace:
            self._df = c_df
        else:
            return DataM(c_df, shape=self.shape)

    def timestamps_cmp(
        self, op: Callable, timestamp: float, inplace: bool = False, compare_to: str = None
    ):
        """
        Filter interactions based on timestamp.

        :param op: Comparison operator. Keep only the interactions for which op(t, timestamp) 
                   evaluates to True.
        :param timestamp: Timestamp to compare against in seconds from epoch.
        :param inplace: Modify the data matrix in place. If False, returns a new object.
        :param compare_to: If specified, compares the timestamp to a related interaction time 
                           instead. Must be one of user-min", "user-max", "user-median" or 
                           "user-mean". For example, "user-max" filters all interactions where 
                           op(max(t_1, t_2, ...), timestamp) is True, where t_1, t_2, ... are 
                           all interaction times of that same user.
        """
        logger.debug(f"Performing {op.__name__}(t, timestamp)")
        assert compare_to in [None, "user-min", "user-max", "user-median", "user-mean"]
        if compare_to is None:
            mask = op(self._df[TIMESTAMP_IX], timestamp)
        else:
            compare_to = compare_to.split("-")[1]
            ts_grouped_by_user = self._df.groupby(USER_IX)[TIMESTAMP_IX]
            ts_user = getattr(ts_grouped_by_user, compare_to)()
            mask = op(self._df[USER_IX].map(ts_user), timestamp)
        return self._apply_mask(mask, inplace=inplace)

    def timestamps_gt(self, timestamp: float, inplace: bool = False, compare_to: str = None):
        """Keep only interactions where t > timestamp. See `timestamps_cmp` for more info."""
        return self.timestamps_cmp(operator.gt, timestamp, inplace, compare_to)

    def timestamps_lt(self, timestamp: float, inplace: bool = False, compare_to: str = None):
        """Keep only interactions where t < timestamp. See `timestamps_cmp` for more info."""
        return self.timestamps_cmp(operator.lt, timestamp, inplace, compare_to)

    def timestamps_gte(self, timestamp: float, inplace: bool = False, compare_to: str = None):
        """Keep only interactions where t >= timestamp. See `timestamps_cmp` for more info."""
        return self.timestamps_cmp(operator.ge, timestamp, inplace, compare_to)

    def timestamps_lte(self, timestamp: float, inplace: bool = False, compare_to: str = None):
        """Keep only interactions where t <= timestamp. See `timestamps_cmp` for more info."""
        return self.timestamps_cmp(operator.le, timestamp, inplace, compare_to)

    def users_in(self, U: Union[Set[int], List[int]], inplace=False):
        """Keep only interactions by one of the specified users."""
        logger.debug("Performing users_in comparison")

        mask = self._df[USER_IX].isin(U)

        return self._apply_mask(mask, inplace=inplace)

    def indices_in(self, u_i_lists: Tuple[List[int], List[int]], inplace=False):
        """Keep only interactions between the specified user-item combinations."""
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
    def user_history(self) -> Set[Tuple[int, List[int]]]:
        """The set of all active users and the items they've interacted with."""
        return groupby2(*self.indices)

    @property
    def active_users(self) -> Set[int]:
        """The set of all users with at least one interaction."""
        U, _ = self.indices
        return set(U)

    @property
    def active_user_count(self) -> int:
        """The number of users with at least one interaction."""
        U, _ = self.indices
        return len(set(U))

    @property
    def binary_values(self) -> scipy.sparse.csr_matrix:
        """
        All user-item interactions as a sparse, binary matrix of size (users, items).

        An entry is 1 if there is at least one interaction between that user and item 
        and either:
            - No interaction values are known, or
            - The sum of interaction values for that user and item is strictly positive

        In all other cases the entry is 0.
        """
        values = self.values
        values[values > 0] = 1
        values[values < 0] = 0
        values.eliminate_zeros()
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
