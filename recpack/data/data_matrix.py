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

    The data is stored in as a dataframe, properties as well as functions
    are provided to access this data in intuitive ways.

    TODO: add usage example here?

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

        Each entry is the sum of interaction values for that user and item.
        If the value_ix is not present in the dataframe,
        the entry is the total number of interactions between that user and item.

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

        TODO: the names of these fields are not rendered as the the defaults they are.
        """
        return self._df.copy()

    @property
    def timestamps(self) -> pd.Series:
        """
        Timestamps of interactions as a pandas Series, indexed by user and item id.
        """
        if TIMESTAMP_IX not in self._df:
            raise AttributeError(
                "No timestamp column, so timestamps could not be retrieved"
            )
        index = pd.MultiIndex.from_frame(self._df[[USER_IX, ITEM_IX]])
        return self._df[[TIMESTAMP_IX]].set_index(index)[TIMESTAMP_IX]

    def eliminate_timestamps(self, inplace: bool = False):
        """
        Remove all timestamp information.

        :type inplace: bool
        :param inplace: Modify the data matrix in place. If False, returns a new object.
        """
        df = self._df
        if TIMESTAMP_IX in df:
            df = df.drop(columns=[TIMESTAMP_IX], inplace=inplace, errors="ignore")
        return None if inplace else DataM(df, shape=self.shape)

    @property
    def indices(self) -> Tuple[List[int], List[int]]:
        """
        Return all user-item combinations that have at least one interaction.

        Returns a tuple of a list of user indices, and a list of item indices
        """
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

    def timestamps_cmp(self, op: Callable, timestamp: float, inplace: bool = False):
        """
        Filter interactions based on timestamp.

        :param op: Comparison operator.
            Keep only interactions for which op(t, timestamp) is True.
        :param timestamp: Timestamp to compare against in seconds from epoch.
        :param inplace: Modify the data matrix in place. If False, returns a new object.
        """
        logger.debug(f"Performing {op.__name__}(t, timestamp)")

        mask = op(self._df[TIMESTAMP_IX], timestamp)

        return self._apply_mask(mask, inplace=inplace)

    def timestamps_gt(self, timestamp: float, inplace: bool = False):
        """select interactions after a given timestamp.

        Performs timestamps_cmp operation to select rows for which t > timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new DataM object
        :rtype: Union[DataM, None]
        """
        return self.timestamps_cmp(operator.gt, timestamp, inplace)

    def timestamps_lt(self, timestamp: float, inplace: bool = False):
        """select interactions up to a given timestamp.

        Performs timestamps_cmp operation to select rows for which t < timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new DataM object
        :rtype: Union[DataM, None]
        """
        return self.timestamps_cmp(operator.lt, timestamp, inplace)

    def timestamps_gte(self, timestamp: float, inplace: bool = False):
        """select interactions after and including a given timestamp.

        Performs timestamps_cmp operation to select rows for which t >= timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new DataM object
        :rtype: Union[DataM, None]
        """
        return self.timestamps_cmp(operator.ge, timestamp, inplace)

    def timestamps_lte(self, timestamp: float, inplace: bool = False):
        """select interactions up to and including a given timestamp.

        Performs timestamps_cmp operation to select rows for which t <= timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new DataM object
        :rtype: Union[DataM, None]
        """
        return self.timestamps_cmp(operator.le, timestamp, inplace)

    def users_in(self, U: Union[Set[int], List[int]], inplace=False):
        """Keep only interactions by one of the specified users.

        :param U: A Set or List of users to select the interactions from.
        :type U: Union[Set[int], List[int]]
        :param inplace: Apply the selection in place or not, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new DataM object
        :rtype: Union[DataM, None]
        """
        logger.debug("Performing users_in comparison")

        mask = self._df[USER_IX].isin(U)

        return self._apply_mask(mask, inplace=inplace)

    def indices_in(self, u_i_lists: Tuple[List[int], List[int]], inplace=False):
        """Select interactions between the specified user-item combinations.

        :param u_i_lists: two lists as a tuple, the first list are the indices of users,
                    and the second are indices of items,
                    both should be of the same length.
        :type u_i_lists: Tuple[List[int], List[int]]
        :param inplace: Apply the selection in place to the object,
                            defaults to False
        :type inplace: bool, optional
        :return: None if inplace is True,
            otherwise a new DataM object with the selection of events.
        :rtype: Union[DataM, None]
        """
        logger.debug("Performing indices_in comparison")

        # Data is temporarily duplicated across a MultiIndex and
        #   the [USER_IX, ITEM_IX] columns for fast multi-indexing.
        # This index can be dropped safely,
        #   as the data is still there in the original columns.
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
    def interaction_count(self) -> int:
        """The total number of interactions."""
        return len(self._df)

    @property
    def binary_values(self) -> scipy.sparse.csr_matrix:
        """
        All user-item interactions as a sparse, binary matrix of size (users, items).

        An entry is 1 if there is at least one interaction between that user and item
        and either:
            - The value_ix is not present in the dataframe,
            - The sum of interaction values for that user and item is strictly positive

        In all other cases the entry is 0.
        """
        values = self.values
        values[values > 0] = 1
        values[values < 0] = 0
        values.eliminate_zeros()
        return values

    def copy(self):
        """Create a copy of this dataM object.

        :return: Copy of this object
        :rtype: DataM
        """
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
        """Create a DataM object based on a dataframe

        .. deprecated:: 0.0.1
            This function will be removed in 1.0.0,
            the functionality is replaced by the __init__ operation.

        """
        return DataM(df, item_ix, user_ix, value_ix, timestamp_ix, shape)
