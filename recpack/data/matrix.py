import logging
import operator
from typing import List, Set, Tuple, Union, Callable, Any

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


logger = logging.getLogger("recpack")


class DataMatrix:
    pass


class InteractionMatrix(DataMatrix):
    """
    Stores information about interactions between users and items.

    The data is stored in as a dataframe, properties as well as functions
    are provided to access this data in intuitive ways.

    If a user interacted with an example more than once,
    there should be two rows for this user-item pair.

    TODO: add usage example here?

    :param df: Dataframe containing user-item interactions. Must contain at least
               item ids and user ids.
    :param item_ix: Item ids column name
    :param user_ix: User ids column name
    :param timestamp_ix: Interaction timestamps column name
    :param shape: The desired shape of the matrix, i.e. the number of users and items.
                  If no shape is specified, the number of users will be equal to the
                  maximum user id plus one, the number of items to the maximum item
                  id plus one.
    """

    ITEM_IX = "iid"
    USER_IX = "uid"
    TIMESTAMP_IX = "ts"
    INTERACTION_IX = "interactionid"

    def __init__(
        self,
        df: pd.DataFrame,
        item_ix: str,
        user_ix: str,
        timestamp_ix: str = None,
        shape: Tuple[int, int] = None,
    ):

        # Give each interaction a unique id,
        # this will allow selection of specific events
        if InteractionMatrix.INTERACTION_IX in df.columns:
            pass
        else:
            df = df.reset_index().rename(
                columns={"index": InteractionMatrix.INTERACTION_IX}
            )

        col_mapper = {
            item_ix: InteractionMatrix.ITEM_IX,
            user_ix: InteractionMatrix.USER_IX,
        }

        if timestamp_ix is not None:
            col_mapper[timestamp_ix] = InteractionMatrix.TIMESTAMP_IX
            self.has_timestamps = True
        else:
            self.has_timestamps = False

        self._df = df.rename(columns=col_mapper)

        num_users = (
            self._df[InteractionMatrix.USER_IX].max() + 1 if shape is None else shape[0]
        )
        num_items = (
            self._df[InteractionMatrix.ITEM_IX].max() + 1 if shape is None else shape[1]
        )

        self.shape = (num_users, num_items)

    def copy(self):
        """Create a copy of this dataM object.

        :return: Copy of this object
        :rtype: InteractionMatrix
        """
        timestamp_ix = self.TIMESTAMP_IX if self.has_timestamps else None
        return InteractionMatrix(
            self._df.copy(),
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
            timestamp_ix=timestamp_ix,
            shape=self.shape,
        )

    @property
    def values(self) -> csr_matrix:
        """
        All user-item interactions as a sparse matrix of size (|users|, |items|).

        Each entry is the sum of interaction values for that user and item.
        If the value_ix is not present in the dataframe,
        the entry is the total number of interactions between that user and item.

        If there are no interactions between a user and item, the entry is 0.
        """
        values = np.ones(self._df.shape[0])
        indices = self._df[
            [InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX]
        ].values
        indices = indices[:, 0], indices[:, 1]

        matrix = csr_matrix((values, indices), shape=self.shape, dtype=np.int32)
        return matrix

    def get_timestamp(self, interactionid: int) -> int:
        """Return the timestamp of a specific interaction

        :param interactionid: the interaction id in the dataframe
            to fetch the timestamp from.
        :type interactionid: int
        :raises AttributeError: Raised if the object does not have timestamps
        :return: The timestamp of the fetched id
        :rtype: int
        """
        if not self.has_timestamps:
            raise AttributeError(
                "No timestamp column, so timestamps could not be retrieved"
            )
        return self._df.loc[
            self._df[InteractionMatrix.INTERACTION_IX] == interactionid,
            InteractionMatrix.TIMESTAMP_IX,
        ].values[0]

    # TODO This doesn't work with duplicates. FIX!
    @property
    def timestamps(self) -> pd.Series:
        """Timestamps of interactions as a pandas Series, indexed by user and item id.

        :raises AttributeError: If there is no timestamp column
        :return: Series of interactions with multi index on user, item ids
        :rtype: pd.Series
        """
        if not self.has_timestamps:
            raise AttributeError(
                "No timestamp column, so timestamps could not be retrieved"
            )
        index = pd.MultiIndex.from_frame(
            self._df[[InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX]]
        )
        return self._df[[InteractionMatrix.TIMESTAMP_IX]].set_index(index)[
            InteractionMatrix.TIMESTAMP_IX
        ]

    def eliminate_timestamps(self, inplace: bool = False):
        """
        Remove all timestamp information.

        :type inplace: bool
        :param inplace: Modify the data matrix in place. If False, returns a new object.
        """
        interaction_m = self if inplace else self.copy()

        if InteractionMatrix.TIMESTAMP_IX in interaction_m._df:
            interaction_m._df.drop(
                columns=[InteractionMatrix.TIMESTAMP_IX],
                inplace=True,
                errors="ignore",
            )

        # Set boolean to False
        interaction_m.has_timestamps = False
        return None if inplace else interaction_m

    @property
    def indices(self) -> Tuple[List[int], List[int]]:
        """
        Return all user-item combinations that have at least one interaction.

        Returns a tuple of a list of user indices, and a list of item indices
        """
        return self.values.nonzero()

    def _apply_mask(self, mask, inplace=False):
        interaction_m = self if inplace else self.copy()

        c_df = interaction_m._df[mask]

        interaction_m._df = c_df
        return None if inplace else interaction_m

    def timestamps_cmp(self, op: Callable, timestamp: float, inplace: bool = False):
        """
        Filter interactions based on timestamp.

        :param op: Comparison operator.
            Keep only interactions for which op(t, timestamp) is True.
        :param timestamp: Timestamp to compare against in seconds from epoch.
        :param inplace: Modify the data matrix in place. If False, returns a new object.
        """
        logger.debug(f"Performing {op.__name__}(t, timestamp)")

        mask = op(self._df[InteractionMatrix.TIMESTAMP_IX], timestamp)

        return self._apply_mask(mask, inplace=inplace)

    def timestamps_gt(self, timestamp: float, inplace: bool = False):
        """select interactions after a given timestamp.

        Performs timestamps_cmp operation to select rows for which t > timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
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
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
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
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
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
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self.timestamps_cmp(operator.le, timestamp, inplace)

    def users_in(self, U: Union[Set[int], List[int]], inplace=False):
        """Keep only interactions by one of the specified users.

        :param U: A Set or List of users to select the interactions from.
        :type U: Union[Set[int], List[int]]
        :param inplace: Apply the selection in place or not, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        logger.debug("Performing users_in comparison")

        mask = self._df[InteractionMatrix.USER_IX].isin(U)

        return self._apply_mask(mask, inplace=inplace)

    def interactions_in(self, interaction_ids: List[int], inplace: bool = False):
        """Select the interactions by their interaction ids

        :param interaction_ids: A list of interaction ids
        :type interaction_ids: List[int]
        :param inplace: Apply the selection in place,
            or return a new InteractionMatrix object, defaults to False
        :type inplace: bool, optional
        :return: None if inplace, otherwise new InteractionMatrix
            object with the selected interactions
        :rtype: Union[None, InteractionMatrix]
        """
        logger.debug("Performing interactions_in comparison")

        mask = self._df[InteractionMatrix.INTERACTION_IX].isin(interaction_ids)

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
            otherwise a new InteractionMatrix object with the selection of events.
        :rtype: Union[InteractionMatrix, None]
        """
        logger.debug("Performing indices_in comparison")

        interaction_m = self if inplace else self.copy()

        # Data is temporarily duplicated across a MultiIndex and
        #   the [USER_IX, ITEM_IX] columns for fast multi-indexing.
        # This index can be dropped safely,
        #   as the data is still there in the original columns.
        index = pd.MultiIndex.from_frame(
            interaction_m._df[[InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX]]
        )
        tuples = list(zip(*u_i_lists))
        c_df = interaction_m._df.set_index(index)
        c_df = c_df.loc[tuples]
        c_df.reset_index(drop=True, inplace=True)

        interaction_m._df = c_df

        return None if inplace else interaction_m

    @property
    def binary_user_history(self):
        """The unique interactions per user

        :yield: tuples of user, list of distinct items the user interacted with.
        :rtype: List[Tuple[int, List[int]]]
        """
        df = self._df.drop_duplicates(
            [InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX]
        )
        for uid, user_history in df.groupby(InteractionMatrix.USER_IX):
            yield (uid, user_history[InteractionMatrix.ITEM_IX].values)

    @property
    def interaction_history(self):
        """The interactions per user

        :yield: tuples of user, list of interaction ids
            for each interaction of the user.
        :rtype: List[Tuple[int, List[int]]]
        """
        for uid, user_history in self._df.groupby("uid"):
            yield (uid, user_history[InteractionMatrix.INTERACTION_IX].values)

    @property
    def sorted_interaction_history(self):
        """The interactions per user, sorted by timestamp (ascending).

        :raises AttributeError: If there is no timestamp column can't sort
        :yield: tuple of user id, list of interaction ids sorted by timestamp
        :rtype: List[Tuple[int, List[int]]]
        """
        if not self.has_timestamps:
            raise AttributeError(
                "InteractionMatrix is missing timestamps. "
                "Cannot sort user history without timestamps."
            )
        for uid, user_history in self._df.groupby("uid"):
            yield (
                uid,
                user_history.sort_values("ts", ascending=True)[
                    InteractionMatrix.INTERACTION_IX
                ].values,
            )

    @property
    def active_users(self) -> Set[int]:
        """The set of all users with at least one interaction."""
        U, _ = self.indices
        return set(U)

    @property
    def num_active_users(self) -> int:
        """The number of users with at least one interaction."""
        U, _ = self.indices
        return len(set(U))

    @property
    def num_interactions(self) -> int:
        """The total number of interactions."""
        return len(self._df)

    @property
    def binary_values(self) -> csr_matrix:
        """
        All user-item interactions as a sparse, binary matrix of size (users, items).

        An entry is 1 if there is at least one interaction between that user and item
        and either:
            - The value_ix is not present in the dataframe,
            - The sum of interaction values for that user and item is strictly positive

        In all other cases the entry is 0.
        """
        return to_binary(self.values)

    @classmethod
    def from_csr_matrix(cls, X: csr_matrix) -> "InteractionMatrix":
        """
        Create an InteractionMatrix from a csr_matrix containing interactions.
        WARNING: No timestamps can be passed this way!

        :return: [description]
        :rtype: [type]
        """
        # First extract easy interactions, only one occurence.
        uids, iids = (X == 1).nonzero()

        # Next extract multiple interactions for a user-item pair.
        multiple_uids, multiple_iids = (X > 1).nonzero()

        for uid, iid in zip(multiple_uids, multiple_iids):
            interaction_cnt = X[uid, iid]

            uids = np.append(uids, interaction_cnt * [uid])
            iids = np.append(iids, interaction_cnt * [iid])
            # iids.extend(interaction_cnt * [iid])

        df = pd.DataFrame({cls.USER_IX: uids, cls.ITEM_IX: iids})

        return InteractionMatrix(df, cls.ITEM_IX, cls.USER_IX, shape=X.shape)


"""
Conversion and validation of the various matrix data types supported by recpack.

In this module the Matrix type is defined, as the union of the InteractionMatrix object,
and csr_matrix, the typically used sparse represenation.

This allows you to use the classes that support Matrix as parameter type
to be used without the use of the InteractionMatrix object.
"""
Matrix = Union[InteractionMatrix, csr_matrix]
_supported_types = Matrix.__args__  # type: ignore


def to_csr_matrix(
    X: Union[Matrix, Tuple[Matrix, ...]], binary: bool = False
) -> csr_matrix:
    """
    Convert a matrix-like object to a scipy csr_matrix.

    :param X: Matrix-like object or tuple of objects to convert
    :param binary: Ensure matrix is binary, sets non-zero values to 1 if not
    :raises: UnsupportedTypeError
    """
    if isinstance(X, (tuple, list)):
        return type(X)(to_csr_matrix(x, binary=binary) for x in X)
    if isinstance(X, csr_matrix):
        res = X
    elif isinstance(X, InteractionMatrix):
        res = X.values
    else:
        raise UnsupportedTypeError(X)
    return to_binary(res) if binary else res


def to_binary(X: csr_matrix) -> csr_matrix:
    """
    Converts a matrix to binary by setting all non-zero values to 1.
    """
    X_binary = X.astype(bool).astype(X.dtype)

    return X_binary


def _is_supported(t: Any) -> bool:
    """
    Returns whether a given matrix type is supported by recpack.
    """
    if not isinstance(t, type):
        t = type(t)
    return issubclass(t, _supported_types)


class UnsupportedTypeError(Exception):
    """
    Raised when a matrix of type not supported by recpack is received.

    :param X: The matrix object received
    """

    def __init__(self, X: Any):
        assert not _is_supported(X)
        super().__init__(
            "Recpack only supports matrix types {}. Received {}.".format(
                ", ".join(t.__name__ for t in _supported_types), type(X).__name__
            )
        )
