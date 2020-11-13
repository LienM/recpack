import logging
from abc import ABC, abstractmethod
from typing import Tuple
import math

import numpy as np
import scipy.sparse

from tqdm.auto import tqdm

from recpack.data.data_matrix import DataM, USER_IX, ITEM_IX, TIMESTAMP_IX

logger = logging.getLogger("recpack")


class Splitter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def split(self, data):
        pass

    @property
    def name(self):
        return None


class UserSplitter(Splitter):
    def __init__(self, users_in, users_out):
        super().__init__()
        self.users_in = users_in
        self.users_out = users_out

    def name(self):
        return f"user_splitter"

    def split(self, data):
        """
        Splits a user-interaction matrix by the user indices.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
        :return: A tuple containing the training, validation and test data objects in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`, class:`recpack.DataM`)
        """
        tr_data = data.users_in(self.users_in)
        te_data = data.users_in(self.users_out)

        logger.debug(f"{self.name} - Split successful")

        return tr_data, te_data


class StrongGeneralizationSplitter(Splitter):
    """
    Implementation of strong generalization split. All events for a user will always occur only in the same split.

    :param in_perc: The percentage of the data to make training data.
    :type in_perc: `float`

    :param seed: Seed for randomisation in function, set this to a specified value in order to get expected results.
    :type seed: `int`

    :param error_margin: The allowed difference between in_perc and the actual split percentage.
    :type error_margin: `float`
    """

    def __init__(self, in_perc=0.7, seed=None, error_margin=0.01):
        super().__init__()
        self.in_perc = in_perc
        self.out_perc = 1 - in_perc
        self.seed = seed
        self.error_margin = error_margin

    @property
    def name(self):
        return f"strong_generalization_tr_{self.in_perc*100}_te_{(self.out_perc)*100}"

    def split(self, data):
        """
        Splits a user-interaction matrix by randomly shuffling users,
        and filling up the different data slices with the right percentages of data.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
        :return: A tuple containing the training, validation and test data objects in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`, class:`recpack.DataM`)
        """
        sp_mat = data.values

        users = list(set(sp_mat.nonzero()[0]))

        nr_users = len(users)

        # Seed to make testing possible
        if self.seed is not None:
            np.random.seed(self.seed)

        for i in range(0, 5):
            # Try five times
            np.random.shuffle(users)

            tr_cut = int(np.floor(nr_users * self.in_perc))

            users_in = users[:tr_cut]
            users_out = users[tr_cut:]

            total_interactions = sp_mat.nnz
            data_in_cnt = sp_mat[users_in, :].nnz

            real_perc = data_in_cnt / total_interactions

            within_margin = np.isclose(
                real_perc, self.in_perc, atol=self.error_margin)

            if within_margin:
                logger.debug(f"{self.name} - Iteration {i} - Within margin")
                break
            else:
                logger.debug(
                    f"{self.name} - Iteration {i} - Not within margin")

        u_splitter = UserSplitter(users_in, users_out)

        logger.debug(f"{self.name} - Split successful")

        return u_splitter.split(data)


class ItemSplitter(Splitter):
    pass


class InteractionSplitter(Splitter):
    pass


class PercentageInteractionSplitter(Splitter):
    # TODO Add documentation
    def __init__(self, in_perc, seed=None):
        super().__init__()
        self.in_perc = in_perc

        self.seed = seed if seed else 12345

    def split(self, data):
        """
        Split the input data based on the parameters set at construction.
        Returns 2 sparse matrices in and out
        """

        tr_u, tr_i = [], []
        te_u, te_i = [], []

        for u, user_history in tqdm(
                data.user_history, desc="split user ratings"):
            rstate = np.random.RandomState(self.seed + u)

            rstate.shuffle(user_history)
            hist_len = len(user_history)
            cut = int(np.ceil(hist_len * self.in_perc))

            tr_i.extend(user_history[:cut])
            tr_u.extend([u] * len(user_history[:cut]))

            te_i.extend(user_history[cut:])
            te_u.extend([u] * len(user_history[cut:]))

        tr_data = data.indices_in((tr_u, tr_i))
        te_data = data.indices_in((te_u, te_i))

        logger.debug(f"{self.name} - Split successful")

        return tr_data, te_data


class TimestampSplitter(Splitter):
    """
    Use this class if you want to split your data on a timestamp.
    Training data will be all data in the interval [t-t_alpha, t),
    the test data will be the data in the interval [t, t+t_delta).
    If t_alpha or t_delta are omitted, all data before/after t is used.

    :param t: epoch timestamp to split on
    :type t: int
    :param t_delta: seconds past t to consider as test data (default is None, all data >= t is considered)
    :type t_delta: int
    :param t_alpha: seconds before t to use as training data (default is None -> all data < t is considered)
    :type t_alpha: int
    :param split_user: Allow user data to be split. If False, data from the same user is kept together,
                       the time of the most recent action determines which split the user is placed in.
    :type split_user: bool
    """

    def __init__(self, t, t_delta=None, t_alpha=None, split_user=True):
        super().__init__()
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha
        self.split_user = split_user

    @property
    def name(self):
        base_name = f"timed_split_t{self.t}"
        if self.t_delta:
            base_name += f"_t_delta_{self.t_delta}"
        if self.t_alpha:
            base_name += f"_t_alpha_{self.t_alpha}"

        return base_name

    def split(self, data: DataM) -> Tuple[DataM, DataM]:
        """
        Split the input data by using the timestamps provided in the timestamp field of data.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
        :return: A tuple containing the training and test data objects, in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`)
        """

        if self.t_alpha is None:
            # Holds indices where timestamp < t
            tr_data = data.timestamps_lt(self.t, split_user=self.split_user)
        else:
            # Holds indices in interval: t-t_alpha =< timestamp < t
            tr_data = data.timestamps_lt(
                self.t, split_user=self.split_user).timestamps_gte(
                self.t - self.t_alpha, split_user=self.split_user)

        if self.t_delta is None:
            # Holds indices where timestamp >= t
            te_data = data.timestamps_gte(self.t, split_user=self.split_user)
        else:
            # Get indices where timestamp >= t and timestamp < t + t_delta
            te_data = data.timestamps_gte(
                self.t, split_user=self.split_user).timestamps_lt(
                self.t + self.t_delta, split_user=self.split_user)

        logger.debug(f"{self.name} - Split successful")

        return tr_data, te_data


class MostRecentSplitter(Splitter):
    """
    Splits the n most recent actions of each user into a separate set.

    :param n: Number of most recent actions. If negative, all but the |n| earliest 
              actions are split off.
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    @property
    def name(self):
        return f"most_recent_split_n{self.n}"

    def split(self, data: DataM) -> Tuple[DataM, DataM]:
        """
        Returns a data matrix with all but the n most recent actions of each user 
        and a data matrix with the n most recent actions of each user.

        :param data: Data matrix to be split. Must contain timestamps.
        :return: A tuple containing a matrix with all but n most recent and matrix 
                 with n most recent actions of each user.
        """
        df = data.dataframe
        pos = df.groupby(USER_IX)[TIMESTAMP_IX].rank(method="first")
        cnt = df[USER_IX].map(df[USER_IX].value_counts())
        if self.n >= 0:
            mask = pos <= cnt - self.n
        else:
            mask = pos <= -self.n

        logger.debug(f"{self.name} - Split successful")
        return DataM(df[mask], shape=data.shape), DataM(df[~mask], shape=data.shape)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, scipy.sparse.csr_matrix):
        raise ValueError("Matrix given must be of CSR format.")
    csr.data[csr.indptr[row]: csr.indptr[row + 1]] = value


class FoldIterator:
    def __init__(self, data_m_in, data_m_out, batch_size=1000):
        self.data_m_in = data_m_in
        self.data_m_out = data_m_out
        self.batch_size = batch_size
        assert self.batch_size > 0  # Avoid inf loops

        self.data_m_in.eliminate_timestamps()
        self.data_m_out.eliminate_timestamps()

        # users need history, but absence of true labels is allowed (some
        # metrics don't require true labels).
        self.users = list(sorted(set(self.data_m_in.indices[0])))

    def __iter__(self):
        self.batch_generator = batch(self.users, self.batch_size)
        return self

    def __next__(self):
        user_batch = np.array(next(self.batch_generator))
        fold_in = self.data_m_in.values[user_batch, :]
        fold_out = self.data_m_out.values[user_batch, :]

        return fold_in, fold_out, user_batch

    def __len__(self):
        return math.ceil(len(self.users) / self.batch_size)
