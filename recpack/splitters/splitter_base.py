import logging
from abc import ABC, abstractmethod
from typing import Tuple
import math

import numpy as np
import scipy.sparse

from tqdm.auto import tqdm

from recpack.data.data_matrix import DataM

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

        shape = data.shape

        # We require a full data split.
        assert shape[0] == (len(self.users_in) + len(self.users_out))

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

        nr_users = sp_mat.shape[0]

        users = list(range(0, nr_users))

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

            within_margin = np.isclose(real_perc, self.in_perc, atol=self.error_margin)

            if within_margin:
                logger.debug(f"{self.name} - Iteration {i} - Within margin")
                break
            else:
                logger.debug(f"{self.name} - Iteration {i} - Not within margin")

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

        for u, user_history in tqdm(data.user_history, desc="split user ratings"):
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
    Training data will be all data in the interval [t_alpha, t),
    the test data will be the data in the interval [t, t+t_delta).
    If t_alpha or t_delta are omitted, all data before/after t is used.

    :param t: epoch timestamp to split on
    :type t: int
    :param t_delta: seconds past t to consider as test data (default is None, all data >= t is considered)
    :type t_delta: int
    :param t_alpha: seconds before t to use as training data (default is None -> all data < t is considered)
    :type t_alpha: int
    """

    def __init__(self, t, t_delta=None, t_alpha=None):
        super().__init__()
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

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
        :return: A tuple containing the training, validation and test data objects in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`, class:`recpack.DataM`)
        """

        if self.t_alpha is None:
            # Holds indices where timestamp < t
            tr_data = data.timestamps_lt(self.t)
        else:
            # Holds indices in interval: t-t_alpha =< timestamp < t
            tr_data = data.timestamps_lt(self.t).timestamps_gte(self.t - self.t_alpha)

        if self.t_delta is None:
            # Holds indices where timestamp >= t
            te_data = data.timestamps_gte(self.t)
        else:
            # Get indices where timestamp >= t and timestamp < t + t_delta
            te_data = data.timestamps_gte(self.t).timestamps_lt(self.t + self.t_delta)

        logger.debug(f"{self.name} - Split successful")

        return tr_data, te_data


class LeaveLastOneOutSplitter(Splitter):
    """For each user, the last item is kept as out and the rest as in.
    """

    @property
    def name(self):
        return "leave_last_one_out_splitter"
    
    def split(self, data: DataM) -> Tuple[DataM, DataM]:
        # Get the ranks of interactions, from last to first. 
        ranked_by_ts = data.timestamps.groupby('uid').rank(ascending=False, method='first')
        # the input dataframe is all except the last one
        in_df = ranked_by_ts[ranked_by_ts != 1].reset_index()
        in_indices = list(zip(*in_df.loc[:, ['uid', 'iid']].values))

        # The output dataframe is the last interaction
        out_df = ranked_by_ts[ranked_by_ts == 1].reset_index()
        out_indices = list(zip(*out_df.loc[:, ['uid', 'iid']].values))

        in_ = data.indices_in(in_indices)
        out_ = data.indices_in(out_indices)
        return in_, out_


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

        # users need history, but absence of true labels is allowed (some metrics don't require true labels).
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
