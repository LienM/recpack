from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import scipy.sparse

from recpack.data_matrix import DataM


def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, scipy.sparse.csr_matrix):
        raise ValueError("Matrix given must be of CSR format.")
    csr.data[csr.indptr[row]: csr.indptr[row + 1]] = value


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
        sp_mat = data.values

        # We require a full data split.
        assert shape[0] == (len(self.users_in) + len(self.users_out))

        U, I = sp_mat.nonzero()

        tr_u, tr_i = zip(*filter(lambda x: x[0] in self.users_in, zip(U, I)))
        te_u, te_i = zip(*filter(lambda x: x[0] in self.users_out, zip(U, I)))

        tr_data = data.indices_in((tr_u, tr_i))
        te_data = data.indices_in((te_u, te_i))

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

        for _ in range(0, 5):
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
                continue

        u_splitter = UserSplitter(users_in, users_out)

        return u_splitter.split(data)


class ItemSplitter(Splitter):
    pass


class InteractionSplitter(Splitter):
    pass


class PercentageInteractionSplitter(Splitter):
    # TODO Add documentation
    def __init__(self, in_perc, seed=None):
        self.in_perc = in_perc

        self.seed = seed if seed else 12345

    def split(self, data):
        """
        Split the input data based on the parameters set at construction.
        Returns 2 sparse matrices in and out
        """

        sp_mat = data.values

        users = sp_mat.shape[0]

        tr_u, tr_i = [], []
        te_u, te_i = [], []

        for u in range(0, users):
            _, user_history = sp_mat[u, :].nonzero()

            rstate = np.random.RandomState(self.seed)
            rstate.shuffle(user_history)
            hist_len = len(user_history)
            cut = int(np.ceil(hist_len * self.in_perc))

            tr_i.extend(user_history[:cut])
            tr_u.extend([u] * len(user_history[:cut]))

            te_i.extend(user_history[cut:])
            te_u.extend([u] * len(user_history[cut:]))

        tr_data = data.indices_in((tr_u, tr_i))
        te_data = data.indices_in((te_u, te_i))

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
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

    @property
    def name(self):
        return f"timed_split_t{self.t}_t_delta_{self.t_delta}_t_alpha_{self.t_alpha}"

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

        return tr_data, te_data


class FoldIterator:
    def __init__(self, data_m_in, data_m_out, batch_size=1000):
        self.data_m_in = data_m_in
        self.data_m_out = data_m_out
        self._index = 0
        self._max_index = data_m_in.shape[0]
        self.batch_size = batch_size
        assert self.batch_size > 0  # Avoid inf loops

        self.sp_mat_in = self.data_m_in.values
        self.sp_mat_out = self.data_m_out.values

    def __iter__(self):
        return self

    def __next__(self):
        # While loop to make it possible to skip any cases where user has no items in in or out set.
        while True:
            # Yield multi line fold.
            if self._index < self._max_index:
                start = self._index
                end = self._index + self.batch_size
                # make sure we don't go out of range
                if end >= self._max_index:
                    end = self._max_index

                fold_in = self.sp_mat_in[start:end]
                fold_out = self.sp_mat_out[start:end]

                # Filter out users with missing data in either in or out.

                # Get row sum for both in and out
                in_sum = fold_in.sum(1)
                out_sum = fold_out.sum(1)
                # Rows with sum == 0 are rows that should be removed in both in and out.
                rows_to_delete = []
                for i in range(len(in_sum)):
                    if in_sum[i, 0] == 0 or out_sum[i, 0] == 0:
                        rows_to_delete.append(i)
                # Set 0 values for rows to delete
                if len(rows_to_delete) > 0:
                    for r_to_del in rows_to_delete:
                        csr_row_set_nz_to_val(fold_in, r_to_del, value=0)
                        csr_row_set_nz_to_val(fold_out, r_to_del, value=0)
                    fold_in.eliminate_zeros()
                    fold_out.eliminate_zeros()
                    # Remove rows with 0 values
                    fold_in = fold_in[fold_in.getnnz(1) > 0]
                    fold_out = fold_out[fold_out.getnnz(1) > 0]
                # If no matrix is left over, continue to next batch without returning.
                if fold_in.nnz == 0:
                    self._index = end
                    continue

                self._index = end
                # Get a list of users we return as recommendations
                users = np.array([i for i in range(start, end)])
                users = list(set(users) - set(users[rows_to_delete]))

                return fold_in, fold_out, users
            raise StopIteration
