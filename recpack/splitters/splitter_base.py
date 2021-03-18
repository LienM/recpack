import logging
from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Union
import math

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from recpack.data.matrix import InteractionMatrix

logger = logging.getLogger("recpack")


class Splitter(ABC):
    """Splitters facilitate splitting data in specific ways.

    Each splitter implements a split method, which splits data into two outputs.

    """

    def __init__(self):
        pass

    @abstractmethod
    def split(self, data: InteractionMatrix):
        """Split data

        :param data: Interactions to split
        :type data: InteractionMatrix
        """
        pass

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def identifier(self):
        """String identifier of the splitter object,
        contains name and parameter values."""
        paramstring = ",".join((f"{k}={v}" for k, v in self.__dict__.items()))
        return self.name + f"({paramstring})"


class UserSplitter(Splitter):
    """Split data by the user identifiers of the interactions.

    Users in ``users_in`` will get put into the `data_in` matrix,
    users in ``users_out`` get put into the `data_out` matrix

    :param users_in: Users for whom the events get put into the `in_matrix`
    :type users_in: Union[Set[int], List[int]]
    :param users_out: Users for whom the events get put into the `out_matrix`
    :type users_out: Union[Set[int], List[int]]
    """

    def __init__(
        self,
        users_in: Union[Set[int], List[int]],
        users_out: Union[Set[int], List[int]],
    ):
        super().__init__()
        self.users_in = users_in
        self.users_out = users_out

    def split(
        self, data: InteractionMatrix
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """
        Splits a user-interaction matrix by the interactions' user indices.

        :param data: Matrix with item interactions to be split.
        :type data: InteractionMatrix
        :return: A tuple with the first value containing
            the interactions of ``users_in``,
            and the second with the interactions of ``users_out``.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        data_in = data.users_in(self.users_in)
        data_out = data.users_in(self.users_out)

        logger.debug(f"{self.identifier} - Split successful")

        return data_in, data_out


class StrongGeneralizationSplitter(Splitter):
    """Randomly divides the users into two sets,
    events for a user will always occur only in one split.

    :param in_frac: The fraction of the data to end up in the first matrix.
        Defaults to 0.7
    :type in_frac: float, optional

    :param seed: Seed for randomisation in function,
        set this to a specified value in order to get expected results.
    :type seed: `int`, optional

    :param error_margin: The allowed difference between ``in_frac``
        and the actual split fraction.
        Defaults to 0.01
    :type error_margin: float, optional
    """

    def __init__(self, in_frac=0.7, seed=None, error_margin=0.01):
        super().__init__()
        self.in_frac = in_frac
        self.out_frac = 1 - in_frac
        self.seed = seed
        self.error_margin = error_margin

    def split(self, data):
        """Splits the user-interaction matrix.

        :param data: Matrix of interactions to be split.
        :type data: InteractionMatrix
        :return: A tuple containing the `data_in` and `data_out` objects.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        sp_mat = data.values

        users = list(set(sp_mat.nonzero()[0]))

        nr_users = len(users)

        # Seed to make testing possible
        if self.seed is not None:
            np.random.seed(self.seed)

        # Try five times
        for i in range(0, 5):
            np.random.shuffle(users)

            in_cut = int(np.floor(nr_users * self.in_frac))

            users_in = users[:in_cut]
            users_out = users[in_cut:]

            total_interactions = sp_mat.nnz
            data_in_cnt = sp_mat[users_in, :].nnz

            real_frac = data_in_cnt / total_interactions

            within_margin = np.isclose(real_frac, self.in_frac, atol=self.error_margin)

            if within_margin:
                logger.debug(f"{self.identifier} - Iteration {i} - Within margin")
                break
            else:
                logger.debug(f"{self.identifier} - Iteration {i} - Not within margin")

        u_splitter = UserSplitter(users_in, users_out)
        ret = u_splitter.split(data)

        logger.debug(f"{self.identifier} - Split successful")
        return ret


class UserInteractionTimeSplitter(Splitter):
    """Split users based on the time of their most recent interactions.

    `data_in` are the interactions of users
    whose last interaction occurs before ``t``.
    `data_out` are all interactions of the other users.

    A user's interactions will all be placed in one of the two sets,
    never in both.

    :param t: Timestamp to split on. In seconds since epoch.
    :type t: int
    """

    def __init__(self, t: float):
        super().__init__()
        self.t = t

    def split(
        self, data: InteractionMatrix
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits the input matrix into `data_in` and `data_out`.

        :param data: Data matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        :return: Tuple containing `data_in`, `data_out`
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        in_users = []
        out_users = []

        for uid, user_history in tqdm(data.sorted_interaction_history):

            last_interaction = user_history[-1]

            last_interaction_time = data.get_timestamp(last_interaction)

            if last_interaction_time < self.t:
                in_users.append(uid)

            else:
                out_users.append(uid)

        data_in = data.users_in(in_users)
        data_out = data.users_in(out_users)

        logger.debug(f"{self.identifier} - Split successful")
        return data_in, data_out


class FractionInteractionSplitter(Splitter):
    """Splits data randomly, such that ``in_fraction``
    of interactions is put in `data_in` and the rest in `data_out`

    :param in_frac: Fraction of events to end up in the `data_in` matrix.
    :type in_frac: float
    :param seed: Seed to allow for consistent behaviour. Defaults to 42.
    :type seed: int, optional
    """

    def __init__(self, in_frac, seed: int = 42):
        super().__init__()
        self.in_frac = in_frac

        self.seed = seed

    def split(
        self, data: InteractionMatrix
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Split the input data based on the parameters set at construction.
        Returns 2 sparse matrices in and out.

        :param data: Matrix with interactions to split.
        :type data: InteractionMatrix
        :returns: Tuple of data_in and data_out.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """

        in_interactions = []
        out_interactions = []

        for u, interaction_history in tqdm(data.interaction_history):

            rstate = np.random.RandomState(self.seed + u)

            rstate.shuffle(interaction_history)
            hist_len = len(interaction_history)
            cut = int(np.ceil(hist_len * self.in_frac))

            in_interactions.extend(interaction_history[:cut])
            out_interactions.extend(interaction_history[cut:])

        data_in = data.interactions_in(in_interactions)
        data_out = data.interactions_in(out_interactions)

        logger.debug(f"{self.identifier} - Split successful")

        return data_in, data_out


class TimestampSplitter(Splitter):
    """Splits data on a timestamp into two matrices.

    `data_in` will be all data in the interval ``[t-t_alpha, t[``,
    and `data_out` will be the data in the interval ``[t, t+t_delta[``.
    If ``t_alpha`` or ``t_delta`` are omitted, they are assumed infinity.

    :param t: epoch timestamp to split on
    :type t: int
    :param t_delta: seconds past t to consider as in data.
        Defaults to None
    :type t_delta: int, optional
    :param t_alpha: seconds before t to use as out data.
        Defaults to None
    :type t_alpha: int, optional
    """

    def __init__(self, t, t_delta=None, t_alpha=None):
        super().__init__()
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

    def split(
        self, data: InteractionMatrix
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Split the input data by using the timestamps
        provided in the timestamp field of data.

        :param data: Matrix with interactions to be split.
            ``data.has_timestamps`` should evaluate to True.
        :type data: InteractionMatrix
        :return: A tuple containing the `data_in` and `data_out` matrices.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        assert data.has_timestamps

        if self.t_alpha is None:
            # timestamp < t
            data_in = data.timestamps_lt(self.t)
        else:
            # t-t_alpha =< timestamp < t
            data_in = data.timestamps_lt(self.t).timestamps_gte(self.t - self.t_alpha)

        if self.t_delta is None:
            # timestamp >= t
            data_out = data.timestamps_gte(self.t)
        else:
            # timestamp >= t and timestamp < t + t_delta
            data_out = data.timestamps_gte(self.t).timestamps_lt(self.t + self.t_delta)

        logger.debug(f"{self.identifier} - Split successful")

        return data_in, data_out


class MostRecentSplitter(Splitter):
    """Splits the n most recent actions into `data_out`,
    and earlier interactions into `data_in`.

    :param n: Number of most recent actions.
        If negative, all but the ``n`` earliest
        actions are split off.
    :type n: int
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def split(
        self, data: InteractionMatrix
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """
        Returns a data matrix with all but the n most recent actions of each user
        and a data matrix with the n most recent actions of each user.

        :param data: Data matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        :return: A tuple containing a matrix with all but n most recent and matrix
                 with n most recent actions of each user.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        assert data.has_timestamps

        in_interactions = []
        out_interactions = []

        for u, interaction_history in tqdm(data.sorted_interaction_history):

            in_interactions.extend(interaction_history[: -self.n])
            out_interactions.extend(interaction_history[-self.n :])

        data_in = data.interactions_in(in_interactions)
        data_out = data.interactions_in(out_interactions)

        logger.debug(f"{self.identifier} - Split successful")

        return data_in, data_out


def yield_batches(iterable, n=1):
    """Helper to generate batches from an iterable.

    TODO: Move this to a util file

    :param iterable: iterable to get batches from
    :type iterable: Iterable
    :param n: size of batch, defaults to 1
    :type n: int, optional
    :yield: Batch of length n of the iterable values
    :rtype: Iterator[Any]
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def csr_row_set_nz_to_val(csr: csr_matrix, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, csr_matrix):
        raise ValueError("Matrix given must be of CSR format.")
    csr.data[csr.indptr[row] : csr.indptr[row + 1]] = value


class FoldIterator:
    def __init__(self, data_m_in, data_m_out, batch_size=1000):
        self.data_m_in = data_m_in
        self.data_m_out = data_m_out
        self.batch_size = batch_size
        assert self.batch_size > 0  # Avoid inf loops

        self.data_m_in.eliminate_timestamps(inplace=True)
        self.data_m_out.eliminate_timestamps(inplace=True)

        # users need history, but absence of true labels is allowed (some
        # metrics don't require true labels).
        self.users = list(sorted(set(self.data_m_in.indices[0])))

    def __iter__(self):
        self.batch_generator = yield_batches(self.users, self.batch_size)
        return self

    def __next__(self):
        user_batch = np.array(next(self.batch_generator))
        fold_in = self.data_m_in.values[user_batch, :]
        fold_out = self.data_m_out.values[user_batch, :]

        return fold_in, fold_out, user_batch

    def __len__(self):
        return math.ceil(len(self.users) / self.batch_size)
