# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Union
import math

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from recpack.matrix import InteractionMatrix


logger = logging.getLogger("recpack")


class Splitter(ABC):
    """Base class for defining a splitter.

    Each splitter implements a split method, which splits data into two outputs.
    """

    @abstractmethod
    def split(self, data: InteractionMatrix):
        """Abstract method to be implemented by the child class.

        Splits dataset into two based on a splitting condition.

        :param data: Interactions to split
        :type data: InteractionMatrix
        """
        raise NotImplementedError()

    @property
    def name(self):
        """The name of the splitter."""
        return self.__class__.__name__

    @property
    def identifier(self):
        """String identifier of the splitter object,
        contains name and parameter values."""
        paramstring = ",".join((f"{k}={v}" for k, v in self.__dict__.items()))
        return self.name + f"({paramstring})"


class UserSplitter(Splitter):
    """Split data by the user identifiers of the interactions.

    Users in ``users_in`` are assigned to the first return value,
    users in ``users_out`` are assigned to the second return value.

    :param users_in: Users for whom the events are assigned to the `in_matrix`
    :type users_in: Union[Set[int], List[int]]
    :param users_out: Users for whom the events are assigned to the `out_matrix`
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

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits data by the user identifiers of the interactions.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        :return: A 2-tuple: the first value contains
            the interactions of ``users_in``,
            the second the interactions of ``users_out``.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        data_in = data.users_in(self.users_in)
        data_out = data.users_in(self.users_out)

        logger.debug(f"{self.identifier} - Split successful")

        return data_in, data_out


class StrongGeneralizationSplitter(Splitter):
    """Randomly splits the users into two sets so that
    interactions for a user will always occur only in one split.

    :param in_frac: The fraction of interactions that are assigned
        to the first value in the output tuple. Defaults to 0.7.
    :type in_frac: float, optional
    :param seed: Seed the random generator. Set this value
        if you require reproducible results.
    :type seed: int, optional
    :param error_margin: The allowed error between ``in_frac``
        and the actual split fraction. Defaults to 0.01.
    :type error_margin: float, optional
    """

    def __init__(self, in_frac=0.7, seed=None, error_margin=0.01):
        super().__init__()
        self.in_frac = in_frac
        self.out_frac = 1 - in_frac

        if seed is None:
            # Set seed if it was not set before.
            seed = np.random.get_state()[1][0]

        self.seed = seed
        self.error_margin = error_margin

    def split(self, data):
        """Randomly splits the users into two sets so that
            interactions for a user will always occur only in one split.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        :return: A 2-tuple containing the ``data_in`` and ``data_out`` matrices.
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

    Interactions of users whose last interaction occurred before ``t``
    are assigned to the first return value.
    Interactions of all the other users are assigned to the second return
    value.

    A user can only occur in one of both sets.

    :param t: Timestamp to split on in seconds since epoch.
    :type t: int
    """

    def __init__(self, t: float):
        super().__init__()
        self.t = t

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits users based on the time of their most recent interactions.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        :return: A 2-tuple containing the ``data_in`` and ``data_out`` matrices.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """

        max_ts_per_user = data.timestamps.groupby(level=0).max()

        filt = max_ts_per_user < self.t

        in_users = max_ts_per_user[filt].index.get_level_values(0).values.tolist()
        out_users = max_ts_per_user[~filt].index.get_level_values(0).values.tolist()

        # in_users = []
        # out_users = []

        data_in = data.users_in(in_users)
        data_out = data.users_in(out_users)

        logger.debug(f"{self.identifier} - Split successful")
        return data_in, data_out


class FractionInteractionSplitter(Splitter):
    """Split data randomly, such that ``in_fraction``
    of interactions are assigned to the first return value and the remainder to the second.

    :param in_frac: Fraction of events to end up in the first return value.
    :type in_frac: float
    :param seed: Seed the random generator. Set this value
        if you require reproducible results.
        Defaults to None, which results in a random seed.
    :type seed: int, optional
    """

    def __init__(self, in_frac, seed: int = None):
        super().__init__()
        self.in_frac = in_frac

        if seed is None:
            # Set seed if it was not set before.
            seed = np.random.get_state()[1][0]

        self.seed = seed

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits data randomly, such that ``in_fraction``
        of interactions are assigned to ``data_in`` and the remainder to ``data_out``.


        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        :return: A 2-tuple containing the ``data_in`` and ``data_out`` matrices.
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
    """Split data so that the first return value contains interactions in ``[t-delta_in, t[``,
    and the second those in ``[t, t+delta_out[``.

    If ``delta_in`` or ``delta_out`` are omitted, they are assumed to have a value of infinity.
    A user can occur in both return values.

    :param t: Timestamp to split on in seconds since epoch.
    :type t: int
    :param delta_out: Seconds past t. Upper bound on the timestamp
        of interactions in the second return value. Defaults to None (infinity).
    :type delta_out: int, optional
    :param delta_in: Seconds before t. Lower bound on the timestamp
        of interactions in the first return value. Defaults to None (infinity).
    :type delta_in: int, optional
    """

    def __init__(self, t, delta_out=None, delta_in=None):
        super().__init__()
        self.t = t
        self.delta_out = delta_out
        self.delta_in = delta_in

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits data so that ``data_in`` contains interactions in ``[t-delta_in, t[``,
        and ``data_out`` those in ``[t, t+delta_out[``.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        :return: A 2-tuple containing the ``data_in`` and ``data_out`` matrices.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        assert data.has_timestamps

        if self.delta_in is None:
            # timestamp < t
            data_in = data.timestamps_lt(self.t)
        else:
            # t-delta_in =< timestamp < t
            data_in = data.timestamps_lt(self.t).timestamps_gte(self.t - self.delta_in)

        if self.delta_out is None:
            # timestamp >= t
            data_out = data.timestamps_gte(self.t)
        else:
            # timestamp >= t and timestamp < t + delta_out
            data_out = data.timestamps_gte(self.t).timestamps_lt(self.t + self.delta_out)

        logger.debug(f"{self.identifier} - Split successful")

        return data_in, data_out


class MostRecentSplitter(Splitter):
    """Splits the n most recent interactions of a user into the second return value,
    and earlier interactions into the first.

    :param n: Number of most recent actions to assign to the second return value.
        If ``n`` is negative, all but the ``n`` earliest
        interactions are assigned to the second return value.
    :type n: int
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits the n most recent interactions of a user into ``data_out``,
        and earlier interactions into ``data_in``.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        :return: A 2-tuple containing the ``data_in`` and ``data_out`` matrices.
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


def csr_row_set_nz_to_val(csr: csr_matrix, row, value=0):
    """Set all nonzero elements to the given value. Useful to set to 0 mostly."""
    if not isinstance(csr, csr_matrix):
        raise ValueError("Matrix given must be of CSR format.")
    csr.data[csr.indptr[row] : csr.indptr[row + 1]] = value
