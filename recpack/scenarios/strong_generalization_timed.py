# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.matrix import InteractionMatrix
from recpack.scenarios import Scenario
from recpack.scenarios.splitters import TimestampSplitter, StrongGeneralizationSplitter


class StrongGeneralizationTimed(Scenario):
    """Predict future interactions for previously unseen users.

    - :attr:`full_training_data` contains interactions from ``frac_users_in`` of the users.
      Only interactions whose timestamps are in the interval ``[t - delta_in, t[`` are used.

    - :attr:`test_data_in` contains data from the ``1-frac_users_in`` users
      for which the events' timestamps are in ``[t - delta_in, t[``.

    - :attr:`test_data_out` contains interactions of the test users
      with timestamps in ``[t, t + delta_out [``

    If validation data is requested, 80% of the full training users are used
    as validation training users, the remaining 20% are used for validation evaluation:

    - :attr:`validation_training_data` contains interactions of validation training users
      with timestamps in ``[t_validation - delta_in, t_validation[``

    - :attr:`validation_data_in` contains all interactions
      of the validation evaluation users
      with timestamps in ``[t_validation - delta_in, t_validation[``.

    - :attr:`validation_data_out` are the interactions
      of the validation evaluation users,
      with timestamps in ``[t_validation, min(t, t_validation + delta_out)[``

    .. warning::

        The scenario can only be used when the dataset has timestamp information.

    **Example**

    As an example, we split this data with ``frac_users_in = 5/6``, ``t = 4``,
    ``t_validation = 2``, ``delta_in = None (infinity)``,
    ``delta_out = 2`` and ``validation = True``::

        time    0   1   2   3   4   5
        Alice   X   X
        Bob         X   X   X   X
        Carol   X   X       X       X
        Dave    X       X   X
        Erin    X   X   X           X
        Frank   X   X   X   X

    would yield full_training_data::

        time    0   1   2   3   4   5
        Alice   X   X
        Bob         X   X   X   X
        Dave    X       X   X
        Erin    X   X   X
        Frank   X   X   X   X

    validation_training_data::

        time    0   1   2   3   4   5
        Alice   X   X
        Dave    X       X   X
        Erin    X   X   X
        Frank   X   X   X   X

    validation_data_in::

        time    0   1   2   3   4   5
        Bob         X

    validation_data_out::

        time    0   1   2   3   4   5
        Bob             X   X

    test_data_in::

        time    0   1   2   3   4   5
        Carol   X   X       X

    test_data_out::

        time    0   1   2   3   4   5
        Carol                       X

    :param frac_users_in: The fraction of users to use as training users.
    :type frac_users_in: float
    :param t: Timestamp to split the interactions of the test users into
        :attr:`test_data_out` and :attr:`test_data_in`; and select
        :attr:`full_training_data` out of all interactions of the training users.
    :type t: int
    :param t_validation: Timestamp to split the interactions of the validation users
        into :attr:`validation_data_out` and :attr:`validation_data_in`; and select
        :attr:`validation_training_data` out of all interactions of the training users.
        Required if validation is True.
    :type t_validation: int, optional
    :param delta_out: Size of interval in seconds for the target datasets.
        Both sets will contain interactions that occurred within ``delta_out`` seconds
        after the splitting timestamp.
        Defaults to None (all interactions past the splitting timestamp).
    :type delta_out: int, optional
    :param delta_in: Size of interval in seconds for
        :attr:`full_training_data`, :attr:`validation_training_data`,
        :attr:`validation_data_in` and :attr:`test_data_in`.
        All sets will contain interactions that occurred within ``delta_out`` seconds
        before the splitting timestamp.
        Defaults to None (all interactions past the splitting timestamp).
    :type delta_in: int, optional
    :param validation: Assign a portion of the training dataset to validation data
        if True,
        else split without validation data into only a training and test dataset.
    :type validation: boolean, optional
    :param seed: The seed to use for the random components of the splitter.
        If None, a random seed will be used. Defaults to None
    :type seed: int, optional
    """

    def __init__(
        self,
        frac_users_in,
        t,
        t_validation=None,
        delta_out=None,
        delta_in=None,
        validation=False,
        seed=None,
    ):
        super().__init__(validation=validation, seed=seed)
        self.frac_users_in = frac_users_in
        self.t = t
        self.delta_out = delta_out
        self.delta_in = delta_in
        self.t_validation = t_validation
        if self.validation and not self.t_validation:
            raise Exception("t_validation should be provided when using validation split.")

        self.timestamp_spl = TimestampSplitter(t, delta_out, delta_in)

        self.strong_gen = StrongGeneralizationSplitter(frac_users_in, seed=self.seed)

        if self.validation:
            assert self.t_validation < self.t
            self.validation_time_splitter = TimestampSplitter(t_validation, delta_out, delta_in)

    def _split(self, data: InteractionMatrix):
        """Splits data into non overlapping train, validation and test user sets
            and uses temporal information to split
            validation and test into in and out folds.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        # Split across user dimension
        tr_val_data, te_data = self.strong_gen.split(data)

        # Make sure only interactions before t are used for training /
        # validation
        self._full_train_X, _ = self.timestamp_spl.split(tr_val_data)
        self._test_data_in, self._test_data_out = self.timestamp_spl.split(te_data)

        if self.validation:
            train_data, validation_data = self.validation_splitter.split(self._full_train_X)
            # Split validation data into input and output on t_validation
            (
                self._validation_data_in,
                self._validation_data_out,
            ) = self.validation_time_splitter.split(validation_data)
            # select the right train data.
            self._validation_train_X, _ = self.validation_time_splitter.split(train_data)
