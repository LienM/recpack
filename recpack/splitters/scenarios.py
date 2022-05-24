from recpack.data.matrix import InteractionMatrix
from recpack.splitters.scenario_base import Scenario
import recpack.splitters.splitter_base as splitter_base


class StrongGeneralization(Scenario):
    """Split your data so that a user can only be in one of
    training, validation or test set.

    A scenario is stateful. After calling :attr:`split` on your dataset,
    the datasets can be retrieved under
    :attr:`full_training_data`, :attr:`validation_training_data`,
    :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.

    During splitting each user is randomly assigned to one of three groups of users:
    training, validation and testing.

    The full training data contains ``frac_users_train`` of the users.
    If validation data is requested validation_training data contains
    ``frac_users_train`` * 0.8 of the users,
    the remaining 20% are assigned to validation data.

    Test and validation users' interactions are split into a `data_in` (fold-in) and
    `data_out` (held-out) set.
    ``frac_interactions_in`` of interactions are assigned to fold-in, the remainder
    to the held-out set.

    Strong generalization is considered more robust, harder and more realistic than
    :class:`recpack.splitters.scenarios.WeakGeneralization`

    **Example**

    As an example, we split this interaction matrix with ``frac_users_train = 5/6``,
    ``frac_interactions_in = 0.75`` and ``validation = True``::

        item    0   1   2   3   4   5
        user1   X   X
        user2       X   X   X   X
        user3   X   X       X       X
        user4   X       X   X
        user5   X   X   X
        user6   X   X   X   X

    would yield full_training_data::

        item    0   1   2   3   4   5
        user1   X   X
        user3   X   X       X       X
        user4   X       X   X
        user5   X   X   X
        user6   X   X   X   X

    validation_training_data::

        item    0   1   2   3   4   5
        user1   X   X
        user3   X   X       X       X
        user5   X   X   X
        user6   X   X   X   X

    validation_data_in::
        user4   X           X

    validation_data_out::
        user4           X

    test_data_in::

        item    0   1   2   3   4   5
        user2       X       X   X

    test_data_out::

        item    0   1   2   3   4   5
        user2           X

    :param frac_users_train: Fraction of users assigned to the full training dataset.
        Between 0 and 1.
    :type frac_users_train: float
    :param frac_interactions_in: Fraction of the users'
        interactions to be used as fold-in set (user history). Between 0 and 1.
    :type frac_interactions_in: float
    :param validation: Assign a portion of the full training dataset
        to validation datasets if True,
        else split without validation data into only a training and test dataset.
    :type validation: boolean, optional
    :param seed: The seed to use for the random components of the splitter.
        If None, a random seed will be used. Defaults to None
    :type seed: int, optional
    """

    def __init__(
        self,
        frac_users_train: float,
        frac_interactions_in: float,
        validation=False,
        seed=None,
    ):
        super().__init__(validation=validation, seed=seed)
        self.frac_users_train = frac_users_train
        self.frac_interactions_in = frac_interactions_in

        self.strong_gen = splitter_base.StrongGeneralizationSplitter(frac_users_train, seed=self.seed)
        self.interaction_split = splitter_base.FractionInteractionSplitter(frac_interactions_in, seed=self.seed)

    def _split(self, data: InteractionMatrix) -> None:
        """Splits your data so that a user can only be in one of
            training, validation or test set.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        """
        self._full_train_X, test_data = self.strong_gen.split(data)

        if self.validation:
            self._validation_train_X, validation_data = self.validation_splitter.split(self._full_train_X)

            (
                self._validation_data_in,
                self._validation_data_out,
            ) = self.interaction_split.split(validation_data)

        self._test_data_in, self._test_data_out = self.interaction_split.split(test_data)


# TODO: docs for Weak generalization suck
class WeakGeneralization(Scenario):
    """Applies user based hold-out sampling.

    A scenario is stateful. After calling :attr:`split` on your dataset,
    the datasets can be retrieved under
    :attr:`full_training_data`, :attr:`validation_training_data`,
    :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.

    .. note::
        This scenario duplicates interactions across training, validation
        and test datasets:

    for each user their events ``I_u`` are distributed over the datasets as follows:

    - :attr:`full_training_data` (``IT_u``) contains ``frac_data_in * |I_u|`` (rounded up)
      of the user's interactions in the full dataset.
    - :attr:`validation_training_data` contains ``frac_data_in * |IT_u|`` (rounded up)
      of the user's interactions in the full training dataset.
    - :attr:`validation_data_in` contains the same
      events as :attr:`validation_training_data`
      for users with enough interactions to also
      have one in the validation data out dataset.
    - :attr:`validation_data_in` contains the remaining ``(1 - frac_data_in) * |IT_u|``
      (rounded down) of the user's interactions in the full training dataset.
    - :attr:`test_data_in` contains the same events as :attr:`test_training_data`
      for users with enough interactions to also have an event in the test out dataset.
    - :attr:`test_training_data` contains the remaining ``(1 - frac_data_in) * |I_u|``
      of the user's interactions in the full dataset.

    **Example**

    As an example, splitting following data with ``data_in_frac = 0.5``,
    and ``validation = True``::

        item    0   1   2   3   4   5
        user1   X   X   X
        user2           X   X   X   X

    would yield full_training_data::

        item    0   1   2   3   4   5
        user1   X   X
        user2           X       X

    validation_training_data::

        item    0   1   2   3   4   5
        user1       X
        user2           X

    validation_data_in::

        item    0   1   2   3   4   5
        user1       X
        user2           X

    validation_data_out::

        item    0   1   2   3   4   5
        user1   X
        user2                   X

    test_data_in::

        item    0   1   2   3   4   5
        user1   X   X
        user2           X       X

    test_data_out::

        item    0   1   2   3   4   5
        user1           X
        user2               X       X

    :param frac_data_in: Fraction of interactions per user used for
        training. The interactions are randomly chosen.
    :type frac_data_in: float
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
        frac_data_in: float,
        validation=False,
        seed=None,
    ):
        super().__init__(validation=validation, seed=seed)

        self.frac_data_in = frac_data_in

        self.interaction_split = splitter_base.FractionInteractionSplitter(self.frac_data_in, seed=self.seed)

        if validation:
            self.validation_splitter = splitter_base.FractionInteractionSplitter(
                self.frac_data_in,
                seed=self.seed,
            )

    def _split(self, data: InteractionMatrix):
        """Splits your data so that a user has interactions in all of
            training, validation and test set.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        """
        self._full_train_X, self._test_data_out = self.interaction_split.split(data)

        if self.validation:
            (
                self._validation_train_X,
                self._validation_data_out,
            ) = self.validation_splitter.split(self._full_train_X)
            self._validation_data_in = self._validation_train_X.copy()

        self._test_data_in = self._full_train_X.copy()


class Timed(Scenario):
    """Split your dataset into a training, validation and test dataset
        based on the timestamp of the interaction.

    A scenario is stateful. After calling :attr:`split` on your dataset,
    the datasets can be retrieved under
    :attr:`full_training_data`, :attr:`validation_training_data`,
    :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.

    .. note::
        This scenario duplicates interactions across training, validation
        and test datasets:

    - :attr:`full_training_data` is constructed by using
      all interactions whose timestamps
      are in the interval ``[t - delta_in, t[``
    - :attr:`full_training_data` are all interactions
      with timestamps in ``[t_validation - delta_in, t_validation[``.
    - :attr:`validation_data_in` are interactions with timestamps in
      ``[t_validation - delta_in, t_validation[``
    - :attr:`validation_data_out` are interactions with timestamps in
      ``[t_validation, min(t, t_validation + delta_out)[``.
    - :attr:`test_data_in` are events with timestamps in  ``[t - delta_in, t[``.
    - :attr:`test_data_out` are events with timestamps in ``[t, t + delta_out[``.

    **Example**

    As an example, we split this data with ``t = 4``, ``t_validation = 2``
    ````delta_in = None (infinity)``, delta_out = 2``, and ``validation = True``::

        time    0   1   2   3   4   5   6
        user1   X   X               X
        user2       X   X   X   X
        user3   X   X       X       X   X

    would yield full_training_data::

        time    0   1   2   3   4   5   6
        user1   X   X
        user2       X   X   X
        user3   X   X       X

    validation_training_data::

        time    0   1   2   3   4   5   6
        user1   X   X
        user2       X
        user3   X   X

    validation_data_in::

        time    0   1   2   3   4   5   6
        user2       X
        user3   X   X

    validation_data_out::

        time    0   1   2   3   4   5   6
        user2           X   X
        user3           X

    test_data_in::

        time    0   1   2   3   4   5   6
        user1   X   X
        user3   X   X       X

    test_data_out::

        time    0   1   2   3   4   5   6
        user1                       X
        user3                       X   X

    :param t: Timestamp to split target dataset :attr:`test_data_out`
        from the remainder of the data.
    :type t: int
    :param t_validation: Timestamp to split :attr:`validation_data_out`
        from :attr:`validation_training_data`. Required if validation is True.
    :type t_validation: int, optional
    :param delta_out: Size of interval in seconds for
        both :attr:`validation_data_out` and :attr:`test_data_out`.
        Both sets will contain interactions that occurred within ``delta_out`` seconds
        after the splitting timestamp.
        Defaults to None (acting as infinity).
    :type delta_out: int, optional
    :param delta_in: Size of interval in seconds for
        :attr:`full_training_data`, :attr:`validation_training_data`,
        :attr:`validation_data_in` and :attr:`test_data_in`.
        All sets will contain interactions that occurred within ``delta_out`` seconds
        before the splitting timestamp.
        Defaults to None (acting as infinity).
    :type delta_in: int, optional
    :param validation: Assign a portion of the full training dataset to validation data
        if True, else split without validation data
        into only a training and test dataset.
    :type validation: boolean, optional
    :param seed: Seed for randomisation parts of the scenario.
        Timed scenario is deterministic, so changing seed should not matter.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional

    """

    def __init__(
        self,
        t,
        t_validation=None,
        delta_out=None,
        delta_in=None,
        validation=False,
        seed=None,
    ):
        super().__init__(validation=validation, seed=seed)
        self.t = t
        self.delta_out = delta_out
        self.delta_in = delta_in
        self.t_validation = t_validation
        if self.validation and not self.t_validation:
            raise Exception("t_validation should be provided when requesting a validation dataset.")

        self.timestamp_spl = splitter_base.TimestampSplitter(t, delta_out, delta_in)

        if self.validation:
            assert self.t_validation < self.t
            # Override the validation splitter to a timed splitter.
            self.validation_time_splitter = splitter_base.TimestampSplitter(t_validation, delta_out, delta_in)

    def _split(self, data):
        """Splits your dataset into a training, validation and test dataset
            based on the timestamp of the interaction.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        self._full_train_X, self._test_data_out = self.timestamp_spl.split(data)
        self._test_data_in = self._full_train_X.copy()

        if self.validation:
            (
                self._validation_train_X,
                self._validation_data_out,
            ) = self.validation_time_splitter.split(self._full_train_X)
            self._validation_data_in = self._validation_train_X.copy()


# TODO: how do you define the validation_train data?
class StrongGeneralizationTimed(Scenario):
    """Splits data into non overlapping train, validation and test user sets
    and uses temporal information to split validation and test into in and out folds.

    A scenario is stateful. After calling :attr:`split` on your dataset,
    the datasets can be retrieved under
    :attr:`full_training_data`, :attr:`validation_training_data`,
    :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.

    - :attr:`full_training_data` contains interactions
      from ``frac_users_in`` of the users.
      Only interactions whose timestamps are
      in the interval ``t - delta_in, t[`` are used.

    - :attr:`test_data_in` contains data from the ``1-frac_users_in`` users
      for which the events' timestamps are in ``[t - delta_in, t[``.

    - :attr:`test_data_out` contains interactions of the test users
      with timestamps in ``[t, t + delta_out [``

    If validation data is requested, 80% of the full training users are used
    as validation training users, the remaining 20% are used for validation evaluation:

    - :attr:`validation_data_in` contains all interactions
      of the validation evaluation users
      with timestamps in ``[t_validation - delta_in, t_validation[``.
    - :attr:`validation_data_out` are the interactions
      of the validation evaluation users,
      with timestamps in ``[t_validation, min(t, t_validation + delta_out)[``

    **Example**

    As an example, we split this data with ``frac_users_in = 5/6``, ``t = 4``,
    ``t_validation = 2``, ``delta_in = None (infinity)``,
    ``delta_out = 2`` and ``validation = True``::

        time    0   1   2   3   4   5
        user1   X   X
        user2       X   X   X   X
        user3   X   X       X       X
        user4   X       X   X
        user5   X   X   X           X
        user6   X   X   X   X

    would yield full_training_data::

        time    0   1   2   3   4   5
        user1   X   X
        user2       X   X   X   X
        user4   X       X   X
        user5   X   X   X
        user6   X   X   X   X

    validation_training_data::

        time    0   1   2   3   4   5
        user1   X   X
        user4   X       X   X
        user5   X   X   X
        user6   X   X   X   X

    validation_data_in::

        time    0   1   2   3   4   5
        user2       X

    validation_data_out::
        time    0   1   2   3   4   5
        user2           X   X

    test_data_in::

        time    0   1   2   3   4   5
        user3   X   X       X

    test_data_out::

        time    0   1   2   3   4   5
        user3                       X

    :param frac_users_in: The fraction of users to use
        for the training(_validation) dataset.
    :type frac_users_in: float
    :param t: Timestamp to split the interactions of the test users into
        :attr:`test_data_out` and :attr:`test_data_in`; and select
        :attr:`training_data` out of all interactions of the training users
        if validation is False.
    :type t: int
    :param t_validation: Timestamp to split the interactions of the validation users
        into :attr:`validation_data_out` and :attr:`validation_data_in`; and select
        :attr:`validation_training_data` out of all interactions of the training users
        if validation is True. Required if validation is True.
    :type t_validation: int, optional
    :param delta_out: Size of interval in seconds for
        both :attr:`validation_data_out` and :attr:`test_data_out`.
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

        self.timestamp_spl = splitter_base.TimestampSplitter(t, delta_out, delta_in)

        self.strong_gen = splitter_base.StrongGeneralizationSplitter(frac_users_in, seed=self.seed)

        if self.validation:
            assert self.t_validation < self.t
            self.validation_time_splitter = splitter_base.TimestampSplitter(t_validation, delta_out, delta_in)

    def _split(self, data):
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


class StrongGeneralizationTimedMostRecent(Scenario):
    """Split users into non-overlapping training,
    validation and test user sets based on the time of
    their most recent interaction.

    A scenario is stateful. After calling :attr:`split` on your dataset,
    the datasets can be retrieved under
    :attr:`full_training_data`, :attr:`validation_training_data`,
    :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.


    Test data contains all users whose most recent interactions was after ``t``:

    - :attr:`test_data_out` contains the ``n`` most recent interactions of
      a user whose most recent interactions was after ``t``.
    - :attr:`test_data_in` contains all earlier interactions of the test users.

    - :attr:`full_training_data` contains events from all users
    whose most recent interaction was before ``t``

    If validation is True, validation evaluation users are those whose most recent interaction
    is after ``t_validation``.

    - :attr:`validation_training_data` contains users whose most
    recent interaction happened before ``t_validation``.

    - :attr:`validation_data_out` contains the ``n`` most recent interactions of
      a user whose most recent interactions was in the interval ``[t_validation, t[``.
    - :attr:`validaton_data_in` contains all earlier interactions of the
      validation_evaluation users.

    **Example**

    As an example, splitting following data with ``t = 4``, ``t_validation=2``,
    ``n = 1`` and ``validation=True``::

        time    0   1   2   3   4   5
        user1   X   X
        user2       X   X   X   X
        user3       X   X   X


    would yield full_training_data::

        time    0   1   2   3   4   5
        user1   X   X
        user3       X   X   X

    validation_training_data::

        time    0   1   2   3   4   5
        user1   X   X

    validation_data_in::

        time    0   1   2   3   4   5
        user3       X

    validation_data_out::

        time    0   1   2   3   4   5
        user3           X   X

    test_data_in::

        time    0   1   2   3   4   5
        user1
        user2       X   X   X

    test_data_out::

        time    0   1   2   3   4   5
        user1
        user2                   X

    :param t: Users whose last action has ``time >= t`` are placed in the test set,
              all other users are placed in the training or validation sets.
    :param t_validation: Users whose last action has ``time >= t_validation`` and
                         ``time < t`` are put in the validation set. Users whose
                         last action has ``time < t_validation`` are put in train.
                         Only required if validation is True.
    :param n: The n most recent user actions to split off. If a negative number,
              split off all but the ``n`` earliest actions.
    :param validation: Assign a portion of the full training dataset
        to validation data if True,
        else split without validation data into only a training and test dataset.
    :type validation: boolean, optional
    :param seed: Seed for randomisation parts of the scenario.
        This scenario is deterministic, so changing seed should not matter.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional

    """

    def __init__(
        self,
        t: float,
        t_validation: float = None,
        n: int = 1,
        validation: bool = False,
        seed=None,
    ):
        super().__init__(validation=validation, seed=seed)
        self.t = t
        self.t_validation = t_validation
        self.n = n
        if self.validation and not self.t_validation:
            raise Exception("t_validation should be provided when using validation split.")

        self.user_splitter_test = splitter_base.UserInteractionTimeSplitter(t)
        if self.validation:
            assert self.t_validation < self.t
            self.user_splitter_val = splitter_base.UserInteractionTimeSplitter(t_validation)
        self.most_recent_splitter = splitter_base.MostRecentSplitter(n)

    def _split(self, data):
        """Splits users into non-overlapping training,
            validation and test user sets based on the time of
            their most recent interaction.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        self._full_train_X, te_data = self.user_splitter_test.split(data)

        self._test_data_in, self._test_data_out = self.most_recent_splitter.split(te_data)

        if self.validation:
            self._validation_train_X, val_data = self.user_splitter_val.split(self._full_train_X)
            (
                self._validation_data_in,
                self._validation_data_out,
            ) = self.most_recent_splitter.split(val_data)


class LastItemPrediction(Scenario):
    """Split data so that a user's previous ``n_most_recent`` interactions are used to
    predict their last interaction. Trains on all other interactions.

    A scenario is stateful. After calling :attr:`split` on your dataset,
    the datasets can be retrieved under
    :attr:`full_training_data`, :attr:`validation_training_data`,
    :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.

    - :attr:`full_training_data` contains all but the most recent
      interaction for each of the users.
    - :attr: `validation_training_data` contains
      all but the most recent interaction of each user in the full training dataset.
    - :attr:`validation_data_in` contains the ``n_most_recent`` interactions before
      the last interaction of each user in the full training dataset.
    - :attr:`validation_data_out` contains the most recent interaction
      of each user in the full training dataset

    - :attr:`test_data_in`: contains the ``n_most_recent`` interactions before
      the last interaction of all users.
    - :attr:`test_data_out` contains the last interaction of all users.

    **Example**

    As an example, we split this data with
    ``validation = True`` and ``n_most_recent = 1``::

        time    0   1   2   3   4   5
        user1   X   X   X
        user2       X   X   X   X

    would yield full_training_data::

        time    0   1   2   3   4   5
        user1   X   X
        user2       X   X   X

    validation_training_data::

        time    0   1   2   3   4   5
        user1   X
        user2       X   X

    validation_data_in::

        time    0   1   2   3   4   5
        user1   X
        user2           X

    validation_data_out::

        time    0   1   2   3   4   5
        user1       X
        user2               X

    test_data_in::

        time    0   1   2   3   4   5
        user1       X
        user2          X

    test_data_out::

        time    0   1   2   3   4   5
        user1           X
        user2               X


    :param validation: construct validation datasets if True,
        else split without validation data into only a full training and test dataset.
    :type validation: boolean, optional
    :param seed: Seed for randomisation parts of the scenario.
        This scenario is deterministic, so changing seed should not matter.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional
    :param n_most_recent: How much of the historic events to use as input history.
        Defaults to 1
    :type n_most_recent: int, optional
    """

    def __init__(self, validation=False, seed=None, n_most_recent=1):
        super().__init__(validation=validation, seed=seed)
        self.most_recent_splitter = splitter_base.MostRecentSplitter(1)
        self.n_most_recent = n_most_recent
        self.history_splitter = splitter_base.MostRecentSplitter(n_most_recent)

    def _split(self, data):
        """Splits data so that a user's second to last interaction is used to predict their last interaction.
            Trains on all other interactions.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        self._full_train_X, self._test_data_out = self.most_recent_splitter.split(data)
        # Select n most recent as test_in
        _, self._test_data_in = self.history_splitter.split(self._full_train_X)

        if self.validation:
            # Select the 1 event to use as val_out per user
            (
                self._validation_train_X,
                self._validation_data_out,
            ) = self.most_recent_splitter.split(self._full_train_X)

            # Select n items to use as val_in per user
            _, self._validation_data_in = self.history_splitter.split(self._validation_train_X)


class NextItemPrediction(LastItemPrediction):
    def __init__(self, validation=False, seed=None, n_most_recent=1):
        raise DeprecationWarning(
            "NextItemPrediction will be deprecated in favour of " "the more descriptive LastItemPrediction"
        )

        super.__init__(self, validation, seed, n_most_recent)
