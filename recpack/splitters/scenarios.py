from recpack.data.matrix import InteractionMatrix
from recpack.splitters.scenario_base import Scenario
import recpack.splitters.splitter_base as splitter_base


class StrongGeneralization(Scenario):
    """Split your data so that a user can only be in one of
    training, validation or test set.

    A scenario is stateful. After calling ``split`` on your dataset,
    the training, validation and test dataset can be retrieved under
    :attr:`training_data`, :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.

    During splitting each user is randomly assigned to one of the three groups of users.
    When no validation data is required, training data contains ``frac_users_train``
    of the users.
    If validation data is requested training data contains ``frac_users_train`` * 0.8
    of the users, the remaining 20% is assigned to validation data.

    Test and validation users' interactions are split into a `data_in` (fold-in) and `data_out` (held-out) set.
    ``frac_interactions_in`` of interactions are assigned to fold-in, the remainder
    to the held-out set.

    Strong generalization is considered more robust, harder and more realistic than
    :class:`class recpack.splitters.scenarios.WeakGeneralization`

    **Example**

    As an example, we split this interaction matrix with ``frac_users_train = 0.5``,
    ``frac_interactions_in = 0.75`` and ``validation=False``::

        item    0   1   2   3   4   5
        user1   X   X
        user2       X   X   X   X

    would yield training_data::

        item    0   1   2   3   4   5
        user1   X   X
        user2

    test_data_in::

        item    0   1   2   3   4   5
        user1
        user2       X       X   X

    test_data_out::

        item    0   1   2   3   4   5
        user1
        user2           X

    :param frac_users_train: Fraction of users assigned to the training dataset. Between 0 and 1.
    :type frac_users_train: float
    :param frac_interactions_in: Fraction of the users'
        interactions to be used as fold-in set (user history). Between 0 and 1.
    :type frac_interactions_in: float
    :param validation: Assign a portion of the training dataset to validation data if True,
        else split without validation data into only a training and test dataset.
    :type validation: boolean, optional
    """

    def __init__(
        self, frac_users_train: float, frac_interactions_in: float, validation=False
    ):
        super().__init__(validation=validation)
        self.frac_users_train = frac_users_train
        self.frac_interactions_in = frac_interactions_in

        self.strong_gen = splitter_base.StrongGeneralizationSplitter(frac_users_train)
        self.interaction_split = splitter_base.FractionInteractionSplitter(
            frac_interactions_in
        )

    def _split(self, data: InteractionMatrix) -> None:
        """Splits your data so that a user can only be in one of
            training, validation or test set.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        """
        train_val_data, test_data = self.strong_gen.split(data)

        if self.validation:
            self.train_X, validation_data = self.validation_splitter.split(
                train_val_data
            )
            (
                self._validation_data_in,
                self._validation_data_out,
            ) = self.interaction_split.split(validation_data)
        else:
            self.train_X = train_val_data

        self._test_data_in, self._test_data_out = self.interaction_split.split(
            test_data
        )


class WeakGeneralization(Scenario):
    """Split your data so that a user has interactions in all of
    training, validation and test set.

    A scenario is stateful. After calling ``split`` on your dataset,
    the training, validation and test dataset can be retrieved under
    :attr:`training_data`, :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.

    ! Note ! This scenario duplicates interactions across training, validation
    and test datasets:

    - Training interactions are also used as :attr:`validation_data_in` to predict :attr:`validation_data_out`.
    - The union of training and validation interactions is used as :attr:`test_data_in` to predict
    :attr:`test_data_out`.

    The training dataset contains ``frac_interactions_train``
    of a user's interactions.
    The validation_data_out dataset contains ``frac_interactions_validation``
    of a user's interactions.
    The test_data_out dataset contains the remaining interactions.

    This requires ``frac_interactions_train + frac_interactions_validation`` < 1.

    If ``validation == False``, ``frac_interactions_validation`` is 0.

    **Example**

    As an example, plitting following data with ``frac_users_train = 0.333``,
    ``frac_interactions_validation = 0.333`` and ``validation = True``
    and ``validation=False``::

        item    0   1   2   3   4   5
        user1   X   X   X
        user2           X   X   X

    would yield training_data::

        item    0   1   2   3   4   5
        user1       X
        user2               X

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
        user2               X

    :param frac_interactions_train: Fraction of interactions per user used for
        training. The interactions are randomly chosen.
    :type frac_interactions_train: float
    :param frac_interactions_validation: Fraction of interactions per user used
        for validation_data_out. The interactions are randomly chosen.
    :type frac_interactions_validation: float, optional
    :param validation: Assign a portion of the training dataset to validation data if True,
        else split without validation data into only a training and test dataset.
    :type validation: boolean, optional
    """

    def __init__(
        self,
        frac_interactions_train: float,
        frac_interactions_validation: float = 0,
        validation=False,
    ):

        super().__init__(validation=validation)
        self.frac_interactions_train = frac_interactions_train

        self.interaction_split = splitter_base.FractionInteractionSplitter(
            frac_interactions_train + frac_interactions_validation
        )

        assert (frac_interactions_train + frac_interactions_validation) < 1

        if frac_interactions_validation > 0 and not validation:
            raise ValueError(
                "If validation=False, frac_interactions_validation should be zero."
            )

        if validation:
            self.frac_interactions_validation = frac_interactions_validation
            self.validation_splitter = splitter_base.FractionInteractionSplitter(
                frac_interactions_train
                / (frac_interactions_train + frac_interactions_validation)
            )

    def _split(self, data: InteractionMatrix):
        """Splits your data so that a user has interactions in all of
            training, validation and test set.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        """
        train_val_data, self._test_data_out = self.interaction_split.split(data)

        if self.validation:
            self.train_X, self._validation_data_out = self.validation_splitter.split(
                train_val_data
            )
            self._validation_data_in = self.train_X.copy()
        else:
            self.train_X = train_val_data

        self._test_data_in = train_val_data.copy()


class Timed(Scenario):
    """Split your dataset into a training, validation and test dataset
        based on the timestamp of the interaction.

    A scenario is stateful. After calling ``split`` on your dataset,
    the training, validation and test dataset can be retrieved under
    :attr:`training_data`, :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.

    ! Note ! This scenario duplicates interactions across training, validation
    and test datasets:

    - Training data is constructed by using all interactions whose timestamps
    are in the interval ``[t - delta_in, t[`` or when validation is True, with timestamps in
    ``[t_validation - delta_in, t_validation[``.
    - Validation in data are interactions with timestamps in
      ``[t_validation - delta_in, t_validation[``.
    - Validation out data are interactions with timestamps in
      ``[t_validation, min(t, t_validation + delta_out)[``.
    - Test data in are those events with timestamps in  ``[t - delta_in, t[``.
    - Test data out are events with timestamps in ``[t, t + delta_out[``.

    **Example**

    As an example, we split this data with ``t = 2``,
    ````delta_in = None (infinity)``, delta_out = 2``, and ``validation=False``::

        time    0   1   2   3   4   5
        user1   X   X
        user2       X   X   X   X

    would yield training_data::

        time    0   1   2   3   4   5
        user1   X   X
        user2       X

    test_data_in::

        time    0   1   2   3   4   5
        user1   X   X
        user2       X

    test_data_out::

        time    0   1   2   3   4   5
        user1
        user2           X   X


    :param t: Timestamp to split on in seconds since epoch.
    :type t: int
    :param delta_out: Seconds past splitting timestamp to use for hold-out datasets.
        Defaults to None (infinity).
    :type delta_out: int, optional
    :param delta_in: Seconds before splitting timestamp to use for training or fold-in datasets.
        Defaults to None (infinity).
    :type delta_in: int, optional
    :param t_validation: Timestamp to split on in seconds since epoch.
        Validation interactions in [t, min(t_validation, t+delta_out)]
        are added to the validation_data_out dataset.
        Is required if validation is True.
    :type t_validation: int, optional
    :param validation: Assign a portion of the training dataset to validation data if True,
        else split without validation data into only a training and test dataset.
    :type validation: boolean, optional
    """

    def __init__(
        self, t, t_validation=None, delta_out=None, delta_in=None, validation=False
    ):
        super().__init__(validation=validation)
        self.t = t
        self.delta_out = delta_out
        self.delta_in = delta_in
        self.t_validation = t_validation
        if self.validation and not self.t_validation:
            raise Exception(
                "t_validation should be provided when requesting a validation dataset."
            )

        self.timestamp_spl = splitter_base.TimestampSplitter(t, delta_out, delta_in)

        if self.validation:
            assert self.t_validation < self.t
            # Override the validation splitter to a timed splitter.
            self.validation_time_splitter = splitter_base.TimestampSplitter(
                t_validation, delta_out, delta_in
            )

    def _split(self, data):
        """Splits your dataset into a training, validation and test dataset
            based on the timestamp of the interaction.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        lt_t, gt_t = self.timestamp_spl.split(data)

        if self.validation:
            (
                self.train_X,
                self._validation_data_out,
            ) = self.validation_time_splitter.split(lt_t)
            self._validation_data_in = self.train_X.copy()
        else:
            self.train_X = lt_t

        self._test_data_in = lt_t
        self._test_data_out = gt_t


class StrongGeneralizationTimed(Scenario):
    """Splits data into non overlapping train, validation and test user sets
    and uses temporal information to split validation and test into in and out folds.

    A scenario is stateful. After calling ``split`` on your dataset,
    the training, validation and test dataset can be retrieved under
    :attr:`training_data`, :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.

    :attr:`training_data` contains ``frac_users_in`` of the users
    if no validation data is requested.
    Training data is constructed by using all interactions of training users whose timestamps
    are in the interval ``t - delta_in, t[``.

    If validation data is requested, 80% of ``frac_users_in`` users are used as training users,
    and the remaining 20% as validation users:
    - :attr:`validation_data_in` contains all interactions of the validation users
    with timestamps in ``[t_validation - delta_in, t_validation[``.
    - :attr:`validation_data_out` are the interactions of the validation users,
    with timestamps in ``[t_validation, min(t, t_validation + delta_out)[``

    Test users are the remaining ``1-frac_users_in`` of users.
    Similar to validation data the input and output folds are constructed
    based on the timestamp split:

    - :attr:`test_data_in`: contains interactions of the test users
    with timestamps in ``[t - delta_in, t[``.
    - :attr:`test_data_out` contains interactions of the test users
      with timestamps in ``[t, t + delta_out [``

    **Example**

    As an example, we split this data with ``t = 2``,
    ``delta_in = None (infinity)``, ``delta_out = 2`` and ``validation=False``::

        time    0   1   2   3   4   5
        user1   X   X
        user2       X   X   X   X

    would yield training_data::

        time    0   1   2   3   4   5
        user1   X   X
        user2

    test_data_in::

        time    0   1   2   3   4   5
        user1
        user2       X

    test_data_out::

        time    0   1   2   3   4   5
        user1
        user2           X   X


    :param frac_users_in: The fraction of users to use
        for the training(_validation) dataset.
    :type frac_users_in: float
    :param t: Timestamp to split on in seconds since epoch.
    :type t: int
    :param delta_out: Seconds past splitting timestamp to use for hold-out datasets.
        Defaults to None (infinity).
    :type delta_out: int, optional
    :param delta_in: Seconds before splitting timestamp to use for training or fold-in datasets.
        Defaults to None (infinity).
    :type delta_in: int, optional
    :param t_validation: Timestamp to split on in seconds since epoch.
        Validation interactions in [t, min(t_validation, t+delta_out)]
        are added to the :attr:`validation_data_out` dataset.
        Is required if validation is True.
    :type t_validation: int, optional
    :param validation: Assign a portion of the training dataset to validation data if True,
        else split without validation data into only a training and test dataset.
    :type validation: boolean, optional
    """

    def __init__(
        self,
        frac_users_in,
        t,
        t_validation=None,
        delta_out=None,
        delta_in=None,
        validation=False,
    ):
        super().__init__(validation=validation)
        self.frac_users_in = frac_users_in
        self.t = t
        self.delta_out = delta_out
        self.delta_in = delta_in
        self.t_validation = t_validation
        if self.validation and not self.t_validation:
            raise Exception(
                "t_validation should be provided when using validation split."
            )

        self.timestamp_spl = splitter_base.TimestampSplitter(t, delta_out, delta_in)

        self.strong_gen = splitter_base.StrongGeneralizationSplitter(frac_users_in)

        if self.validation:
            assert self.t_validation < self.t
            self.validation_time_splitter = splitter_base.TimestampSplitter(
                t_validation, delta_out, delta_in
            )

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
        tr_val_data, _ = self.timestamp_spl.split(tr_val_data)

        if self.validation:
            train_data, validation_data = self.validation_splitter.split(tr_val_data)
            # Split validation data into input and output on t_validation
            (
                self._validation_data_in,
                self._validation_data_out,
            ) = self.validation_time_splitter.split(validation_data)
            # select the right train data.
            self.train_X, _ = self.validation_time_splitter.split(train_data)
        else:
            self.train_X = tr_val_data

        self._test_data_in, self._test_data_out = self.timestamp_spl.split(te_data)


class StrongGeneralizationTimedMostRecent(Scenario):
    """Split users into non-overlapping training,
    validation and test user sets based on the time of
    their most recent interaction.

    A scenario is stateful. After calling ``split`` on your dataset,
    the training, validation and test dataset can be retrieved under
    :attr:`training_data`, :attr:`validation_data` (:attr:`validation_data_in`, :attr:`validation_data_out`)
    and :attr:`test_data` (:attr:`test_data_in`, :attr:`test_data_out`) respectively.


    Test data contains all users whose most recent interactions was after ``t``:
    - :attr:`test_data_out` contains the ``n`` most recent interactions of
    a user whose most recent interactions was after ``t``.
    - :attr:`test_data_in` contains all earlier interactions of the test users.

    If validation is True, the training data contains users
    whose most recent interaction happened before ``t_validation``.
    Otherwise it contains users whose most recent interaction happened before ``t``.

    In the case where validation is True, the validation data contains all
    users whose most recent interaction was in the interval ``[t_validation, t[``:
    - :attr:`validation_data_out` contains the ``n`` most recent interactions of
    a user whose most recent interactions was in the interval ``[t_validation, t[``.
    - :attr:`validaton_data_in` contains all earlier interactions of the validation users.

    **Example**

    As an example, splitting following data with ``t = 2``,
    ``n = 1`` and ``validation=False``::

        time    0   1   2   3   4   5
        user1   X   X
        user2       X   X   X   X

    would yield training_data::

        time    0   1   2   3   4   5
        user1   X   X
        user2

    test_data_in::

        time    0   1   2   3   4   5
        user1
        user2       X   X   X

    test_data_out::

        time    0   1   2   3   4   5
        user1
        user2                   X

    :param t: Users whose last action has time >= t are placed in the test set,
              all other users are placed in the training or validation sets.
    :param t_validation: Users whose last action has time >= t_validation and
                         time < t are put in the validation set. Users whose
                         last action has time < t_validation are put in train.
                         Only required if validation is True.
    :param n: The n most recent user actions to split off. If a negative number,
              split off all but the ``n`` earliest actions.
    :param validation: Assign a portion of the training dataset to validation data if True,
        else split without validation data into only a training and test dataset.
    :type validation: boolean, optional
    """

    def __init__(
        self, t: float, t_validation: float = None, n: int = 1, validation: bool = False
    ):
        super().__init__(validation=validation)
        self.t = t
        self.t_validation = t_validation
        self.n = n
        if self.validation and not self.t_validation:
            raise Exception(
                "t_validation should be provided when using validation split."
            )

        self.user_splitter_test = splitter_base.UserInteractionTimeSplitter(t)
        if self.validation:
            assert self.t_validation < self.t
            self.user_splitter_val = splitter_base.UserInteractionTimeSplitter(
                t_validation
            )
        self.most_recent_splitter = splitter_base.MostRecentSplitter(n)

    def _split(self, data):
        """Splits users into non-overlapping training,
            validation and test user sets based on the time of
            their most recent interaction.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        tr_val_data, te_data = self.user_splitter_test.split(data)

        self._test_data_in, self._test_data_out = self.most_recent_splitter.split(
            te_data
        )

        if self.validation:
            self.train_X, val_data = self.user_splitter_val.split(tr_val_data)
            (
                self._validation_data_in,
                self._validation_data_out,
            ) = self.most_recent_splitter.split(val_data)
        else:
            self.train_X = tr_val_data
