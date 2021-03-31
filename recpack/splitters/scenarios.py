from recpack.data.matrix import InteractionMatrix
from recpack.splitters.scenario_base import Scenario
import recpack.splitters.splitter_base as splitter_base


class StrongGeneralization(Scenario):
    """Splits your data so that a user can only be in one of
    training, validation or test.

    During splitting each user is randomly assigned to one of the three groups of users.
    Training data will contain ``frac_users_train``
    fraction of the users, without validation.
    If validation is requested the Training data
    is split into 80% training data, and 20% validation data.

    Test and validation users have their interactions split over an
    input and output matrix,
    based on the ``frac_interactions_in`` parameter.

    Strong generalization is considered more robust, harder and more realistic than
    :class:`class recpack.splitters.scenarios.WeakGeneralization`

    **Example**

    As an example, splitting following data with ``frac_users_train = 0.5``,
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

    :param frac_users_train: Fraction of users used for training, between 0 and 1.
    :type frac_users_train: float
    :param frac_interactions_in: Fraction of the user's
        interactions to be used as history.
    :type frac_interactions_in: float
    :param validation: Split
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
        """Split the data and make ready for use.

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

        self._test_data_in, self._test_data_out = self.interaction_split.split(test_data)


class WeakGeneralization(Scenario):
    """Weak generalization splits the data by
    dividing the interactions of a user over the three splits.

    Interactions are split per user into three splits,
    such that training dataset contains ``frac_interactions_train``
    fraction of the interactions,
    validation contains ``frac_interactions_validation`` fraction
    of the interactions, and test contains the remaining interactions.

    ``frac_interactions_train + frac_interactions_validation``
    is required to be less than 1.

    If ``validation == False``, ``frac_interactions_validation`` is 0.

    **Validation data**
    The training dataset is used as input set for the validation data.
    The validation output dataset are the events sampled in the previous step.

    **Test Data**
    The union of train samples and validation samples are
    used as test input dataset.
    The test output dataset are the test samples.

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

    :param frac_interactions_train: Percentage of interactions per user used for
        training. The interactions are randomly chosen.
    :type frac_interactions_train: float
    :param frac_interactions_validation: Percentage of interactions per user used
        for validation. The interactions are randomly chosen.
    :type frac_interactions_validation: float, optional
    :param validation: Split includes validation sets as well
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

        if frac_interactions_validation > 0:
            # TODO Maybe I should raise errors instead?
            assert validation

        if validation:
            self.frac_interactions_validation = frac_interactions_validation
            self.validation_splitter = splitter_base.FractionInteractionSplitter(
                frac_interactions_train
                / (frac_interactions_train + frac_interactions_validation)
            )

    def _split(self, data: InteractionMatrix):

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
    """Splits the data on temporal information.

    **Training Data**
    Training data is constructed by using all events whose timestamps
    are in the interval ``[t - alpha, t[``.
    Or if validation is requested, with timestamps in
    ``[t_validation - alpha, t_validation [``


    **Validation data**
    If ``validation == True`` validation data is constructed based
    on the timestamps of events,
    and split arround ``t_validation``.

    - Validation input is data with timestamps in
      ``[t_validation - alpha, t_validation [``
    - Validation expected output is data with timestamps in
      ``[t_validation, min(t, t_validation + delta) [``

    **Test Data**
    The test data contains all events, split on timestamp ``t``.

    - Test input are those events with timestamps in  ``[t - alpha, t[``
    - Test expected output are events with timestamps in ``[t, t + delta[``,

    **Example**

    As an example, splitting following data with ``t = 2``,
    ``delta = 2``, ``alpha = inf`` and ``validation=False``::

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


    :param t: Timestamp (as seconds since epoch) to split
        training data from test data.
    :type t: int
    :param t_delta: Bound on the timestamps of expected output in the test data,
        default is inf.
    :type t_delta: int, optional
    :param t_alpha: Bound on the timestamps of the training data, default is inf.
    :type t_alpha: int, optional
    :param t_validation: the timestamp to split training data from validation data.
        Is required if validation is True
    :type t_validation: int, optional
    :param validation: If true validation data will be generated, otherwise it won't
    :type validation: boolean, optional
    """

    def __init__(
        self, t, t_validation=None, t_delta=None, t_alpha=None, validation=False
    ):
        super().__init__(validation=validation)
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha
        self.t_validation = t_validation
        if self.validation and not self.t_validation:
            raise Exception(
                "t_validation should be provided when using validation split."
            )

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)

        if self.validation:
            assert self.t_validation < self.t
            # Override the validation splitter to a timed splitter.
            self.validation_time_splitter = splitter_base.TimestampSplitter(
                t_validation, t_delta, t_alpha
            )

    def _split(self, data):
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
    """Splits data into non overlapping train, validation and test user sets,
    and uses temporal information to split in and out folds.

    **Train Data**
    Training data contains a ``frac_users_in`` fraction of the users
    if no validation is requested.
    If validation is requested, 80% of these users are used as training users,
    and the remaining 20% as validation users.
    Training data is constructed by using all events from those users whose timestamps
    are in the interval ``t - alpha, t[``. Or if validation is requested,
    with timestamps in ``[t_validation - alpha, t_validation [``

    **Validation data**
    If ``validation == True``,
    validation users are the 20% of users from the training split.
    Validation data is constructed from the events of these users split on
    their interaction timestamps.

    - Validation input are the events whose timestamps are in
      ``[t_validation - alpha, t_validation [``
    - Validation expected output are the events from the validation users,
      whose timestamps are in [t_validation, min(t, t_validation + delta) [``

    **Test Data**
    Test users are the remaining ``1-frac_users_in`` fraction of the users.
    Similar to validation data the input and output folds are constructed
    based on the timestamp split.

    - Test input contains events from the test users
      whose timestamps are in ``[t - alpha, t[``
    - Test expected output contains events from test users
      with timestamps in ``[t, t + delta [``

    **Example**

    As an example, splitting following data with ``t = 2``,
    ``delta = 2``, ``alpha = inf`` and ``validation=False``::

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
    :param t: Timestamp (in seconds since epoch) to split the data on.
    :type t: int
    :param t_delta: Bound on the timestamps of expected output of the test data,
        default is inf.
    :type t_delta: int, optional
    :param t_alpha: Bound on the timestamps of training data, default is inf.
    :type t_alpha: int, optional
    :param t_validation: the timestamp to split training data from validation data.
        Is required if validation is True
    :type t_validation: int, optional
    :param validation: Split
    :type validation: boolean, optional

    """

    def __init__(
        self,
        frac_users_in,
        t,
        t_validation=None,
        t_delta=None,
        t_alpha=None,
        validation=False,
    ):
        super().__init__(validation=validation)
        self.frac_users_in = frac_users_in
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha
        self.t_validation = t_validation
        if self.validation and not self.t_validation:
            raise Exception(
                "t_validation should be provided when using validation split."
            )

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)

        self.strong_gen = splitter_base.StrongGeneralizationSplitter(frac_users_in)

        if self.validation:
            assert self.t_validation < self.t
            self.validation_time_splitter = splitter_base.TimestampSplitter(
                t_validation, t_delta, t_alpha
            )

    def _split(self, data):
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
    """Splits users into non overlapping train,
    validation and test user sets based on the time of
    their most recent action.

    **Train Data**
    If validation is enabled, the train data contains users
    whose most recent interaction happened before ``t_validation``.
    Otherwise it contains users whose most recent interaction happened before ``t``.

    **Validation Data**
    If validation is requested, contains users with their most recent interaction
    in the interval ``[t_validation, t[``.

    - Validation expected output are the ``n`` most recent events for each user
    - Validation input are the remaining events for each user.

    **Test Data**
    Test data contains all users whose most recent interactions was after ``t``.

    - Test expected output (test_data_out) contains the ``n`` most recent events
    - Test input (test_data_in) contains all earlier events for the test users.

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
    :param validation: Create validation sets
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
        tr_val_data, te_data = self.user_splitter_test.split(data)

        self._test_data_in, self._test_data_out = self.most_recent_splitter.split(te_data)

        if self.validation:
            self.train_X, val_data = self.user_splitter_val.split(tr_val_data)
            (
                self._validation_data_in,
                self._validation_data_out,
            ) = self.most_recent_splitter.split(val_data)
        else:
            self.train_X = tr_val_data
