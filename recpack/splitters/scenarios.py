from warnings import warn
from recpack.splitters.scenario_base import Scenario
import recpack.splitters.splitter_base as splitter_base


class StrongGeneralization(Scenario):
    def __init__(
        self, perc_users_train: float, perc_interactions_in: float, validation=False
    ):
        """
        Strong generalization splits your data so that a user can only be in one of
        training, validation or test.
        This is considered more robust, harder and more realistic than weak generalization.

        :param perc_users_train: Percentage of users used for training, between 0 and 1.
        :type perc_users_train: float
        :param perc_interactions_in: Percentage of the user's interactions to be used as history.
        :type perc_interactions_in: float
        :param validation: Split
        :type validation: boolean, optional
        """
        super().__init__(validation=validation)
        self.perc_users_train = perc_users_train
        self.perc_interactions_in = perc_interactions_in

        self.strong_gen = splitter_base.StrongGeneralizationSplitter(perc_users_train)
        self.interaction_split = splitter_base.PercentageInteractionSplitter(
            perc_interactions_in
        )

    def split(self, data):
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

        self.test_data_in, self.test_data_out = self.interaction_split.split(test_data)

        self.validate()


class WeakGeneralization(Scenario):
    def __init__(
        self,
        perc_interactions_train: float,
        perc_interactions_validation: float = 0,
        validation=False,
    ):
        """
        Weak generalization splits the data such that all users in the test set
        are also in the training set.

        :param perc_interactions_train: Percentage of interactions per user used for
            training. The interactions are randomly chosen.
        :type perc_interactions_train: float
        :param perc_interactions_validation: Percentage of interactions per user used
            for validation. The interactions are randomly chosen.
        :type perc_interactions_validation: float, optional
        :param validation: Split includes validation sets as well
        :type validation: boolean, optional
        """
        super().__init__(validation=validation)
        self.perc_interactions_train = perc_interactions_train

        self.interaction_split = splitter_base.PercentageInteractionSplitter(
            perc_interactions_train + perc_interactions_validation
        )

        assert (perc_interactions_train + perc_interactions_validation) < 1

        if perc_interactions_validation > 0:
            # TODO Maybe I should raise errors instead?
            assert validation

        if validation:
            self.perc_interactions_validation = perc_interactions_validation
            self.validation_splitter = splitter_base.PercentageInteractionSplitter(
                perc_interactions_train
                / (perc_interactions_train + perc_interactions_validation)
            )

    def split(self, data):

        train_val_data, self.test_data_out = self.interaction_split.split(data)

        if self.validation:
            self.train_X, self._validation_data_out = self.validation_splitter.split(
                train_val_data
            )
            self._validation_data_in = self.train_X.copy()
        else:
            self.train_X = train_val_data

        self.test_data_in = train_val_data.copy()

        self.validate()


class Timed(Scenario):
    def __init__(
        self, t, t_validation=None, t_delta=None, t_alpha=None, validation=False
    ):
        """
        the data is only split on time.

        If there is validation data, training data is all
        data for which timestamps fall in [t_validation - alpha, t_validation [
        If there is not, training data is all data in [t - alpha, t[.

        Test_out data is all data in [t, t + delta[,
        and test_in is all data in [t - alpha, t[
        Alpha and delta are default inf.

        For validation data if requested,
        the input is all data in [t_validation - alpha, t_validation [,
        and output is all data in [t_validation, min(t, t_validation + delta) [

        :param t: Time (as seconds since epoch) to split along.
        :type t: int
        :param t_delta: Allows bounding of testing interval, default is inf.
        :type t_delta: int, optional
        :param t_alpha: Allows bounding of training interval, default is inf.
        :type t_alpha: int, optional
        :param t_validation: the timestamp to split training interval on,
                            to generate a validation dataset.
                            Is required if validation is True
        :type t_validation: int, optional
        :param validation: Split
        :type validation: boolean, optional
        """
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

    def split(self, data):
        lt_t, gt_t = self.timestamp_spl.split(data)

        if self.validation:
            (
                self.train_X,
                self._validation_data_out,
            ) = self.validation_time_splitter.split(lt_t)
            self._validation_data_in = self.train_X.copy()
        else:
            self.train_X = lt_t

        self.test_data_in = lt_t
        self.test_data_out = gt_t

        self.validate()


class StrongGeneralizationTimed(Scenario):
    """
    The data is split into non overlapping train, validation and test sets.
    Where a user can have interactions in only 1 of the sets.

    The data is further split by the provided timestamps,
    to get a more realistic train and test scenario.

    test_users is a sample of all users of 1-`perc_users_in`.
    train_users are the remaining users (if no validation)
    or a sample of 0.8 of the remaining users (if validation)
    Validation users are the 0.2 of the remaining users
    after the train test split.

    If there is validation data, training data is all
    data for which user is in training_users
    and timestamps fall in [t_validation - alpha, t_validation [
    If there is not, timestamp constraint is [t - alpha, t[.

    Test_out data is all data for which user is in test_users
    and timestamp is in [t, t + delta [,
    and test_in is all data for which user is in test_users
    and timestamp in [t - alpha, t[
    Alpha and delta are default inf.

    For validation data if requested,
    the input is all data for which user is in validation_users
    and timestamp in [t_validation - alpha, t_validation [,
    and output is all data
    for which user is in validation_users
    and timestamp in [t_validation, min(t, t_validation + delta) [

    Split data in strong generalization fashion, also splitting on timestamp.
    training_data = user in set of sampled test users + event time < t
    test_data_in = user not in the set of test users + event_time < t
    test_data_out = user not in the set of test users + event_time > t

    :param perc_users_in: The fraction of users to use
                            for the training(_validation) dataset.
    :type t: float
    :param t: Time (as seconds since epoch) to split along.
    :type t: int
    :param t_delta: Allows bounding of testing interval, default is inf.
    :type t_delta: int, optional
    :param t_alpha: Allows bounding of training interval, default is inf.
    :type t_alpha: int, optional
    :param t_validation: the timestamp to split training interval on,
                        to generate a validation dataset.
                        Is required if validation is True
    :type t_validation: int, optional
    :param validation: Split
    :type validation: boolean, optional

    """

    def __init__(
        self,
        perc_users_in,
        t,
        t_validation=None,
        t_delta=None,
        t_alpha=None,
        validation=False,
    ):
        super().__init__(validation=validation)
        self.perc_users_in = perc_users_in
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha
        self.t_validation = t_validation
        if self.validation and not self.t_validation:
            raise Exception(
                "t_validation should be provided when using validation split."
            )

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)

        self.strong_gen = splitter_base.StrongGeneralizationSplitter(perc_users_in)

        if self.validation:
            assert self.t_validation < self.t
            self.validation_time_splitter = splitter_base.TimestampSplitter(
                t_validation, t_delta, t_alpha
            )

    def split(self, data):
        # Split across user dimension
        tr_val_data, te_data = self.strong_gen.split(data)
        # Make sure only interactions before t are used for training /
        # validation
        tr_val_data, _ = self.timestamp_spl.split(tr_val_data)

        if self.validation:
            # Split 80-20 train and val data.
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

        self.test_data_in, self.test_data_out = self.timestamp_spl.split(te_data)

        self.validate()


class StrongGeneralizationTimedMostRecent(Scenario):
    """
    Users are split into train, validation and test sets based on the time of 
    their most recent action. A user can have interactions in only 1 of the sets.

    The test set is further divided in a test_data_in set containing all but the 
    n most recent actions, and a test_data_out set containing the n most recent 
    actions of the test users. The validation set (if required) is split the same.

    As an example, splitting following data on t = 2 with n = 1, w/o validation

        time    0   1   2   3   4   5
        user1     X   X   
        user2         X   X   X   X

    would yield training_data:

        time    0   1   2   3   4   5
        user1     X   X   
        user2      

    test_data_in:

        time    0   1   2   3   4   5
        user1         
        user2         X   X   X

    test_data_out:

        time    0   1   2   3   4   5
        user1         
        user2                     X

    :param t: Users whose last action has time >= t are placed in the test set,
              all other users are placed in the training or validation sets.
    :param t_validation: Users whose last action has time >= t_validation and 
                         time < t are put in the validation set. Users whose 
                         last action has time < t_validation are put in train.
                         Only required if validation is True.
    :param n: The n most recent user actions to split off. If a negative number,
              split off all but the |n| earliest actions.
    :param validation: Create validation sets
    """

    def __init__(self, t: float, t_validation: float = None, n: int = 1, 
                 validation: bool = False):
        super().__init__(validation=validation)
        self.t = t
        self.t_validation = t_validation
        self.n = n
        if self.validation and not self.t_validation:
            raise Exception(
                "t_validation should be provided when using validation split.")

        self.user_splitter_test = splitter_base.UserInteractionTimeSplitter(t, on="max")
        if self.validation:
            assert self.t_validation < self.t
            self.user_splitter_val = splitter_base.UserInteractionTimeSplitter(
                t_validation, on="max"
            )
        self.most_recent_splitter = splitter_base.MostRecentSplitter(n)

    def split(self, data):
        tr_val_data, te_data = self.user_splitter_test.split(data)

        self.test_data_in, self.test_data_out = self.most_recent_splitter.split(te_data)

        if self.validation:
            self.train_X, val_data = self.user_splitter_val.split(tr_val_data)
            self._validation_data_in, self._validation_data_out = \
                self.most_recent_splitter.split(val_data)
        else:
            self.train_X = tr_val_data

        self.validate()


class TimedOutOfDomainPredictAndEvaluate(Scenario):
    """
    Class to split data from 2 input datatypes,
    one of the data types will be used as training data,
    the other one will be used as test data.
    An example use case is training a model on pageviews,
    and evaluating with purchases as test_in, test_out and split on time

    training_data = data_1 & timestamp < t
    test_data_in = data_2 & timestamp < t
    test_data_out = data_2 & timestamp > t
    """

    # TODO Make this more standard.

    def __init__(
        self, t, t_validation=None, t_alpha=None, t_delta=None, validation=False
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
            self.validation_time_splitter = splitter_base.TimestampSplitter(
                t_validation, t_delta, t_alpha
            )

    def split(self, data, data_2):
        d_1_lt, _ = self.timestamp_spl.split(data)
        d_2_lt, d_2_gt = self.timestamp_spl.split(data_2)

        self.test_data_in, self.test_data_out = (d_2_lt, d_2_gt)

        if self.validation:
            d_1_lt_val, _ = self.validation_time_splitter.split(d_1_lt)
            d_2_lt_val, d_2_gt_val = self.validation_time_splitter.split(d_2_lt)
            self.train_X = d_1_lt_val

            self._validation_data_in, self._validation_data_out = (
                d_2_lt_val,
                d_2_gt_val,
            )
        else:
            self.train_X = d_1_lt

        self.validate()


class TrainInTimedOutOfDomainEvaluate(Scenario):
    """
    Class to split two input datatypes for training and testing.
    train_data = data_1 & timestamp < t
    test_in = train_data
    test_out = data_2 & timestamp > t
    """

    # TODO Make this more standard.

    def __init__(
        self, t, t_validation=None, t_alpha=None, t_delta=None, validation=False
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
            self.validation_time_splitter = splitter_base.TimestampSplitter(
                t_validation, t_delta, t_alpha
            )

    def split(self, data, data_2):
        d_1_lt, _ = self.timestamp_spl.split(data)
        d_2_lt, d_2_gt = self.timestamp_spl.split(data_2)

        self.test_data_in, self.test_data_out = (d_1_lt, d_2_gt)

        if self.validation:
            d_1_lt_val, _ = self.validation_time_splitter.split(d_1_lt)
            d_2_lt_val, d_2_gt_val = self.validation_time_splitter.split(d_2_lt)
            self.train_X = d_1_lt_val

            self._validation_data_in, self._validation_data_out = (
                d_1_lt_val,
                d_2_gt_val,
            )
        else:
            self.train_X = d_1_lt

        self.validate()


class TrainInTimedOutOfDomainWithLabelsEvaluate(TrainInTimedOutOfDomainEvaluate):
    """
    Same as TrainInTimedOutOfDomainEvaluate but with historical data_2 as labels.
    """

    def __init__(
        self, t, t_validation=None, t_alpha=None, t_delta=None, validation=False
    ):
        super().__init__(t, t_validation, t_alpha, t_delta, validation=validation)

    def split(self, data, data_2):
        d_1_lt, _ = self.timestamp_spl.split(data)
        d_2_lt, d_2_gt = self.timestamp_spl.split(data_2)

        self.test_data_in, self.test_data_out = (d_1_lt, d_2_gt)

        if self.validation:
            d_1_lt_val, _ = self.validation_time_splitter.split(d_1_lt)
            d_2_lt_val, d_2_gt_val = self.validation_time_splitter.split(d_2_lt)
            self.train_X = d_1_lt_val

            self._validation_data_in, self._validation_data_out = (
                d_1_lt_val,
                d_2_gt_val,
            )

            train_users = list(self.train_X.active_users)
            self.train_y = d_2_lt_val.users_in(train_users)

        else:
            self.train_X = d_1_lt
            self.train_y = d_2_lt

        self.validate()
