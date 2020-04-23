from recpack.splitters.scenario_base import Scenario
import recpack.splitters.splitter_base as splitter_base
from recpack.utils import get_logger


class StrongGeneralization(Scenario):
    def __init__(self, perc_users_in):
        super().__init__()
        self.perc_users_in = perc_users_in

        self.strong_gen = splitter_base.StrongGeneralizationSplitter(perc_users_in)
        self.interaction_split = splitter_base.InteractionSplitter()

    def split(self, data):
        self.training_data, te_data = self.strong_gen.split(data)
        self.test_data_in, self.test_data_out = self.interaction_split.split(te_data)


class TrainingInTestOutTimed(Scenario):

    def __init__(self, t, t_delta=None, t_alpha=None):
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)

    def split(self, data):
        self.training_data, self.test_data_out = self.timestamp_spl.split(data)
        self.test_data_in = self.training_data


class StrongGeneralizationTimed(Scenario):
    """
    Split data in strong generalization fashion, also splitting on timestamp.
    training_data = user in set of sampled test users + event time < t
    test_data_in = user not in the set of test users + event_time < t
    test_data_out = user not in the set of test users + event_time > t
    """
    # TODO Make this more standard.

    def __init__(self, perc_users_in, t, t_delta=None, t_alpha=None):
        super().__init__()
        self.perc_users_in = perc_users_in
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)
        self.strong_gen = splitter_base.StrongGeneralizationSplitter(perc_users_in)

    def split(self, data, data_2):
        # Split across user dimension
        tr_data, te_data = self.strong_gen.split(data)
        # Make sure only interactions before t are used for training
        self.training_data, _ = self.timestamp_spl.split(tr_data)

        self.test_data_in, self.test_data_out = self.timestamp_spl.split(te_data)


class TimedOutOfDomainPredictAndEvaluate(Scenario):
    """
    Class to split data from 2 input datatypes, one of the data types will be used as training data,
    the other one will be used as test data.
    An example use case is training a model on pageviews,
    and evaluating with purchases as test_in, test_out and split on time

    training_data = data_1 & timestamp < t
    test_data_in = data_2 & timestamp < t
    test_data_out = data_2 & timestamp > t
    """
    # TODO Make this more standard.

    def __init__(self, t, t_alpha=None, t_delta=None):
        super().__init__()
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)

    def split(self, data, data_2):
        self.training_data, _ = self.timestamp_spl.split(data)
        self.test_data_in, self.test_data_out = self.timestamp_spl.split(data_2)


class TrainInTimedOutOfDomainEvaluate(Scenario):
    """
    Class to split two input datatypes for training and testing.
    train_data = data_1 & timestamp < t
    test_in = train_data
    test_out = data_2 & timestamp > t
    """
    # TODO Make this more standard.

    def __init__(self, t, t_alpha=None, t_delta=None):
        super().__init__()
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)

    def split(self, data, data_2):
        self.training_data, _ = self.timestamp_spl.split(data)
        _, self.test_data_out = self.timestamp_spl.split(data_2)
        self.test_data_in = self.training_data
