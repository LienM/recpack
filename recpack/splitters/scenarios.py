from recpack.splitters.scenario_base import Scenario
import recpack.splitters.splitter_base as splitter_base


class StrongGeneralizationTimed(Scenario):
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
    # TODO Make this more standard.

    def __init__(self, t, t_alpha=None, t_delta=None):
        super().__init__()
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)

    def split(self, data, data_2):
        self.training_data, _ = self.timestamp_spl(data)
        self.test_data_in, self.test_data_out = self.timestamp_spl(data_2)


class TrainInTimedOutOfDomainEvaluate(Scenario):
    # TODO Make this more standard.

    def __init__(self, t, t_alpha=None, t_delta=None):
        super().__init__()
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

        self.timestamp_spl = splitter_base.TimestampSplitter(t, t_delta, t_alpha)

    def split(self, data, data_2):
        self.training_data, _ = self.timestamp_spl(data)
        _, self.test_data_out = self.timestamp_spl(data_2)
        self.test_data_in = self.training_data
