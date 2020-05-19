import math

import numpy as np
import scipy.sparse

# from functools import cached_property
from cached_property import cached_property

from recpack.utils import to_tuple

from recpack.splitters.splitter_base import FoldIterator

from tqdm.auto import tqdm


class ExperimentMeta(type):
    BASE_CLASSES = ["IExperiment", "Experiment", "LoggingExperiment"]

    def __new__(meta, name, bases, attrs):
        cls = super().__new__(meta, name, bases, attrs)
        if name not in ExperimentMeta.BASE_CLASSES:
            print(meta, name, bases, attrs)
        return cls


class IExperiment(metaclass=ExperimentMeta):
    """
    Extend from this class to create experiments from scratch. The Experiment class already has the basic functionallity.
    Every step of the process can be overriden individually to create a custom experiment structure.
    Keyword arguments are injected into the respective functions from the command line. For example, preprocess can
    take a path as parameter in the subclass, this will then be injected from the command line automatically.
    In other words, when calling preprocess, only the arguments defined in Experiment need to be given in Python.

    Parameters starting with underscore are not logged, but can be supplied through the command line.
    """
    def __init__(self):
        super().__init__()
        self.parameters = dict()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def identifier(self):
        paramstring = "_".join((f"{k}_{v}" for k, v in self.parameters.items()))
        return self.name + "__" + paramstring

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def preprocess(self):
        """
        Return one or two datasets of type np.ndarray.
        """
        raise NotImplementedError("Need to override Experiment.preprocess")

    def split(self, X, y=None):
        """
        Return train and test sets.
        Train can be single variable, or tuple with features and labels.
        Test set is always a tuple of history and ground truth.
        """
        raise NotImplementedError("Need to override Experiment.split")

    def fit(self, X, y=None, **fit_params):
        raise NotImplementedError("Need to override Experiment.fit")

    def predict(self, X, **predict_params):
        raise NotImplementedError("Need to override Experiment.predict")

    def iter_predict(self, X, y, **predict_params):
        raise NotImplementedError("Need to override Experiment.iter_predict")

    def transform_predictions(self, user_iterator):
        """
        Apply business rules after the recommendation step and before evaluation.
        These may influence the calculated performance.
        Example:
        def transform_predictions(self, user_iterator, no_repeats=False):
            user_iterator = super().transform_predictions(user_iterator)
            if no_repeats:
                user_iterator = NoRepeats(user_iterator)
            yield from user_iterator
        """
        yield from user_iterator

    def eval_user(self, X, y_true, y_pred, user_id=None):
        raise NotImplementedError("Need to override Experiment.eval_user")

    def run(self):
        raise NotImplementedError("Need to override Experiment.run")


class Experiment(IExperiment):
    """
    Extend from this class to create experiments with already some default behavior. Uses pipelines and metrics@K.
    Every step of the process can be overriden individually to create a custom experiment structure.
    Keyword arguments are injected into the respective functions from the command line. For example, preprocess can
    take a path as parameter in the subclass, this will then be injected from the command line automatically.
    In other words, when calling preprocess, only the arguments defined in Experiment need to be given in Python.

    Parameters starting with underscore are not logged, but can be supplied through the command line.
    """
    def __init__(self, _batch_size=1000):
        super().__init__()
        self.batch_size = _batch_size

    @property
    def name(self):
        return self.__class__.__name__

    @cached_property
    def pipeline(self):
        raise NotImplementedError("Need to override Experiment.pipeline")

    @cached_property
    def metrics(self):
        return []

    @cached_property
    def K_values(self):
        return [1, 5, 10, 20, 50, 100]

    def preprocess(self):
        """
        Return one or two datasets of type np.ndarray.
        """
        raise NotImplementedError("Need to override Experiment.preprocess")

    def split(self, X, y=None):
        """
        Return train and test sets.
        Train can be single variable, or tuple with features and labels.
        Test set is always a tuple of history and ground truth.
        """
        return (X, None), (X, y)

    def fit(self, X, y=None, **fit_params):
        self.pipeline.fit(X, y, **fit_params)

    def predict(self, X, **predict_params):
        return self.pipeline.predict(X, **predict_params)

    def iter_predict(self, X, y, **predict_params):
        Xt = X
        for _, name, transform in self.pipeline._iter(with_final=False):
            Xt = transform.transform(Xt)

        for _in, _out, user_ids in FoldIterator(Xt, y, batch_size=self.batch_size):
            y_pred = self.pipeline.steps[-1][-1].predict(_in, **predict_params)
            if scipy.sparse.issparse(y_pred):
                # to dense format
                y_pred = y_pred.toarray()
            else:
                # ensure ndarray instead of matrix
                y_pred = np.asarray(y_pred)

            for user_id, y_hist_u, y_true_u, y_pred_u in zip(user_ids, _in, _out, y_pred):
                yield user_id, y_hist_u, y_true_u, y_pred_u

    def transform_predictions(self, user_iterator):
        """
        Apply business rules after the recommendation step and before evaluation.
        These may influence the calculated performance.
        Example:
        def transform_predictions(self, user_iterator, no_repeats=False):
            user_iterator = super().transform_predictions(user_iterator)
            if no_repeats:
                user_iterator = NoRepeats(user_iterator)
            yield from user_iterator
        """
        yield from user_iterator

    def eval_user(self, X, y_true, y_pred, user_id=None):
        pass
        # print("X", X)
        # print("y_true", y_true)
        # print("y_pred", y_pred)
        # print("user_id", user_id)

    def run(self):
        data = to_tuple(self.preprocess())
        train, (X_test, y_test) = self.split(*data)
        train = to_tuple(train)

        self.fit(*train)
        user_iterator = self.iter_predict(X_test, y_test)

        # TODO: can be parallel maybe
        for user_id, y_hist_u, y_true_u, y_pred_u in tqdm(user_iterator, total=len(set(X_test.indices[0]))):
            self.eval_user(y_hist_u, y_true_u, y_pred_u, user_id=user_id)


# TODO:
#  - metrics and evaluation
#  - parse command (class name) and run if provided
#  - parse parameters and inject based on diff with interface (Experiment)
#  - find better solution to use of cached_property
#  - logging with wandb and ExperimentContext (refactor into LoggingExperiment?)



