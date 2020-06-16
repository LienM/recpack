import random
import csv
import os
import datetime

import numpy as np
import scipy.sparse

from typing import List

from recpack.utils import to_tuple, sparse_to_csv, dict_to_csv

from recpack.splitters.splitter_base import FoldIterator

from tqdm.auto import tqdm
import functools


BASE_OUTPUT_DIR = "output/"


def provider(f):
    cache = dict()
    @functools.wraps(f)
    def wrapper(self, **kwargs):
        if self not in cache:
            cache[self] = f(self, **kwargs)
        return cache[self]
    return wrapper


class IDataSource(object):
    def __init__(self):
        super().__init__()

    @property
    def data_name(self):
        raise NotImplementedError("Need to override data_name")

    def get_params(self):
        params = super().get_params() if hasattr(super(), "get_params") else dict()
        params['data_source'] = self.data_name
        return params

    def preprocess(self):
        """
        Return one or two datasets of type np.ndarray.
        """
        raise NotImplementedError("Need to override Experiment.preprocess")


class ISceneario(object):
    def __init__(self):
        super().__init__()

    @property
    def scenario_name(self):
        raise NotImplementedError("Need to override scenario_name")

    def get_params(self):
        params = super().get_params() if hasattr(super(), "get_params") else dict()
        params['scenario'] = self.scenario_name
        return params

    def split(self, X, y=None):
        """
        Return train and test sets.
        Train can be single variable, or tuple with features and labels.
        Test set is always a tuple of history and ground truth.
        """
        raise NotImplementedError("Need to override Experiment.split")

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


class IExperiment(IDataSource, ISceneario):
    """
    Extend from this class to create experiments from scratch. The Experiment class already has the basic functionallity.
    Every step of the process can be overriden individually to create a custom experiment structure.
    Keyword arguments are injected into the respective functions from the command line. For example, preprocess can
    take a path as parameter in the subclass, this will then be injected from the command line automatically.
    In other words, when calling preprocess, only the arguments defined in Experiment need to be given in Python.

    Parameters starting with underscore are not included in the identifier, but can still be supplied through the
    command line and are still logged as parameters.
    """
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def identifier(self):
        identifying_params = {k: v for k, v in self.get_params().items() if not k[0] == "_"}
        paramstring = "_".join((f"{k}_{v}" for k, v in sorted(identifying_params.items())))
        return self.name + "__" + paramstring

    def get_output_file(self, filename):
        raise NotImplementedError("Need to override Experiment.get_output_file")

    def get_params(self):
        return super().get_params() if hasattr(super(), "get_params") else dict()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def fit(self, X, y=None):
        raise NotImplementedError("Need to override Experiment.fit")

    def predict(self, X):
        raise NotImplementedError("Need to override Experiment.predict")

    def iter_predict(self, X, y, **predict_params):
        raise NotImplementedError("Need to override Experiment.iter_predict")

    # def eval_user(self, X, y_true, y_pred, user_id=None):
    #     raise NotImplementedError("Need to override Experiment.eval_user")
    #
    # def eval_batch(self, X, y_true, y_pred, user_ids=None):
    #     raise NotImplementedError("Need to override Experiment.eval_batch")

    def generate_recommendations(self, y_pred):
        pass

    def save(self, hist, y_true, y_pred):
        pass

    def process_predictions(self, X_test, y_test, batch_iterator):
        pass

    def run(self):
        raise NotImplementedError("Need to override Experiment.run")


class Experiment(IExperiment):
    """
    Extend from this class to create experiments with already some default behavior. Uses pipelines and metrics@K.
    Every step of the process can be overriden individually to create a custom experiment structure.
    Keyword arguments are injected into the respective functions from the command line. For example, preprocess can
    take a path as parameter in the subclass, this will then be injected from the command line automatically.
    In other words, when calling preprocess, only the arguments defined in Experiment need to be given in Python.

    Parameters starting with underscore are not included in the identifier, but can still be supplied through the
    command line and are still logged as parameters.
    """
    def __init__(self, _seed: int = None, _batch_size: int = 1000):
        super().__init__()
        self.seed = random.randint(0, 2**32-1) if _seed is None else _seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.batch_size = _batch_size

        self.statistics = dict()
        self.statistics["start_time"] = datetime.datetime.now().replace(microsecond=0)

        self.output_path = os.path.join(BASE_OUTPUT_DIR, str(self.identifier), str(self.seed))

        os.makedirs(self.output_path)

    @property
    def name(self):
        return self.__class__.__name__

    @functools.lru_cache(maxsize=None)
    def get_params(self):
        params = super().get_params()
        params['_seed'] = self.seed
        return params

    @provider
    def pipeline(self):
        raise NotImplementedError("Need to override Experiment.pipeline")

    def get_output_file(self, filename):
        return os.path.join(self.output_path, filename)

    # @provider
    # def metrics(self):
    #     return []
    #
    # @provider
    # def k_values(self):
    #     return [1, 5, 10, 20, 50, 100]

    # @lru_cache(maxsize=None)
    # def _metrics(self):
    #     metrics = list()
    #     for metric in self.metrics():
    #         for k in self.K_values():
    #             metrics.append(metric(k))
    #     return metrics

    # @provider
    # def metrics(self, metrics: List[str], k_values: List[int]):
    #     metrics = list()
    #     for metric in metrics:
    #         for k in k_values:
    #             metrics.append(metric(k))
    #     return metrics

    def split(self, X, y=None):
        """
        Return train and test sets.
        Train can be single variable, or tuple with features and labels.
        Test set is always a tuple of history and ground truth.
        """
        return (X, None), (X, y)

    def fit_params(self):
        return dict()

    def fit(self, X, y=None):
        start = datetime.datetime.now()
        self.pipeline().fit(X, y, **self.fit_params())
        end = datetime.datetime.now()
        self.statistics["train_duration"] = end - start

    def predict_params(self):
        return dict()

    def predict(self, X):
        return self.pipeline().predict(X, **self.predict_params())

    def iter_predict(self, X, y, **predict_params):
        Xt = X
        for _, name, transform in self.pipeline()._iter(with_final=False):
            Xt = transform.transform(Xt)

        for _in, _out, user_ids in tqdm(FoldIterator(Xt, y, batch_size=self.batch_size)):
            y_pred = self.pipeline().steps[-1][-1].predict(_in, **predict_params)
            if scipy.sparse.issparse(y_pred):
                # to dense format
                y_pred = y_pred.toarray()
            else:
                # ensure ndarray instead of matrix
                y_pred = np.asarray(y_pred)

            yield user_ids, _in, _out, y_pred
            # for user_id, y_hist_u, y_true_u, y_pred_u in zip(user_ids, _in, _out, y_pred):
            #     yield user_id, y_hist_u, y_true_u, y_pred_u

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

    # def eval_user(self, X, y_true, y_pred, user_id=None):
    #     pass

    # def eval_batch(self, X, y_true, y_pred, user_ids=None):
    #     if not scipy.sparse.issparse(y_pred):
    #         y_pred = scipy.sparse.csr_matrix(y_pred)
    #
    #     for metric in self._metrics():
    #         metric.update(y_pred, y_true)
    #
    #     # TODO: can be parallel maybe
    #     # for user_id, y_hist_u, y_true_u, y_pred_u in tqdm(user_iterator, total=len(set(X_test.indices[0]))):
    #     #     self.eval_user(y_hist_u, y_true_u, y_pred_u, user_id=user_id)

    def generate_recommendations(self, y_pred, _max_k: int = 100):
        """ Extract the sparse top-K recommendations from a dense list of predictions for each user"""
        items = np.argpartition(y_pred, -_max_k)[:, -_max_k:]

        U, I, V = [], [], []
        for ix in range(y_pred.shape[0]):
            U.extend([ix] * _max_k)
            I.extend(items[ix])
            V.extend(y_pred[ix, items[ix]])

        y_pred_top_K = scipy.sparse.lil_matrix(
            (V, (U, I)), dtype=y_pred.dtype, shape=y_pred.shape
        )
        return y_pred_top_K

    def save(self, hist, y_true, y_pred):
        # save params
        cleaned_params = {k[1:] if k[0] == "_" else k: v for k, v in self.get_params().items()}
        dict_to_csv(cleaned_params, self.get_output_file("params.csv"))

        # save statistics
        dict_to_csv(self.statistics, self.get_output_file("statistics.csv"))

        # save results
        sparse_to_csv(hist, self.get_output_file("history.csv"))
        sparse_to_csv(y_true, self.get_output_file("y_true.csv"))
        sparse_to_csv(y_pred, self.get_output_file("y_pred.csv"))

    def process_predictions(self, X_test, y_test, batch_iterator):
        recommendations = scipy.sparse.lil_matrix(y_test.shape)
        for user_ids, X, y_true, y_pred in batch_iterator:
            recommendations[user_ids] = self.generate_recommendations(y_pred)

        self.save(X_test, y_test, recommendations)

    def run(self):
        data = to_tuple(self.preprocess())
        train, (X_test, y_test) = self.split(*data)
        # train = to_tuple(train)
        train = map(lambda x: x.values, to_tuple(train))

        self.fit(*train)

        batch_iterator = self.iter_predict(X_test, y_test)
        batch_iterator = self.transform_predictions(batch_iterator)

        self.process_predictions(X_test, y_test, batch_iterator)


# TODO:
#  - metrics and evaluation



