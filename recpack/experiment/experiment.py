import random
import os
import datetime

import numpy as np
import scipy.sparse

from sklearn.pipeline import Pipeline
from recpack.data.data_matrix import DataM

from recpack.data.data_source import DataSource

from recpack.utils import to_tuple, dict_to_csv, InteractionsCSVWriter

from recpack.splitters.splitter_base import FoldIterator
from recpack.experiment.transform_predictions import FilterHistory

from tqdm.auto import tqdm
import functools

from recpack.utils.globals import (
    BASE_OUTPUT_DIR, PARAMS_FILE, STATISTICS_FILE, HISTORY_FILE, Y_TRUE_FILE, Y_PRED_FILE,
    METRICS_FILE
)


def provider(f):
    cache = dict()
    @functools.wraps(f)
    def wrapper(self, **kwargs):
        if self not in cache:
            cache[self] = f(self, **kwargs)
        return cache[self]
    return wrapper


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


class IEvaluator(object):
    def __init__(self):
        super().__init__()

    def evaluate(self, X, y, recommendations):
        """ Evaluate recommendations. X is the history and y the left out 'ground truth'. """
        pass


class IExperiment(DataSource, ISceneario, IEvaluator):
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
        # TODO: decouple from parameters (join identifier of algo, scenario and datasource)
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

    def recommender(self):
        raise NotImplementedError("Need to override Experiment.recommender")

    def transformers(self):
        """ Return a list of (name, transformer) pairs to be applied before the recommendation. """
        raise NotImplementedError("Need to override Experiment.transformers")

    def pipeline(self):
        raise NotImplementedError("Need to override Experiment.pipeline")

    def fit(self, X, y=None):
        raise NotImplementedError("Need to override Experiment.fit")

    def predict(self, X):
        raise NotImplementedError("Need to override Experiment.predict")

    def iter_predict(self, X, y, **predict_params):
        raise NotImplementedError("Need to override Experiment.iter_predict")

    def transform_predictions(self, batch_iterator):
        raise NotImplementedError("Need to override Experiment.transform_predictions")

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
        params['algorithm'] = self.algorithm_name
        return params

    @provider
    def pipeline(self, _cache=False):
        return Pipeline(
            steps=self.transformers() + [
                ("recommender", self.recommender()),
            ],
            memory="cache" if _cache else None
        )

    @provider
    def transformers(self):
        """ Return a list of (name, transformer) pairs to be applied before the recommendation. """
        return []

    @property
    def algorithm_name(self):
        return self.recommender().name

    def get_output_file(self, filename):
        return os.path.join(self.output_path, filename)

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

    def transform_predictions(self, batch_iterator, filter_history: bool = True):
        """
        Apply business rules after the recommendation step and before evaluation.
        These may influence the calculated performance.
        filter_history indicates whether interactions in the training history should be filtered from the recommendations.
        """
        batch_iterator = super().transform_predictions(batch_iterator)
        if filter_history:
            batch_iterator = FilterHistory(batch_iterator)
        yield from batch_iterator

    def generate_recommendations(self, y_pred, _max_k: int = 100):
        """ Extract the sparse top-K recommendations from a dense list of predictions for each user"""
        if _max_k < 0:
            return scipy.sparse.csr_matrix(y_pred)
        else:
            items = np.argpartition(y_pred, -_max_k)[:, -_max_k:]

        U, I, V = [], [], []
        for ix in range(y_pred.shape[0]):
            U.extend([ix] * _max_k)
            I.extend(items[ix])
            V.extend(y_pred[ix, items[ix]])

        y_pred_top_K = scipy.sparse.csr_matrix(
            (V, (U, I)), dtype=y_pred.dtype, shape=y_pred.shape
        )
        return y_pred_top_K

    def save(self, hist, y_true, y_pred):
        # save params
        cleaned_params = {k[1:] if k[0] == "_" else k: v for k, v in self.get_params().items()}
        dict_to_csv(cleaned_params, self.get_output_file(PARAMS_FILE))

        # save statistics
        dict_to_csv(self.statistics, self.get_output_file(STATISTICS_FILE))

        # save results
        writer = InteractionsCSVWriter(user_id_mapping=self.get_user_id_mapping(), item_id_mapping=self.get_item_id_mapping())
        writer.sparse_to_csv(hist, self.get_output_file(HISTORY_FILE), values=False)
        writer.sparse_to_csv(y_true, self.get_output_file(Y_TRUE_FILE), values=False)
        writer.sparse_to_csv(y_pred, self.get_output_file(Y_PRED_FILE))

    def process_predictions(self, X_test, y_test, batch_iterator):
        recommendations = scipy.sparse.lil_matrix(y_test.shape)
        for user_ids, X, y_true, y_pred in batch_iterator:
            recommendations[user_ids] = self.generate_recommendations(y_pred)

        return recommendations

    def run(self):
        data = to_tuple(self.preprocess())
        train, (X_test, y_test) = self.split(*data)
        # train = to_tuple(train)
        train = map(lambda x: x.values, to_tuple(train))

        self.fit(*train)

        batch_iterator = self.iter_predict(X_test, y_test)
        batch_iterator = self.transform_predictions(batch_iterator)

        recommendations = self.process_predictions(X_test, y_test, batch_iterator)
        self.save(X_test.values, y_test.values, recommendations)

        # HACK to remove zero rows
        # TODO: should map ids again for propper saving of detailed metrics
        X_test = DataM(X_test.values[X_test.values.getnnz(axis=1) > 0])
        y_test = DataM(y_test.values[y_test.values.getnnz(axis=1) > 0])
        recommendations = recommendations[recommendations.getnnz(axis=1) > 0]
        # END of hack

        data = self.evaluate(X_test.values, y_test.values, recommendations)

        dict_to_csv(data, self.get_output_file(METRICS_FILE))



# TODO:
#  - preprocessing options (binary values, etc)
#  - auto generate sweep file
#  - change saving structure to: data/scenario/algo/algo_params/seed
#  - allow to inject default values through functions (to override specific parts)
#  - group parameters per class? (easier to derive unique name for algo and generate sweep file)
#  - entry point for eval?



