import logging
import os.path

import pandas as pd
import numpy as np

from recpack.experiment.experiment import (
    Experiment, IExperiment, provider, IScenario, IEvaluator
)
from recpack.experiment.globals import HISTORY_FILE, Y_TRUE_FILE, Y_PRED_FILE, METRICS_FILE
from recpack.experiment.CLI import CLI
from recpack.experiment.transform_predictions import NoRepeats
from recpack.splitters.scenarios import StrongGeneralization
from recpack.data.datasets.movielens import ML20MDataSource
from recpack.data.data_source import DataSource

from recpack.algorithms.baseline import Popularity
from recpack.algorithms.similarity.nearest_neighbour import ItemKNN, CosineSimilarity
from recpack.algorithms.similarity.shared_account import SharedAccount, Agg
from recpack.algorithms.similarity.linear import EASE, EASE_AVG, EASE_Intercept, EASE_AVG_Int

from recpack.metrics import NDCGK, RecallK, MeanReciprocalRankK

from recpack.utils import (
    logger, csv_to_sparse, dict_to_csv,
    USER_KEY, ITEM_KEY, VALUE_KEY
)

logger.setLevel(logging.DEBUG)


class StrongGeneralizationScenario(IScenario):
    @property
    def scenario_name(self):
        return "StrongGeneralization"

    def split(self, X, y=None, test_users=10000, history_ratio=0.8):
        user_train_ratio = (X.shape[0] - test_users) / X.shape[0]
        scenario = StrongGeneralization(user_train_ratio, history_ratio)
        scenario.split(X)
        return scenario.training_data, scenario.test_data

    def transform_predictions(self, batch_iterator, allow_repeats: bool = False):
        if not allow_repeats:
            batch_iterator = NoRepeats(batch_iterator)
        yield from batch_iterator


cli = CLI()


class Evaluator(IEvaluator):
    @provider
    def metrics(self):
        return [
            NDCGK,
            RecallK,
            # MeanReciprocalRankK
        ]

    @provider
    def k_values(self):
        return [
            1, 5, 20, 50, 100
        ]

    @provider
    def _metrics(self):
        metrics = list()
        for m in self.metrics():
            for k in self.k_values():
                metrics.append(m(k))

        return metrics

    def evaluate(self, X, y, recommendations):
        results = dict()
        for metric in self._metrics():
            scores = metric.fit_transform(recommendations, y)
            value = scores.per_user_average()

            results[metric.name] = value

        return results


class Eval(Evaluator, DataSource, cli.Command, mask=IEvaluator):
    user_id = USER_KEY
    item_id = ITEM_KEY
    value_id = VALUE_KEY

    def __init__(self, _path: str):
        super().__init__()
        self.path = _path

    def load_df(self):
        df_X = pd.read_csv(os.path.join(self.path, HISTORY_FILE))
        df_y = pd.read_csv(os.path.join(self.path, Y_TRUE_FILE))
        df_pred = pd.read_csv(os.path.join(self.path, Y_PRED_FILE))
        return df_X, df_y, df_pred

    def run(self):
        X, y, y_pred = map(lambda x: x.values, self.preprocess())
        data = self.evaluate(X, y, y_pred)

        dict_to_csv(data, os.path.join(self.path, METRICS_FILE))


class MLExperiment(ML20MDataSource, StrongGeneralizationScenario, Evaluator, Experiment):
    pass


class Pop(MLExperiment, cli.Command, mask=IExperiment):
    @provider
    def recommender(self, k=200):
        return Popularity(K=k)


class IKNN(MLExperiment, cli.Command, mask=IExperiment):
    @provider
    def recommender(self, k=200, normalize: bool = False):
        return ItemKNN(K=k, normalize=normalize)


class Cosine(MLExperiment, cli.Command, mask=IExperiment):
    @provider
    def recommender(self, normalize: bool = False):
        return CosineSimilarity(normalize=normalize)


class Ease(MLExperiment, cli.Command, mask=IExperiment):
    @provider
    def recommender(self, l2: float = 500, alpha: float = 0):
        return EASE(l2=l2, alpha=alpha)


class EaseAvg(MLExperiment, cli.Command, mask=IExperiment):
    @provider
    def recommender(self, l2: float = 1):
        return EASE_AVG(l2=l2)


class EaseInt(MLExperiment, cli.Command, mask=IExperiment):
    @provider
    def recommender(self, l2: float = 1, alpha: float = 0):
        return EASE_Intercept(l2=l2, alpha=alpha)


class EaseAvgInt(MLExperiment, cli.Command, mask=IExperiment):
    @provider
    def recommender(self, l2: float = 0.1):
        return EASE_AVG_Int(l2=l2)


class SA_EASE(MLExperiment, cli.Command, mask=IExperiment):
    @provider
    def recommender(self, l2: float = 500, p: float=0.75):
        alg = EASE(l2=l2)
        return SharedAccount(alg, p=p)


class SA_IKNN(MLExperiment, cli.Command, mask=IExperiment):
    @provider
    def recommender(self, k: int = 200, p: float = 0.75, agg: Agg = Agg.Adj, normalize: bool = False):
        alg = ItemKNN(K=k, normalize=normalize)
        return SharedAccount(alg, p=p, agg=agg)


cli.run()
