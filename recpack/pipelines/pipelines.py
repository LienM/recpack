from collections import defaultdict

import scipy.sparse
import numpy as np

from recpack.metrics import RecallK, MeanReciprocalRankK, NDCGK
from recpack.utils import get_logger
from recpack.data_matrix import DataM
import recpack.experiment as experiment
from recpack.splitters.splitter_base import FoldIterator

from tqdm.auto import tqdm


class MetricRegistry:
    """
    Register metrics here for clean showing later on.
    """

    METRICS = {"NDCG": NDCGK, "Recall": RecallK, "MRR": MeanReciprocalRankK}

    def __init__(self, algorithms, metric_names, K_values):
        self.registry = defaultdict(dict)
        self.K_values = K_values
        self.metric_names = metric_names

        self.algorithms = algorithms
        self.logger = get_logger()

        for algo in algorithms:
            for m in self.metric_names:
                for K in K_values:
                    self._create(algo.identifier, m, K)

    def _create(self, algorithm_name, metric_name, K):
        metric = self.METRICS[metric_name](K)
        self.registry[algorithm_name][f"{metric_name}_K_{K}"] = metric
        self.logger.debug(f"Metric {metric_name} created for algorithm {algorithm_name}")
        return

    def __getitem__(self, key):
        return self.registry[key]

    def register_from_factory(self, metric_factory, identifier, K_values):
        for algo in self.algorithms:
            for K in K_values:
                self.register(metric_factory.create(K), algo.identifier, f"{identifier}@{K}")

    def register(self, metric, algorithm_name, metric_name):
        self.logger.debug(f"Metric {metric_name} created for algorithm {algorithm_name}")
        self.registry[algorithm_name][metric_name] = metric

    @property
    def metrics(self):
        results = defaultdict(dict)
        for key in self.registry:
            for k in self.registry[key]:
                results[key][k] = self.registry[key][k].value
        return results

    @property
    def number_of_users_evaluated(self):
        results = defaultdict(dict)
        for key in self.registry:
            for k in self.registry[key]:
                results[key][k] = self.registry[key][k].num_users
        return results


class Pipeline(object):

    def __init__(self, algorithms, metric_names, K_values):
        """
        Performs all steps in order and holds on to results.

        :param algorithms: List of algorithms to evaluate in this pipeline
        :type algorithms: `list(recpack.algorithms.Model)`

        :param metric_names: The names of metrics to compute in this pipeline.
                            Allowed values are: `NDCG`, `Recall` and `MRR`
        :type metric_names: `list(string)`

        :param K_values: The K values for each of the metrics
        :type K_values: `list(int)`
        """
        self.algorithms = algorithms

        self.metric_names = metric_names
        self.K_values = K_values
        self.metric_registry = MetricRegistry(algorithms, metric_names, K_values)

    def run(self, train_X, test_data_in, test_data_out, train_y=None, batch_size=1000):
        """
        Run the pipeline with the input data.
        This will use the different components in the pipeline to:
        1. Split data into train, validation, test
        2. Train models
        3. Split test data into in and out
        4. Evaluate models
        5. Store metrics

        :param data: The first data object to use. This data will be used in the splitters.
        :type data: `recpack.DataM`
        :param data_2: Additional data.
                       If the splitter expects a second data object to generate the train, validation and test,
                       you should use this one to give it that information.
        :type data_2: `recpack.DataM`
        """
        X = train_X.binary_values
        y = train_y.binary_values if train_y else None

        self.train(X, y)
        self.eval(test_data_in, test_data_out, batch_size)

    def train(self, X, y=None):
        for algo in self.algorithms:
            get_logger().debug(f"Training algo {algo.identifier}")
            if y is not None:
                algo.fit(X, y)
            else:
                algo.fit(X)

    def eval(self, data_in, data_out, batch_size):
        for _in, _out, user_ids in tqdm(FoldIterator(data_in, data_out, batch_size=batch_size)):
            get_logger().debug(f"start evaluation batch")
            for algo in self.algorithms:
                metrics = self.metric_registry[algo.identifier]

                get_logger().debug(f"predicting batch with algo {algo.identifier}")
                X_pred = algo.predict(_in, user_ids=user_ids)
                if scipy.sparse.issparse(X_pred):
                    # to dense format
                    X_pred = X_pred.toarray()
                else:
                    # ensure ndarray instead of matrix
                    X_pred = np.asarray(X_pred)
                get_logger().debug(f"finished predicting batch with algo {algo.identifier}")

                for metric in metrics.values():
                    metric.update(X_pred, _out)

            get_logger().debug(f"end evaluation batch")

    def get(self):
        return self.metric_registry.metrics

    def get_number_of_users_evaluated(self):
        return self.metric_registry.number_of_users_evaluated


class LoggingPipeline(Pipeline):

    def __init__(self, algorithms, metric_names, K_values):
        """
        Performs the steps like Pipeline, but with logging using experiment context.
        """
        super().__init__(algorithms, metric_names, K_values)

    def train(self, X, y=None):
        # first experiment forks current one, following ones for parent of current
        above = 0

        for algo in self.algorithms:
            # Only pass the sparse training interaction matrix to algo
            experiment.fork_experiment(algo.identifier, above)
            above = 1
            experiment.log_param("algorithm", algo.name)
            for param, value in algo.get_params().items():
                experiment.log_param(param, value)

            get_logger().debug(f"Training algo {algo.identifier}")
            if y is not None:
                algo.fit(X, y)
            else:
                algo.fit(X)

    def eval(self, data_in, data_out, batch_size):
        super().eval(data_in, data_out, batch_size)

        # log results
        metrics = self.metric_registry.metrics
        for algo in self.algorithms:
            experiment.set_experiment(algo.identifier)
            for metric, value in metrics[algo.identifier].items():
                experiment.log_result(metric, value)
