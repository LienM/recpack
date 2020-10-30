import logging
from collections import defaultdict
from typing import Tuple, Union

import scipy.sparse
from tqdm.auto import tqdm

from recpack.metrics import METRICS
from recpack.data.data_matrix import DataM
from recpack.splitters.splitter_base import FoldIterator


logger = logging.getLogger("recpack")


class MetricRegistry:
    """
    Register metrics here for clean showing later on.
    """


    def __init__(self, algorithms, metric_names, K_values):
        self.registry = defaultdict(dict)
        self.K_values = K_values
        self.metric_names = metric_names

        self.algorithms = algorithms

        for algo in algorithms:
            for m in self.metric_names:
                for K in K_values:
                    self._create(algo.identifier, m, K)

    def _create(self, algorithm_name, metric_name, K):
        metric = METRICS[metric_name](K)
        self.registry[algorithm_name][f"{metric_name}_K_{K}"] = metric
        logger.debug(f"Metric {metric_name} created for algorithm {algorithm_name}")
        return

    def __getitem__(self, key):
        return self.registry[key]

    def register_from_factory(self, metric_factory, identifier, K_values):
        for algo in self.algorithms:
            for K in K_values:
                self.register(metric_factory.create(K), algo.identifier, f"{identifier}@{K}")

    def register(self, metric, algorithm_name, metric_name):
        logger.debug(f"Metric {metric_name} created for algorithm {algorithm_name}")
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

    def run(self, train_data: Union[Tuple[DataM, DataM], DataM], test_data: Tuple[DataM, DataM], validation_data: Tuple[DataM, DataM] = None, batch_size=1000):
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

        if isinstance(train_data, DataM):
            X = train_data.binary_values
            y = None
        else:
            X = train_data[0].binary_values
            y = train_data[1].binary_values

        self.train(X, y=y, validation_data=validation_data)
        self.eval(test_data[0], test_data[1], batch_size)

    def train(self, X, y=None, validation_data=None):
        for algo in self.algorithms:
            logger.debug(f"Training algo {algo.identifier}")
            if y is not None and validation_data is not None:
                algo.fit(X, y=y, validation_data=validation_data)
            elif y is not None:
                algo.fit(X, y=y)
            elif validation_data is not None:
                algo.fit(X, validation_data=validation_data)
            else:
                algo.fit(X)

    def eval(self, data_m_in, data_m_out, batch_size):

        for algo in self.algorithms:
            metrics = self.metric_registry[algo.identifier]

            X_pred = data_m_in.values
            y_pred = algo.predict(X_pred)

            if not scipy.sparse.issparse(y_pred):
                y_pred = scipy.sparse.csr_matrix(y_pred)

            y_true = data_m_out.binary_values

            for metric in metrics.values():
                metric.calculate(y_true, y_pred)

    def get(self):
        return self.metric_registry.metrics

    def get_number_of_users_evaluated(self):
        return self.metric_registry.number_of_users_evaluated
