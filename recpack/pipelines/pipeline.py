import logging
from collections import defaultdict
from typing import Tuple, Union, Dict, List, Any

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from recpack.algorithms.base import Algorithm, TorchMLAlgorithm
from recpack.data.matrix import InteractionMatrix
from recpack.metrics.base import Metric
from recpack.pipelines.registries import (
    ALGORITHM_REGISTRY,
    METRIC_REGISTRY,
    AlgorithmEntry,
    MetricEntry,
    OptimisationMetricEntry,
)
from recpack.postprocessing.postprocessors import Postprocessor


logger = logging.getLogger("recpack")


class MetricAccumulator:
    """Accumulates metrics and
    provides methods to aggregate results into usable formats.
    """

    def __init__(self):
        self.acc = defaultdict(dict)

    def __getitem__(self, key):
        return self.acc[key]

    def add(self, metric, algorithm_name, metric_name):
        logger.debug(f"Metric {metric_name} created for algorithm {algorithm_name}")
        self.acc[algorithm_name][metric_name] = metric

    @property
    def metrics(self):
        results = defaultdict(dict)
        for key in self.acc:
            for k in self.acc[key]:
                results[key][k] = self.acc[key][k].value
        return results

    @property
    def num_users(self):
        results = defaultdict(dict)
        for key in self.acc:
            for k in self.acc[key]:
                results[key][k] = self.acc[key][k].num_users
        return results


class Pipeline(object):
    """Performs all steps in order and holds on to results.

    Pipeline is run per algorithm.
    First grid parameters are optimised by training on train_data and
    evaluation on validation_data.
    Next, unless the model requires validation data,
    the model with optimised parameters is retrained
    on the combination of train_data and validation_data.
    Predictions are then generated with test_in as input and evaluated on test_out.

    Results can be accessed via the :meth:`get_metrics` method.

    # TODO results_directory is not documented
    :param algorithm_entries: List of AlgorithmEntry objects to evaluate in this pipeline.
        An AlgorithmEntry defines which algorithm to train, with which fixed parameters
        (params) and which parameters to optimize (grid).
    :type algorithm_entries: List[AlgorithmEntry]
    :param metric_entries: List of MetricEntry objects to evaluate each algorithm on.
        A MetricEntry defines which metric and value of the parameter K
        (number of recommendations).
    :type metric_entries: List[MetricEntry]
    :param train_data: The data to train models on.
    :type train_data: InteractionMatrix
    :param validation_data: The data to use for optimising parameters,
        can be None only if none of the algorithms require optimisation.
    :type validation_data: Union[Tuple[InteractionMatrix, InteractionMatrix], None]
    :param test_data: The data to perform evaluation, as (`test_in`, `test_out`) tuple.
    :type: Tuple[InteractionMatrix, InteractionMatrix]
    :param optimisation_metric: The metric to optimise each algorithm on.
    :param post_processor: A postprocessor instance to apply filters
        on the recommendation scores.
    :type post_processor: Postprocessor
    """

    def __init__(
        self,
        results_directory: str,
        algorithm_entries: List[AlgorithmEntry],
        metric_entries: List[MetricEntry],
        full_training_data: InteractionMatrix,
        validation_training_data: Union[InteractionMatrix, None],
        validation_data: Union[Tuple[InteractionMatrix, InteractionMatrix], None],
        test_data: Tuple[InteractionMatrix, InteractionMatrix],
        optimisation_metric_entry: Union[OptimisationMetricEntry, None],
        # TODO Nullable or not?
        post_processor: Union[Postprocessor, None],
        # TODO Document
        recommend_history: bool
    ):
        self.results_directory = results_directory
        self.algorithm_entries = algorithm_entries
        self.metric_entries = metric_entries
        self.full_training_data = full_training_data
        self.validation_training_data = validation_training_data
        self.validation_data = validation_data
        self.test_data_in, self.test_data_out = test_data
        self.optimisation_metric_entry = optimisation_metric_entry
        self.post_processor = post_processor
        self.recommend_history = recommend_history

        self._metric_acc = MetricAccumulator()
        # All dataframes from optimisation are accumulated in this list
        # To give it back to the user.
        self._optimisation_results = []

    def run(self):
        """Runs the pipeline."""
        for algorithm_entry in tqdm(self.algorithm_entries):
            # Check whether we need to optimize hyperparameters
            if algorithm_entry.optimise:
                params = self._optimise(algorithm_entry)
            else:
                params = algorithm_entry.params

            algorithm = ALGORITHM_REGISTRY.get(algorithm_entry.name)(**params)
            # Train the final version of the algorithm
            self._train(algorithm, params, self.full_training_data)
            # Make predictions
            X_pred = self._predict_and_postprocess(algorithm, self.test_data_in)

            for metric_entry in self.metric_entries:
                # m = self._evaluate(metric_entry, self.test_data_out.binary_values, X_pred)
                metric = METRIC_REGISTRY.get(metric_entry.name)(K=metric_entry.K)
                metric.calculate(self.test_data_out.binary_values, X_pred)
                self._metric_acc.add(metric, algorithm.identifier, metric.name)

    def _train(self, algorithm: Algorithm, params: Dict[str, Any], training_data: InteractionMatrix) -> Algorithm:
        if isinstance(algorithm, TorchMLAlgorithm):
            algorithm.fit(training_data, self.validation_data)
        else:
            algorithm.fit(training_data)
        return algorithm

    # TODO Work with csr_matrices
    def _predict_and_postprocess(self, algorithm: Algorithm, data_in: InteractionMatrix) -> csr_matrix:
        X_pred = algorithm.predict(data_in)
        X_pred = self.post_processor.process(X_pred)

        if not self.recommend_history:
            X_pred = X_pred - X_pred.multiply(data_in.binary_values)

        return X_pred

    def _evaluate(self, metric: MetricEntry, X_true: csr_matrix, X_pred: csr_matrix) -> Metric:
        m = METRIC_REGISTRY.get(metric.name)(K=metric.K)
        m.calculate(X_true, X_pred)
        return m

    def _optimise(self, algorithm_entry: AlgorithmEntry) -> Dict[str, Any]:
        results = []
        # Construct grid:
        optimisation_params = ParameterGrid(algorithm_entry.grid)
        for p in optimisation_params:
            algorithm = ALGORITHM_REGISTRY.get(algorithm_entry.name)(**p)
            # Train with given hyperparameter instantiations
            self._train(algorithm, p, self.validation_training_data)
            # Makre predictions and postprocess
            validation_data_in, validation_data_out = self.validation_data
            X_pred_val = self._predict_and_postprocess(algorithm, validation_data_in)
            # Compute optimisation metric value
            optimisation_metric = METRIC_REGISTRY.get(self.optimisation_metric_entry.name)(K=self.optimisation_metric_entry.K)
            optimisation_metric.calculate(validation_data_out.binary_values, X_pred_val)
            # optimisation_metric = self._evaluate(self.optimisation_metric_entry, validation_data_out.binary_values, X_pred_val)
            # metric = evaluate_recommendations(target, recommendations, self.optimisation_metric)

            params = p

            results.append(
                {
                    "identifier": algorithm.identifier,
                    "params": {**p, **algorithm_entry.params},
                    optimisation_metric.name: optimisation_metric.value,
                }
            )

        # Sort by metric value
        optimal_params = sorted(
            results,
            key=lambda x: x[optimisation_metric.name],
            reverse=~self.optimisation_metric_entry.minimise,
        )[0]["params"]

        self._optimisation_results.extend(results)
        return optimal_params

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get the metrics for the pipeline.

        Returns a nested dict, with structure:
        <algorithm> -> <metric> -> value

        It can be easily rendered into a well readable table using pandas::

            import pandas as pd

            pd.DataFrame.from_dict(pipeline.get_metrics())

        :return: Metric values as a nested dict.
        :rtype: Dict[str, Dict[str, float]]
        """
        return self._metric_acc.metrics

    def save_metrics(self) -> None:
        """Save the metrics in a json file

        The file will be saved in the experiment directory.
        """
        df = pd.DataFrame.from_dict(self.get_metrics())
        df.to_json(f"{self.results_directory}/results.json")

    def get_num_users(self) -> int:
        """Get the amount of users used in the evaluation.

        :return: The number of users used in the evaluation.
        :rtype: int
        """
        return self._metric_acc.num_users

    @property
    def optimisation_results(self):
        """Contains a result for each of the hyperparameter combinations tried out,
        for each of the algorithms evaluated.
        """
        return pd.DataFrame.from_records(self._optimisation_results)
