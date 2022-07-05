import logging
from collections import defaultdict
from typing import Tuple, Union, Dict, List, Any, Optional

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from recpack.algorithms.base import Algorithm, TorchMLAlgorithm
from recpack.pipelines.registries import (
    ALGORITHM_REGISTRY,
    METRIC_REGISTRY,
    AlgorithmEntry,
    MetricEntry,
    OptimisationMetricEntry,
)
from recpack.postprocessing.postprocessors import Postprocessor

from recpack.matrix import InteractionMatrix

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
    """Performs optimisation, training, prediction and evaluation, keeping track of results.

    Pipeline is run per algorithm.
    First grid parameters are optimised by training on `validation_training_data` and
    evaluating using `validation_data`.
    Next, unless the model is based on :class:`recpack.algorithms.TorchMLAlgorithm`,
    the model with optimised parameters is retrained on `full_training_data`.
    The final evaluation happens using the `test_data`.

    Results can be accessed via the :meth:`get_metrics` method.

    :param results_directory: Path to a directory in which to save results of the pipeline
        when save_metrics() is called.
    :type results_directory: string
    :param algorithm_entries: List of AlgorithmEntry objects to evaluate in this pipeline.
        An AlgorithmEntry defines which algorithm to train, with which fixed parameters
        (params) and which parameters to optimize (grid).
    :type algorithm_entries: List[AlgorithmEntry]
    :param metric_entries: List of MetricEntry objects to evaluate each algorithm on.
        A MetricEntry defines which metric and value of the parameter K
        (number of recommendations).
    :type metric_entries: List[MetricEntry]
    :param full_training_data: The data to train models on, in the final evaluation.
    :type full_training_data: InteractionMatrix
    :param validation_training_data: The data to train models on when optimising parameters.
    :type validation_training_data: InteractionMatrix
    :param validation_data: The data to use for optimising parameters,
        can be None only if none of the algorithms require optimisation.
    :type validation_data: Union[Tuple[InteractionMatrix, InteractionMatrix], None]
    :param test_data: The data to perform evaluation, as (`test_in`, `test_out`) tuple.
    :type: Tuple[InteractionMatrix, InteractionMatrix]
    :param optimisation_metric: The metric to optimise each algorithm on.
    :type optimisation_metric: Union[OptimisationMetricEntry, None]
    :param post_processor: A postprocessor instance to apply filters
        on the recommendation scores.
    :type post_processor: Postprocessor
    :param remove_history: Boolean to configure if the recommendations can include already interacted with items.
    :type remove_history: Boolean
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
        post_processor: Postprocessor,
        remove_history: bool,
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
        self.remove_history = remove_history

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
            if isinstance(algorithm, TorchMLAlgorithm):
                self._train(algorithm, self.validation_training_data)
            else:
                self._train(algorithm, self.full_training_data)
            # Make predictions
            X_pred = self._predict_and_postprocess(algorithm, self.test_data_in)

            for metric_entry in self.metric_entries:
                metric = METRIC_REGISTRY.get(metric_entry.name)(K=metric_entry.K)
                metric.calculate(self.test_data_out.binary_values, X_pred)
                self._metric_acc.add(metric, algorithm.identifier, metric.name)

    def _train(self, algorithm: Algorithm, training_data: InteractionMatrix) -> Algorithm:
        if isinstance(algorithm, TorchMLAlgorithm):
            algorithm.fit(training_data, self.validation_data)
        else:
            algorithm.fit(training_data)
        return algorithm

    # TODO Work with csr_matrices
    def _predict_and_postprocess(self, algorithm: Algorithm, data_in: InteractionMatrix) -> csr_matrix:
        X_pred = algorithm.predict(data_in)

        if self.remove_history:
            X_pred = X_pred - X_pred.multiply(data_in.binary_values)

        X_pred = self.post_processor.process(X_pred)

        return X_pred

    def _optimise(self, algorithm_entry: AlgorithmEntry) -> Dict[str, Any]:
        results = []
        # Construct grid:
        optimisation_params = ParameterGrid(algorithm_entry.grid)
        for p in optimisation_params:
            algorithm = ALGORITHM_REGISTRY.get(algorithm_entry.name)(**p, **algorithm_entry.params)
            # Train with given hyperparameter instantiations
            self._train(algorithm, self.validation_training_data)
            # Make predictions and postprocess
            validation_data_in, validation_data_out = self.validation_data
            X_pred_val = self._predict_and_postprocess(algorithm, validation_data_in)
            # Compute optimisation metric value
            optimisation_metric = METRIC_REGISTRY.get(self.optimisation_metric_entry.name)(
                K=self.optimisation_metric_entry.K
            )
            optimisation_metric.calculate(validation_data_out.binary_values, X_pred_val)

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
            reverse=not self.optimisation_metric_entry.minimise,
        )[0]["params"]

        self._optimisation_results.extend(results)
        return optimal_params

    def get_metrics(self, short: Optional[bool] = False) -> pd.DataFrame:
        """Get the metrics for the pipeline.

        :param short: If short is True, only the algorithm names are returned, and not the parameters.
            Defaults to False
        :type short: bool, optional
        :return: Algorithms and their respective performance.
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self._metric_acc.metrics).T
        if short:
            # Parameters are between (), so if we split on the (,
            # we can get the algorithm name by taking the first of the splits.
            df.index = df.index.map(lambda x: x.split("(")[0])
        return df

    def save_metrics(self) -> None:
        """Save the metrics in a json file

        The file will be saved in the experiment directory.
        """
        df = self.get_metrics()
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
