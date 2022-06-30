import logging
from collections import defaultdict, namedtuple
from collections.abc import Iterable
import datetime
import os
from typing import Tuple, Union, Dict, List, Optional
import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.matrix import InteractionMatrix
from recpack.scenarios import Scenario
from recpack.pipelines.registries import MetricRegistry, AlgorithmRegistry

logger = logging.getLogger("recpack")


ALGORITHM_REGISTRY = AlgorithmRegistry()
"""Registry for algorithms.

Contains the Recpack algorithms by default,
and allows registration of new algorithms via the `register` function.

Example::

    from recpack.pipelines import ALGORITHM_REGISTRY

    # Construct an ItemKNN object with parameter K=20
    algo = ALGORITHM_REGISTRY.get('ItemKNN')(K=20)

    from recpack.algorithms import ItemKNN
    ALGORITHM_REGISTRY.register('HelloWorld', ItemKNN)

    # Also construct an ItemKNN object with parameter K=20
    algo = ALGORITHM_REGISTRY.get('HelloWorld')(K=20)
"""


METRIC_REGISTRY = MetricRegistry()
"""Registry for metrics.

Contains the Recpack metrics by default,
and allows registration of new metrics via the `register` function.

Example::

    from recpack.pipelines import METRIC_REGISTRY

    # Construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('Recall')(K=20)

    from recpack.algorithms import Recall
    METRIC_REGISTRY.register('HelloWorld', Recall)

    # Also construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('HelloWorld')(K=20)

"""


class MetricAccumulator:
    """
    Add metrics here for clean showing later on.
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


MetricEntry = namedtuple("MetricEntry", ["name", "K"])
OptimisationMetricEntry = namedtuple("OptimisationMetricEntry", ["name", "K", "minimise"])
AlgorithmEntry = namedtuple("AlgorithmEntry", ["name", "grid", "params"])


class Pipeline(object):
    """Trains and evaluates the algorithms specified for the scenario given, for all metrics.

    Pipeline is run per algorithm.
    First grid parameters are optimised by training on validation_training_data and
    evaluation on validation_data.
    Next, unless the model requires validation data,
    the model with optimised parameters is retrained
    on full_training_data.
    Predictions are then generated with test_in as input and evaluated on test_out.

    Results can be accessed via the :meth:`get_metrics` method.

    :param results_directory: Absolute path to the directory in which results will be stored.
    :type results_directory: str
    :param algorithms: List of AlgorithmEntry objects to evaluate in this pipeline.
    An AlgorithmEntry defines which algorithm to train, with which fixed parameters (params)
    and which parameters to optimize (grid).
    :type algorithms: List[AlgorithmEntry]
    :param metrics: List of MetricEntry objects to evaluate each algorithm on.
    A MetricEntry defines which metric and value of the parameter K (number of recommendations).
    :type metrics: List[MetricEntry]
    :param full_training_data: Training data used in final training.
    :type full_training_data: InteractionMatrix
    :param validation_training_data: Training data used for hyperparameter optimization.
    :type validation_training_data: Union[InteractionMatrix, None]
    :param validation_data: The data to use for optimising parameters,
    can be None only if none of the algorithms require optimisation.
    :type validation_data: Union[Tuple[InteractionMatrix, InteractionMatrix], None]
    :param test_data: The data to perform evaluation, as (`test_in`, `test_out`) tuple.
    :type test_data: Tuple[InteractionMatrix, InteractionMatrix]
    :param optimisation_metric: The metric to optimise each algorithm on.
    :type optimisation_metric: Union[OptimisationMetricEntry, None]
    """

    def __init__(
        self,
        results_directory: str,
        algorithms: List[AlgorithmEntry],
        metrics: List[MetricEntry],
        full_training_data: InteractionMatrix,
        validation_training_data: Union[InteractionMatrix, None],
        validation_data: Union[Tuple[InteractionMatrix, InteractionMatrix], None],
        test_data: Tuple[InteractionMatrix, InteractionMatrix],
        optimisation_metric: Union[OptimisationMetricEntry, None],
    ):
        self.results_directory = results_directory
        self.algorithms = algorithms
        self.metrics = metrics
        self.full_training_data = full_training_data
        self.validation_training_data = validation_training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.optimisation_metric = optimisation_metric

        self._metric_acc = MetricAccumulator()
        # All dataframes from optimisation are accumulated in this list
        # To give it back to the user.
        self._optimisation_results = []

    def _optimise(self, algorithm, metric_entry: OptimisationMetricEntry):
        # TODO: investigate using advanced optimisers

        if len(algorithm.grid) == 0:
            return algorithm.params

        results = []
        # Construct grid:
        optimisation_params = ParameterGrid(algorithm.grid)
        for p in optimisation_params:
            algo = ALGORITHM_REGISTRY.get(algorithm.name)(**p, **algorithm.params)

            if isinstance(algo, TorchMLAlgorithm):
                # Training a TorchMLAlgorithm requires validation data for early stopping.
                # TODO Technically we only need to use validation_training_data here when early stopping
                # If we run for a fixed number of iterations, it doesn't matter anyway.
                algo.fit(self.validation_training_data, self.validation_data)
            else:
                algo.fit(self.validation_training_data)
            metric = METRIC_REGISTRY.get(metric_entry.name)(K=metric_entry.K)

            prediction = algo.predict(self.validation_data[0])

            metric.calculate(self.validation_data[1].binary_values, prediction)

            results.append(
                {
                    "identifier": algo.identifier,
                    "params": {**p, **algorithm.params},
                    metric_entry.name: metric.value,
                }
            )

        # Sort by metric value
        optimal_params = sorted(results, key=lambda x: x[metric_entry.name], reverse=~metric_entry.minimise,)[
            0
        ]["params"]

        self._optimisation_results.extend(results)
        return optimal_params

    def run(self):
        """Runs the pipeline."""

        for algorithm in tqdm(self.algorithms):
            # optimisation:
            optimal_params = self._optimise(algorithm, self.optimisation_metric)

            # Train again.
            algo = ALGORITHM_REGISTRY.get(algorithm.name)(**optimal_params)
            if isinstance(algo, TorchMLAlgorithm):
                # TODO: Optimise this retraining step, by returning trained model?
                algo.fit(self.validation_training_data, self.validation_data)
            else:
                algo.fit(self.full_training_data)

            # Evaluate
            test_in, test_out = self.test_data
            recommendations = algo.predict(test_in)
            for metric in self.metrics:
                m = METRIC_REGISTRY.get(metric.name)(K=metric.K)
                m.calculate(test_out.binary_values, recommendations)
                self._metric_acc.add(m, algo.identifier, m.name)

    def get_metrics(self, short: Optional[bool] = False) -> pd.DataFrame:
        """Get the metrics for the pipeline.

        :param short: If short is True, only the algorithm names are returned, and not the parameters.
            Defaults to False
        :type short: Optional[bool]
        :return: Algorithms and their respective performance.
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self._metric_acc.metrics).T
        if short:
            # Parameters are between (), so if we split on the (,
            # we can get the algorithm name by taking the first of the splits.
            df.index = df.index.map(lambda x: x.split("(")[0])
        return df

    def get_metrics_dataframe(self, short: Optional[bool] = False):
        """Get a dataframe with the algorithms as keys.

        :param short: If short is True, only the algorithm names are returned, and not the parameters.
            Defaults to False
        :type short: Optional[bool]
        """
        df = pd.DataFrame.from_dict(self.get_metrics()).T
        if short:
            # Parameters are between (), so if we split on the (,
            # we can get the algorithm name by taking the first of the splits.
            df.index = df.index.map(lambda x: x.split("(")[0])
        return df

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


class PipelineBuilder(object):
    """Builder to facilitate construction of pipelines.

    The builder contains functions to set specific values for the pipeline.
    Save and Load make it possible to easily recreate pipelines.

    :param folder_name: The name of the folder where pipeline information will be stored.
        If no name is specified, the timestamp of creation is used.
    :type folder_name: Optional[str]
    :param base_path: The base_path to store pipeline in,
        defaults to the current working directory.
    :type base_path: Optional[str]
    """

    def __init__(self, folder_name: str = None, base_path: str = None):

        self.folder_name = folder_name
        if self.folder_name is None:
            self.folder_name = datetime.datetime.now().isoformat()

        self.base_path = base_path or os.getcwd()

        self.metrics = {}
        self.algorithms = []

        self._config_file_path = f"{self.base_path}/{self.folder_name}/config.yaml"
        self._full_train_file_path = f"{self.base_path}/{self.folder_name}/full_train"
        self._validation_train_file_path = f"{self.base_path}/{self.folder_name}/validation_train"
        self._test_in_file_path = f"{self.base_path}/{self.folder_name}/test_in"
        self._test_out_file_path = f"{self.base_path}/{self.folder_name}/test_out"
        self._validation_in_file_path = f"{self.base_path}/{self.folder_name}/validation_in"
        self._validation_out_file_path = f"{self.base_path}/{self.folder_name}/validation_out"
        # TODO Rename this to better reflect the contents
        self.results_directory = f"{self.base_path}/{self.folder_name}"

    def _parse_arg(self, arg: Union[type, str]) -> str:
        if type(arg) == type:
            arg = arg.__name__

        elif type(arg) != str:
            raise TypeError(f"Argument should be string or type, not {type(arg)}!")

        return arg

    def add_metric(self, metric: Union[str, type], K: Union[List, int]):
        """Register a metric to evaluate

        :param metric: Metric name or type.
        :type metric: Union[str, type]
        :param K: The K value(s) used to construct metrics.
            If it is a list, for each value a metric is added.
        :type K: Union[List, int]
        :raises ValueError: If metric can't be resolved to a key
            in the `METRIC_REGISTRY`.
        """

        # Make it so it's possible to add metrics by their class as well.
        metric = self._parse_arg(metric)

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric} could not be resolved.")

        if isinstance(K, Iterable):
            for k in K:
                self.add_metric(metric, k)
        else:
            # Check if metric already exists
            metric_name = f"{metric}_{K}"

            if metric_name in self.metrics:
                logger.warning(f"Metric {metric_name} already exists.")
            else:
                self.metrics[metric_name] = MetricEntry(metric, K)

    def add_algorithm(self, algorithm: Union[str, type], grid: Dict = {}, params: Dict = {}):
        """Add an algorithm to run.

        Parameters in grid will be optimised during running of pipeline.

        :param algorithm: Algorithm name or algorithm type.
        :type algorithm: Union[str, type]
        :param grid: Parameters to optimise, and the values to use in grid search,
            defaults to {}
        :type grid: dict, optional
        :param params: The key-values that are set fixed for running, defaults to {}
        :type params: dict, optional
        :raises ValueError: If algorithm can't be resolved to a key
            in the `ALGORITHM_REGISTRY`.
        """
        algorithm = self._parse_arg(algorithm)

        if algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm {algorithm} could not be resolved.")

        self.algorithms.append(AlgorithmEntry(algorithm, grid, params))

    def set_optimisation_metric(self, metric: Union[str, type], K: int, minimise=False):
        """Set the metric for optimisation of parameters in algorithms.

        If the metric is not implemented by default in recpack,
        you should register it in the `METRIC_REGISTRY`

        :param metric: metric name or metric type
        :type metric: Union[str, type]
        :param K: The K value for the metric
        :type K: int
        :param minimise: If True minimal value for metric is better, defaults to False
        :type minimise: bool, optional
        :raises ValueError: If metric can't be resolved to a key
            in the `METRIC_REGISTRY`.
        """
        metric = self._parse_arg(metric)

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"metric {metric} could not be resolved.")

        self.optimisation_metric = OptimisationMetricEntry(metric, K, minimise)

    def set_validation_training_data(self, validation_training_data: InteractionMatrix):
        """Set the training dataset used for validation.

        :param validation_training_data: The interaction matrix to use during training when optimizing hyperparameters.
        :type validation_training_data: InteractionMatrix
        """
        self.validation_training_data = validation_training_data

    def set_full_training_data(self, full_training_data: InteractionMatrix):
        """Set the complete training dataset.

        :param full_training_data: The interaction matrix to use during training.
        :type full_training_data: InteractionMatrix
        """
        self.full_training_data = full_training_data

    def set_validation_data(self, validation_data: Tuple[InteractionMatrix, InteractionMatrix]):
        # TODO Support csr_matrix
        """Set the validation datasets.

        Validation data should be a tuple of InteractionMatrices.

        :param validation_data: The tuple of validation data,
            as (validation_in, validation_out) tuple.
        :type validation_data: Tuple[InteractionMatrix, InteractionMatrix]
        :raises ValueError: If tuple does not contain two InteractionMatrices.
        """
        if not len(validation_data) == 2:
            raise ValueError("Incorrect value, expected tuple with data_in and data_out")
        self.validation_data = validation_data

    def set_test_data(self, test_data: Tuple[InteractionMatrix, InteractionMatrix]):
        """Set the test datasets.

        Test data should be a tuple of InteractionMatrices.

        :param test_data: The tuple of test data, as (test_in, test_out) tuple.
        :type test_data: Tuple[InteractionMatrix, InteractionMatrix]
        :raises ValueError: If tuple does not contain two InteractionMatrices.
        """
        if not len(test_data) == 2:
            raise ValueError("Incorrect value, expected tuple with data_in and data_out")

        self.test_data = test_data

    def set_data_from_scenario(self, scenario: Scenario):
        """Set the train, validation and test data based by
        extracting them from the scenario."""

        self.set_full_training_data(scenario.full_training_data)
        self.set_test_data(scenario.test_data)
        if scenario.validation:
            self.set_validation_training_data(scenario.validation_training_data)
            self.set_validation_data(scenario.validation_data)

    def _check_readiness(self):
        if len(self.metrics) == 0:
            raise RuntimeError("No metrics specified, can't construct pipeline")

        if len(self.algorithms) == 0:
            raise RuntimeError("No algorithms specified, can't construct pipeline")

        if not hasattr(self, "optimisation_metric") and np.any([len(algo.grid) > 0 for algo in self.algorithms]):
            raise RuntimeError(
                "No optimisation metric selected to perform requested hyperparameter optimisation,"
                "can't construct pipeline."
            )

        # Check availability of data
        if not hasattr(self, "full_training_data"):
            raise RuntimeError("No full training data available, can't construct pipeline.")

        if not hasattr(self, "test_data"):
            raise RuntimeError("No test data available, can't construct pipeline.")

        # If there are parameters to optimise,
        # there needs to be validation data available.
        if self._requires_validation() and not hasattr(self, "validation_data"):
            raise RuntimeError(
                "No validation data available to perform the requested hyperparameter optimisation"
                ", can't construct pipeline."
            )

        if self._requires_validation() and not hasattr(self, "validation_training_data"):
            raise RuntimeError(
                "No validation training data available to perform the requested hyperparameter optimisation"
                ", can't construct pipeline."
            )

        # Validate shape is correct
        shape = self.full_training_data.shape

        if any([d.shape != shape for d in self.test_data]):
            raise RuntimeError("Shape mismatch between test and training data")

        if hasattr(self, "validation_data") and any([d.shape != shape for d in self.validation_data]):
            raise RuntimeError("Shape mismatch between validation and training data")

    def _requires_validation(self) -> bool:
        return any([len(algo.grid) > 0 for algo in self.algorithms]) or any(
            [issubclass(ALGORITHM_REGISTRY.get(algo.name), TorchMLAlgorithm) for algo in self.algorithms]
        )

    def build(self) -> Pipeline:
        """Construct a pipeline object, given the set values.

        If required fields are not set, raises an error.

        :return: The constructed pipeline.
        :rtype: Pipeline
        """

        self._check_readiness()

        return Pipeline(
            self.results_directory,
            self.algorithms,
            list(self.metrics.values()),
            self.full_training_data,
            getattr(self, "validation_training_data", None),
            getattr(self, "validation_data", None),
            self.test_data,
            getattr(self, "optimisation_metric", None),
        )

    def _save_data(self):
        self.full_training_data.save(self._full_train_file_path)

        self.test_data[0].save(self._test_in_file_path)
        self.test_data[1].save(self._test_out_file_path)

        if hasattr(self, "validation_data"):
            self.validation_data[0].save(self._validation_in_file_path)
            self.validation_data[1].save(self._validation_out_file_path)
            self.validation_training_data.save(self._validation_train_file_path)

    @property
    def _pipeline_config(self):
        d = {}
        d["algorithms"] = [algo._asdict() for algo in self.algorithms]
        d["metrics"] = [m._asdict() for m in self.metrics.values()]
        d["optimisation_metric"] = self.optimisation_metric._asdict() if hasattr(self, "optimisation_metric") else None

        return d

    def save(self):
        """Save the pipeline settings to file"""
        self._check_readiness()

        # Make sure folder exists
        if not os.path.exists(f"{self.base_path}/{self.folder_name}"):
            os.makedirs(f"{self.base_path}/{self.folder_name}")

        self._save_data()

        with open(self._config_file_path, "w") as outfile:
            outfile.write(yaml.safe_dump(self._pipeline_config))

    def _load_data(self, d):
        self.full_training_data = InteractionMatrix.load(self._full_train_file_path)

        self.test_data = (
            InteractionMatrix.load(self._test_in_file_path),
            InteractionMatrix.load(self._test_out_file_path),
        )

        try:
            self.validation_data = (
                InteractionMatrix.load(self._validation_in_file_path),
                InteractionMatrix.load(self._validation_out_file_path),
            )
            self.validation_training_data = InteractionMatrix.load(self._validation_train_file_path)
        except FileNotFoundError:
            pass

    def load(self):
        # TODO Split into load settings and main function
        """Load the settings from file into the correct attributes."""

        with open(self._config_file_path, "r") as infile:
            d = yaml.safe_load(infile)

        for m in d["metrics"]:
            self.add_metric(m["name"], m["K"])

        for a in d["algorithms"]:
            self.add_algorithm(a["name"], a["grid"], a["params"])

        if d["optimisation_metric"]:
            opt = d["optimisation_metric"]
            self.set_optimisation_metric(opt["name"], opt["K"], opt["minimise"])

        self._load_data(d)
