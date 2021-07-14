import logging
from collections import defaultdict, Counter, Iterable
from dataclasses import asdict, dataclass
import datetime
import os
from typing import Tuple, Union, Dict, Any, List, Callable
import yaml

import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import recpack.algorithms
import recpack.metrics
from recpack.data.matrix import Matrix, InteractionMatrix


logger = logging.getLogger("recpack")


class Registry:
    def __init__(self, src):
        self.registered = {}
        self.src = src

    def __getitem__(self, key: str) -> type:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        try:
            self.get(key)
            return True
        except AttributeError:
            return False

    def get(self, name: str) -> type:
        if name in self.registered:
            return self.registered[name]
        else:
            return getattr(self.src, name)

    def register(self, name: str, c: type):
        self.registered[name] = c


class AlgorithmRegistry(Registry):
    def __init__(self):
        super().__init__(recpack.algorithms)


class MetricRegistry(Registry):
    def __init__(self):
        super().__init__(recpack.metrics)


ALGORITHM_REGISTRY = AlgorithmRegistry()
METRIC_REGISTRY = MetricRegistry()


class MetricAccumulator:
    """
    Register metrics here for clean showing later on.
    """

    def __init__(self):
        self.registry = defaultdict(dict)

    def __getitem__(self, key):
        return self.registry[key]

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
    def num_users(self):
        results = defaultdict(dict)
        for key in self.registry:
            for k in self.registry[key]:
                results[key][k] = self.registry[key][k].num_users
        return results


@dataclass
class MetricEntry:
    name: str
    K: int

    def __hash__(self):
        return hash(self.name) + hash(self.K)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "MetricEntry":
        return MetricEntry(**d)


@dataclass
class OptimisationMetricEntry(MetricEntry):
    minimise: bool = False


@dataclass
class AlgorithmEntry:
    name: str
    grid: Dict[str, Any]
    params: Dict[str, Any]

    def to_dict(self):
        return asdict(self)

    def __hash__(self):
        return hash(str(self.to_dict()))

    @classmethod
    def from_dict(cls, d):
        return AlgorithmEntry(**d)


class Pipeline(object):
    """Performs all steps in order and holds on to results.

    Pipeline is run per algorithm.
    First grid parameters are optimised by training on train_data and
    evaluation on validation_data.
    Next, unless the model requires validation data,
    the model with optimised parameters is retrained
    on the combination of train_data and validation_data.
    Predictions are then generated with test_in as input and evaluated on test_out.

    # TODO Method was renamed
    Results can be accessed via the :meth:`get` method.

    :param algorithms: List of algorithms to evaluate in this pipeline.
    :type algorithms: List[AlgorithmEntry]
    :param metric_names: List of Metric entries to evaluate each algorithm on.
    :type metric_names: List[MetricEntry]
    :param train_data: The data to train models on.
    :type train_data: InteractionMatrix
    :param validation_data: The data to use for optimising parameters,
        can be None only if none of the algorithms require optimisation.
    :type validation_data: Union[Tuple[InteractionMatrix, InteractionMatrix], None]
    :param test_data: The data to perform evaluation, as (`test_in`, `test_out`) tuple.
    :type: Tuple[InteractionMatrix, InteractionMatrix]
    :param optimisation_metric: The metric to optimise each algorithm on.
    """

    def __init__(
        self,
        algorithms: List[AlgorithmEntry],
        metrics: List[MetricEntry],
        train_data: InteractionMatrix,
        validation_data: Union[Tuple[InteractionMatrix, InteractionMatrix], None],
        test_data: Tuple[InteractionMatrix, InteractionMatrix],
        optimisation_metric: Union[OptimisationMetricEntry, None],
    ):
        self.algorithms = algorithms
        self.metrics = metrics
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.optimisation_metric = optimisation_metric

        self._results = MetricAccumulator()

    def _optimise(self, algorithm, metric_entry: OptimisationMetricEntry):
        # TODO: investigate using advanced optimisers

        if len(algorithm.grid) == 0:
            return algorithm.params

        results = []
        # Construct grid:
        optimisation_params = ParameterGrid(algorithm.grid)
        for p in optimisation_params:
            algo = ALGORITHM_REGISTRY.get(algorithm.name)(**p, **algorithm.params)

            # validation data in case of TorchML!
            if isinstance(algo, recpack.algorithms.base.TorchMLAlgorithm):
                algo.fit(self.train_data, self.validation_data)
            else:
                algo.fit(self.train_data)
            metric = METRIC_REGISTRY.get(metric_entry.name)(K=metric_entry.K)

            prediction = algo.predict(self.validation_data[0])

            # TODO: Binary values or values?
            metric.calculate(self.validation_data[1].values, prediction)

            results.append(
                {
                    "identifier": algo.identifier,
                    "params": {**p, **algorithm.params},
                    "value": metric.value,
                }
            )

        # Sort by metric value
        optimal_params = sorted(
            results, key=lambda x: x["value"], reverse=~metric_entry.minimise
        )[0]["params"]
        return optimal_params

    def run(self):
        """Runs the pipeline."""

        for algorithm in tqdm(self.algorithms):
            # optimisation:
            optimal_params = self._optimise(algorithm, self.optimisation_metric)

            # Train again.
            algo = ALGORITHM_REGISTRY.get(algorithm.name)(**optimal_params)
            if isinstance(algo, recpack.algorithms.base.TorchMLAlgorithm):
                # TODO: Optimise this retraining step, by returning trained model?
                algo.fit(self.train_data, self.validation_data)
            else:
                algo.fit(
                    self.train_data
                    if self.validation_data is None
                    else self.train_data
                    + self.validation_data[0]
                    + self.validation_data[1]
                )

            # Evaluate
            test_in, test_out = self.test_data
            recommendations = algo.predict(test_in)
            for metric in self.metrics:
                m = METRIC_REGISTRY.get(metric.name)(K=metric.K)
                m.calculate(test_out.binary_values, recommendations)

                self._results.register(m, algo.identifier, m.name)

    def get_metrics(self):
        return self._results.metrics

    def get_num_users(self):
        return self._results.num_users


class PipelineBuilder(object):
    """Builder to facilitate construction of pipelines.

    The builder contains functions to set specific values for the pipeline.
    Save and Load make it possible to easily recreate pipelines.

    :param name: The name of the pipeline.
        The filename the pipeline is saved to is based on this name.
        If no name is specified, the timestamp of creation is used.
    :type name: string
    :param path: The path to store pipeline in,
        defaults to the current working directory.
    :type path: str
    """

    # TODO: input validation of metrics / algorithms / data
    def __init__(self, name=None, path=os.getcwd()):

        self.name = name
        if self.name is None:
            self.name = datetime.datetime.now().isoformat()

        self.path = path

        self.metrics = []
        self.algorithms = []

        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.optimisation_metric = None

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
        if type(metric) == type:
            metric = metric.__name__
        elif type(metric) != str:
            raise TypeError(f"Metric should be string or type, not {type(metric)}!")

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric} could not be resolved.")

        if isinstance(K, Iterable):
            for k in K:
                self.add_metric(metric, k)
        else:
            self.metrics.append(MetricEntry(metric, K))

    def add_algorithm(
        self, algorithm: Union[str, type], grid: Dict = {}, params: Dict = {}
    ):
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
        if type(algorithm) == type:
            algorithm = algorithm.__name__

        elif type(algorithm) != str:
            raise TypeError(
                f"Algorithm should be string or type, not {type(algorithm)}!"
            )

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
        # Make it so it's possible to add metrics by their class as well.
        if type(metric) == type:
            metric = metric.__name__

        elif type(metric) != str:
            raise ValueError(f"Metric should be string or type, not {type(metric)}!")

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"metric {metric} could not be resolved.")

        self.optimisation_metric = OptimisationMetricEntry(metric, K, minimise)

    def set_train_data(self, train_data: InteractionMatrix):
        """Set the training dataset.

        :param train_data: The interaction matrix to use during training.
        :type train_data: InteractionMatrix
        """
        self.train_data = train_data

    def set_validation_data(self, validation_data: Tuple[Matrix, Matrix]):
        # TODO Can these be both csr_matrix as InteractionMatrix, or only InteractionMatrix?
        """Set the validation datasets.

        Validation data should be a tuple of InteractionMatrices.

        :param validation_data: The tuple of validation data,
            as (validation_in, validation_out) tuple.
        :type validation_data: Tuple[InteractionMatrix, InteractionMatrix]
        :raises ValueError: If tuple does not contain two InteractionMatrices.
        """
        if not len(validation_data) == 2:
            raise ValueError(
                "Incorrect value, expected tuple with data_in and data_out"
            )
        self.validation_data = validation_data

    def set_test_data(self, test_data: Tuple[InteractionMatrix, InteractionMatrix]):
        """Set the test datasets.

        Test data should be a tuple of InteractionMatrices.

        :param test_data: The tuple of test data, as (test_in, test_out) tuple.
        :type test_data: Tuple[InteractionMatrix, InteractionMatrix]
        :raises ValueError: If tuple does not contain two InteractionMatrices.
        """
        if not len(test_data) == 2:
            raise ValueError(
                "Incorrect value, expected tuple with data_in and data_out"
            )

        self.test_data = test_data

    def _check_readiness(self):
        if len(self.metrics) == 0:
            raise RuntimeError("No metrics specified, can't construct pipeline")

        if len(self.algorithms) == 0:
            raise RuntimeError("No algorithms specified, can't construct pipeline")

        # Check that there is an optimisation criterion
        # if there are algorithms to optimise
        # TODO: If none specified? Use last one?
        if self.optimisation_metric is None and np.any(
            [len(algo.grid) > 0 for algo in self.algorithms]
        ):
            raise RuntimeError(
                "No optimisation metric selected to perform requested hyperparameter optimisation,"
                "can't construct pipeline."
            )

        # Check availability of data
        if self.train_data is None:
            raise RuntimeError("No training data available, can't construct pipeline.")

        if self.test_data is None:
            raise RuntimeError("No test data available, can't construct pipeline.")

        # If there are parameters to optimise,
        # there needs to be validation data available.
        if self.validation_data is None and np.any(
            [len(algo.grid) > 0 for algo in self.algorithms]
        ):
            raise RuntimeError(
                "No validation data available to perform the requested hyper parameter optimisation"
                ", can't construct pipeline."
            )

        # Validate shape is correct
        shape = self.train_data.shape

        if any([d.shape != shape for d in self.test_data]):
            raise RuntimeError("Shape mismatch between test and training data")

        if self.validation_data is not None and any(
            [d.shape != shape for d in self.validation_data]
        ):
            raise RuntimeError("Shape mismatch between validation and training data")

    def build(self) -> Pipeline:
        """Construct a pipeline object, given the set values.

        If required fields are not set, raises an error.

        :return: The constructed pipeline.
        :rtype: Pipeline
        """

        self._check_readiness()

        # Check duplicated metrics
        cm = Counter(self.metrics)
        for key, count in cm.items():
            if count > 1:
                logger.warning(
                    f"Found duplicated metric: {key}, "
                    "deduplicating before creation of pipeline."
                )

        metrics = list(cm.keys())

        # Check duplicated algorithms
        ca = Counter(self.algorithms)
        for key, count in cm.items():
            if count > 1:
                logger.warning(
                    f"Found duplicated algorithm: {key}, "
                    "deduplicating before creation of pipeline."
                )

        algorithms = list(ca.keys())

        return Pipeline(
            algorithms,
            metrics,
            self.train_data,
            self.validation_data,
            self.test_data,
            self.optimisation_metric,
        )

    @property
    def _config_file_path(self):
        return f"{self.path}/{self.name}/config.yaml"

    @property
    def _train_file_path(self):
        return f"{self.path}/{self.name}/train"

    @property
    def _test_in_file_path(self):
        return f"{self.path}/{self.name}/test_in"

    @property
    def _test_out_file_path(self):
        return f"{self.path}/{self.name}/test_out"

    @property
    def _validation_in_file_path(self):
        return f"{self.path}/{self.name}/validation_in"

    @property
    def _validation_out_file_path(self):
        return f"{self.path}/{self.name}/validation_out"

    def _construct_config_dict(self):
        dict_to_dump = {}
        dict_to_dump["algorithms"] = [algo.to_dict() for algo in self.algorithms]
        dict_to_dump["metrics"] = [m.to_dict() for m in self.metrics]

        dict_to_dump["optimisation_metric"] = (
            self.optimisation_metric.to_dict()
            if self.optimisation_metric is not None
            else None
        )

        dict_to_dump["train_data"] = {"filename": self._train_file_path}

        dict_to_dump["test_data"] = {
            "data_in": {"filename": self._test_in_file_path},
            "data_out": {"filename": self._test_out_file_path},
        }

        dict_to_dump["validation_data"] = (
            {
                "data_in": {"filename": self._validation_in_file_path},
                "data_out": {"filename": self._validation_out_file_path},
            }
            if self.validation_data
            else None
        )
        return dict_to_dump

    def _save_data(self):
        self.train_data.save(self._train_file_path)

        self.test_data[0].save(self._test_in_file_path)
        self.test_data[1].save(self._test_out_file_path)

        if self.validation_data:
            self.validation_data[0].save(self._validation_in_file_path)
            self.validation_data[1].save(self._validation_out_file_path)

    def save(self):
        """Save the pipeline settings to file"""
        self._check_readiness()

        # Make sure folder exists
        if not os.path.exists(f"{self.path}/{self.name}"):
            os.makedirs(f"{self.path}/{self.name}")

        self._save_data()
        d = self._construct_config_dict()

        with open(self._config_file_path, "w") as outfile:
            outfile.write(yaml.safe_dump(d))

    def _load_data(self, d):
        self.train_data = InteractionMatrix.load(d["train_data"]["filename"])

        self.test_data = (
            InteractionMatrix.load(d["test_data"]["data_in"]["filename"]),
            InteractionMatrix.load(d["test_data"]["data_out"]["filename"]),
        )

        if d["validation_data"]:
            self.validation_data = (
                InteractionMatrix.load(d["validation_data"]["data_in"]["filename"]),
                InteractionMatrix.load(d["validation_data"]["data_out"]["filename"]),
            )

    def load(self):
        """Load the settings from file into the correct members."""

        with open(self._config_file_path, "r") as infile:

            d = yaml.safe_load(infile)

        self.metrics = [MetricEntry(**m) for m in d["metrics"]]
        self.algorithms = [AlgorithmEntry(**a) for a in d["algorithms"]]
        self.OptimisationMetricEntry = (
            # TODO Load does not exist
            OptimisationMetricEntry.load(d["optimisation_metric"])
            if d["optimisation_metric"]
            else None
        )

        self._load_data(d)
