import logging
from collections import defaultdict, Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
import datetime
import os
from typing import Tuple, Union, Dict, Any
import yaml

import numpy as np

# import scipy.sparse
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import recpack.algorithms

# from recpack.metrics import METRICS
from recpack.data.matrix import Matrix, InteractionMatrix

# from recpack.splitters.splitter_base import FoldIterator

logger = logging.getLogger("recpack")


class Registry:
    def __init__(self):
        self.registered = {}

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        try:
            self.get(key)
            return True
        except AttributeError:
            return False

    def get(self, name):
        raise NotImplementedError

    def register(self, name, c):
        self.registered[name] = c


class AlgorithmRegistry(Registry):
    def get(self, name):
        if name in self.registered:
            return self.registered[name]
        else:
            return getattr(recpack.algorithms, name)


class MetricRegistry(Registry):
    def get(self, name):
        if name in self.registered:
            return self.registered[name]
        else:
            return getattr(recpack.metrics, name)


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
    def number_of_users_evaluated(self):
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
    def __init__(
        self,
        algorithms,
        metrics,
        train_data,
        validation_data,
        test_data,
        optimisation_metric,
    ):
        """
        Performs all steps in order and holds on to results.

        :param algorithms: List of algorithms to evaluate in this pipeline
        :type algorithms: `list(recpack.algorithms.Model)`

        :param metric_names: The names of metrics to compute in this pipeline.
                            Allowed values can be found in :ref:`recpack.metrics`
        :type metric_names: `list(string)`

        :param K_values: The K values for each of the metrics
        :type K_values: `list(int)`
        """
        self.algorithms = algorithms
        self.metrics = metrics
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.optimisation_metric = optimisation_metric
        self.metric_registry = MetricAccumulator()

    def _optimise(self, algorithm, metric_entry: OptimisationMetricEntry):
        # TODO: investigate using advanced optimisers
        # Construct grid:
        results = []
        if len(algorithm.grid) == 0:
            return algorithm.params
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
        """
        Runs the pipeline.

        This will use the different components in the pipeline to:
        1. Train models
        2. Evaluate models
        3. return metrics

        :param train_data: Training data. If given a tuple the second matrix will
                           be used as targets.
        :param test_data: Test data, (in, out) tuple.
        :param validation_data: Validation data, (in, out) tuple. Optional.
        """
        # optimisation phase:
        for algorithm in tqdm(self.algorithms):
            optimal_params = self._optimise(algorithm, self.optimisation_metric)

            # Train again.
            algo = ALGORITHM_REGISTRY.get(algorithm.name)(**optimal_params)
            # algo.fit(union(self.train_data, self.validation_data))
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
                m = getattr(recpack.metrics, metric.name)(metric.K)
                m.calculate(test_out.binary_values, recommendations)

                self.metric_registry.register(m, algo.identifier, m.name)

    def get(self):
        return self.metric_registry.metrics

    def get_number_of_users_evaluated(self):
        return self.metric_registry.number_of_users_evaluated


class PipelineBuilder(object):
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

    def add_metric(self, metric: str, K):
        """Register a metric to evaluate

        :param metric: [description]
        :type metric: [type]
        :param K: [description]
        :type K: [type]
        :return: [description]
        :rtype: [type]
        """

        # Make it so it's possible to add metrics by their class as well.
        if type(metric) == type:
            metric = metric.__name__

        if type(metric) != str:
            raise ValueError(f"metric should be string or type, not {type(metric)}!")

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"metric {metric} could not be resolved.")

        if isinstance(K, Iterable):
            for k in K:
                self.add_metric(metric, k)
        else:
            self.metrics.append(MetricEntry(metric, K))

    def add_algorithm(self, algorithm, grid={}, params={}):
        """Register an algorithm to run.

        :param algorithm: [description]
        :type algorithm: [type]
        :param grid: [description], defaults to {}
        :type grid: dict, optional
        :param params: [description], defaults to {}
        :type params: dict, optional
        """
        if type(algorithm) == type:
            algorithm = algorithm.__name__

        if type(algorithm) != str:
            raise ValueError(
                f"algorithm should be string or type, not {type(algorithm)}!"
            )

        if algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(f"algorithm {algorithm} could not be resolved.")

        if type(algorithm) != str:
            raise ValueError("algorithm name should be string!")
            # TODO: Accept strings and class names?
        self.algorithms.append(AlgorithmEntry(algorithm, grid, params))

    def set_optimisation_metric(self, metric, K, minimise=False):
        # Make it so it's possible to add metrics by their class as well.
        if type(metric) == type:
            metric = metric.__name__

        if type(metric) != str:
            raise ValueError(f"metric should be string or type, not {type(metric)}!")

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"metric {metric} could not be resolved.")

        self.optimisation_metric = OptimisationMetricEntry(metric, K, minimise)

    def set_train_data(self, train_data: Matrix):
        self.train_data = train_data

    def set_validation_data(self, validation_data: Matrix):
        if not len(validation_data) == 2:
            raise RuntimeError(
                "Incorrect value, expected tuple with data_in and data_out"
            )
        self.validation_data = validation_data

    def set_test_data(self, test_data: Matrix):
        if not len(test_data) == 2:
            raise RuntimeError(
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
            raise RuntimeError("No optimisation metric selected")

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
                "No validation data available, can't construct pipeline."
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
        """Save the pipeline settings to file

        :return: [description]
        :rtype: [type]
        """
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
            OptimisationMetricEntry.load(d["optimisation_metric"])
            if d["optimisation_metric"]
            else None
        )

        self._load_data(d)
