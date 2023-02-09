# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from collections.abc import Iterable
import datetime
import os
from typing import Tuple, Union, Dict, List, Optional, Any
import warnings

import numpy as np

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.matrix import InteractionMatrix
from recpack.pipelines.hyperparameter_optimisation import OptimisationInfo, GridSearchInfo
from recpack.pipelines.pipeline import Pipeline
from recpack.pipelines.registries import (
    ALGORITHM_REGISTRY,
    METRIC_REGISTRY,
    AlgorithmEntry,
    MetricEntry,
    OptimisationMetricEntry,
)
from recpack.postprocessing.filters import PostFilter
from recpack.postprocessing.postprocessors import Postprocessor
from recpack.scenarios import Scenario

logger = logging.getLogger("recpack")


class PipelineBuilder(object):
    """Builder to facilitate construction of pipelines.

    The builder contains functions to set specific values for the pipeline.
    Save and Load make it possible to easily recreate pipelines.

    To disable history filtering in the pipeline, set the :attr:`remove_history` attribute to False.::

        pipeline_builder.remove_history = False

    :param folder_name: The name of the folder where pipeline
        information will be stored.
        If no name is specified, the timestamp of creation is used.
    :type folder_name: str, optional
    :param base_path: The base_path to store pipeline in,
        defaults to the current working directory.
    :type base_path: str, optional
    """

    def __init__(self, folder_name: Optional[str] = None, base_path: Optional[str] = None):

        self.folder_name = folder_name
        if self.folder_name is None:
            self.folder_name = datetime.datetime.now().isoformat()

        self.base_path = base_path or os.getcwd()

        self.metric_entries = {}
        self.algorithm_entries = []
        self.post_processor = Postprocessor()

        self.remove_history = True

        self.results_directory = f"{self.base_path}/{self.folder_name}"

    def _arg_to_str(self, arg: Union[type, str]) -> str:
        if type(arg) == type:
            arg = arg.__name__

        elif type(arg) != str:
            raise TypeError(f"Argument should be string or type, not {type(arg)}!")

        return arg

    def add_metric(self, metric: Union[str, type], K: Optional[Union[List, int]] = None):
        """Register a metric to evaluate

        :param metric: Metric name or type.
        :type metric: Union[str, type]
        :param K: The K value(s) used to construct metrics.
            If it is a list, for each value a metric is added.
        :type K: Optional[Union[List, int]], optional
        :raises ValueError: If metric can't be resolved to a key
            in the ``METRIC_REGISTRY``.
        """

        # Make it so it's possible to add metrics by their class as well.
        metric = self._arg_to_str(metric)

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric} could not be resolved.")

        if isinstance(K, Iterable):
            for k in K:
                self.add_metric(metric, k)
        elif K is not None:
            # TODO Should we validate these K values to see if they make sense?
            # Check if metric already exists
            metric_name = f"{metric}_{K}"

            if metric_name in self.metric_entries:
                logger.warning(f"Metric {metric_name} already exists.")
            else:
                self.metric_entries[metric_name] = MetricEntry(metric, K)
        else:
            # Bit of a hack to pass none, but it's the best I can do I think.
            self.metric_entries[metric] = MetricEntry(metric, K)

    def add_algorithm(
        self,
        algorithm: Union[str, type],
        grid: Optional[Dict[str, List]] = None,
        params: Optional[Dict[str, Any]] = None,
        optimisation_info: Optional[OptimisationInfo] = None,
    ):
        """Add an algorithm to use in the pipeline.

        If the algorithm is not implemented by default in recpack,
        you should register it in the ``ALGORITHM_REGISTRY``

        :param algorithm: Algorithm class name or type of the algorithm to add.
        :type algorithm: Union[str, type]
        :param grid: [DEPRECATED] Parameters to optimise,
            the dict will be turned into a grid such that each combination of values
            is used. Defaults to None
        :type grid: Dict[str, List], optional
        :param params: The fixed parameters for running the algorithm, represented as a key-value dictionary.
            Defaults to None
        :type params: Dict[str, Any], optional
        :param optimisation_info: Optimisation info,
            contains information for the optimiser to define the parameter space.
        :type optimisation_info: OptimisationInfo
        :raises ValueError: If the passed algorithm can't be resolved to a key
            in the ``ALGORITHM_REGISTRY``.
        """
        algorithm = self._arg_to_str(algorithm)

        if grid is not None:
            optimisation_info = GridSearchInfo(grid)

            warnings.warn(
                "Grid parameter for add_algorithm function will be deprecated in favour of optimisation_info."
            )

        if algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm {algorithm} could not be resolved.")

        self.algorithm_entries.append(AlgorithmEntry(algorithm, optimisation_info or None, params or {}))

    def add_post_filter(self, filter: PostFilter) -> None:
        """Add a filter which will be applied
            on the recommendation scores before prediction.

        :param filter: Filter to apply, cannot be of type RemoveHistory
        :type filter: PostFilter
        """
        self.post_processor.add_filter(filter)

    def set_optimisation_metric(self, metric: Union[str, type], K: int, minimise=False):
        """Set the metric for optimisation of parameters in algorithms.

        If the metric is not implemented by default in recpack,
        you should register it in the ``METRIC_REGISTRY``

        :param metric: metric name or metric type
        :type metric: Union[str, type]
        :param K: The K value for the metric
        :type K: int
        :param minimise: If True minimal value for metric is better, defaults to False
        :type minimise: bool, optional
        :raises ValueError: If metric can't be resolved to a key
            in the ``METRIC_REGISTRY``.
        """
        metric = self._arg_to_str(metric)

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"metric {metric} could not be resolved.")

        self.optimisation_metric = OptimisationMetricEntry(metric, K, minimise)

    def set_full_training_data(self, train_data: InteractionMatrix):
        """Set the full_training dataset.
        This dataset is used for the final training
        before evaluation on the test dataset.

        :param train_data: The interaction matrix to use for training.
        :type train_data: InteractionMatrix
        """
        self.full_training_data = train_data

    def set_validation_training_data(self, train_data: InteractionMatrix):
        """Set the validation training dataset.
        This dataset is used for training models during parameter optimisation,
        or for incrementally trained models.

        :param train_data: The interaction matrix to use for training.
        :type train_data: InteractionMatrix
        """
        self.validation_training_data = train_data

    def set_validation_data(self, validation_data: Tuple[InteractionMatrix, InteractionMatrix]):
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

    @property
    def remove_history(self):
        """``True`` to enable removal of a user's previous interactions,
        ```False``` to disable. Defaults to ``True``.
        """
        return self._remove_history

    @remove_history.setter
    def remove_history(self, value):
        """Pass ``True`` to enable removal of a user's previous interactions,
        ```False``` to disable. Defaults to ``True``.
        """
        self._remove_history = value

    def _check_readiness(self):
        if len(self.metric_entries) == 0:
            raise RuntimeError("No metrics specified, can't construct pipeline")

        if len(self.algorithm_entries) == 0:
            raise RuntimeError("No algorithms specified, can't construct pipeline")

        if not hasattr(self, "optimisation_metric") and np.any([algo.optimise for algo in self.algorithm_entries]):
            raise RuntimeError(
                "No optimisation metric selected to perform "
                "requested hyperparameter optimisation, "
                "can't construct pipeline."
            )

        # Check availability of data
        if not hasattr(self, "full_training_data"):
            raise RuntimeError("No full training data available, can't construct pipeline.")

        if not hasattr(self, "test_data"):
            raise RuntimeError("No test data available, can't construct pipeline.")

        # If there are parameters to optimise,
        # there needs to be validation data available.
        if not hasattr(self, "validation_data") and self._requires_validation_data():
            raise RuntimeError(
                "No validation data available to perform "
                "the requested hyperparameter optimisation, "
                "can't construct pipeline."
            )

        if not hasattr(self, "validation_training_data") and self._requires_validation_data():
            raise RuntimeError(
                "No validation training data available to perform "
                "the requested hyperparameter optimisation, "
                "can't construct pipeline."
            )

        # Validate shape is correct
        shape = self.full_training_data.shape

        if any([d.shape != shape for d in self.test_data]):
            raise RuntimeError("Shape mismatch between test and training data")

        if hasattr(self, "validation_data") and any([d.shape != shape for d in self.validation_data]):
            raise RuntimeError("Shape mismatch between validation and training data")

        if hasattr(self, "validation_training_data") and self.validation_training_data.shape != shape:
            raise RuntimeError("Shape mismatch between validation training data and full training data")

    def _requires_validation_data(self) -> bool:
        return any([algo.optimise for algo in self.algorithm_entries]) or any(
            [issubclass(ALGORITHM_REGISTRY.get(algo.name), TorchMLAlgorithm) for algo in self.algorithm_entries]
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
            self.algorithm_entries,
            list(self.metric_entries.values()),
            self.full_training_data,
            self.validation_training_data if hasattr(self, "validation_training_data") else None,
            self.validation_data if hasattr(self, "validation_data") else None,
            self.test_data,
            self.optimisation_metric if hasattr(self, "optimisation_metric") else None,
            self.post_processor,
            self.remove_history,
        )
