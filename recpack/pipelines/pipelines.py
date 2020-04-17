from collections import defaultdict
from recpack.evaluate.metrics import RecallK, MeanReciprocalRankK, NDCGK
from recpack.splits.splits import TrainValidationSplitTwoDataInputs
from recpack.splitters.scenario import *


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

        for algo in algorithms:
            for m in self.metric_names:
                for K in K_values:
                    self._create(algo.name, m, K)

    def _create(self, algorithm_name, metric_name, K):
        metric = self.METRICS[metric_name](K)
        self.registry[algorithm_name][f"{metric_name}_K_{K}"] = metric
        return

    def __getitem__(self, key):
        return self.registry[key]

    def register_from_factory(self, metric_factory, identifier, K_values):
        for algo in self.algorithms:
            for K in K_values:
                self.register(metric_factory.create(K), algo.name, f"{identifier}@{K}")

    def register(self, metric, algorithm, identifier):
        print(f"registered {algorithm} - {identifier}")
        self.registry[algorithm][identifier] = metric

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


class ScenarioPipeline:

    def __init__(self, scenario, algorithms, metric_names, K_values):
        """
        Performs all steps in order and holds on to results.

        :param scenario: Splitter object which will split input data into train, validation and test data
        :type scenario: `recpack.splitters.Scenario`

        :param algorithms: List of algorithms to evaluate in this pipeline
        :type algorithms: `list(recpack.algorithms.Model)`

        :param metric_names: The names of metrics to compute in this pipeline.
                            Allowed values are: `NDCG`, `Recall` and `MRR`
        :type metric_names: `list(string)`

        :param K_values: The K values for each of the metrics
        :type K_values: `list(int)`
        """
        self.scenario = scenario
        self.algorithms = algorithms

        self.metric_names = metric_names
        self.K_values = K_values
        self.metric_registry = MetricRegistry(algorithms, metric_names, K_values)

    def run(self, data, data_2=None):
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
        self.scenario.split(data, data_2)

        for algo in self.algorithms:
            # Only pass the sparse training interaction matrix to algo
            algo.fit(self.scenario.training_data.values)

        for _in, _out, users in self.scenario.test_iterator:
            for algo in self.algorithms:

                metrics = self.metric_registry[algo.name]
                X_pred = algo.predict(_in)

                for metric in metrics.values():
                    metric.update(X_pred, _out, users)

    def get(self):
        return self.metric_registry.metrics

    def get_number_of_users_evaluated(self):
        return self.metric_registry.number_of_users_evaluated


class Pipeline:
    """
    Performs all steps in order and holds on to results.

    :param splitter: Splitter object which will split input data into train, validation and test data
    :type splitter: `recpack.splits.Splitter`

    :param algorithms: List of algorithms to evaluate in this pipeline
    :type algorithms: `list(recpack.algorithms.Model)`

    :param evaluator: Evaluator object, takes as input the train, validation and test data
                      to generate in and out matrices for evaluation.
    :type evaluator: `recpack.evaluate.Evaluator`

    :param metric_names: The names of metrics to compute in this pipeline.
                         Allowed values are: `NDCG`, `Recall` and `MRR`
    :type metric_names: `list(string)`

    :param K_values: The K values for each of the metrics
    :type K_values: `list(int)`
    """

    def __init__(self, splitter, algorithms, evaluator, metric_names, K_values):
        self.splitter = splitter
        self.algorithms = algorithms
        self.evaluator = evaluator
        self.metric_names = metric_names
        self.K_values = K_values
        self.metric_registry = MetricRegistry(algorithms, metric_names, K_values)

    def run(self, data, data_2=None):
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

        # Check if splitter expects 2nd data source:
        if issubclass(type(self.splitter), TrainValidationSplitTwoDataInputs):
            assert data_2 is not None
            tr_data, val_data, te_data = self.splitter.split(data, data_2)
        else:
            tr_data, val_data, te_data = self.splitter.split(data)

        for algo in self.algorithms:
            # Only pass the sparse training interaction matrix to algo
            algo.fit(tr_data.values)

        self.evaluator.split(tr_data, val_data, te_data)

        for _in, _out, users in self.evaluator:
            for algo in self.algorithms:

                metrics = self.metric_registry[algo.name]
                X_pred = algo.predict(_in)

                for metric in metrics.values():

                    metric.update(X_pred, _out, users)

    def get(self):
        return self.metric_registry.metrics

    def get_number_of_users_evaluated(self):
        return self.metric_registry.number_of_users_evaluated


class ParameterGeneratorPipeline(Pipeline):
    """
    Construct a new pipeline for each set of parameters generated by the generator member.
    This makes it possible to test different seeds, different splits, ...

    :param parameterGenerator: Generator object which will generate parameters for each of the pipelines to run.
    :type parameterGenerator: `recpack.pipelines.ParameterGenerator`

    :param splitter_class: Class which will be used to create new splitters with different parameters.
    :type splitter: subclass of `recpack.splits.Splitter`

    :param algorithms: List of algorithms to evaluate in this pipeline
    :type algorithms: `list(recpack.algorithms.Model)`

    :param evaluator_class: Evaluator class, used to generate evaluators for the pipelines.
    :type evaluator: subclass of `recpack.evaluate.Evaluator`

    :param metric_names: The names of metrics to compute in this pipeline.
                         Allowed values are: `NDCG`, `Recall` and `MRR`
    :type metric_names: `list(string)`

    :param K_values: The K values for each of the metrics
    :type K_values: `list(int)`

    """
    def __init__(self, parameterGenerator, splitter_class, algorithms, evaluator_class, metric_names, K_values):
        self.splitter_class = splitter_class
        self.algorithms = algorithms
        self.evaluator_class = evaluator_class
        self.metric_names = metric_names
        self.K_values = K_values

        self.parameterGenerator = parameterGenerator

        # Per value in parameter iterator a new metric registry will be registered.
        self.metric_registries = []

    def run(self, data, validation_data=None):

        for params in self.parameterGenerator:
            splitter = self.splitter_class(**params.splitter_params)
            # TODO batch size
            evaluator = self.evaluator_class(**params.evaluator_params)

            pipeline = Pipeline(splitter, self.algorithms, evaluator, self.metric_names, self.K_values)

            pipeline.run(data, validation_data)

            # TODO: Add the parameters used somewhere.
            self.metric_registries.append(pipeline.metric_registry)

    def get_aggregated(self):
        """
        Aggregate metrics from different runs into a single metric registry.
        """
        raise NotImplementedError("Comming soon")
        pass

    def get(self):
        """
        Returns a list of metric registry metrics (in order of execution)
        """
        return [m.metrics for m in self.metric_registries]

    def get_number_of_users_evaluated(self):
        """
        Returns the number of users used for evaluation in each of the slices
        """
        return [m.number_of_users_evaluated for m in self.metric_registries]
