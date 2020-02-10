from collections import defaultdict
from recpack.evaluate import RecallK, MeanReciprocalRankK, NDCGK


class MetricRegistry:

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

    @property
    def metrics(self):
        results = defaultdict(dict)
        for key in self.registry:
            for k in self.registry[key]:
                results[key][k] = self.registry[key][k].value
        return results


class Pipeline:
    """
    Performs all steps in order and holds on to results. 
    """

    def __init__(self, splitter, algorithms, evaluator, metric_names, K_values):
        self.splitter = splitter
        self.algorithms = algorithms
        self.evaluator = evaluator
        self.metric_names = metric_names
        self.K_values = K_values
        self.metric_registry = MetricRegistry(algorithms, metric_names, K_values)

    def run(self, sp_mat):

        tr_mat, te_mat = self.splitter.split(sp_mat)

        for algo in self.algorithms:
            algo.fit(tr_mat)

        self.evaluator.split(te_mat)

        for _in, _out in self.evaluator:
            for algo in self.algorithms:

                metrics = self.metric_registry[algo.name]
                X_pred = algo.predict(_in)

                for metric in metrics.values():

                    metric.update(X_pred, _out)

    def get(self):
        return self.metric_registry.metrics

