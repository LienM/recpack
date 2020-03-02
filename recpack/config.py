import yaml
import recpack.splits
import recpack.algorithms
import recpack.evaluate


class PipelineConfig:
    def __init__(self, config_file):
        self.config = yaml.safe_load(config_file)
        self.validate()

    def validate(self):
        assert "splitter" in self.config
        assert "type" in self.config["splitter"]
        assert self.config["splitter"]["type"] in recpack.splits.SPLITTERS

        assert "evaluator" in self.config
        assert "type" in self.config["evaluator"]
        assert self.config["evaluator"]["type"] in recpack.evaluate.EVALUATORS

        assert "metrics" in self.config
        # TODO make testable that the metric is actually usable
        assert "K_values" in self.config

        assert "algorithms" in self.config
        assert type(self.config["algorithms"]) == list

        for algo_c in self.config["algorithms"]:
            assert "type" in algo_c
            assert algo_c["type"] in recpack.algorithms.ALGORITHMS

    def get_algorithms(self):
        for algo_c in self.config["algorithms"]:
            algo_type = algo_c["type"]
            algo_params = algo_c.get("params", {})
            yield algo_type, algo_params

    def get_evaluator(self):
        return self.config["evaluator"]["type"], self.config["evaluator"].get("params", {})

    def get_splitter(self):
        return self.config["splitter"]["type"], self.config["splitter"].get("params", {})

    def get_metrics(self):
        return self.config["metrics"]

    def get_K_values(self):
        return self.config["K_values"]


class ParameterGeneratorPipelineConfig(PipelineConfig):

    def validate(self):

        assert "parameter_generator" in self.config
        assert "type" in self.config["parameter_generator"]
        assert self.config["parameter_generator"]["type"] in recpack.pipelines.PARAMETER_GENERATOS

        assert "splitter" in self.config
        assert "type" in self.config["splitter"]
        assert self.config["splitter"]["type"] in recpack.splits.SPLITTERS

        assert "evaluator" in self.config
        assert "type" in self.config["evaluator"]
        assert self.config["evaluator"]["type"] in recpack.evaluate.EVALUATORS

        assert "metrics" in self.config
        # TODO make testable that the metric is actually usable
        assert "K_values" in self.config

        assert "algorithms" in self.config
        assert type(self.config["algorithms"]) == list

        for algo_c in self.config["algorithms"]:
            assert "type" in algo_c
            assert algo_c["type"] in recpack.algorithms.ALGORITHMS

    def get_parameter_generator(self):
        return self.config["parameter_generator"]["type"], self.config["parameter_generator"].get("params", {})

    def get_evaluator(self):
        return self.config["evaluator"]["type"]

    def get_splitter(self):
        return self.config["splitter"]["type"]
