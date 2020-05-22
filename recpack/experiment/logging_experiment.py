from recpack.experiment.experiment import Experiment
import recpack.experiment as experiment

WANDB = True

try:
    import wandb
except ImportError:
    WANDB = False


class LoggingExperiment(Experiment):
    def __init__(self, project="uncategorized", **kwargs):
        super().__init__(**kwargs)
        experiment.init(self.name)

        for param, value in self.get_params().items():
            print(param, value)

        for param, value in self.get_params().items():
            experiment.log_param(param, value)
        # if WANDB:
        #     self.run = wandb.init(project=project, name=self.identifier)
        #     for param, value in self.get_params().items():
        #         self.run.config.__setattr__(param, value)

    def get_params(self):
        """ Return parameters as dict (for logging). """
        return super().get_params()

    def run(self):
        super().run()

        for metric in self._metrics:
            experiment.log_result(metric, metric.value)

        # if WANDB:
        #     self.run.log(self.results)
