from recpack.experiment.experiment import Experiment

WANDB = True

try:
    import wandb
except ImportError:
    WANDB = False


class LoggingExperiment(Experiment):
    def __init__(self, project="uncategorized", **kwargs):
        super().__init__(**kwargs)
        for param, value in self.get_params().items():
            print(param, value)

        # if WANDB:
        #     run = wandb.init(project=project, name=self.identifier)
        #     for param, value in self.get_params().items():
        #         run.config.__setattr__(param, value)

    def get_params(self):
        """ Return parameters as dict (for logging). """
        return super().get_params()
