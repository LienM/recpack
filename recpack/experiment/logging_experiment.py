from recpack.experiment.experiment import Experiment

WANDB = True

try:
    import wandb
except ImportError:
    WANDB = False


class LoggingExperiment(Experiment):
    def __init__(self, project="uncategorized", **kwargs):
        super().__init__(**kwargs)
        if WANDB:
            run = wandb.init(project=project, name=self.identifier)
            for param, value in self.parameters.items():
                run.config.__setattr__(param, value)
