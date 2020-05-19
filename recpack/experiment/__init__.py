""" Globally accessible experiment context. Through these functions, parameters and results can be logged.
They will be stored at the end of the program (atexit). """

from recpack.experiment.ExperimentContext import ExperimentContext
from recpack.experiment.experiment import Experiment


rootEC = ExperimentContext("")          # root ec
currentEC = rootEC
ECMap = dict()


def ECRequired(f):
    def wrapper(*args, **kwargs):
        if currentEC is None:
            raise RuntimeError("Set ExperimentContext first with experiment.new_experiment of experiment.set_experiment")
        return f(*args, **kwargs)

    return wrapper


@ECRequired
def log_param(name, value):
    return currentEC.log_param(name, value)


@ECRequired
def log_result(name, value):
    return currentEC.log_result(name, value)

@ECRequired
def log_file(name, path):
    return currentEC.log_file(name, path)


def init(name, wandb=False):
    global WANDB_ENABLED
    if rootEC.name:
        raise RuntimeError("Experiment already labeled")
    assert name not in ECMap
    rootEC.name = name
    ECMap[name] = rootEC
    if wandb:
        ExperimentContext.enable_wandb()


def _fork_experiment(name, to_fork_ec):
    global currentEC
    ec = ExperimentContext(name, to_fork_ec)
    ECMap[name] = ec
    currentEC = ec


def fork_experiment(name, above=0):
    """ Fork the current experiment or follow its parent `above` times. """
    global currentEC
    to_fork_ec = currentEC
    for i in range(above):
        if not to_fork_ec.parent:
            raise RuntimeError(f"Can't go {above} levels higher from current experiment.")
        to_fork_ec = to_fork_ec.parent
    return _fork_experiment(name, to_fork_ec)


def fork_root_experiment(name):
    """ Make a direct fork of the root experiment. """
    return _fork_experiment(name, rootEC)


def set_experiment(name):
    """ Switch to a different experiment by name. """
    global currentEC
    currentEC = ECMap[name]
    return currentEC


