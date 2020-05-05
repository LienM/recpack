
from recpack.experiment.ExperimentContext import ExperimentContext


rootEC = ExperimentContext("")          # root ec
currentEC = rootEC
ECMap = {"_root": rootEC}               # rootEC encoded as _root


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


def new_experiment(name):
    global currentEC
    assert name not in ECMap
    ec = ExperimentContext(name)
    ECMap[name] = ec
    currentEC = ec
    return ec


def fork_experiment(name, above=0):
    global currentEC
    to_fork_ec = currentEC
    for i in range(above):
        if not to_fork_ec.parent:
            raise RuntimeError(f"Can't go {above} levels higher from current experiment.")
        to_fork_ec = to_fork_ec.parent
    ec = ExperimentContext(name, to_fork_ec)
    ECMap[name] = ec
    currentEC = ec
    return ec


def fork_root_experiment(name):
    return fork_experiment(name, to_fork="_root")


def set_experiment(name):
    global currentEC
    currentEC = ECMap[name]
    return currentEC


