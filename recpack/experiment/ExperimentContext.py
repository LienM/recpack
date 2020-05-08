import os
from datetime import datetime
import csv
import logging
import atexit


BASE_OUTPUT_DIR = "output/"
PARAMS_FILENAME = "params.csv"
RESULTS_FILENAME = "results.csv"

WANDB_ENABLED = False


class ExperimentContext(object):
    """
    Provides a context for the currently running experiment. This allows for global parameter and result logging.
    See also __init__.py of this package.
    """
    def __init__(self, name, parent=None):
        self._name = name
        self.parent = parent

        self._params = dict()
        self._results = dict()
        self._files = dict()
        self._start_time = datetime.now().replace(microsecond=0)

        self._output_path = os.path.join(BASE_OUTPUT_DIR, str(self.name) + "_" +self._start_time.isoformat())
        if name:
            os.makedirs(self.output_path)
            atexit.register(self.save)

    @property
    def name(self):
        if self.parent and self.parent.name:
            return self.parent.name + "_" + self._name
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def root(self):
        root = self
        while root.parent:
            root = root.parent
        return root

    @property
    def root_name(self):
        return self.root.name

    @property
    def output_path(self):
        return self._output_path

    @property
    def params_path(self):
        return os.path.join(self.output_path, PARAMS_FILENAME)

    @property
    def results_path(self):
        return os.path.join(self.output_path, RESULTS_FILENAME)

    @property
    def params(self):
        ret = self.parent.params.copy() if self.parent else dict()
        ret.update(self._params)
        return ret

    @property
    def results(self):
        ret = self.parent.results.copy() if self.parent else dict()
        ret.update(self._results)
        return ret

    @property
    def files(self):
        ret = self.parent.files.copy() if self.parent else dict()
        ret.update(self._files)
        return ret

    def save(self):
        # local
        for filepath, data in [(self.params_path, self.params), (self.results_path, self.results)]:
            with open(filepath, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                writer.writeheader()
                writer.writerow(data)

        # files
        for name, path in self.files.items():
            os.symlink(path, os.path.join(self.output_path, name))

        # wanddb
        if WANDB_ENABLED:
            import wandb
            project = self.root_name
            if len(project) == 0:
                project = "uncategorized"
            run = wandb.init(project=project, name=self.name[len(self.root_name)+1:], reinit=True)
            with run:
                for param, value in self.params.items():
                    run.config.__setattr__(param, value)

                run.log(self.results)

    def log_param(self, name, value):
        if name in self._params:
            logging.warning(f"{name} already logged in parameters of experiment {self.name}, overwriting")
        self._params[name] = value
        return value

    def log_result(self, name, value):
        if name in self._results:
            logging.warning(f"{name} already logged in results of experiment {self.name}, overwriting")
        self._results[name] = value
        return value

    def log_file(self, name, path):
        if name in self._results:
            logging.warning(f"{name} already logged in files of experiment {self.name}, overwriting")
        self._files[name] = os.path.abspath(path)
        return path


