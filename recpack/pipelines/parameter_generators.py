class Params:
    """
    Simple struct for storing parameters.
    """
    def __init__(self, splitter_params=[], evaluator_params=[]):
        self.splitter_params = splitter_params
        self.evaluator_params = evaluator_params


class ParameterIterator:
    """
    Iterator to get the parameters out of the ParameterGenerators
    """
    def __init__(self, parameter_generator):
        self._param_generator = parameter_generator
        self._index = 0
        self._max_index = len(parameter_generator)

    def __next__(self):
        if self._index < self._max_index:
            params = self._param_generator.get(self._index)
            self._index += 1
            return params
        else:
            raise StopIteration


class ParameterGenerator:
    """
    Parameter generator main class.
    Sub classes should implement get and __len__() methods
    """
    def __init__(self):
        pass

    def get(self, i):
        pass

    def __len__(self):
        return 0

    def __iter__(self):
        return ParameterIterator(self)


class TemporalSWParameterGenerator(ParameterGenerator):
    """
    Parameter generator to use with ParameterGeneratorPipeline to evaluate algorithms using a sliding window approach.
    SW approach detailed in paper by Olivier Jeunen (http://adrem.uantwerpen.be/bibrem/pubs/OfflineEvalJeunen2018.pdf).

    :param t_0: The epoch timestamp of the start of the first window.
                Typically equal to the minimal timestamp in your dataset.
    :type t_0: `int`

    :param interval_between_t: The number of seconds from previous slice in each subsequent slice.
    :type interval_between_t: `int`

    :param nr_t: The number of slices to generate
    :type nr_t: `int`

    :param t_delta: The size in seconds of the test part of the data.
    :type t_delta: `int`

    :param t_alpha: The size in seconds of the train part of the data.
    :type t_alpha: `int`

    """
    def __init__(self, t_0, interval_between_t, nr_t, t_delta=None, t_alpha=None):
        self.t_0 = t_0
        self.t_delta = t_delta
        self.t_alpha = t_alpha
        self.interval_between_t = interval_between_t
        self.nr_t = nr_t

    def get(self, i):
        # Do i + 1 because in the first iteration t should be interval away from t_0
        splitter_params = [self.t_0 + ((i+1) * self.interval_between_t), self.t_delta, self.t_alpha]
        return Params(splitter_params=splitter_params)

    def __len__(self):
        return self.nr_t
