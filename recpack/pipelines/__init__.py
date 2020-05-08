from recpack.pipelines.pipelines import Pipeline, LoggingPipeline
from recpack.pipelines.parameter_generators import ParameterGenerator, TemporalSWParameterGenerator, SplitterGridSearchGenerator

PARAMETER_GENERATORS = {
    "TemporalSWParameterGenerator": TemporalSWParameterGenerator,
    "SplitterGridSearchGenerator": SplitterGridSearchGenerator
}


def get_parameter_generator(generator_name):
    return PARAMETER_GENERATORS[generator_name]
