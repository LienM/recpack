from .pipelines import Pipeline, ParameterGeneratorPipeline
from .parameter_generators import ParameterGenerator, TemporalSWParameterGenerator

PARAMETER_GENERATORS = {
    "TemporalSWParameterGenerator": TemporalSWParameterGenerator
}


def get_parameter_generator(generator_name):
    return PARAMETER_GENERATORS[generator_name]
