from .pipelines import Pipeline, ParameterGeneratorPipeline
from .parameter_generators import ParameterGenerator, TemporalSWParameterGenerator

PARAMETER_GENERATOS = {
    "TemporalSWParrameterGenerator": TemporalSWParameterGenerator
}


def get_parameter_generator(generator_name):
    return PARAMETER_GENERATOS[generator_name]
