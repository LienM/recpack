from .splits import (
    SeparateDataForValidationAndTestSplit,
    SeparateDataForValidationAndTestTimedSplit,
    PredefinedUserSplit,
    StrongGeneralization,
    TimedSplit,
    WeakGeneralization
)

SPLITTERS = {
    'PredefinedUserSplit': PredefinedUserSplit,
    'StrongGeneralization': StrongGeneralization,
    'TimedSplit': TimedSplit,
    'WeakGeneralization': WeakGeneralization,
    'SeparateDataForValidationAndTestSplit': SeparateDataForValidationAndTestSplit,
    'SeparateDataForValidationAndTestTimedSplit': SeparateDataForValidationAndTestTimedSplit,
}


def get_splitter(splitter_name):
    return SPLITTERS[splitter_name]
