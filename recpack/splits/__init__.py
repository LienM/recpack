from .splits import (
    SeparateDataForValidationAndTestSplit,
    SeparateDataForValidationAndTestTimedSplit,
    PredefinedUserSplit,
    StrongGeneralization,
    TimedSplit,
    StrongGeneralizationTimedSplit,
    WeakGeneralization
)

SPLITTERS = {
    'PredefinedUserSplit': PredefinedUserSplit,
    'StrongGeneralization': StrongGeneralization,
    'TimedSplit': TimedSplit,
    'StrongGeneralizationTimedSplit': StrongGeneralizationTimedSplit,
    'WeakGeneralization': WeakGeneralization,
    'SeparateDataForValidationAndTestSplit': SeparateDataForValidationAndTestSplit,
    'SeparateDataForValidationAndTestTimedSplit': SeparateDataForValidationAndTestTimedSplit,
}


def get_splitter(splitter_name):
    return SPLITTERS[splitter_name]
