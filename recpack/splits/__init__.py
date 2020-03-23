from .splits import (
    SeparateDataForValidationAndTestSplit,
    SeparateDataForValidationAndTestTimedSplit,
    PredefinedUserSplit,
    StrongGeneralizationSplit,
    TimedSplit,
    StrongGeneralizationTimedSplit,
    WeakGeneralizationSplit
)

SPLITTERS = {
    'PredefinedUserSplit': PredefinedUserSplit,
    'StrongGeneralizationSplit': StrongGeneralizationSplit,
    'TimedSplit': TimedSplit,
    'StrongGeneralizationTimedSplit': StrongGeneralizationTimedSplit,
    'WeakGeneralizationSplit': WeakGeneralizationSplit,
    'SeparateDataForValidationAndTestSplit': SeparateDataForValidationAndTestSplit,
    'SeparateDataForValidationAndTestTimedSplit': SeparateDataForValidationAndTestTimedSplit,
}


def get_splitter(splitter_name):
    return SPLITTERS[splitter_name]
