from recpack.evaluate.evaluators import (
    FoldInPercentageEvaluator,
    LeavePOutCrossValidationEvaluator,
    TrainingInTestOutEvaluator,
    TimedSplitEvaluator,
)

EVALUATORS = {
    'FoldInPercentage': FoldInPercentageEvaluator,
    'LPOCV': LeavePOutCrossValidationEvaluator,
    'TrainingInTestOut': TrainingInTestOutEvaluator,
    'TimedSplit': TimedSplitEvaluator
}


def get_evaluator(evaluator_name):
    return EVALUATORS[evaluator_name]
