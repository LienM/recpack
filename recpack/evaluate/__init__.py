from .evaluate import (
    FoldInPercentageEvaluator,
    LeavePOutCrossValidationEvaluator,
    TrainingInTestOutEvaluator,
    TimedSplitEvaluator
)
from .metrics import RecallK, NDCGK, MeanReciprocalRankK

EVALUATORS = {
    'FoldInPercentage': FoldInPercentageEvaluator,
    'LPOCV': LeavePOutCrossValidationEvaluator,
    'TrainingInTestOut': TrainingInTestOutEvaluator,
    'TimedSplit': TimedSplitEvaluator
}


def get_evaluator(evaluator_name):
    return EVALUATORS[evaluator_name]
