from .evaluate import (
    FoldInPercentageEvaluator,
    LeavePOutCrossValidationEvaluator,
    TrainingInTestOutEvaluator,
    TimedSplitEvaluator,
    MutexEvaluator,
)
from .metrics import RecallK, NDCGK, MeanReciprocalRankK

EVALUATORS = {
    'FoldInPercentage': FoldInPercentageEvaluator,
    'LPOCV': LeavePOutCrossValidationEvaluator,
    'TrainingInTestOut': TrainingInTestOutEvaluator,
    'TimedSplit': TimedSplitEvaluator,
    'MutexEvaluator': MutexEvaluator
}


def get_evaluator(evaluator_name):
    return EVALUATORS[evaluator_name]
