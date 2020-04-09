from .evaluate import (
    FoldInPercentageEvaluator,
    LeavePOutCrossValidationEvaluator,
    TrainingInTestOutEvaluator,
    TimedSplitEvaluator,
)
from .metrics import RecallK, NDCGK, MeanReciprocalRankK, MutexMetric

EVALUATORS = {
    'FoldInPercentage': FoldInPercentageEvaluator,
    'LPOCV': LeavePOutCrossValidationEvaluator,
    'TrainingInTestOut': TrainingInTestOutEvaluator,
    'TimedSplit': TimedSplitEvaluator,
    'MutexMetric': MutexMetric
}


def get_evaluator(evaluator_name):
    return EVALUATORS[evaluator_name]
