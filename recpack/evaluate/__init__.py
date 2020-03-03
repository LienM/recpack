from .evaluate import FoldInPercentage, LeavePOutCrossValidation, TrainingInTestOutEvaluator, TimedTestSplit
from .metrics import RecallK, NDCGK, MeanReciprocalRankK

EVALUATORS = {
    'FoldInPercentage': FoldInPercentage,
    'LPOCV': LeavePOutCrossValidation,
    'TrainingInTestOut': TrainingInTestOutEvaluator,
    'TimedTestSplit': TimedTestSplit
}


def get_evaluator(evaluator_name):
    return EVALUATORS[evaluator_name]
