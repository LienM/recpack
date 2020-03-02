from .evaluate import FoldInPercentage, LeavePOutCrossValidation, TrainingInTestOutEvaluator
from .metrics import RecallK, NDCGK, MeanReciprocalRankK

EVALUATORS = {
    'FoldInPercentage': FoldInPercentage,
    'LPOCV': LeavePOutCrossValidation,
    'TrainingInTestOut': TrainingInTestOutEvaluator,
}


def get_evaluator(evaluator_name):
    return EVALUATORS[evaluator_name]
