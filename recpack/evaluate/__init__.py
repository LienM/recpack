from recpack.evaluate.evaluators import (
    FoldInPercentageEvaluator,
    LeavePOutCrossValidationEvaluator,
    TrainingInTestOutEvaluator,
    TimedSplitEvaluator,
)
# from recpack.evaluate.metrics import RecallK, NDCGK, MeanReciprocalRankK, MutexMetric

EVALUATORS = {
    'FoldInPercentage': FoldInPercentageEvaluator,
    'LPOCV': LeavePOutCrossValidationEvaluator,
    'TrainingInTestOut': TrainingInTestOutEvaluator,
    'TimedSplit': TimedSplitEvaluator,
    # 'MutexMetric': MutexMetric
}


def get_evaluator(evaluator_name):
    return EVALUATORS[evaluator_name]
