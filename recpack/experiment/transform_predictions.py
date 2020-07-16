import numpy as np


def FilterHistory(batch_iterator):
    """
    Example business rule to apply in transform_predictions
    Lower prediction scores of items in history by 10^10.
    """
    for user_ids, X, y_true, y_pred in batch_iterator:
        yield user_ids, X, y_true, np.asarray(y_pred - 1e10 * X)
