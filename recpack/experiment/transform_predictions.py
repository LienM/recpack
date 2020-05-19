



def NoRepeats(user_iterator):
    """
    Example business rule to apply in transform_predictions
    Lower prediction scores of items in history by 10^10.
    """
    for user_id, y_hist_u, y_true_u, y_pred_u in user_iterator:
        yield user_id, y_hist_u, y_true_u, y_pred_u - 1e10 * y_hist_u