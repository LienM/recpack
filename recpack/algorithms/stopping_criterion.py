import logging
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix

from recpack.algorithms.loss_functions import bpr_loss_metric, warp_loss_metric
from recpack.metrics.dcg import ndcg_k
from recpack.metrics.recall import recall_k

logger = logging.getLogger("recpack")

# TODO: Add to docs


class StoppingCriterion:
    def __init__(
        self,
        loss_function: Callable,
        minimize: bool = False,
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.01,
        **kwargs,
    ):
        """StoppingCriterion provides a wrapper around any loss function
        used in the validation stage of an iterative algorithm.

        A loss function can be maximized or minimized.
        If stop_early is true, then an EarlyStoppingException
        is raised when there were at least {max_iter_no_change}
        iterations with no improvements greater than {min_improvement}.

        :param loss_function: Metric function used in validation,
                                required interface of func(X_true, X_pred)
        :type loss_function: Callable
        :param minimize: True if smaller values of loss_function are better,
            defaults to False
        :type minimize: bool, optional
        :param stop_early: Use early stopping to halt learning when overfitting,
            defaults to False
        :type stop_early: bool, optional
        :param max_iter_no_change: Amount of iterations with no improvements greater
            than {min_improvement} before we stop early, defaults to 5
        :type max_iter_no_change: int, optional
        :param min_improvement: Improvements smaller than {min_improvement}
            are not counted as actual improvements, defaults to 0.01
        :type min_improvement: float, optional
        :param kwargs: The keyword arguments to be passed to the loss function
        :type kwargs: dict, optional
        """
        self.best_value = np.inf if minimize else -np.inf
        self.loss_function = loss_function
        self.minimize = minimize
        self.stop_early = stop_early
        # In scikit-learn this is n_iter_no_change but I find that misleading
        self.max_iter_no_change = max_iter_no_change
        self.n_iter_no_change = 0
        self.min_improvement = min_improvement

        self.kwargs = kwargs

    def update(self, X_true: csr_matrix, X_pred: csr_matrix) -> bool:
        """Update StoppingCriterion value.

        All args and kwargs are passed to the loss function
        that is maximized/minimized.

        :raises EarlyStoppingException: When early stopping condition is met,
            as configured in __init__.
        :return: True if value is better than the previous best value, False if not.
        :rtype: bool
        """
        loss = self.loss_function(X_true, X_pred, **self.kwargs)

        if self.minimize:
            # If we try to minimize, smaller values of loss are better.
            better = loss < self.best_value and (
                abs(loss - self.best_value) > self.min_improvement
            )
        else:
            # If we try to maximize, larger values of loss are better.
            better = loss > self.best_value and (
                abs(loss - self.best_value) > self.min_improvement
            )

        if self.stop_early and not better:
            # Decrease in performance also counts as no change.
            self.n_iter_no_change += 1

        logger.info(
            f"StoppingCriterion has value {loss}, which is "
            f"{'better' if better else 'worse'} than previous iterations."
        )

        if better:
            # Reset iterations w/o change.
            self.n_iter_no_change = 0
            self.best_value = loss
            return True
        else:
            if self.n_iter_no_change >= self.max_iter_no_change:
                raise EarlyStoppingException(
                    f"No improvements in the last {self.n_iter_no_change} iterations."
                )

        return False

    FUNCTIONS = {
        "bpr": {
            "loss_function": bpr_loss_metric,
            "minimize": True,
            "batch_size": 1000,
        },
        "recall": {"loss_function": recall_k, "minimize": False, "k": 50},
        "ndcg": {"loss_function": ndcg_k, "minimize": False, "k": 50},
        "warp": {"loss_function": warp_loss_metric, "minimize": True},
    }

    @classmethod
    def create(cls, criterion_name: str, **kwargs):
        """Construct a StoppingCriterion instance,
            based on the name of the loss function.

        BPR and WARP loss will minimize a loss function,
        Recall and NDCG will optimise a ranking metric,
            with default @K=50

        keyword arguments of the criteria can be set by passing
        them as kwarg to this create function.
        # TODO add example

        :param criterion_name: Name of the criterion to use,
            one of ["bpr", "warp", "recall", "ndcg"]
        :type criterion_name: str
        :param kwargs: Keyword arguments to pass to the criterion when calling it,
            useful to pass hyperparameters to the loss functions.
        :raises ValueError: If the requested criterion
            is not one of tha allowed options.
        :return: The constructed stopping criterion
        :rtype: [type]
        """

        # TODO: In other similar checks we raise a Value error, should be unified.
        if criterion_name not in cls.FUNCTIONS:
            raise ValueError(f"stopping criterion {criterion_name} not supported")

        return StoppingCriterion(**cls.FUNCTIONS[criterion_name], **kwargs)


class EarlyStoppingException(Exception):
    """Raised when Early Stopping condition is met."""

    pass
