# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix

from recpack.algorithms.loss_functions import bpr_loss_wrapper, warp_loss_wrapper
from recpack.metrics.dcg import ndcg_k
from recpack.metrics.recall import recall_k
from recpack.metrics.precision import precision_k

logger = logging.getLogger("recpack")


class StoppingCriterion:
    """StoppingCriterion provides a wrapper around any loss function
    used in the validation stage of an iterative algorithm.

    A loss function can be maximized or minimized.
    If stop_early is true, then an EarlyStoppingException
    is raised when there were at least ``max_iter_no_change``
    iterations with no improvements greater than {min_improvement}.

    **Example**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms.stopping_criterion import StoppingCriterion
        X_true = csr_matrix(np.array([[1, 0, 1], [1, 1, 0], [1, 1, 0]]))
        X_pred = csr_matrix(np.array([[.9, .5, .2], [.3, .2, .05], [.1, .1, .9]]))

        # Creating a simple loss function
        # That computes the sum of the absolute error at each position in the matrix
        def my_loss(X_true, X_pred):
            error = np.abs(X_true - X_pred)
            return np.sum(error)

        # construct StoppingCriterion
        sc = StoppingCriterion(my_loss, minimize=True)

        # Update stopping criterion
        better = sc.update(X_true, X_pred)

        # Since this was the first update, it will be better than what was before
        assert better

        # Trying again with the same input, will give no improvement
        better = sc.update(X_true, X_pred)
        assert not better

        # Constructing a better prediction matrix
        # This would usually be done by a learning algorithm
        X_pred = csr_matrix(np.array([[.9, .3, .8], [.4, .5, .05], [.4, .4, .5]]))
        better = sc.update(X_true, X_pred)
        assert better

        # The best value for the loss function can also be retrieved
        print(sc.best_value)

    :param loss_function: Metric function used in validation,
                            function should take two positional arguments
                            ``X_true`` and ``X_pred``,
                            and should return a single float value,
                            which will be optimised.
    :type loss_function: Callable
    :param minimize: True if smaller values of loss_function are better,
        defaults to False.
    :type minimize: bool, optional
    :param stop_early: Use early stopping to halt learning when overfitting,
        defaults to False
    :type stop_early: bool, optional
    :param max_iter_no_change: Amount of iterations with no improvements greater
        than ``min_improvement`` before we stop early, defaults to 5
    :type max_iter_no_change: int, optional
    :param min_improvement: Improvements smaller than min_improvement
        are not counted as actual improvements, defaults to 0.01
    :type min_improvement: float, optional
    :param kwargs: The keyword arguments to be passed to the loss function
    :type kwargs: dict, optional
    """

    FUNCTIONS = {
        "bpr": {
            "loss_function": bpr_loss_wrapper,
            "minimize": True,
            "batch_size": 1000,
        },
        "recall": {"loss_function": recall_k, "minimize": False, "k": 50},
        "ndcg": {"loss_function": ndcg_k, "minimize": False, "k": 50},
        "warp": {"loss_function": warp_loss_wrapper, "minimize": True},
        "precision": {"loss_function": precision_k, "minimize": False},
    }
    """Available loss function options.

    These values can be passed to the ``create`` method,
    and will create the corresponding stopping criterion.

    Available loss functions:

    - bpr : Bayesian Personalised Ranking loss, will be minimized.
    - warp : Weighted Approximate-Rank Pairwise loss, will be minimized.
    - recall: Recall@k metric, will be maximized.
    - ndcg: Normalized Discounted Cumulative Gain, will be maximized.
    """

    def __init__(
        self,
        loss_function: Callable,
        minimize: bool = False,
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        **kwargs,
    ):

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
        """Update StoppingCriterion value based on
        expected and predicted interactions.

        The two matrices and kwargs set during init of StoppingCriterion are
        passed to the value function, the computed loss or metric is compared
        to the previous best value.
        If new value is better (bigger if ``minimize == False``, smaller otherwise),
        the new value is marked as best,
        and True is returned.

        Logs an info statement with the computed loss value,
        and if it was better.

        :raises EarlyStoppingException: When early stopping condition is met,
            if early stopping is enabled.
        :return: True if value is better than the previous best value, False if not.
        :rtype: bool
        """
        loss = self.loss_function(X_true, X_pred, **self.kwargs)

        if self.minimize:
            # If we try to minimize, smaller values of loss are better.
            better = loss <= self.best_value and (
                abs(loss - self.best_value) > self.min_improvement
            )
        else:
            # If we try to maximize, larger values of loss are better.
            better = loss >= self.best_value and (
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

    @classmethod
    def create(cls, criterion_name: str, **kwargs) -> "StoppingCriterion":
        """Construct a StoppingCriterion instance,
        based on the name of the loss function.

        BPR and WARP loss will minimize a loss function,
        Recall and NormalizedDiscountedCumulativeGain will optimise a ranking metric,
        with default @K=50

        keyword arguments of the criteria can be set by passing
        them as kwarg to this create function.

        **Example**::

            import numpy as np
            from scipy.sparse import csr_matrix
            from recpack.algorithms.stopping_criterion import StoppingCriterion
            X_true = csr_matrix(np.array([[1, 0, 1], [1, 1, 0], [1, 1, 0]]))
            X_pred = csr_matrix(np.array([[.9, .5, .2], [.3, .2, .05], [.1, .1, .9]]))

            # construct StoppingCriterion
            # setting k to 2, such that when ndcg function is called,
            # k=2 will be passed to it
            sc = StoppingCriterion.create('ndcg', k=2)

            # Update gives us if it was better or not
            better = sc.update(X_true, X_pred)


        :param criterion_name: Name of the criterion to use,
            one of ["bpr", "warp", "recall", "ndcg"]
        :type criterion_name: str
        :param kwargs: Keyword arguments to pass to the criterion when calling it,
            useful to pass hyperparameters to the loss functions.
        :raises ValueError: If the requested criterion
            is not one of tha allowed options.
        :return: The constructed stopping criterion
        :rtype: StoppingCriterion
        """

        if criterion_name not in cls.FUNCTIONS:
            raise ValueError(f"stopping criterion {criterion_name} not supported")

        # We will combine the two dicts, with kwargs getting precendence.
        return StoppingCriterion(**{**cls.FUNCTIONS[criterion_name], **kwargs})


class EarlyStoppingException(Exception):
    """Raised when Early Stopping condition is met."""

    pass
