import logging
from typing import Callable
from math import ceil


import numpy as np
from scipy.sparse import csr_matrix

import torch

from recpack.algorithms.loss_functions import bpr_loss_metric
from recpack.metrics.dcg import ndcg_k
from recpack.metrics.recall import recall_k
from recpack.metrics.base import MetricTopK

logger = logging.getLogger("recpack")


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


def naive_sparse2tensor(data: csr_matrix) -> torch.Tensor:
    """
    Naively converts sparse csr_matrix to torch Tensor.

    :param data: CSR matrix to convert
    :type data: csr_matrix
    :return: Torch Tensor representation of the matrix.
    :rtype: torch.Tensor
    """
    return torch.FloatTensor(data.toarray())


def naive_tensor2sparse(tensor: torch.Tensor) -> csr_matrix:
    """
    Converts torch Tensor to sparse csr_matrix.

    :param tensor: Torch Tensor representation of the matrix to convert.
    :type tensor: torch.Tensor
    :return: CSR matrix representation of the matrix.
    :rtype: csr_matrix
    """
    return csr_matrix(tensor.detach().numpy())


def get_users(data):
    return list(set(data.nonzero()[0]))


def get_batches(users, batch_size=1000):
    return [
        users[i * batch_size: min((i * batch_size) + batch_size, len(users))]
        for i in range(ceil(len(users) / batch_size))
    ]


def get_knn(data: csr_matrix, k: int) -> csr_matrix:
    """
    Return csr_matrix of top K items for every user.
    @param data: Predicted affinity of users for items.
    @param k: Value for K; k could be None.
    @return: Sparse matrix containing values of top K predictions.
    """
    metric = MetricTopK(k)
    top_K_ranks = metric.get_top_K_ranks(data)
    top_K_ranks[top_K_ranks > 0] = 1  # ranks to binary

    return top_K_ranks.multiply(data)  # elementwise multiplication


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
        """
        StoppingCriterion provides a wrapper around any loss function
        used in the validation stage of an iterative algorithm.
        A loss function can be maximized or minimized.
        If stop_early is true, then an EarlyStoppingException
        is raised when there were at least {max_iter_no_change}
        iterations with no improvements greater than {min_improvement}.

        :param loss_function: Metric function used in validation,
                                required interface of func(X_true, X_pred)
        :type loss_function: Callable
        :param minimize: True if smaller values of loss_function are better, defaults to False
        :type minimize: bool, optional
        :param stop_early: Use early stopping to halt learning when overfitting, defaults to False
        :type stop_early: bool, optional
        :param max_iter_no_change: Amount of iterations with no improvements greater than {min_improvement} before we stop early, defaults to 5
        :type max_iter_no_change: int, optional
        :param min_improvement: Improvements smaller than {min_improvement} are not counted as actual improvements, defaults to 0.01
        :type min_improvement: float, optional
        :param kwargs: The keyword arguments to be passed to the loss function
        :type kwargs: dict, optional
        """
        self.best_value = np.inf if minimize else -np.inf
        self.loss_function = loss_function
        self.minimize = minimize
        self.stop_early = stop_early
        self.max_iter_no_change = max_iter_no_change  # In scikit-learn this is n_iter_no_change but I find that misleading
        self.n_iter_no_change = 0
        self.min_improvement = min_improvement

        self.kwargs = kwargs

    def update(self, X_true: csr_matrix, X_pred: csr_matrix) -> bool:
        """
        Update StoppingCriterion value.
        All args and kwargs are passed to the loss function
        that is maximized/minimized.

        :raises EarlyStoppingException: When early stopping condition is met, as configured in __init__.
        :return: True if value is better than the previous best value, False if not.
        :rtype: bool
        """
        loss = self.loss_function(X_true, X_pred, **self.kwargs)

        if self.minimize:
            # If we try to minimize, smaller values of loss are better.
            better = loss < self.best_value
        else:
            # If we try to maximize, larger values of loss are better.
            better = loss > self.best_value

        if self.stop_early:
            if not better:
                # Decrease in performance also counts as no change.
                self.n_iter_no_change += 1
            else:
                min_change_made = abs(loss - self.best_value) > self.min_improvement
                if not min_change_made:
                    self.n_iter_no_change += 1

        logger.info(
            f"StoppingCriterion has value {loss}, which is {'better' if better else 'worse'} than previous iterations."
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
    }

    @classmethod
    def create(cls, criterion_name, **kwargs):

        if criterion_name not in cls.FUNCTIONS:
            raise RuntimeError(f"stopping criterion {criterion_name} not supported")

        return StoppingCriterion(**cls.FUNCTIONS[criterion_name], **kwargs)


class EarlyStoppingException(Exception):
    """
    Raised when Early Stopping condition is met.
    """

    pass
