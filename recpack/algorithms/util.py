import logging
from typing import Callable
from math import ceil


import numpy as np
from scipy.sparse import csr_matrix

import torch


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


class StoppingCriterion:
    def __init__(
        self,
        loss_function: Callable,
        minimize: bool = False,
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        tol: float = 0.01,
    ):
        """
        StoppingCriterion provides a wrapper around any loss function
        used in the validation stage of an iterative algorithm.
        A loss function can be maximized or minimized.
        If stop_early is true, then an EarlyStoppingException
        is raised when there were at least {max_iter_no_change}
        iterations with no improvements greater than {tol}.

        :param loss_function: Metric function used in validation
        :type loss_function: Callable
        :param minimize: True if smaller values of loss_function are better, defaults to False
        :type minimize: bool, optional
        :param stop_early: Use early stopping to halt learning when overfitting, defaults to False
        :type stop_early: bool, optional
        :param max_iter_no_change: Amount of iterations with no improvements greater than {tol} before we stop early, defaults to 5
        :type max_iter_no_change: int, optional
        :param tol: Improvements smaller than {tol} are not counted as actual improvements, defaults to 0.01
        :type tol: float, optional
        """
        self.best_value = np.inf if minimize else -np.inf
        self.loss_function = loss_function
        self.minimize = minimize
        self.stop_early = stop_early
        self.max_iter_no_change = max_iter_no_change  # In scikit-learn this is n_iter_no_change but I find that misleading
        self.n_iter_no_change = 0
        self.tol = tol  # min_change seems like a more appropriate name

    def update(self, *args, **kwargs) -> bool:
        """
        Update StoppingCriterion value.
        All args and kwargs are passed to the loss function
        that is maximized/minimized.

        :raises EarlyStoppingException: When early stopping condition is met, as configured in __init__.
        :return: True if value is better than the previous best value, False if not.
        :rtype: bool
        """
        loss = self.loss_function(*args, **kwargs)

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
                min_change_made = abs(loss - self.best_value) > self.tol
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


class EarlyStoppingException(Exception):
    """
    Raised when Early Stopping condition is met.
    """
    pass
