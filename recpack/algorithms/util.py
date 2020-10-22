import logging
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix

import torch


logger = logging.getLogger("recpack")


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


def naive_sparse2tensor(data: csr_matrix):
    return torch.FloatTensor(data.toarray())


def naive_tensor2sparse(tensor: torch.Tensor):
    return csr_matrix(tensor.detach().numpy())


class StoppingCriterion:
    """
    StoppingCriterion provides a wrapper around any loss function
    """

    # TODO Add documentation
    def __init__(
        self,
        loss_function: Callable,
        minimize: bool = False,
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        tol: float = 0.01,
    ):
        self.best_value = np.inf if minimize else -np.inf
        self.loss_function = loss_function
        self.minimize = minimize
        self.stop_early = stop_early
        self.max_iter_no_change = max_iter_no_change  # In scikit-learn this is n_iter_no_change but I find that misleading
        self.n_iter_no_change = 0
        self.tol = tol  # min_change seems like a more appropriate name

    def update(self, *args, **kwargs):
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
    pass
