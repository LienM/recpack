import numpy
import pytest

from scipy.sparse import csr_matrix
from torch import Tensor

from recpack.algorithms.util import (
    naive_sparse2tensor,
    naive_tensor2sparse,
)
from recpack.algorithms.stopping_criterion import StoppingCriterion, EarlyStoppingException
from recpack.metrics.recall import recall_k
from recpack.metrics.dcg import ndcg_k


def loss_function(l):
    i = iter(l)

    def inner(X_true, X_pred):
        return next(i)

    return inner


def test_stopping_criterion_raise():
    l = [1] * 6

    crit = StoppingCriterion(
        loss_function(l),
        minimize=False,
        stop_early=True,
        max_iter_no_change=3,
        min_improvement=0.01,
    )
    # Setting X_pred and X_true to None,
    # because they are not needed in these tests to be actual values
    X_pred = None
    X_true = None
    # First update successful
    crit.update(X_true, X_pred)
    # No change
    crit.update(X_true, X_pred)
    # No change
    crit.update(X_true, X_pred)

    with pytest.raises(EarlyStoppingException):
        crit.update(X_true, X_pred)


def test_stopping_criterion_raise2():
    l = [1, 1, 2, 2, 2, 2]

    crit = StoppingCriterion(
        loss_function(l),
        minimize=True,
        stop_early=True,
        max_iter_no_change=3,
        min_improvement=0.01,
    )

    # Setting X_pred and X_true to None,
    # because they are not needed in these tests to be actual values
    X_pred = None
    X_true = None

    # First update successful
    crit.update(X_true, X_pred)
    # No change
    crit.update(X_true, X_pred)
    # Bad change
    crit.update(X_true, X_pred)

    with pytest.raises(EarlyStoppingException):
        crit.update(X_true, X_pred)


def test_stopping_criterion_improves_maximize():
    l = [1, 2, 3]

    crit = StoppingCriterion(
        loss_function(l),
        minimize=False,
        stop_early=True,
        max_iter_no_change=3,
        min_improvement=0.01,
    )

    # Setting X_pred and X_true to None,
    # because they are not needed in these tests to be actual values
    X_pred = None
    X_true = None

    # First update successful
    crit.update(X_true, X_pred)
    assert crit.best_value == l[0]
    # Second update successful
    crit.update(X_true, X_pred)
    assert crit.best_value == l[1]
    # Third update successful
    crit.update(X_true, X_pred)
    assert crit.best_value == l[2]


def test_stopping_criterion_improves_minimize():
    l = [3, 2, 1]

    crit = StoppingCriterion(
        loss_function(l),
        minimize=True,
        stop_early=True,
        max_iter_no_change=3,
        min_improvement=0.01,
    )

    # Setting X_pred and X_true to None,
    # because they are not needed in these tests to be actual values
    X_pred = None
    X_true = None

    # First update successful
    crit.update(X_true, X_pred)
    assert crit.best_value == l[0]
    # Second update successful
    crit.update(X_true, X_pred)
    assert crit.best_value == l[1]
    # Third update successful
    crit.update(X_true, X_pred)
    assert crit.best_value == l[2]


@pytest.mark.parametrize(
    "criterion, expected_function", [("recall", recall_k), ("ndcg", ndcg_k)]
)
def test_stopping_criterion_create(criterion, expected_function):
    c = StoppingCriterion.create(criterion)

    assert c.loss_function == expected_function


def test_kwargs_criterion_create():
    c = StoppingCriterion.create("recall")

    assert "k" in c.kwargs


def test_csr_tensor_conversions(larger_matrix):
    assert isinstance(larger_matrix, csr_matrix)
    tensor = naive_sparse2tensor(larger_matrix)
    assert isinstance(tensor, Tensor)

    csr_again = naive_tensor2sparse(tensor)

    assert isinstance(csr_again, csr_matrix)

