import pytest

from recpack.algorithms.util import StoppingCriterion, EarlyStoppingException


def loss_function(l):
    i = iter(l)

    def inner():
        return next(i)

    return inner


def test_stopping_criterion_raise():
    l = [1] * 6

    crit = StoppingCriterion(
        loss_function(l),
        minimize=False,
        stop_early=True,
        max_iter_no_change=3,
        tol=0.01,
    )

    # First update successful
    crit.update()
    # No change
    crit.update()
    # No change
    crit.update()

    with pytest.raises(EarlyStoppingException):
        crit.update()


def test_stopping_criterion_raise2():
    l = [1, 1, 2, 2, 2, 2]

    crit = StoppingCriterion(
        loss_function(l),
        minimize=True,
        stop_early=True,
        max_iter_no_change=3,
        tol=0.01,
    )

    # First update successful
    crit.update()
    # No change
    crit.update()
    # Bad change
    crit.update()

    with pytest.raises(EarlyStoppingException):
        crit.update()


def test_stopping_criterion_improves_maximize():
    l = [1, 2, 3]

    crit = StoppingCriterion(
        loss_function(l),
        minimize=False,
        stop_early=True,
        max_iter_no_change=3,
        tol=0.01,
    )

    # First update successful
    crit.update()
    assert crit.best_value == l[0]
    # Second update successful
    crit.update()
    assert crit.best_value == l[1]
    # Third update successful
    crit.update()
    assert crit.best_value == l[2]


def test_stopping_criterion_improves_minimize():
    l = [3, 2, 1]

    crit = StoppingCriterion(
        loss_function(l),
        minimize=True,
        stop_early=True,
        max_iter_no_change=3,
        tol=0.01,
    )

    # First update successful
    crit.update()
    assert crit.best_value == l[0]
    # Second update successful
    crit.update()
    assert crit.best_value == l[1]
    # Third update successful
    crit.update()
    assert crit.best_value == l[2]
