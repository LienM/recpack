# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy
import pytest
from scipy.sparse import csr_matrix

from recpack.metrics import PercentileRanking


def test_perc_ranking():
    values = [1, 1, 1, 1]
    users = [0, 0, 0, 0]
    items = [0, 2, 3, 7]
    y_true = csr_matrix((values, (users, items)), shape=(2, 10))

    values_pred = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    users_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    items_pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_pred = csr_matrix((values_pred, (users_pred, items_pred)), shape=(1, 10))

    pr = PercentileRanking()
    pr.calculate(y_true, y_pred)

    manual_numerator = 0 + 20 + 30 + 70  # %
    manual_denominator = 1 + 1 + 1 + 1

    numpy.testing.assert_almost_equal(pr.value, manual_numerator / manual_denominator)


@pytest.mark.parametrize("seed", list(range(5)))
def test_perc_ranking_sparse(seed):
    numpy.random.seed(seed)
    N = 10000
    num_users = 500
    num_items = 50
    # fmt:off
    y_true = csr_matrix(
        (
            [1 for i in range(N)],
            (
                [numpy.random.randint(0, num_users) for _ in range(N)],
                [numpy.random.randint(0, num_items) for _ in range(N)],
            )
        )
    )

    y_true[y_true > 0] = 1

    y_pred = csr_matrix(
        (
            [numpy.random.rand() for i in range(N)],
            (
                [numpy.random.randint(0, num_users) for _ in range(N)],
                [numpy.random.randint(0, num_items) for _ in range(N)],
            )
        )
    )
    # fmt:on
    num_users, num_items = y_pred.shape
    # Compute ranking per user_item pair:
    ranks = {}
    for u in range(num_users):
        ranks[u] = {}
        for ix, i in enumerate(numpy.flip(numpy.argsort(y_pred[u, :].toarray()[0]))):
            ranks[u][i] = ix

    numerator = 0
    denominator = 0
    for u, i in zip(*y_true.nonzero()):
        numerator += ranks[u][i] * 100 / num_items
        denominator += 1

    m = PercentileRanking()
    m.calculate(y_true, y_pred)

    # In the random situation these values are almost equal,
    # but can diverge a bit when there are fewer users
    numpy.testing.assert_almost_equal(m.value, numerator / denominator, 0)
