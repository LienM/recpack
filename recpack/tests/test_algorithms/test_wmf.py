# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest
import numpy as np
from scipy.sparse import csr_matrix

from recpack.algorithms import WeightedMatrixFactorization
from recpack.matrix import to_binary


@pytest.mark.parametrize("num_users", [4, 5, 10])  # , 4, 5, 10])
def test_wmf_different_shapes(num_users):
    # Regression Test: Used to throw an error with some sizes

    wmf = WeightedMatrixFactorization(confidence_scheme="minimal", num_components=3, iterations=100)

    values = np.random.randint(0, 2, size=20)
    users = np.random.randint(0, num_users, size=20)
    items = np.random.randint(0, 3, size=20)
    test_matrix = csr_matrix((values, (users, items)))

    test_matrix = to_binary(test_matrix)

    # Test if the internal factor matrices are correctly fitted
    wmf.fit(test_matrix)


@pytest.mark.parametrize("cs", ["log-scaling", "minimal"])
def test_wmf(cs):
    wmf = WeightedMatrixFactorization(confidence_scheme=cs, num_components=3, iterations=200)

    values = [2, 5, 4, 1, 3, 4, 3]
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    test_matrix = csr_matrix((values, (users, items)))

    # Test if the internal factor matrices are correctly fitted
    wmf.fit(test_matrix)

    should_converge_to = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]

    u = wmf.user_factors_.detach().cpu().numpy()
    i = wmf.item_factors_.detach().cpu().numpy()

    dotproduct = u.dot(i.T)
    np.testing.assert_almost_equal(dotproduct, should_converge_to, decimal=1)

    # Test the prediction
    values_pred = [1, 2, 3]
    users_pred = [1, 0, 1]
    items_pred = [0, 0, 1]
    pred_matrix = csr_matrix((values_pred, (users_pred, items_pred)), shape=test_matrix.shape)
    prediction = wmf.predict(pred_matrix)

    exp_values = [1, 0, 0, 1, 1, 0]
    exp_users = [0, 0, 0, 1, 1, 1]
    exp_items = [0, 1, 2, 0, 1, 2]
    expected_prediction = csr_matrix((exp_values, (exp_users, exp_items)), shape=test_matrix.shape)
    np.testing.assert_almost_equal(prediction.toarray(), expected_prediction.toarray(), decimal=1)


def test_wmf_invalid_confidence_scheme():
    with pytest.raises(ValueError):
        _ = WeightedMatrixFactorization(confidence_scheme="Nonsense")
