import pytest
import numpy
from scipy.sparse import csr_matrix

from recpack.algorithms import WeightedMatrixFactorization


@pytest.mark.parametrize("cs", ["log-scaling", "minimal"])
def test_wmf(cs):
    wmf = WeightedMatrixFactorization(
        confidence_scheme=cs, num_components=10, iterations=500
    )

    values = [2, 5, 4, 1, 3, 4, 3]
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    test_matrix = csr_matrix((values, (users, items)))

    # Test if the internal factor matrices are correctly fitted
    wmf.fit(test_matrix)

    should_converge_to = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    dotproduct = wmf.user_factors_.dot(wmf.item_factors_.T)
    numpy.testing.assert_almost_equal(dotproduct, should_converge_to, decimal=1)

    # Test the prediction
    values_pred = [1, 2, 3]
    users_pred = [1, 0, 1]
    items_pred = [0, 0, 1]
    pred_matrix = csr_matrix(
        (values_pred, (users_pred, items_pred)), shape=test_matrix.shape
    )
    prediction = wmf.predict(pred_matrix)

    exp_values = [1, 0, 0, 1, 1, 0]
    exp_users = [0, 0, 0, 1, 1, 1]
    exp_items = [0, 1, 2, 0, 1, 2]
    expected_prediction = csr_matrix(
        (exp_values, (exp_users, exp_items)), shape=test_matrix.shape
    )
    numpy.testing.assert_almost_equal(
        prediction.toarray(), expected_prediction.toarray(), decimal=1
    )


def test_linear_equation():
    Y = numpy.array([[1, 0], [5, 5], [0, 1]])
    YtY = Y.T.dot(Y)

    values = [2, 5, 4, 1, 3, 4, 3]
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    test_matrix = csr_matrix((values, (users, items)))

    wmf = WeightedMatrixFactorization(num_components=2, regularization=0.1)
    result = wmf._linear_equation(Y, YtY, test_matrix, 0)

    numpy.testing.assert_almost_equal(result, numpy.array([-0.66, 0.86]), decimal=2)


def test_wmf_invalid_confidence_scheme():
    with pytest.raises(ValueError):
        _ = WeightedMatrixFactorization(confidence_scheme="Nonsense")
