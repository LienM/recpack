import numpy
from scipy.sparse import csr_matrix
from recpack.algorithms import WeightedMatrixFactorizationAlgorithm


def test_wmf():
    wmf = WeightedMatrixFactorizationAlgorithm(cs='log-scaling', factors=10, iterations=10000)

    values = [2, 5, 4, 1, 3, 4, 3]
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    test_matrix = csr_matrix((values, (users, items)))

    # Test if the internal factor matrices are correctly fitted
    wmf.fit(test_matrix)

    should_converge_to = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    dotproduct = wmf.user_factors_.dot(wmf.item_factors_.T)
    numpy.testing.assert_almost_equal(dotproduct, should_converge_to, decimal=2)

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
    numpy.testing.assert_almost_equal(prediction.toarray(), expected_prediction.toarray(), decimal=2)
