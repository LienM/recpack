import scipy.sparse

import pytest


@pytest.fixture(scope="function")
def data():
    # TODO move this test input to a conftest file as a fixture
    input_dict = {
        "userId": [2, 2, 2, 0, 0, 0],
        "movieId": [1, 3, 4, 0, 2, 4],
        "values": [1, 2, 1, 1, 1, 2],
    }

    matrix = scipy.sparse.csr_matrix(
        (input_dict["values"], (input_dict["userId"], input_dict["movieId"])),
        shape=(10, 5)
    )
    return matrix


@pytest.fixture(scope="function")
def X_pred():
    pred_users, pred_items, pred_values = (
        [0, 0, 0, 2, 2, 2],
        [0, 2, 3, 1, 3, 4],
        [0.3, 0.2, 0.1, 0.23, 0.3, 0.5],
    )

    pred = scipy.sparse.csr_matrix(
        (pred_values, (pred_users, pred_items)), shape=(10, 5)
    )

    return pred


@pytest.fixture(scope="function")
def X_true():
    true_users, true_items = [0, 0, 2, 2, 2], [0, 2, 0, 1, 3]

    true_data = scipy.sparse.csr_matrix(
        ([1 for i in range(len(true_users))], (true_users, true_items)), shape=(10, 5)
    )

    return true_data


@pytest.fixture(scope="function")
def X_true_simplified():
    true_users, true_items = [0, 2], [2, 4]

    true_data = scipy.sparse.csr_matrix(
        ([1 for i in range(len(true_users))], (true_users, true_items)), shape=(10, 5)
    )

    return true_data


@pytest.fixture(scope="function")
def item_features():
    items, features = [0, 1, 2, 3, 4], [0, 1, 0, 1, 0]

    feature_mat = scipy.sparse.csr_matrix(
        (
            [1 for i in range(len(items))], 
            (items, features)
        ),
        shape=(5, 2)
    )
    return feature_mat