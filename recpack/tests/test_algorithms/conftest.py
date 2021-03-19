import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


from recpack.data.matrix import InteractionMatrix


@pytest.fixture(scope="function")
def pageviews():
    pv_users, pv_items, pv_values = (
        [0, 0, 0, 2, 2, 2],
        [0, 2, 3, 1, 3, 4],
        [1, 2, 1, 1, 1, 1],
    )

    pv = sp.csr_matrix((pv_values, (pv_users, pv_items)), shape=(10, 5))

    return pv


@pytest.fixture(scope="function")
def pageviews_for_pairwise():
    pv_users, pv_items, pv_values = (
        [0, 0, 0, 1, 1, 1, 3, 3, 4, 4],
        [0, 1, 2, 0, 1, 2, 3, 4, 3, 4],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )

    pv = sp.csr_matrix((pv_values, (pv_users, pv_items)), shape=(10, 5))

    return pv


@pytest.fixture(scope="function")
def pageviews_interaction_m(pageviews):
    return InteractionMatrix.from_csr_matrix(pageviews)


@pytest.fixture(scope="function")
def purchases():
    pur_users, pur_items, pur_values = (
        [0, 2],
        [0, 4],
        [1, 1],
    )

    pur = sp.csr_matrix((pur_values, (pur_users, pur_items)), shape=(10, 5))

    return pur


@pytest.fixture(scope="function")
def labels():
    labels = sp.csr_matrix(([1], ([0], [0])), shape=(1, 5))

    return labels


@pytest.fixture(scope="function")
def labels_more_durable_items():
    labels = sp.csr_matrix(([1, 1], ([0, 0], [0, 3])), shape=(1, 5))

    return labels


@pytest.fixture(scope="function")
def metadata():
    data_dict = {
        "title": ["item_1", "Rosemary", "", "item_2", "Pepper"],
        "item_id": [0, 1, 2, 3, 4],
    }

    return pd.DataFrame.from_dict(data_dict)


@pytest.fixture(scope="function")
def metadata_tags_matrix():
    """constructs a matrix, with 3 tags: sport, news and celeb
    encoded as columns 1, 2 and 3 in the matrix"""
    items, tags, values = (
        [0, 0, 1, 2, 3, 4],
        [0, 1, 1, 0, 2, 1],
        [1, 1, 1, 1, 1, 1],
    )

    mat = sp.csr_matrix((values, (items, tags)), shape=(5, 3))
    return mat


@pytest.fixture(scope="function")
def larger_matrix():
    num_interactions = 2000
    num_users = 500
    num_items = 500

    np.random.seed(400)

    pv_users, pv_items, pv_values = (
        [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        [1] * num_interactions,
    )

    pv = sp.csr_matrix(
        (pv_values, (pv_users, pv_items)), shape=(num_users + 200, num_items)
    )

    return pv


@pytest.fixture(scope="function")
def data():
    pred_users, pred_items, pred_values = (
        [0, 0, 0, 2, 2, 2],
        [0, 2, 3, 1, 3, 4],
        [0.3, 0.2, 0.1, 0.23, 0.3, 0.5],
    )

    pred = sp.csr_matrix((pred_values, (pred_users, pred_items)), shape=(10, 5))

    return pred
