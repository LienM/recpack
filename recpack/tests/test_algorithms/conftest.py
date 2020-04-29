import scipy.sparse as sp
import pandas as pd
import pytest


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
        "item_id": [0,1,2,3,4],
    }

    return pd.DataFrame.from_dict(data_dict)
    