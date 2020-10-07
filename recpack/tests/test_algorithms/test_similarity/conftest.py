import scipy.sparse as sp
import pytest


@pytest.fixture(scope="function")
def pageviews_for_pairwise():
    pv_users, pv_items, pv_values = (
        [0, 0, 0, 1, 1, 1, 3, 3, 4, 4],
        [0, 1, 2, 0, 1, 2, 3, 4, 3, 4],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )

    pv = sp.csr_matrix((pv_values, (pv_users, pv_items)), shape=(10, 5))

    return pv
