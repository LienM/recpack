import numpy as np
from recpack.algorithms import NMFItemToItem, SVDItemToItem


def test_nmf_item_to_item(pageviews):
    a = NMFItemToItem(2)

    a.fit(pageviews)
    assert a.similarity_matrix_ is not None
    np.testing.assert_array_equal(a.similarity_matrix_.diagonal(), 0)
    n_items = pageviews.shape[1]
    assert a.similarity_matrix_.shape == (n_items, n_items)

    prediction = a.predict(pageviews[2])
    assert prediction.shape == (1, n_items)
    assert prediction.nonzero() != []


def test_svd_item_to_item(pageviews):
    a = SVDItemToItem(2)

    a.fit(pageviews)
    assert a.similarity_matrix_ is not None

    n_items = pageviews.shape[1]
    assert a.similarity_matrix_.shape == (n_items, n_items)
    np.testing.assert_array_equal(a.similarity_matrix_.diagonal(), 0)

    prediction = a.predict(pageviews[2])
    assert prediction.shape == (1, n_items)
    assert prediction.nonzero() != []
