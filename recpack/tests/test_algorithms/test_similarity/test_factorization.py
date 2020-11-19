import pytest
import sklearn
from recpack.algorithms import NMF, SVD, NMFItemToItem, SVDItemToItem


def test_nmf(pageviews):
    a = NMF(2)
    a.fit(pageviews)

    assert a.user_features_ is not None
    assert a.item_features_ is not None

    prediction = a.predict(pageviews)
    assert prediction.shape == pageviews.shape

    # Reconstruction expected, so items with 1 should have high scores.
    assert prediction[0, 0] > prediction[0, 1]
    assert prediction[0, 2] > prediction[0, 1]
    assert prediction[0, 3] > prediction[0, 1]

    assert prediction[2, 1] > prediction[2, 0]
    assert prediction[2, 1] > prediction[2, 2]


def test_nmf_item_to_item(pageviews):
    a = NMFItemToItem(2)

    a.fit(pageviews)
    assert a.similarity_matrix_ is not None

    n_items = pageviews.shape[1]
    assert a.similarity_matrix_.shape == (n_items, n_items)

    prediction = a.predict(pageviews[2])
    assert prediction.shape == (1, n_items)
    assert prediction.nonzero() != []


def test_nmf_predict_no_fit(pageviews):
    a = NMF(2)

    with pytest.raises(sklearn.exceptions.NotFittedError):
        a.predict(pageviews[2])


def test_svd(pageviews):
    a = SVD(2)
    a.fit(pageviews)

    assert a.user_features_ is not None
    assert a.item_features_ is not None

    prediction = a.predict(pageviews)
    assert prediction.shape == pageviews.shape

    # Reconstruction expected, so items with 1 should have high scores.
    assert prediction[0, 0] > prediction[0, 1]
    assert prediction[0, 2] > prediction[0, 1]
    assert prediction[0, 3] > prediction[0, 1]

    assert prediction[2, 1] > prediction[2, 0]
    assert prediction[2, 1] > prediction[2, 2]


def test_svd_item_to_item(pageviews):
    a = SVDItemToItem(2)

    a.fit(pageviews)
    assert a.similarity_matrix_ is not None

    n_items = pageviews.shape[1]
    assert a.similarity_matrix_.shape == (n_items, n_items)

    prediction = a.predict(pageviews[2])
    assert prediction.shape == (1, n_items)
    assert prediction.nonzero() != []


def test_svd_predict_no_fit(pageviews):
    a = SVD(2)

    with pytest.raises(sklearn.exceptions.NotFittedError):
        a.predict(pageviews[2])
