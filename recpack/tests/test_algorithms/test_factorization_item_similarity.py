# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
from recpack.algorithms import NMFItemToItem, SVDItemToItem


def test_nmf_item_to_item(X_in):
    a = NMFItemToItem(2)

    a.fit(X_in)
    assert a.similarity_matrix_ is not None
    np.testing.assert_array_equal(a.similarity_matrix_.diagonal(), 0)
    n_items = X_in.shape[1]
    assert a.similarity_matrix_.shape == (n_items, n_items)

    prediction = a.predict(X_in[2])
    assert prediction.shape == (1, n_items)
    assert prediction.nonzero() != []


def test_svd_item_to_item(X_in):
    a = SVDItemToItem(2)

    a.fit(X_in)
    assert a.similarity_matrix_ is not None

    n_items = X_in.shape[1]
    assert a.similarity_matrix_.shape == (n_items, n_items)
    np.testing.assert_array_equal(a.similarity_matrix_.diagonal(), 0)

    prediction = a.predict(X_in[2])
    assert prediction.shape == (1, n_items)
    assert prediction.nonzero() != []
