# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import pytest
import sklearn
from recpack.algorithms import NMF, SVD


def test_nmf(X_in):
    a = NMF(2)
    a.fit(X_in)

    assert a.user_embedding_ is not None
    assert a.item_embedding_ is not None

    prediction = a.predict(X_in)
    assert prediction.shape == X_in.shape

    # Reconstruction expected, so items with 1 should have high scores.
    assert prediction[0, 0] > prediction[0, 1]
    assert prediction[0, 2] > prediction[0, 1]
    assert prediction[0, 3] > prediction[0, 1]

    assert prediction[2, 1] > prediction[2, 0]
    assert prediction[2, 1] > prediction[2, 2]


def test_nmf_predict_no_fit(X_in):
    a = NMF(2)

    with pytest.raises(sklearn.exceptions.NotFittedError):
        a.predict(X_in[2])


def test_svd(X_in):
    a = SVD(2)
    a.fit(X_in)

    assert a.user_embedding_ is not None
    assert a.item_embedding_ is not None

    prediction = a.predict(X_in)
    assert prediction.shape == X_in.shape

    # Reconstruction expected, so items with 1 should have high scores.
    assert prediction[0, 0] > prediction[0, 1]
    assert prediction[0, 2] > prediction[0, 1]
    assert prediction[0, 3] > prediction[0, 1]

    assert prediction[2, 1] > prediction[2, 0]
    assert prediction[2, 1] > prediction[2, 2]


def test_svd_predict_no_fit(X_in):
    a = SVD(2)

    with pytest.raises(sklearn.exceptions.NotFittedError):
        a.predict(X_in[2])
