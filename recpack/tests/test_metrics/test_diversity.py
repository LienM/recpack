# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.metrics.diversity import IntraListDiversityK
import numpy as np


def test_ildK(X_true, X_pred, item_features):
    K = 2

    metric = IntraListDiversityK(K)
    metric.fit(item_features)

    metric.calculate(X_true, X_pred)

    np.testing.assert_almost_equal(metric.value, 1 / 2)


def test_ildK_empty_reco(X_true_unrecommended_user, X_pred, item_features):
    K = 2

    metric = IntraListDiversityK(K)
    metric.fit(item_features)

    metric.calculate(X_true_unrecommended_user, X_pred)

    np.testing.assert_almost_equal(metric.value, 1 / 3)
