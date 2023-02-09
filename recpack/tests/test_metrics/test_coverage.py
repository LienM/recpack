# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
from recpack.metrics.coverage import CoverageK


def test_coverageK(X_pred, X_true):
    K = 2
    metric = CoverageK(K)

    metric.calculate(X_true, X_pred)

    # user 0 gets recommended items 0 and 2
    # user 2 gets recommended items 3 and 4
    # total number of items = 5
    np.testing.assert_almost_equal(metric.value, 4 / 5)

    assert metric.results.shape[0] == 1


def test_coverageK_empty_reco(X_pred, X_true_unrecommended_user):
    K = 2
    metric = CoverageK(K)

    metric.calculate(X_true_unrecommended_user, X_pred)

    # user 0 gets recommended items 0 and 2
    # user 2 gets recommended items 3 and 4
    # total number of items = 5
    np.testing.assert_almost_equal(metric.value, 4 / 5)

    assert metric.results.shape[0] == 1