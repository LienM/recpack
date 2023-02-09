# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy

from recpack.metrics.reciprocal_rank import ReciprocalRankK


def test_rrk(X_pred, X_true):
    K = 2
    metric = ReciprocalRankK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 0.75)


def test_rrk_no_reco(X_pred, X_true_unrecommended_user):
    K = 2
    metric = ReciprocalRankK(K)

    metric.calculate(X_true_unrecommended_user, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 0.5)
