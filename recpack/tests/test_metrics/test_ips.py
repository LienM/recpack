# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from recpack.metrics.ips import IPSMetric, IPSHitRateK
import numpy as np


def test_IPSMetric(X_true):
    metric = IPSMetric()
    metric.fit(X_true)

    assert metric.inverse_propensities.shape == (1, X_true.shape[1])

    # Item 0 has been seen twice, 1,2,3 have been seen once, and item 5 never
    assert metric.inverse_propensities[0, 0] == 5 / 2

    np.testing.assert_array_almost_equal(
        metric.inverse_propensities[0, 1:4], 5)
    assert metric.inverse_propensities[0, 4] == 0


def test_IPSMetric_no_reco(X_true_unrecommended_user):
    metric = IPSMetric()
    metric.fit(X_true_unrecommended_user)

    assert metric.inverse_propensities.shape == (
        1, X_true_unrecommended_user.shape[1])

    # Item 0 has been seen twice, 1,2,3 have been seen once, and item 5 never
    assert metric.inverse_propensities[0, 0] == 6 / 2
    assert metric.inverse_propensities[0, 1] == 6 / 2

    np.testing.assert_array_almost_equal(
        metric.inverse_propensities[0, 2:4], 6)
    assert metric.inverse_propensities[0, 4] == 0


def test_IPSMetric_ip_cap(X_lots_of_items):
    metric = IPSMetric()
    metric.fit(X_lots_of_items)

    assert metric.inverse_propensities.shape == (1, X_lots_of_items.shape[1])

    np.testing.assert_array_equal(metric.inverse_propensities, metric.ip_cap)


def test_IPSHitRate(X_true, X_pred):
    K = 2
    metric = IPSHitRateK(K)

    metric.fit(X_true)

    metric.calculate(X_true, X_pred)

    assert metric.value == 12.5 / 2


def test_IPSHitRate_no_reco(X_true_unrecommended_user, X_pred):
    K = 2
    metric = IPSHitRateK(K)

    metric.fit(X_true_unrecommended_user)

    metric.calculate(X_true_unrecommended_user, X_pred)

    assert metric.value == 15 / 3

# TODO: reimplement SNIPS
# def test_SNIPSHitRate(X_true, X_pred):
#     K = 2
#     metric = SNIPSHitRateK(K)

#     metric.fit(X_true)

#     metric.calculate(X_true, X_pred)

#     assert metric.value == (1 + (2 / 3)) / 3
