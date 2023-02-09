# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy

from recpack.metrics.recall import RecallK, CalibratedRecallK, recall_k, calibrated_recall_k


def test_recallK(X_pred, X_true):
    K = 2
    metric = RecallK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 2 / 3)

    K = 1
    metric = RecallK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 1 / 4)


def test_recallK_no_reco(X_pred, X_true_unrecommended_user):
    K = 2
    metric = RecallK(K)

    metric.calculate(X_true_unrecommended_user, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 4 / 9)

    K = 1
    metric = RecallK(K)

    metric.calculate(X_true_unrecommended_user, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 1 / 6)


def test_calibrated_recallK(X_pred, X_true):
    K = 2
    metric = CalibratedRecallK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 3 / 4)

    K = 1
    metric = CalibratedRecallK(K)

    metric.calculate(X_true, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 1 / 2)


def test_calibrated_recallK_no_reco(X_pred, X_true_unrecommended_user):
    K = 2
    metric = CalibratedRecallK(K)

    metric.calculate(X_true_unrecommended_user, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 3 / 6)

    K = 1
    metric = CalibratedRecallK(K)

    metric.calculate(X_true_unrecommended_user, X_pred)

    numpy.testing.assert_almost_equal(metric.value, 1 / 3)


def test_recallK_func(X_pred, X_true):
    K = 2
    value = recall_k(X_true, X_pred, K)

    numpy.testing.assert_almost_equal(value, 2 / 3)

    K = 1
    value = recall_k(X_true, X_pred, K)

    numpy.testing.assert_almost_equal(value, 1 / 4)


def test_recallK_no_reco_func(X_pred, X_true_unrecommended_user):
    K = 2
    value = recall_k(X_true_unrecommended_user, X_pred, K)

    numpy.testing.assert_almost_equal(value, 4 / 9)

    K = 1
    value = recall_k(X_true_unrecommended_user, X_pred, K)

    numpy.testing.assert_almost_equal(value, 1 / 6)


def test_calibrated_recallK_func(X_pred, X_true):
    K = 2
    value = calibrated_recall_k(X_true, X_pred, K)

    numpy.testing.assert_almost_equal(value, 3 / 4)

    K = 1
    value = calibrated_recall_k(X_true, X_pred, K)

    numpy.testing.assert_almost_equal(value, 1 / 2)


def test_calibrated_recallK_no_reco_func(X_pred, X_true_unrecommended_user):
    K = 2
    value = calibrated_recall_k(X_true_unrecommended_user, X_pred, K)

    numpy.testing.assert_almost_equal(value, 3 / 6)

    K = 1
    value = calibrated_recall_k(X_true_unrecommended_user, X_pred, K)

    numpy.testing.assert_almost_equal(value, 1 / 3)
