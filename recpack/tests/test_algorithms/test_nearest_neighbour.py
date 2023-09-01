# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import math
import operator

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recpack.algorithms import ItemKNN
from recpack.matrix import to_binary
from recpack.algorithms.nearest_neighbour import (
    ItemPNN,
    compute_conditional_probability,
    compute_pearson_similarity,
)


@pytest.fixture(scope="function")
def data():
    values = [1] * 7
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    d = csr_matrix((values, (users, items)), shape=(4, 3))

    return d


@pytest.fixture(scope="function")
def data_empty_col():
    values = [1] * 5
    users = [0, 0, 1, 1, 2]
    items = [1, 2, 2, 1, 2]
    d = csr_matrix((values, (users, items)))

    return d


def test_item_knn(data):

    algo = ItemKNN(K=2)

    algo.fit(data)

    expected_similarities = np.array(
        [
            [0, 0.5, 2 / math.sqrt(6)],
            [0.5, 0, 2 / math.sqrt(6)],
            [2 / math.sqrt(6), 2 / math.sqrt(6), 0],
        ]
    )
    np.testing.assert_almost_equal(algo.similarity_matrix_.toarray(), expected_similarities)

    # Make sure the similarities recommended are the cosine similarities as computed.
    # If we create users with a single item seen in order.
    _in = csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    np.testing.assert_almost_equal(result.toarray(), expected_similarities)

    # Make sure similarities are added correctly.
    _in = csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(1, 3))
    expected_out = [[0.5, 0.5, 4 / math.sqrt(6)]]
    result = algo.predict(_in)
    np.testing.assert_almost_equal(result.toarray(), expected_out)


def test_item_knn_normalize_X(data):

    algo = ItemKNN(K=2, similarity="cosine", normalize_X=True)

    algo.fit(data)

    # data matrix looks like
    # 0 1 1
    # 1 0 1
    # 1 1 1

    # normalized data matrix looks like
    # 0 0.5 0.5
    # 0.5 0 0.5
    # 0.33 0.33 0.33

    # Dot products
    a = 1 / (3 * 3)
    b = 1 / (3 * 3) + 1 / (2 * 2)
    c = (1 / 3 * 3) + 1 / (2 * 2) * 2

    # Item norms
    d = math.sqrt((1 / 2) ** 2 + (1 / 3) ** 2)
    e = d
    f = math.sqrt(2 * (1 / 2) ** 2 + (1 / 3) ** 2)

    expected_similarities = np.array(
        [
            [0, a / (d * e), b / (d * f)],
            [a / (d * e), 0, b / (e * f)],
            [b / (d * f), b / (e * f), 0],
        ]
    )

    np.testing.assert_almost_equal(algo.similarity_matrix_.toarray(), expected_similarities)


def test_item_knn_normalize_sim(data):

    algo = ItemKNN(K=2, normalize_sim=True)

    algo.fit(data)

    np.testing.assert_array_almost_equal(algo.similarity_matrix_.sum(axis=1), 1)


def test_item_knn_empty_col(data_empty_col):
    algo = ItemKNN(K=2)

    algo.fit(data_empty_col)
    expected_similarities = np.array([[0.0, 0.0, 0.0], [0.0, 0, 2 / math.sqrt(6)], [0.0, 2 / math.sqrt(6), 0]])
    np.testing.assert_almost_equal(algo.similarity_matrix_.toarray(), expected_similarities)


def test_item_knn_conditional_probability(data):
    algo = ItemKNN(K=2, similarity="conditional_probability")

    algo.fit(data)
    # similarity is computed as count(i^j) / (count(i) + 1)

    # data matrix looks like
    # 0 1 1
    # 1 0 1
    # 1 1 1

    # cooc = XtX
    # 2 1 2
    # 1 2 2
    # 2 2 3

    # fmt: off
    expected_similarities = np.array(
        [
            [0, 1 / 2, 2 / 2],
            [1 / 2, 0, 2 / 2],
            [2 / 3, 2 / 3, 0]
        ]
    )
    # fmt: on
    np.testing.assert_almost_equal(algo.similarity_matrix_.toarray(), expected_similarities)


@pytest.mark.parametrize("pop_discount", [1, 0.2, 0.5])
def test_item_knn_conditional_probability_w_pop_discount(data, pop_discount):
    algo = ItemKNN(K=2, similarity="conditional_probability", pop_discount=pop_discount)

    algo.fit(data)
    # similarity is computed as count(i^j) / (count(i) * count(j) ^ pop_discount)

    # data matrix looks like
    # 0 1 1
    # 1 0 1
    # 1 1 1

    # cooc = XtX
    # 2 1 2
    # 1 2 2
    # 2 2 3

    # fmt: off
    expected_similarities = np.array(
        [
            [0, 1 / (2 * 2**pop_discount), 2 / (2 * 3**pop_discount)],
            [1 / (2 * 2**pop_discount), 0, 2 / (2 * 3**pop_discount)],
            [2 / (3 * 2**pop_discount), 2 / (3 * 2**pop_discount), 0]
        ]
    )
    # fmt: on
    np.testing.assert_almost_equal(algo.similarity_matrix_.toarray(), expected_similarities)


@pytest.mark.parametrize(
    "K, pdf",
    [
        (1, "uniform"),
        (2, "uniform"),
        (
            1,
            "empirical",
        ),  # (2, "empirical"),  # Cannot run 2 empirical because it leads to fewer nonzero values than needed
        (1, "softmax_empirical"),
        (2, "softmax_empirical"),
    ],
)
def test_item_pnn(data, K, pdf):
    algo = ItemPNN(K=K, similarity="cosine", pdf=pdf)

    algo.fit(data)

    # Test number of nonzeroes
    sims = algo.similarity_matrix_
    binary_sims = to_binary(sims)

    np.testing.assert_array_equal(binary_sims.sum(axis=1).A, K)


def test_item_pnn_uniform_larger(larger_matrix):
    K = 10
    algo = ItemPNN(K=K, similarity="cosine", pdf="uniform")

    algo.fit(larger_matrix)

    # Test number of nonzeroes
    sims = algo.similarity_matrix_.copy()
    binary_sims = to_binary(sims)
    # Test is not exactly the same between two runs
    algo.fit(larger_matrix)
    sims2 = algo.similarity_matrix_.copy()

    assert np.any(np.not_equal(sims.toarray(), sims2.toarray()))


@pytest.mark.parametrize(
    "K, pdf",
    [
        (1, "uniform"),
        (2, "uniform"),
        (
            1,
            "empirical",
        ),  # (2, "empirical"),  # Cannot run 2 empirical because it leads to fewer nonzero values than needed
        (1, "softmax_empirical"),
        (2, "softmax_empirical"),
    ],
)
def test_item_pnn_compute_df(K, pdf):
    algo = ItemPNN(K=K, similarity="cosine", pdf=pdf)

    X = csr_matrix([[0.5, 0.2, 0.3], [0.2, 0.3, 0.5], [0.5, 0.5, 0]])

    p = algo._compute_pdf(pdf, X)

    np.testing.assert_almost_equal(np.sum(p), X.shape[1])
    np.testing.assert_array_almost_equal(np.sum(p, axis=1), 1)


def test_conditional_probability():
    # fmt: off
    X = csr_matrix([
        [1, 0.5],
        [0, 0.25],
        [1, 1]
    ])

    similarity = compute_conditional_probability(X)

    expected_similarity = [
        [0, (0.5 + 1)/2],
        [(1 + 1)/3, 0]
    ]

    np.testing.assert_array_equal(similarity.toarray(), expected_similarity)

    similarity = compute_conditional_probability(X, pop_discount=1)

    expected_similarity = [
        [0, (0.5 + 1) / (2 * 3)],
        [(1 + 1) / (2 * 3), 0]
    ]

    np.testing.assert_array_equal(similarity.toarray(), expected_similarity)

    similarity = compute_conditional_probability(X, pop_discount=0.1)

    expected_similarity = [
        [0, (0.5 + 1) / (2 * 3 ** 0.1)],
        [(1 + 1) / (3 * 2 ** 0.1), 0],
    ]
    # fmt: on

    np.testing.assert_array_almost_equal(similarity.toarray(), expected_similarity)


def test_compute_pearson_similarity():
    # fmt: off
    data = csr_matrix([
        [1, 0, 1, 0],
        [2, 0, 2, 0],
        [0, 3, 0, 3],
        [4, 0, 0, 4],
    ])
    # fmt: on
    corr = compute_pearson_similarity(data)

    assert corr[0, 0] == 0  # no self similarity
    assert corr[0, 1] == 0
    expected_sim_0_2 = ((-1.333 * -0.5) + (-0.333 * 0.5)) / (
        np.sqrt(1.333**2 + 0.333**2 + 1.666**2) * np.sqrt(2 * (0.5) ** 2)
    )
    np.testing.assert_almost_equal(corr[0, 2], expected_sim_0_2, decimal=3)
    expected_sim_0_3 = (1.666 * 0.5) / (np.sqrt(1.333**2 + 0.333**2 + 1.666**2) * np.sqrt(2 * 0.5**2))
    np.testing.assert_almost_equal(corr[0, 3], expected_sim_0_3, decimal=3)


def test_compute_pearson_similarity_binary_matrix():
    data = csr_matrix([[1, 0, 1, 0], [0, 1, 0, 1]])

    with pytest.raises(ValueError) as e:
        compute_pearson_similarity(data)

    assert e.match("binary matrix")
