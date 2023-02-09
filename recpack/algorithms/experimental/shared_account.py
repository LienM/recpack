# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import enum

import numpy as np

from recpack.algorithms.base import ItemSimilarityMatrixAlgorithm
from recpack.matrix import Matrix, to_csr_matrix
from scipy.sparse import csr_matrix


@enum.unique
class Aggregator(enum.Enum):
    Sum = "sum"
    Avg = "avg"
    Adj = "adj"

    def __str__(self):
        return self.value


Agg = Aggregator


class DAMIBCover(ItemSimilarityMatrixAlgorithm):
    """
    DAMIB-Cover Algorithm by Koen Verstrepen et al.
    Only the rescaling os scores is implemented for now.
    The optimal set of explanations is found with the parameter `p`
    by dividing the sum of scores by the size of the set to the power `p`.
    The final score can either be the sum, average or adjusted average
    (with denominator) depending on the `agg` param.
    """

    def __init__(self, algo: ItemSimilarityMatrixAlgorithm, p=0.75, agg: Agg = Agg.Adj):
        super().__init__()
        self.algo = algo
        self.p = p
        self.agg = agg

    def fit(self, X: Matrix):
        return self.algo.fit(X)

    @property
    def similarity_matrix_(self):
        return self.algo.similarity_matrix_

    def predict(self, X: Matrix):
        X = to_csr_matrix(X, binary=True)

        predictions = get_predictions(X, self.similarity_matrix_, self.p, self.agg)

        self._check_prediction(predictions, X)
        return csr_matrix(predictions)


def get_predictions(X, M, p, agg):
    predictions = np.zeros(X.shape, dtype=np.float32)
    # For every user
    for u in set(X.nonzero()[0]):
        # Items this user has interacted with [0, 1, 0] -> indices = 2
        indices = X[u].toarray()[0]
        # [[0 1], [1 0]] (sim) * [0 2] (u)
        similarities = M[indices.astype(bool), :].toarray()
        predictions[u] = get_prediction_u(similarities, p, agg)

    return predictions


def get_prediction_u(similarities, p, agg):
    predictions = np.zeros((similarities.shape[1]), dtype=np.float32)
    filtered = filter_best_subsets(similarities, p)

    for col in range(filtered.shape[1]):
        nonzero = np.count_nonzero(filtered[:, col])
        if nonzero == 0:
            predictions[col] = 0
        elif agg == Agg.Sum:
            # sum
            predictions[col] = np.sum(filtered[:, col])
        elif agg == Agg.Adj:
            # adjusted average
            predictions[col] = np.sum(filtered[:, col]) / nonzero ** p
        elif agg == Agg.Avg:
            # average
            predictions[col] = np.sum(filtered[:, col]) / nonzero
        else:
            raise ValueError("Unknown aggragation method for SA algorithm")

    return predictions


def filter_best_subsets(similarities, p):
    sort_indices = np.empty(similarities.shape, dtype=np.int32)
    for j in range(sort_indices.shape[1]):
        sort_indices[:, j] = np.argsort(-similarities[:, j])

    for col in range(sort_indices.shape[1]):
        order = sort_indices[:, col]
        total = 0
        amount = 0
        for index in order:
            tmp = (total + similarities[index, col]) / (amount + 1) ** p
            if tmp < total:
                break
            else:
                total = tmp
                amount += 1

        similarities[order[amount:], col] = 0

    return similarities
