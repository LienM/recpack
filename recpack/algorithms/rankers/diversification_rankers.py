import scipy.sparse
import numpy as np
from recpack.algorithms.algorithm_base import Ranker
from recpack.algorithms.helpers import get_topK

class GreedyIntraListDiversifier(Ranker):
    def __init__(self, similarity_model, K=100, div_rate=0.1):
        self.similarity_model = similarity_model
        self.div_rate = 0.1
        self.K = K

    def fit(self, X):
        self.similarity_model.fit(X)

    def rank(
        self,
        scores: scipy.sparse.csr_matrix,
        X: scipy.sparse.csr_matrix,
        user_ids: np.array = None
    ):
        assert X.shape == scores.shape
        # Sparse array which will have a 1 if the item is in the topK for that user
        topK_items = scipy.sparse.coo_matrix(X.shape)
        topK_scores = scipy.sparse.csr_matrix(X.shape)

        # A copy which will iteratively get updated.
        working_scores = scores.copy()
        # Fill topK and discount the remaining scores after every iteration
        # TODO? Early breaking criterium if there are not enough scores?
        for _ in range(self.K):
            best_items = get_topK(working_scores, 1)
            topK_scores += best_items
            topK_items += best_items
            topK_items[topK_items > 0] = 1 # Binarize the topK items matrix

            # discount scores based on MRR equation with div_rate = 1-lambda

            working_scores = scores - self.similarity_model.predict(topK_items).multiply(self.div_rate)

            # remove topK from working scores
            working_scores = working_scores - working_scores.multiply(topK_items)
            working_scores[working_scores < 0] = 0

        # After topK, we just add the remaining scores
        topK_scores += working_scores

        return topK_scores
