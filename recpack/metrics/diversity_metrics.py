from recpack.metrics.basic_metrics import MetricK
from recpack.algorithms.algorithm_base import Algorithm
import numpy as np
import scipy.sparse
import logging

class IntraListDistanceK(MetricK):
    def __init__(self, sim_algo: Algorithm, K):
        self.sim_algo = sim_algo
        self.K = K
        self.number_of_users = 0
        self.total_distance = 0

    def _get_idl(self, recommended_items, total_items):
        # Compute the IDL for this list
        # Sum part: SUM(d(i_k, i_l) for i_k in R and l < k)
        # We will compute this sum by constructing a sparse matrix with a 1 on each of the required i_k, i_l tuples
        if len(recommended_items) <= 1:
            # If there are 1 or no items, the intra list distance is 0
            return 0
        coordinates = [(i_k, i_l) for i, i_k in enumerate(recommended_items) for i_l in recommended_items[:i]]
        to_recommend = scipy.sparse.csr_matrix(
            (
                np.ones(len(recommended_items)), 
                (recommended_items, recommended_items)
            ),
            shape=(total_items, total_items)
        )
        logging.debug("coordinates")
        logging.debug(coordinates)
        logging.debug("shape")
        logging.debug(to_recommend.shape)
        # predict gives similarities between 0 and 1 (this is a requirement), distances are computed using 1-sim
        mask = scipy.sparse.csr_matrix((np.ones(len(coordinates)), list(zip(*coordinates))), shape=to_recommend.shape)
        
        similarities = self.sim_algo.predict(to_recommend).multiply(mask)
        t_distance = (mask - similarities).sum()

        idl = 2 / (len(recommended_items)*(len(recommended_items)-1)) * t_distance
        return idl


    def update(self, X_pred, X_true):
        """ Only looks at the predicted items, does not care about the 'true' items.
        """
        # resolve top K items per user
        # Get indices of top K items per user

        # Per user get a set of the topK predicted items
        X_pred_top_K = self.get_topK(X_pred)

        nonzero_users = list(set(X_pred.nonzero()[0]))
        
        for u in nonzero_users:
            recommended_items = list(set(X_pred_top_K[u, :].nonzero()[1]))
            if len(recommended_items) == 0:
                continue

            self.total_distance += self._get_idl(recommended_items, X_pred.shape[1])
            self.number_of_users += 1

    @property
    def value(self):
        if self.number_of_users == 0:
            return 0
        return self.total_distance/self.number_of_users
