from recpack.algorithms.rankers.diversification_rankers import GreedyIntraListDiversifier
from recpack.algorithms.user_item_interactions_algorithms.nn_algorithms import ItemKNN
import scipy.sparse
import numpy as np

def test_greedy_intra_list_diversifier(pageviews):
    sim_algo = ItemKNN(K=2)
    sim_algo.fit(pageviews)

    div_ranker = GreedyIntraListDiversifier(sim_algo, K=3)
    hist = scipy.sparse.csr_matrix(([1,1], ([0,0], [1, 2])), shape=(1,pageviews.shape[1]))
    scores = sim_algo.predict(hist)
    result = div_ranker.rank(scores, hist)
    assert result[0, 0] > result[0, 4]
    assert result[0, 3] > result[0, 0]

    assert result[0, 3] == scores[0, 3]
    assert result[0, 0] == scores[0, 0]
    assert result[0, 4] < scores[0, 4]