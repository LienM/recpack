import numpy


class Metric:
    pass


class RecallK(Metric):

    def __init__(self, K):
        self.K = K
        self.recall = 0
        self.num_users = 0

    def update(self, X_pred, X_true):
        # resolve top K items per user
        # Get indices of top K items per user

        # Per user get a set of the topK predicted items
        topK_items_sets = {
            u: set(best_items_row[-self.K:])
            for u, best_items_row in enumerate(numpy.argpartition(X_pred, -self.K))
        }

        # Per user get a set of interacted items.
        items_sets = {
            u: set(X_true[u].nonzero()[1])
            for u in range(X_true.shape[0])
        }

        for u in topK_items_sets.keys():
            recommended_items = topK_items_sets[u]
            true_items = items_sets[u]

            self.recall += len(recommended_items.intersection(true_items)) / min(self.K, len(true_items))
            self.num_users += 1

        return

    @property
    def value(self):
        return self.recall / self.num_users


class MeanReciprocalRankK(Metric):

    def __init__(self, K):
        self.K = K
        self.rr = 0
        self.num_users = 0

    def update(self, X_pred, X_true):

        topK_list = numpy.argpartition(X_pred, -self.K)[-self.K:]
        sorted_topK_list = topK_list[numpy.argsort(X_pred[topK_list])][::-1]

        items = X_true.indices

        for ix, item in enumerate(sorted_topK_list):
            if item in items:
                self.rr += 1 / (ix + 1)

        self.num_users += 1

        return

    @property
    def value(self):
        return self.rr / self.num_users


class NDCGK(Metric):

    def __init__(self, K):
        self.K = K
        self.NDCG = 0
        self.num_users = 0

        self.discount_template = 1. / numpy.log2(numpy.arange(2, K + 2))

        self.IDCG = {K: self.discount_template[:K].sum()}

    def update(self, X_pred, X_true):

        print(type(X_pred), X_pred)
        print(type(X_true), X_true)
        topK_list = numpy.argpartition(X_pred, -self.K)[-self.K:]
        # Extract top-K highest scores into a sorted list
        sorted_topK_list = topK_list[numpy.argsort(X_pred[topK_list])][::-1]

        items = X_true.indices

        M = len(items)

        if M < self.K:
            IDCG = self.IDCG.get(M)

            if not IDCG:
                IDCG = self.discount_template[:M].sum()
                self.IDCG[M] = IDCG
        else:
            IDCG = self.IDCG[self.K]

        # Compute DCG
        DCG = sum((self.discount_template[rank] * (item in items)) for rank, item in enumerate(sorted_topK_list))

        self.num_users += 1
        self.NDCG += DCG / IDCG

        return

    @property
    def value(self):
        return self.NDCG / self.num_users
