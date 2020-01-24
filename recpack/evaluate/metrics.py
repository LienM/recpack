import numpy


class Metric:
    pass


class RecallK(Metric):

    def __init__(self, K):
        self.K = K
        self.recall = 0
        self.num_users = 0

    def update(self, X_pred, X_true):

        topK_list = numpy.argpartition(X_pred, -self.K)[-self.K:]
        topK_set = set(topK_list)

        items = X_true.indices

        self.recall += len(topK_set.intersection(items)) / min(self.K, len(items))
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
