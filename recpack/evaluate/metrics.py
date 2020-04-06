import numpy
import scipy.sparse

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
        if self.num_users == 0:
            return 0

        return self.recall / self.num_users


class MeanReciprocalRankK(Metric):

    def __init__(self, K):
        self.K = K
        self.rr = 0
        self.num_users = 0

    def update(self, X_pred, X_true):
        # Per user get a sorted list of the topK predicted items
        topK_items = {
            u: best_items_row[-self.K:][numpy.argsort(X_pred[u][best_items_row[-self.K:]])][::-1]
            for u, best_items_row in enumerate(numpy.argpartition(X_pred, -self.K))
        }

        items_sets = {
            u: set(X_true[u].nonzero()[1])
            for u in range(X_true.shape[0])
        }

        for u in topK_items.keys():
            for ix, item in enumerate(topK_items[u]):
                if item in items_sets[u]:
                    self.rr += 1 / (ix + 1)
                    break

            self.num_users += 1

        return

    @property
    def value(self):
        if self.num_users == 0:
            return 0

        return self.rr / self.num_users


class NDCGK(Metric):

    def __init__(self, K):
        self.K = K
        self.NDCG = 0
        self.num_users = 0

        self.discount_template = 1. / numpy.log2(numpy.arange(2, K + 2))

        self.IDCG = {K: self.discount_template[:K].sum()}

    def update(self, X_pred, X_true):

        topK_items = {
            u: best_items_row[-self.K:][numpy.argsort(X_pred[u][best_items_row[-self.K:]])][::-1]
            for u, best_items_row in enumerate(numpy.argpartition(X_pred, -self.K))
        }

        items_sets = {
            u: set(X_true[u].nonzero()[1])
            for u in range(X_true.shape[0])
        }

        for u in topK_items.keys():
            M = len(items_sets[u])
            if M < self.K:
                IDCG = self.IDCG.get(M)

                if not IDCG:
                    IDCG = self.discount_template[:M].sum()
                    self.IDCG[M] = IDCG
            else:
                IDCG = self.IDCG[self.K]

            # Compute DCG
            DCG = sum(
                (self.discount_template[rank] * (item in items_sets[u])) for rank, item in enumerate(topK_items[u])
            )

            self.num_users += 1
            self.NDCG += DCG / IDCG

        return

    @property
    def value(self):
        if self.num_users == 0:
            return 0

        return self.NDCG / self.num_users


class MutexMetric:
    """
    Metric used to evaluate the mutex predictors

    Computes false positives as predictor says it's mutex, but sample shows that the item predicted as mutex has been purchased in the evaluation data
    Computes True negatives as items predicted as non mutex, which are also actually in the evaluation data.
    """
    def __init__(self):
        self.false_positives = 0
        self.positives = 0

        self.true_negatives = 0
        self.negatives = 0

    def update(self, X_pred: numpy.matrix, X_true: scipy.sparse.csr_matrix, users: list) -> None:

        self.positives += X_pred.sum()
        self.negatives += (X_pred.shape[0] * X_pred.shape[1]) - X_pred.sum()

        false_pos = scipy.sparse.csr_matrix(X_pred).multiply(X_true)
        self.false_positives += false_pos.sum()

        negatives = numpy.ones(X_pred.shape) - X_pred
        true_neg = scipy.sparse.csr_matrix(negatives).multiply(X_true)
        self.true_negatives += true_neg.sum()

    @property
    def value(self):
        return (self.false_positives, self.positives, self.true_negatives, self.negatives)
