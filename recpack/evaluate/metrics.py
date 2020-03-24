import enum
import numpy
import scipy.sparse


class Metric:
    pass


class RecallK(Metric):

    def __init__(self, K):
        self.K = K
        self.recall = 0
        self.num_users = 0

    def update(self, X_pred, X_true, users=None):
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

    def update(self, X_pred, X_true, users=None):
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

    def update(self, X_pred, X_true, users=None):

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


class PropensityType(enum.Enum):
    UNIFORM = 1
    GLOBAL = 2
    USER = 3


class SNIPS(Metric):

    def __init__(self, K, inverse_propensities, propensity_type):
        self.K = K
        self.num_users = 0
        self.recall = 0

        self.propensity_type = propensity_type

        self.inverse_propensities = inverse_propensities

    def update(self, X_pred: numpy.matrix, X_true: scipy.sparse.csr_matrix, users: list) -> None:
        """
        Update the internal metrics given a set of recommendations and actual events.
        """
        assert X_pred.shape == X_true.shape
        if self.inverse_propensities is None:
            raise Exception(
                "inverse_propensities should not be None, fit a propensity model before using the SNIPS metric"
            )

        # Top K mask on the X_pred
        topK_item_sets = {
            u: set(best_items_row[-self.K:])
            for u, best_items_row in enumerate(numpy.argpartition(X_pred, -self.K))
        }

        user_indices, item_indices = zip(*[(u, i) for u in topK_item_sets for i in topK_item_sets[u]])
        values = [1 for i in range(len(user_indices))]
        topK_mask = scipy.sparse.csr_matrix((values, (user_indices, item_indices)), shape=X_pred.shape)

        x_pred_top_k = scipy.sparse.csr_matrix(X_pred, shape=X_pred.shape).multiply(topK_mask)
        # binarize the prediction matrix
        x_pred_top_k[x_pred_top_k > 0] = 1

        ip = self.inverse_propensities
        if self.propensity_type == PropensityType.USER:
            # Slice a part of the inverse propensities, so we only have the selected users
            ip = self.inverse_propensities[users]

        X_pred_as_propensity = x_pred_top_k.multiply(ip).tocsr()
        X_true_as_propensity = X_true.multiply(ip).tocsr()

        X_hit_inverse_propensities = X_pred_as_propensity.multiply(X_true)

        for user in range(X_pred.shape[0]):
            # Compute the hit score as the sum of propensities of hit items.
            # (0 contribution to the score if the item is not in X_true)
            hit_score = X_hit_inverse_propensities[user].sum()

            print("hit_score", hit_score)

            # Compute the max possible hit score, as the sum of the propensities of recommended items.
            max_possible_hit_score = X_true_as_propensity[user].sum()

            print("max_possible_hit_score", max_possible_hit_score)
            # Compute the user's actual number of interactions to normalise the metric.
            # And take min with self.K since it will be impossible to hit all of the items if there are more than K
            number_true_interactions = X_true[user].sum()
            print("number_true_interactions", number_true_interactions)

            if number_true_interactions > 0:
                self.num_users += 1
                self.recall += (1/max_possible_hit_score) * hit_score

    @property
    def value(self):
        if self.num_users == 0:
            return 0

        return self.recall / self.num_users

    @property
    def name(self):
        return f"SNIPS@{self.K}"


class SNIPS_factory():
    def __init__(self, propensity_type):
        self.propensity_type = propensity_type
        self.propensities = None
        self.inverse_propensities = None

    def _fit_uniform(self, data):
        nr_items = data.shape[1]
        self.propensities = numpy.array([[1/nr_items for i in range(nr_items)]])
        self.inverse_propensities = 1/self.propensities

    def _fit_global(self, data):
        item_count = data.sum(axis=0)
        total = data.sum()
        self.propensities = item_count/total
        self.inverse_propensities = 1/self.propensities

    def _fit_user(self, data):
        row_sum = data.sum(axis=1)
        self.propensities = scipy.sparse.csr_matrix(data / row_sum)
        self.inverse_propensities = self.propensities.copy()
        self.inverse_propensities.data = 1/self.propensities.data

    def fit(self, data):
        """
        Fit the propensities based on a sparse matrix with recommendation counts
        """
        if self.propensity_type == PropensityType.UNIFORM:
            self._fit_uniform(data)
        elif self.propensity_type == PropensityType.GLOBAL:
            self._fit_global(data)
        elif self.propensity_type == PropensityType.USER:
            self._fit_user(data)
        else:
            raise ValueError(f"Unknown propensity type {self.propensity_type}")

    def create_multipe_SNIPS(self, K_values: list) -> dict:
        snipses = {}
        for K in K_values:
            snipses[f"SNIPS@{K}"] = self.create(K)
        return snipses

    def create(self, K: int) -> SNIPS:
        if self.inverse_propensities is None:
            raise Exception(
                "inverse_propensities should not be None, fit a propensity model before creating SNIPS metrics"
            )
        return SNIPS(K, self.inverse_propensities, self.propensity_type)
