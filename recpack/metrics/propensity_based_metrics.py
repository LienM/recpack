import enum
import numpy
import scipy.sparse

from recpack.metrics.metric import MetricK


class PropensityType(enum.Enum):
    """
    Propensity Type enum.
    - Uniform means each item has the same propensity
    - Global means each item has a single propensity for all users
    - User means that for each user, item combination a unique propensity is computed.
    """

    UNIFORM = 1
    GLOBAL = 2
    USER = 3


class InversePropensity:
    """
    Abstract inverse propensity class
    """

    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

    def get(self, users):
        pass


class SNIPS(MetricK):
    def __init__(self, K: int, inverse_propensities: InversePropensity):
        super().__init__(K)
        self.num_users = 0
        self.score = 0

        self.inverse_propensities = inverse_propensities

    def update(
        self, X_pred: numpy.matrix, X_true: scipy.sparse.csr_matrix
    ) -> None:
        """
        Update the internal metrics given a set of recommendations and actual events.
        """
        assert X_pred.shape == X_true.shape
        if self.inverse_propensities is None:
            raise Exception(
                "inverse_propensities should not be None, fit a propensity model before using the SNIPS metric"
            )

        users = set(X_pred.nonzero()[0]).union(set(X_true.nonzero()[0]))

        # get top K recommendations
        x_pred_top_k = self.get_topK(X_pred)

        # binarize the prediction matrix
        x_pred_top_k[x_pred_top_k > 0] = 1

        # Sort the list of users to avoid weird issues with expected input
        ip = self.inverse_propensities.get(sorted(list(users)))

        X_pred_as_propensity = x_pred_top_k.multiply(ip).tocsr()
        X_true_as_propensity = X_true.multiply(ip).tocsr()

        X_hit_inverse_propensities = X_pred_as_propensity.multiply(X_true)

        for user in users:
            # Compute the hit score as the sum of propensities of hit items.
            # (0 contribution to the score if the item is not in X_true)
            hit_score = X_hit_inverse_propensities[user].sum()

            # Compute the max possible hit score, as the sum of the propensities of recommended items.
            max_possible_hit_score = X_true_as_propensity[user].sum()

            # Compute the user's actual number of interactions to normalize the metric.
            # And take min with self.K since it will be impossible to hit all of the items if there are more than K
            number_true_interactions = X_true[user].sum()

            if number_true_interactions > 0 and max_possible_hit_score > 0:
                self.num_users += 1
                self.score += (1 / max_possible_hit_score) * hit_score

    @property
    def value(self):
        if self.num_users == 0:
            return 0

        return self.score / self.num_users

    @property
    def name(self):
        return f"SNIPS@{self.K}"


IP_CAP = 10000


class UniformInversePropensity:
    """
    Helper class which will do the uniform propensity computation up front
    and return the results
    """

    def __init__(self, data_matrix):
        self.inverse_propensities = 1 / self._get_propensities(data_matrix)
        self.inverse_propensities[self.inverse_propensities > IP_CAP] = IP_CAP

    def get(self, users):
        return self.inverse_propensities

    def _get_propensities(self, data_matrix):
        nr_items = data_matrix.shape[1]
        return numpy.array([[1 / nr_items for i in range(nr_items)]])


class GlobalInversePropensity(InversePropensity):
    """
    Class to compute global propensities and serve them
    """

    def __init__(self, data_matrix):
        self.inverse_propensities = 1 / self._get_propensities(data_matrix)
        self.inverse_propensities[self.inverse_propensities > IP_CAP] = IP_CAP

    def get(self, users):
        return self.inverse_propensities

    def _get_propensities(self, data_matrix):
        item_count = data_matrix.sum(axis=0)
        total = data_matrix.sum()
        propensities = item_count / total
        propensities[propensities == numpy.inf] = 0
        return propensities


class UserInversePropensity(InversePropensity):
    """
    Class to compute propensities on the fly for a set of users.
    """

    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

    def get(self, users):
        # Compute the inverse propensities on the fly
        propensities = self._get_propensities(users)
        inverse_propensities = propensities.copy()
        inverse_propensities.data = 1 / propensities.data
        # Cap the inverse propensity to sensible values,
        # otherwise we will run into division by almost 0 issues.
        inverse_propensities[inverse_propensities > IP_CAP] = IP_CAP
        return inverse_propensities

    def _get_propensities(self, users):
        row_sum = self.data_matrix[users].sum(axis=1)
        user_mask = numpy.zeros((self.data_matrix.shape[0], 1))
        user_mask[users, 0] = 1
        dm_selected_users = self.data_matrix.multiply(user_mask)
        # After removing potentially non zero values by mult with 0, we need to remove them
        dm_selected_users.eliminate_zeros()

        propensities = dm_selected_users.tocsr()
        propensities[users] = scipy.sparse.csr_matrix(propensities[users] / row_sum)

        return propensities


class SNIPSFactory:
    def __init__(self, propensity_type):
        self.propensity_type = propensity_type
        self.inverse_propensities = None

    def fit(self, data):
        """
        Fit the propensities based on a sparse matrix with recommendation counts
        """
        if self.propensity_type == PropensityType.UNIFORM:
            self.inverse_propensities = UniformInversePropensity(data)
        elif self.propensity_type == PropensityType.GLOBAL:
            self.inverse_propensities = GlobalInversePropensity(data)
        elif self.propensity_type == PropensityType.USER:
            self.inverse_propensities = UserInversePropensity(data)
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
        return SNIPS(K, self.inverse_propensities)
