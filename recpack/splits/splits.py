import math
import numpy
import scipy.sparse


class TrainValidationTestSplit:
    def __init__(self):
        pass

    def split(self, data, evaluation_data=None):
        pass

    @property
    def name(self):
        return None


def slice_data(data, indices):
    """
    Helper function to get a new data object with the matrices in data to only contain values at 'indices'
    """
    mask_values = numpy.ones(len(indices[0]))
    mask = scipy.sparse.csr_matrix((mask_values, indices), shape=data.shape)

    new_data = data.copy()
    new_data.apply_mask(mask)
    return new_data


class PredefinedUserSplit(TrainValidationTestSplit):
    """
    Use this class if you want to split your data exactly as it was defined.

    :param tr_indices: User indices to be used for the training set
    :type tr_indices: list(int)
    :param val_indices: User indices to be used for the validation set
    :type val_indices: list(int)
    :param te_indices: User indices to be used for the test set
    :type te_indices: list(int)
    """
    def __init__(self, tr_indices, val_indices, te_indices, split_name):
        self.tr_indices = tr_indices
        self.val_indices = val_indices
        self.te_indices = te_indices
        self.split_name = split_name

    def name(self):
        return f"copy_split_{self.split_name}"

    def split(self, data, evaluation_data=None):
        """
        Splits a user-interaction matrix by the user indices defined for each set upon construction of the object.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
        :param evaluation_data: DataM object unused in this splitter.
        :return: A tuple containing the training, validation and test data objects in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`, class:`recpack.DataM`)
        """

        shape = data.shape
        sp_mat = data.values

        # We require a full data split.
        assert shape[0] == (len(self.tr_indices) + len(self.te_indices) + len(self.val_indices))

        tr_u, tr_i = [], []
        val_u, val_i = [], []
        te_u, te_i = [], []

        for user in self.tr_indices:
            u_nzi = sp_mat[user].indices
            u = [user] * len(u_nzi)
            tr_i.extend(u_nzi)
            tr_u.extend(u)

        for user in self.val_indices:
            u_nzi = sp_mat[user].indices
            u = [user] * len(u_nzi)
            val_i.extend(u_nzi)
            val_u.extend(u)

        for user in self.te_indices:
            u_nzi = sp_mat[user].indices
            u = [user] * len(u_nzi)
            te_i.extend(u_nzi)
            te_u.extend(u)

        tr_data = slice_data(data, (tr_u, tr_i))
        val_data = slice_data(data, (val_u, val_i))
        te_data = slice_data(data, (te_u, te_i))

        return tr_data, val_data, te_data


class StrongGeneralization(TrainValidationTestSplit):
    """
    Implementation of strong generalization split. All events for a user will always occur only in the same split.

    :param tr_perc: The percentage of the data to make training data.
    :type tr_perc: `float`

    :param val_perc: The percentage of the data to make validation data.
    :type val_perc: `float`

    :param seed: Seed for randomisation in function, set this to a specified value in order to get expected results.
    :type seed: `int`
    """

    def __init__(self, tr_perc=0.7, val_perc=0, seed=None):
        self.tr_perc = tr_perc
        self.val_perc = val_perc
        self.test_perc = 1 - tr_perc - val_perc
        self.seed = seed

    @property
    def name(self):
        return f"strong_generalization_tr_{self.tr_perc*100}_val_{(self.val_perc)*100}_te_{(self.te_perc)*100}"

    def split(self, data, validation_data=None):
        """
        Splits a user-interaction matrix by randomly shuffling users,
        and filling up the different data slices with the right percentages of data.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
        :param evaluation_data: DataM object unused in this splitter.
        :return: A tuple containing the training, validation and test data objects in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`, class:`recpack.DataM`)
        """

        shape = data.shape

        sp_mat = data.values
        nonzero_users = list(set(sp_mat.nonzero()[0]))

        # Seed to make testing possible
        if self.seed is not None:
            numpy.random.seed(self.seed)

        numpy.random.shuffle(nonzero_users)
        total_rows = sp_mat.nnz

        row_cnt = 0

        row_cnts = sp_mat.sum(axis=1).A.T[0]

        # These arrays will be used to create masks which will be applied to the data input.
        # This will make it so the mask is applied to both data and timestamps if available.
        tr_u, tr_i = [], []
        val_u, val_i = [], []
        te_u, te_i = [], []

        for user in nonzero_users:

            row_cnt += row_cnts[user]

            u_nzi = sp_mat[user].indices
            u = [user] * len(u_nzi)

            if (row_cnt / total_rows) <= self.tr_perc:
                tr_i.extend(u_nzi)
                tr_u.extend(u)
            elif ((row_cnt / total_rows) - self.tr_perc) <= self.val_perc:
                val_i.extend(u_nzi)
                val_u.extend(u)
            else:
                te_i.extend(u_nzi)
                te_u.extend(u)

        tr_data = slice_data(data, (tr_u, tr_i))
        val_data = slice_data(data, (val_u, val_i))
        te_data = slice_data(data, (te_u, te_i))

        return tr_data, val_data, te_data


class WeakGeneralization(TrainValidationTestSplit):

    def __init__(self, tr_perc=0.7, val_perc=0, seed=None):
        self.tr_perc = tr_perc
        self.val_perc = val_perc
        self.test_perc = 1 - tr_perc - val_perc
        self.seed = seed

    @property
    def name(self):
        return f"weak_generalization_tr_{self.tr_perc*100}_val_{(self.val_perc)*100}_te_{(self.te_perc)*100}"

    def split(self, data, validation_data=None):
        """
        Splits a user-interaction matrix by randomly shuffling interactions,
        and filling up the different data slices with the right percentages of data.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
        :param evaluation_data: DataM object unused in this splitter.
        :return: A tuple containing the training, validation and test data objects in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`, class:`recpack.DataM`)
        """

        shape = data.shape

        sp_mat = data.values
        indices = sp_mat.nonzero()
        number_of_interactions = len(indices[0])
        # each interaction is assigned its rank as value
        # These values will be shuffled, and the corresponding user, item indices
        # will be added to the various output slices.
        interaction_indices = list(range(number_of_interactions))

        # In case of a floating point end of slice we round up.
        tr_start = 0
        tr_end = math.ceil(number_of_interactions * self.tr_perc)

        val_start = tr_end
        val_end = tr_end + math.ceil(number_of_interactions * self.val_perc)

        te_start = val_end
        te_end = number_of_interactions

        # Seed to make testing possible
        if self.seed is not None:
            numpy.random.seed(self.seed)

        numpy.random.shuffle(interaction_indices)

        # Slice the user and item indices by using the list of interaction indices
        tr_u = indices[0][interaction_indices[tr_start:tr_end]]
        tr_i = indices[1][interaction_indices[tr_start:tr_end]]
        val_u = indices[0][interaction_indices[val_start:val_end]]
        val_i = indices[1][interaction_indices[val_start:val_end]]
        te_u = indices[0][interaction_indices[te_start:te_end]]
        te_i = indices[1][interaction_indices[te_start:te_end]]

        tr_data = slice_data(data, (tr_u, tr_i))
        val_data = slice_data(data, (val_u, val_i))
        te_data = slice_data(data, (te_u, te_i))

        return tr_data, val_data, te_data


class TimedSplit(TrainValidationTestSplit):
    """
        Use this class if you want to split your data on a timestamp.
        Training data will be all data before t,
        the test data will be the data in the interval [t, t+t_delta)

        :param t: epoch timestamp to split on
        :type t: int
        :param t_delta: seconds past t to consider as test data (default is None, all data > t is considered)
        :type t_delta: int
    """

    def __init__(self, t, t_delta=None):
        self.t = t
        self.t_delta = t_delta

    @property
    def name(self):
        return f"time_split_t{self.t}_t_delta_{self.t_delta}"

    def split(self, data, evaluation_data=None):
        """
        Split the input data by using the timestamps provided in the tinestamp field of data.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
        :param evaluation_data: DataM object unused in this splitter.
        :return: A tuple containing the training, validation and test data objects in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`, class:`recpack.DataM`)
        """
        # Holds indices where timestamp < t
        tr_u, tr_i = [], []
        # Get indices where timestamp >= t and timestamp < t + t_delta
        te_u, te_i = [], []
        # val indices will be empty, this splitter does not support a validation data slice.
        val_u, val_i = [], []

        for u, i, timestamp in zip(*scipy.sparse.find(data.timestamps)):
            if timestamp < self.t:
                tr_u.append(u)
                tr_i.append(i)

            elif (self.t_delta is None) or (timestamp < (self.t + self.t_delta)):
                te_u.append(u)
                te_i.append(i)

        tr_data = slice_data(data, (tr_u, tr_i))
        val_data = slice_data(data, (val_u, val_i))
        te_data = slice_data(data, (te_u, te_i))

        return tr_data, val_data, te_data


class EvaluationDataForTestSplit(TrainValidationTestSplit):
    """
    Use this class if you want to split your data as training = data and test+val = evaluation_data.

    :param val_perc: the percentage of the evaluation data to use for validation set.
    :type val_perc: `float`
    :param seed: Seed for randomisation in function, set this to a specified value in order to get expected results.
    :type seed: `int`

    """
    def __init__(self, val_perc=0., seed=None):
        self.val_perc = val_perc
        self.seed = seed

    @property
    def name(self):
        return f"evaluation_data_for_test"

    def split(self, data, evaluation_data):
        """
        Split the input data by making train = data and evaluation will be split into validation and test.
        """

        # reuse Strong Generalization splitter
        sgs = StrongGeneralization(0.0, val_perc=self.val_perc, seed=self.seed)

        _, val_data, te_data = sgs.split(evaluation_data)

        tr_data = data.copy()

        return tr_data, val_data, te_data


class EvaluationDataForTestTimedSplit(TrainValidationTestSplit):
    """
        Use this class if you want to split your data on a timestamp.
        Training data will be before t,
        the test data will be the data in the interval [t, t+t_delta)

        :param t: epoch timestamp to split on
        :type t: int
        :param t_delta: seconds past t to consider as test data (default is None, all data > t is considered)
        :type t_delta: int
    """

    def __init__(self, t, t_delta=None):
        self.t = t
        self.t_delta = t_delta

    @property
    def name(self):
        return f"time_split_t{self.t}_t_delta_{self.t_delta}"

    def split(self, data, evaluation_data=None):
        """
        Split the input data by using the timestamps provided in the tinestamp field of data.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be used for train data.
        :type data: class:`recpack.DataM`
        :param evaluation_data: DataM object used to generate train and validation data.
        :return: A tuple containing the training, validation and test data objects in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`, class:`recpack.DataM`)
        """

        # Reuse timed split func.
        tsp = TimedSplit(self.t, self.t_delta)

        tr_data, _, _ = tsp.split(data)
        _, val_data, te_data = tsp.split(evaluation_data)

        return tr_data, val_data, te_data
