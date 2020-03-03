import math
import numpy
import scipy.sparse


class TrainValidationTestSplit:
    def __init__(self):
        pass

    def split(self, data):
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

    def split(self, data):
        """
        Splits a user-interaction matrix by the user indices defined for each set upon construction of the object.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
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

    def split(self, data):
        """
        Splits a user-interaction matrix by randomly shuffling users,
        and filling up the different data slices with the right percentages of data.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
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

    def split(self, data):
        """
        Splits a user-interaction matrix by randomly shuffling interactions,
        and filling up the different data slices with the right percentages of data.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
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
        :param t_delta: seconds past t to consider as test data (default is None, all data >= t is considered)
        :type t_delta: int
        :param t_alpha: seconds before t to use as training data (default is None -> all data < t is considered)
        :type t_alpha: int
    """

    def __init__(self, t, t_delta=None, t_alpha=None):
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

    @property
    def name(self):
        return f"timed_split_t{self.t}_t_delta_{self.t_delta}_t_alpha_{self.t_alpha}"

    def split(self, data):
        """
        Split the input data by using the timestamps provided in the timestamp field of data.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
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
            if timestamp < self.t and ((self.t_alpha is None) or (timestamp >= self.t - self.t_alpha)):
                tr_u.append(u)
                tr_i.append(i)

            elif timestamp > self.t and ((self.t_delta is None) or (timestamp < (self.t + self.t_delta))):
                te_u.append(u)
                te_i.append(i)

        tr_data = slice_data(data, (tr_u, tr_i))
        val_data = slice_data(data, (val_u, val_i))
        te_data = slice_data(data, (te_u, te_i))

        return tr_data, val_data, te_data


class StrongGeneralizationTimedSplit(TrainValidationTestSplit):
    """
        Use this class if you want to split your data on a timestamp and make sure users in test
        do not occur in the training dataset.
        Test users are users with events in the interval [t, t+t_delta)
        The test data will be all data for these test users
        Training data will be all data in interval [t, t_alpha) except data for the test users.

        :param t: epoch timestamp to split on
        :type t: int
        :param t_delta: seconds past t to consider as test interval (default is None, all data >= t is considered)
        :type t_delta: int
        :param t_alpha: seconds before t to use as training interval (default is None -> all data < t is considered)
        :type t_alpha: int
    """

    def __init__(self, t, t_delta=None, t_alpha=None):
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

    @property
    def name(self):
        return f"timed_split_t{self.t}_t_delta_{self.t_delta}_t_alpha_{self.t_alpha}"

    def split(self, data):
        """
        Split the input data by using the timestamps provided in the timestamp field of data.

        :param data: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split.
        :type data: class:`recpack.DataM`
        :return: A tuple containing the training, validation and test data objects in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`, class:`recpack.DataM`)
        """

        # Holds indices of training data
        tr_u, tr_i = [], []
        # Holds indices of test data
        te_u, te_i = [], []
        # val indices will be empty, this splitter does not support a validation data slice.
        val_u, val_i = [], []

        # Get the users who have data in the [t, t+t_delta) interval
        test_users = set()
        for u, i, timestamp in zip(*scipy.sparse.find(data.timestamps)):
            if timestamp > self.t and ((self.t_delta is None) or (timestamp < (self.t + self.t_delta))):
                test_users.add(u)

        for u, i, timestamp in zip(*scipy.sparse.find(data.timestamps)):
            if u in test_users:
                te_u.append(u)
                te_i.append(i)
            elif timestamp < self.t and ((self.t_alpha is None) or (timestamp >= self.t - self.t_alpha)):
                tr_u.append(u)
                tr_i.append(i)

        tr_data = slice_data(data, (tr_u, tr_i))
        val_data = slice_data(data, (val_u, val_i))
        te_data = slice_data(data, (te_u, te_i))

        return tr_data, val_data, te_data


# Splitters that take 2 data objects as input in their split function.
class TrainValidationSplitTwoDataInputs:
    """
    Abstract base class for splitters which take 2 data objects and turn them into train, validation, test data objects.
    This can be useful to encode side info, add a distinct evaluation dataset and others.
    """
    def __init__(self):
        pass

    def split(self, data_1, data_2):
        pass

    @property
    def name(self):
        return None


class SeparateDataForValidationAndTestSplit(TrainValidationSplitTwoDataInputs):
    """
    Use this class if you want to use a separate data object for training and evaluation.

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
        return f"separate_data_for_validation_and_test_{val_perc}"

    def split(self, data_1, data_2):
        """
        Split the input data objects as train = data_1 and val + test = data_2

        :param data_1: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which be used as training data.
        :type data_1: class:`recpack.DataM`
        :param data_2: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be split in validation and test
                     according to the parameter set in the constructor.
        :type data_2: class:`recpack.DataM`
        :return: A tuple containing the training, validation and test data objects in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`, class:`recpack.DataM`)
        """

        # reuse Strong Generalization splitter
        sgs = StrongGeneralization(0.0, val_perc=self.val_perc, seed=self.seed)

        # data_2 is used to generate validation and test data
        _, val_data, te_data = sgs.split(data_2)

        # Training data is the data_1 object.
        tr_data = data_1.copy()

        return tr_data, val_data, te_data


class SeparateDataForValidationAndTestTimedSplit(TrainValidationSplitTwoDataInputs):
    """
    Use this class if you want to use a different data object for training and evaluation, 
    and also split your data on a timestamp.
    Training data will be before t,
    the test data will be the data in the interval [t, t+t_delta)
    No validation data is generated.

    :param t: epoch timestamp to split on
    :type t: int
    :param t_delta: seconds past t to consider as test data (default is None, all data >= t is considered)
    :type t_delta: int

    :param t_alpha: seconds before t to consider as training data (default is None, all data < t is considered)
    :type t_alpha: int
    """

    def __init__(self, t, t_delta=None, t_alpha=None):
        self.t = t
        self.t_delta = t_delta
        self.t_alpha = t_alpha

    @property
    def name(self):
        return f"separate_data_for_validation_and_test_timed_split_t{self.t}_t_delta_{self.t_delta}"

    def split(self, data_1, data_2):
        """
        Split the input data.
        Train = all interactions in data_1 which happen before the timestamp t.
        by using the timestamps provided in the timestamp field of data and validation_data.
        Test = all interaction in data_1 which occurred in the interval [t, t+t_delta).
        Where t and t_delta are defined in the constructor

        :param data_1: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix which will be used for the training data.
        :type data_1: class:`recpack.DataM`
        :param data_2: recpack.DataM containing the data values matrix
                     and potentially timestamp matrix from which test will be extracted.
                     according to the parameter set in the constructor.
        :return: A tuple containing the training, validation and test data objects in that order.
        :rtype: tuple(class:`recpack.DataM`, class:`recpack.DataM`, class:`recpack.DataM`)
        """

        # Reuse timed split func.
        tsp = TimedSplit(self.t, t_delta=self.t_delta, t_alpha=self.t_alpha)

        tr_data, _, _ = tsp.split(data_1)
        _, val_data, te_data = tsp.split(data_2)

        return tr_data, val_data, te_data
