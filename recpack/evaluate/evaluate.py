from collections import defaultdict
from itertools import groupby

import numpy
import scipy.sparse


class Evaluator:
    pass


class LeavePOutCrossValidation(Evaluator):
    # TODO: Implement

    def __init__(self, p):
        self.p = p
        self.sp_mat = sp_mat


class FoldIterator:
    """
    Iterator to get results from a fold instance.
    Fold instance can be any of the Evaluator classes defined.
    """
    def __init__(self, fold_instance, batch_size=1):
        self._fi = fold_instance
        self._index = 0
        self._max_index = fold_instance.sp_mat_in.shape[0]
        self.batch_size = batch_size

    def __next__(self):
        # While loop to make it possible to skip any cases where user has no items in in or out set.
        while(True):
            # Yield multi line fold.
            if self._index < self._max_index:
                start = self._index
                end = self._index + self.batch_size
                # make sure we don't go out of range
                if end >= self._max_index:
                    end = self._max_index

                fold_in = self._fi.sp_mat_in[start: end]
                fold_out = self._fi.sp_mat_out[start: end]

                # TODO Filter out users with all zeros?
                # This does not yet work with batches I think.
                if fold_in.nnz == 0 or fold_out.nnz == 0:
                    self._index = end
                    continue

                self._index = end
                return fold_in, fold_out
            raise StopIteration


class FoldInPercentage(Evaluator):
    """
    Evaluator which generates in matrices by taking `fold_in` percentage of items a user has seen,
    the expected output are the remaining items for that user.

    :param fold_in: The percentage of each user's interactions to add in the input matrix.
    :type fold_in: `float`

    :param seed: The seed for random operations in this class. Very useful when desiring reproducible results.
    :type seed: `int`

    :param batch_size: the number of rows to generate when iterating over the evaluator object.
                       Larger batch_sizes make it possible to optimise computation time.
    :type data: `int`

    """
    def __init__(self, fold_in, seed=None, batch_size=1):
        self.fold_in = fold_in

        self.sp_mat_in = None
        self.sp_mat_out = None

        self.batch_size = batch_size
        self.seed = seed if seed else 12345

    def split(self, tr_data, val_data, te_data, shape=None):
        """
        Split the input data based on the parameters set at construction.
        Returns 2 sparse matrices in and out
        """
        # Set shape based on input data size
        if not shape:
            shape = tr_data.shape
        assert shape == tr_data.shape
        assert shape == val_data.shape
        assert shape == te_data.shape

        # easiest if we extract sparse value matrix from test, since that is the only part of the data we will use.
        sp_mat = te_data.values

        U, I, V = scipy.sparse.find(sp_mat)

        items_by_user_dct = defaultdict(list)

        # Unsure if U and I are sorted, which is a requirement for itertools.groupby.
        # So add them to a defaultdict to be sure
        groups = groupby(zip(U, I, V), lambda x: x[0])
        for key, subiter in groups:
            items_by_user_dct[key].extend(subiter)

        in_fold, out_fold = [], []

        # Seed random
        numpy.random.seed(self.seed)

        for u in items_by_user_dct.keys():
            usr_hist = items_by_user_dct[u]

            _len = len(usr_hist)

            numpy.random.shuffle(usr_hist)

            cut = int(numpy.floor(_len * self.fold_in))

            in_fold.extend(usr_hist[:cut])
            out_fold.extend(usr_hist[cut:])

        U_in, I_in, V_in = zip(*in_fold)

        self.sp_mat_in = scipy.sparse.csr_matrix((V_in, (U_in, I_in)), shape=shape)

        U_out, I_out, V_out = zip(*out_fold)
        self.sp_mat_out = scipy.sparse.csr_matrix((V_out, (U_out, I_out)), shape=shape)

        return self.sp_mat_in, self.sp_mat_out

    def __iter__(self):
        return FoldIterator(self, self.batch_size)


class TrainingInTestOutEvaluator(Evaluator):
    """
    Class to evaluate an algorithm by using training data as input, and test data as expected output.

    :param batch_size: the number of rows to generate when iterating over the evaluator object.
                       Larger batch_sizes make it possible to optimise computation time.
    :type data: `int`
    """

    def __init__(self, batch_size=1):

        self.sp_mat_in = None
        self.sp_mat_out = None

        self.batch_size = batch_size

    def split(self, tr_data, val_data, te_data, shape=None):
        """
        Split the data into in and out matrices.
        The in matrix will be the training data, and the out matrix will be the test data.
        """
        if not shape:
            shape = tr_data.shape
        assert shape == tr_data.shape
        assert shape == val_data.shape
        assert shape == te_data.shape

        U_in, I_in, V_in = [], [], []
        U_out, I_out, V_out = [], [], []

        # We are mostly intersted in user indices in training data, these will be the rows
        # we need to return.
        nonzero_users = list(set(tr_data.values.nonzero()[0]))
        tr_sp_mat = tr_data.values
        te_sp_mat = te_data.values
        for user in nonzero_users:
            # Add the in data
            tr_u_nzi = tr_sp_mat[user].indices
            tr_u = [user] * len(tr_u_nzi)
            tr_vals = tr_sp_mat[user, tr_u_nzi].todense().A[0]

            # Add the out data
            te_u_nzi = te_sp_mat[user].indices
            te_u = [user] * len(te_u_nzi)
            te_vals = te_sp_mat[user, te_u_nzi].todense().A[0]

            if len(tr_u_nzi) > 0 and len(te_u_nzi) > 0:
                # Only use the user if they actually have data for both
                # In and out
                I_in.extend(tr_u_nzi)
                U_in.extend(tr_u)
                V_in.extend(tr_vals)

                I_out.extend(te_u_nzi)
                U_out.extend(te_u)
                V_out.extend(te_vals)

        self.sp_mat_in = scipy.sparse.csr_matrix((V_in, (U_in, I_in)), shape=shape)

        self.sp_mat_out = scipy.sparse.csr_matrix((V_out, (U_out, I_out)), shape=shape)

        return self.sp_mat_in, self.sp_mat_out

    def __iter__(self):
        return FoldIterator(self, self.batch_size)


class TimedTestSplit(Evaluator):
    def __init__(self, t, batch_size=1):
        self.t = t
        self.batch_size = 1

    def __iter__(self):
        return FoldIterator(self, self.batch_size)

    def split(self, tr_data, val_data, te_data, shape=None):
        """
        Split the data into in and out matrices.
        The in matrix will be the training data, and the out matrix will be the test data.
        """
        if not shape:
            shape = tr_data.shape
        assert shape == tr_data.shape
        assert shape == val_data.shape
        assert shape == te_data.shape

        U_in, I_in, V_in = [], [], []
        U_out, I_out, V_out = [], [], []

        # Split test data into before t and after t
        # in will be before t
        # out will be after t.

        nonzero_users = list(set(te_data.values.nonzero()[0]))
        te_sp_mat = te_data.values
        te_timestamps = te_data.timestamps
        for user in nonzero_users:
            # Add the in data
            for item in te_timestamps[user].indices:
                if te_timestamps[user, item] < self.t:
                    U_in.append(user)
                    I_in.append(item)
                    V_in.append(te_sp_mat[user, item])

                else:
                    U_out.append(user)
                    I_out.append(item)
                    V_out.append(te_sp_mat[user, item])

        self.sp_mat_in = scipy.sparse.csr_matrix((V_in, (U_in, I_in)), shape=shape)

        self.sp_mat_out = scipy.sparse.csr_matrix((V_out, (U_out, I_out)), shape=shape)

        return self.sp_mat_in, self.sp_mat_out
