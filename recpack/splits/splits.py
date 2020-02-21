import numpy
import scipy.sparse


class TrainValidationTestSplit:
    def __init__(self):
        pass

    def split(self):
        pass

    @property
    def name(self):
        return None


class PredefinedSplit(TrainValidationTestSplit):
    """
    Use this class if you want to split your data exactly as it was defined.

    :param tr_indices: Indices to be used for the training set
    :type tr_indices: list(int)
    :param val_indices: Indices to be used for the validation set
    :type val_indices: list(int)
    :param te_indices: Indices to be used for the test set
    :type te_indices: list(int)
    """
    def __init__(self, tr_indices, val_indices, te_indices, split_name):
        self.tr_indices = tr_indices
        self.val_indices = val_indices
        self.te_indices = te_indices
        self.split_name = split_name

    def name(self):
        return f"copy_split_{self.split_name}"

    def split(self, sp_mat):
        """
        Splits a user-interaction matrix by the indices defined for each set upon construction of the object.

        :param sp_mat: SciPy sparse matrix containing all data to be split.
        :type sp_mat: class:`scipy.sparse.csr_matrix`
        :return: A tuple containing the training, validation and test matrix in that order.
        :rtype: tuple(class:`scipy.sparse_csr_matrix`, class:`scipy.sparse_csr_matrix`, class:`scipy.sparse_csr_matrix`)
        """

        def create_sp_mat_from_indices(sp_mat, indices):

            new_mat_u, new_mat_i, new_mat_vals = [], [], []

            for user in indices:
                u_nzi = sp_mat[user].indices
                u = [user] * len(u_nzi)
                vals = sp_mat[user, u_nzi].todense().A[0]

                new_mat_i.extend(u_nzi)
                new_mat_u.extend(u)
                new_mat_vals.extend(vals)

            new_mat = scipy.sparse.csr_matrix((new_mat_vals, (new_mat_u, new_mat_i)), shape=shape, dtype=dtype)

            return new_mat

        shape = sp_mat.shape
        dtype = sp_mat.dtype

        assert shape[0] == (len(self.tr_indices) + len(self.te_indices) + len(self.val_indices))

        tr_mat = create_sp_mat_from_indices(sp_mat, self.tr_indices)
        val_mat = create_sp_mat_from_indices(sp_mat, self.val_indices)
        te_mat = create_sp_mat_from_indices(sp_mat, self.te_indices)

        return tr_mat, val_mat, te_mat


class StrongGeneralization(TrainValidationTestSplit):
    def __init__(self, tr_perc=0.7, val_perc=0):
        self.tr_perc = tr_perc
        self.val_perc = val_perc
        self.test_perc = 1 - tr_perc - val_perc

    @property
    def name(self):
        return f"strong_generalization_tr_{self.tr_perc*100}_val_{(self.val_perc)*100}_te_{(self.te_perc)*100}"

    def split(self, sp_mat):

        shape = sp_mat.shape
        dtype = sp_mat.dtype

        nonzero_users = list(set(sp_mat.nonzero()[0]))
        numpy.random.shuffle(nonzero_users)
        total_rows = sp_mat.nnz

        row_cnt = 0

        row_cnts = sp_mat.sum(axis=1).A.T[0]

        tr_u, tr_i, tr_vals = [], [], []
        val_u, val_i, val_vals = [], [], []
        te_u, te_i, te_vals = [], [], []

        for user in nonzero_users:
            u_nzi = sp_mat[user].indices
            u = [user] * len(u_nzi)
            vals = sp_mat[user, u_nzi].todense().A[0]

            if (row_cnt / total_rows) <= self.tr_perc:
                tr_i.extend(u_nzi)
                tr_u.extend(u)
                tr_vals.extend(vals)
            elif ((row_cnt / total_rows) - self.tr_perc) <= self.val_perc:
                val_i.extend(u_nzi)
                val_u.extend(u)
                val_vals.extend(vals)
            else:
                te_i.extend(u_nzi)
                te_u.extend(u)
                te_vals.extend(vals)

            row_cnt += row_cnts[user]

        tr_mat = scipy.sparse.csr_matrix(
            (tr_vals, (tr_u, tr_i)), shape=shape, dtype=dtype,
        )

        val_mat = scipy.sparse.csr_matrix(
            (val_vals, (val_u, val_i)), shape=shape, dtype=dtype
        )

        te_mat = scipy.sparse.csr_matrix(
            (te_vals, (te_u, te_i)), shape=shape, dtype=dtype
        )

        return tr_mat, val_mat, te_mat
