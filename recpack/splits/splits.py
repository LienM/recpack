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

        for ix, user in enumerate(nonzero_users):
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
