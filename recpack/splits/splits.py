import numpy
import scipy.sparse


class TrainTestSplit:
    def __init__(self):
        pass

    def split(self):
        pass


class StrongGeneralization(TrainTestSplit):
    def __init__(self, tr_perc=0.7):
        self.tr_perc = tr_perc

    def split(self, sp_mat):

        shape = sp_mat.shape
        dtype = sp_mat.dtype

        nonzero_users = list(set(sp_mat.nonzero()[0]))
        numpy.random.shuffle(nonzero_users)
        total_rows = sp_mat.nnz

        row_cnt = 0

        row_cnts = sp_mat.sum(axis=1).A.T[0]

        tr_u, tr_i, tr_vals = [], [], []
        te_u, te_i, te_vals = [], [], []

        for ix, user in enumerate(nonzero_users):
            u_nzi = sp_mat[user].indices
            u = [user] * len(u_nzi)
            vals = sp_mat[user, u_nzi].todense().A[0]

            if row_cnt / total_rows >= self.tr_perc:
                te_i.extend(u_nzi)
                te_u.extend(u)
                te_vals.extend(vals)
            else:
                tr_i.extend(u_nzi)
                tr_u.extend(u)
                tr_vals.extend(vals)

            row_cnt += row_cnts[user]

        tr_mat = scipy.sparse.csr_matrix(
            (numpy.array(tr_vals), (numpy.array(tr_u), numpy.array(tr_i))),
            shape=shape,
            dtype=dtype,
        )
        te_mat = scipy.sparse.csr_matrix(
            (te_vals, (te_u, te_i)), shape=shape, dtype=dtype
        )

        return tr_mat, te_mat
