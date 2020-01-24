from collections import defaultdict
from itertools import groupby

import numpy
import scipy.sparse


class Evaluator:
    pass


class LeavePOutCrossValidation(Evaluator):

    def __init__(self, p, sp_mat=None, shape=None):
        self.p = p
        self.sp_mat = sp_mat


class FoldInPercentage(Evaluator):
    class FoldIterator:
        def __init__(self, fold_in_percentage):
            self._fip = fold_in_percentage
            self._index = 0
            self._max_index = fold_in_percentage.sp_mat_tr.shape[0]

        def __next__(self):
            while(True):
                if self._index < self._max_index:
                    fold_in = self._fip.sp_mat_tr[self._index]
                    fold_out = self._fip.sp_mat_te[self._index]

                    if fold_in.nnz == 0 or fold_out.nnz == 0:
                        self._index += 1
                        continue

                    self._index += 1

                    return fold_in[0], fold_out[0]
                raise StopIteration

    def __init__(self, fold_in, sp_mat=None, shape=None, seed=None):
        self.fold_in = fold_in

        self.sp_mat_tr = None
        self.sp_mat_te = None

        if sp_mat is not None:
            sp_mat_tr, sp_mat_te = self.split(sp_mat, shape=shape)

        self.seed = seed if seed else 12345

    def split(self, sp_mat, shape=None):
        """
        Have to set shape specifically else it might turn out too small.
        """
        if not shape:
            shape = sp_mat.shape

        U, I, V = scipy.sparse.find(sp_mat)

        items_by_user_dct = defaultdict(list)

        # Unsure if U and I are sorted, which is a requirement for itertools.groupby. So add them to a defaultdict to be sure
        groups = groupby(zip(U, I, V), lambda x: x[0])
        for key, subiter in groups:
            items_by_user_dct[key].extend(subiter)

        in_fold, out_fold = [], []

        for u in items_by_user_dct.keys():
            usr_hist = items_by_user_dct[u]

            _len = len(usr_hist)

            numpy.random.shuffle(usr_hist)

            cut = int(numpy.floor(_len * self.fold_in))

            in_fold.extend(usr_hist[:cut])
            out_fold.extend(usr_hist[cut:])

        U_tr, I_tr, V_tr = zip(*in_fold)

        sp_mat_tr = scipy.sparse.csr_matrix((V_tr, (U_tr, I_tr)), shape=shape)

        self.sp_mat_tr = sp_mat_tr

        U_te, I_te, V_te = zip(*out_fold)
        sp_mat_te = scipy.sparse.csr_matrix((V_te, (U_te, I_te)), shape=shape)

        self.sp_mat_te = sp_mat_te

        return sp_mat_tr, sp_mat_te

    def __iter__(self):
        return self.FoldIterator(self)
