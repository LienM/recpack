import scipy.sparse
import numpy as np

def get_topK(X_pred: scipy.sparse.csr_matrix, K: int) -> scipy.sparse.csr_matrix:
    # Get nonzero users
    nonzero_users = list(set(X_pred.nonzero()[0]))
    X = X_pred[nonzero_users, :].toarray()

    items = np.argpartition(X, -K)[:, -K:]

    U, I, V = [], [], []

    for ix, user in enumerate(nonzero_users):
        U.extend([user] * K)
        I.extend(items[ix])
        V.extend(X[ix, items[ix]])

    X_pred_top_K = scipy.sparse.csr_matrix(
        (V, (U, I)), dtype=X_pred.dtype, shape=X_pred.shape
    )

    return X_pred_top_K