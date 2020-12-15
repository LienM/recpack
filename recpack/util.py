from collections import defaultdict
import logging

import numpy as np
import scipy.sparse


logger = logging.getLogger("recpack")


def groupby2(keys, values):
    """ A group by of separate lists where order doesn't matter. """
    multidict = defaultdict(list)
    for k, v in zip(keys, values):
        multidict[k].append(v)
    return multidict.items()


def to_tuple(el):
    """ Whether single element or tuple, always returns as tuple. """
    if type(el) == tuple:
        return el
    else:
        return (el,)


def df_to_sparse(df, item_ix, user_ix, value_ix=None, shape=None):
    if value_ix is not None and value_ix in df:
        values = df[value_ix]
    else:
        if value_ix is not None:
            # value_ix provided, but not in df
            logger.warning(
                f"Value column {value_ix} not found in dataframe. Using ones instead."
            )

        num_entries = df.shape[0]
        # Scipy sums up the entries when an index-pair occurs more than once,
        # resulting in the actual counts being stored. Neat!
        values = np.ones(num_entries)

    indices = list(zip(*df.loc[:, [user_ix, item_ix]].values))

    if indices == []:
        indices = [[], []]  # Empty zip does not evaluate right

    if shape is None:
        shape = df[user_ix].max() + 1, df[item_ix].max() + 1
    sparse_matrix = scipy.sparse.csr_matrix(
        (values, indices), shape=shape, dtype=values.dtype
    )

    return sparse_matrix


def get_top_K_ranks(data: scipy.sparse.csr_matrix, k: int = None) -> scipy.sparse.csr_matrix:
    """
    Return csr_matrix of top K item ranks for every user.

    :param data: Predicted affinity of users for items.
    :type data: csr_matrix
    :param k: Value for K; k could be None.
    :type k: int or None
    :return: Sparse matrix containing ranks of top K predictions.
    :rtype: csr_matrix
    """
    U, I, V = [], [], []
    for row_ix, (le, ri) in enumerate(
            zip(data.indptr[:-1], data.indptr[1:])):
        K_row_pick = min(k, ri - le) if k is not None else ri-le

        if K_row_pick != 0:

            top_k_row = data.indices[
                le
                + np.argpartition(data.data[le:ri], list(range(-K_row_pick, 0)))[
                    -K_row_pick:
                ]
            ]

            for rank, col_ix in enumerate(reversed(top_k_row)):
                U.append(row_ix)
                I.append(col_ix)
                V.append(rank + 1)

    data_top_K = scipy.sparse.csr_matrix((V, (U, I)), shape=data.shape)

    return data_top_K


def get_top_K_values(data: scipy.sparse.csr_matrix, k: int = None) -> scipy.sparse.csr_matrix:
    """
    Return csr_matrix of top K items for every user. Which is equal to the K nearest neighbours.
    @param data: Predicted affinity of users for items.
    @param k: Value for K; k could be None.
    @return: Sparse matrix containing values of top K predictions.
    """
    top_K_ranks = get_top_K_ranks(data, k)
    top_K_ranks[top_K_ranks > 0] = 1  # ranks to binary

    return top_K_ranks.multiply(data)  # elementwise multiplication
