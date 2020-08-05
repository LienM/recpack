from collections import defaultdict
import logging

from joblib import Parallel, delayed
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


def parfor(iterator, parallel=True, n_jobs=-1, **kwargs):
    """
    Decorator to turn a for loop into a parallel call from joblib.
    The following code fragments are identical:

    l = list()
    for i in range(5):
        l.append(i)
    # l = [0, 1, 2, 3, 4]

    @parfor(range(5))
    def l(i):
        return i
    # l = [0, 1, 2, 3, 4]

    Do note that, depending on the backend, changes to variables outside the loop will not be reflected!

    Example usage:
    @parfor(enumerate(range(5, 50)))
    def f(i, c):
        time.sleep(1)
        print(i, c)
        return


    @parfor(range(10), parallel=False)
    def f(i):
        time.sleep(1)
        print(i)
    """

    def decorator(body):

        if parallel:
            data = Parallel(n_jobs=n_jobs, **kwargs)(
                delayed(body)(*to_tuple(args)) for args in iterator
            )
            return data
        else:
            # could replace manually doing this with backend=sequential
            data = list()
            for args in iterator:
                data.append(body(*to_tuple(args)))

        return data

    return decorator


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
