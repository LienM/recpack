import numpy as np
import scipy.sparse


def sparse_divide_nonzero(a, b):
    inv_b = b.copy()
    inv_b.data = 1 / inv_b.data
    return a.multiply(inv_b)
