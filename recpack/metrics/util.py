

def sparse_inverse_nonzero(a):
    inv_a = a.copy()
    inv_a.data = 1 / inv_a.data
    return inv_a


def sparse_divide_nonzero(a, b):
    return a.multiply(sparse_inverse_nonzero(b))
