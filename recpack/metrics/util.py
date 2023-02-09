# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from scipy.sparse import csr_matrix


def sparse_inverse_nonzero(a: csr_matrix) -> csr_matrix:
    """Invert nonzero elements of a `scipy.sparse.csr_matrix`.

    :param a: Matrix to invert.
    :type a: csr_matrix
    :return: Matrix with nonzero elements inverted.
    :rtype: csr_matrix
    """
    inv_a = a.copy()
    inv_a.data = 1 / inv_a.data
    return inv_a


def sparse_divide_nonzero(a: csr_matrix, b: csr_matrix) -> csr_matrix:
    """Elementwise divide of nonzero elements of a by nonzero elements of b.

    Elements that were zero in either a or b are zero in the resulting matrix.

    :param a: Numerator.
    :type a: csr_matrix
    :param b: Denominator.
    :type b: csr_matrix
    :return: Result of the elementwise division of matrix a by matrix b.
    :rtype: csr_matrix
    """
    return a.multiply(sparse_inverse_nonzero(b))
