# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from inspect import isgenerator
from itertools import islice
from typing import Iterator, List, Iterable, Union

import numpy as np
from scipy.sparse import csr_matrix
import torch

from recpack.matrix import InteractionMatrix, Matrix, to_binary


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


def naive_sparse2tensor(data: csr_matrix) -> torch.Tensor:
    """Naively converts sparse csr_matrix to torch Tensor.

    :param data: CSR matrix to convert
    :type data: csr_matrix
    :return: Torch Tensor representation of the matrix.
    :rtype: torch.Tensor
    """
    return torch.FloatTensor(data.toarray())


def naive_tensor2sparse(tensor: torch.Tensor) -> csr_matrix:
    """Converts torch Tensor to sparse csr_matrix.

    :param tensor: Torch Tensor representation of the matrix to convert.
    :type tensor: torch.Tensor
    :return: CSR matrix representation of the matrix.
    :rtype: csr_matrix
    """
    return csr_matrix(tensor.detach().numpy())


def get_users(data: Matrix) -> List[int]:
    return list(set(data.nonzero()[0]))


def csr_from_rows(rows: csr_matrix, row_indices: Union[List[int], np.ndarray], shape: tuple) -> csr_matrix:
    """Place a block of rows into an otherwise empty csr_matrix of the given shape.

    Row ``i`` of ``rows`` becomes row ``row_indices[i]`` of the result.
    The result is constructed directly from the CSR internals
    (data, indices and a recomputed indptr), which is much faster than
    allocating a full-size lil_matrix and assigning rows into it.

    :param rows: Matrix with the rows to place, shape (len(row_indices), shape[1]).
    :type rows: csr_matrix
    :param row_indices: For every row in ``rows``, the row index it should get
        in the output matrix. Indices may be in any order, but must be unique.
    :type row_indices: Union[List[int], np.ndarray]
    :param shape: Shape of the output matrix.
    :type shape: tuple
    :return: Matrix of the requested shape, with the given rows filled in
        and every other row empty.
    :rtype: csr_matrix
    """
    row_indices = np.asarray(row_indices)

    # The data arrays can only be reused directly
    # if they are ordered by output row.
    order = np.argsort(row_indices)
    if not np.array_equal(order, np.arange(len(row_indices))):
        rows = rows[order]
        row_indices = row_indices[order]

    row_counts = np.zeros(shape[0], dtype=rows.indptr.dtype)
    row_counts[row_indices] = np.diff(rows.indptr)

    indptr = np.zeros(shape[0] + 1, dtype=rows.indptr.dtype)
    np.cumsum(row_counts, out=indptr[1:])

    return csr_matrix((rows.data, rows.indices, indptr), shape=shape)


def get_batches(iterable: Iterable, batch_size=1000) -> Iterator[List]:
    """Get batches from an iterable.

    The final batch might contain less than batch_size entries, as it will be the remainder.

    :param iterable: List of values that will be split into batches of size `batch_size`
    :type iterable: Iterable
    :param batch_size: Size of each batch, defaults to 1000
    :type batch_size: int, optional
    :yield: Iterator of lists of values
    :rtype: Iterator[List]
    """
    if not isgenerator(iterable):
        iterable = iter(iterable)

    while True:

        batch = list(islice(iterable, 0, batch_size))
        if batch:
            yield batch
        else:
            break


def sample_rows(*args: Matrix, sample_size: int = 1000) -> List[Matrix]:
    """Samples rows from the matrices

    Rows are sampled from the nonzero rows in the first csr_matrix argument.
    The return value will contain a matrix for each of the matrix arguments, with only the sampled rows nonzero.

    :param sample_size: Number of rows to sample, defaults to 1000
    :type sample_size: int, optional
    :return: List of all matrices passed as args
    :rtype: List[Matrix]
    """
    nonzero_users = list(set(args[0].nonzero()[0]))
    users = np.random.choice(nonzero_users, size=min(sample_size, len(nonzero_users)), replace=False)
    sampled_matrices = []

    for mat in args:
        if type(mat) == InteractionMatrix:
            sampled_mat = mat.users_in(users)
        else:
            sampled_mat = csr_from_rows(mat[users, :], users, mat.shape)

        sampled_matrices.append(sampled_mat)

    return sampled_matrices


def union_csr_matrices(a: csr_matrix, b: csr_matrix) -> csr_matrix:
    """Combine entries of 2 binary csr_matrices.


    :param a: Binary csr_matrix
    :type a: csr_matrix
    :param b: Binary csr_matrix
    :type b: csr_matrix
    :return: The union of a and b
    :rtype csr_matrix:
    """
    return to_binary(a + b)


def invert(x: Union[np.ndarray, csr_matrix]) -> Union[np.ndarray, csr_matrix]:
    """Invert an array.

    :param x: [description]
    :type x: [type]
    :return: [description]
    :rtype: [type]
    """
    if isinstance(x, np.ndarray):
        ret = np.zeros(x.shape)
    elif isinstance(x, csr_matrix):
        ret = csr_matrix(x.shape)
    else:
        raise TypeError("Unsupported type for argument x.")
    ret[x.nonzero()] = 1 / x[x.nonzero()]
    return ret
