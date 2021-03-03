import logging
from math import ceil
import numpy as np
from scipy.sparse import csr_matrix
import torch


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


def naive_sparse2tensor(data: csr_matrix) -> torch.Tensor:
    """
    Naively converts sparse csr_matrix to torch Tensor.

    :param data: CSR matrix to convert
    :type data: csr_matrix
    :return: Torch Tensor representation of the matrix.
    :rtype: torch.Tensor
    """
    return torch.FloatTensor(data.toarray())


def naive_tensor2sparse(tensor: torch.Tensor) -> csr_matrix:
    """
    Converts torch Tensor to sparse csr_matrix.

    :param tensor: Torch Tensor representation of the matrix to convert.
    :type tensor: torch.Tensor
    :return: CSR matrix representation of the matrix.
    :rtype: csr_matrix
    """
    return csr_matrix(tensor.detach().numpy())


def get_users(data):
    return list(set(data.nonzero()[0]))


def get_batches(users, batch_size=1000):
    return [
        users[i * batch_size : min((i * batch_size) + batch_size, len(users))]
        for i in range(ceil(len(users) / batch_size))
    ]


def sample(*args: csr_matrix, sample_size: int = 1000):
    """Samples rows from the matrices

    rows are sampled from the nonzero rows in the first csr_matrix argument.
    The return value will contain a matrix for each of the matrix arguments, with only the sampled rows nonzero.
    """
    nonzero_users = list(set(args[0].nonzero()[0]))
    users = np.random.choice(
        nonzero_users, size=min(sample_size, len(nonzero_users)), replace=False
    )

    sampled_matrices = []

    for mat in args:
        sampled_mat = csr_matrix(mat.shape)
        sampled_mat[users, :] = mat[users, :]

        sampled_matrices.append(sampled_mat)

    return sampled_matrices
