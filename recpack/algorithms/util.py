from inspect import isgenerator
from itertools import islice
from typing import Iterable, Iterator, List, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
import torch

from recpack.data.matrix import InteractionMatrix, Matrix, to_binary


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


def get_batches(users: Iterable, batch_size=1000) -> Iterator[List]:
    """Get user ids in batches from a list of ids.

    The list of users will be split into batches of batch_size.
    The final batch might contain less users, as it will be the remainder.

    :param users: list of user ids that will be split
    :type users: Iterable
    :param batch_size: Size of each batch, defaults to 1000
    :type batch_size: int, optional
    :yield: Iterator of lists of users
    :rtype: Iterator[List]
    """
    if not isgenerator(users):
        users = iter(users)

    while True:

        batch = list(islice(users, 0, batch_size))
        if batch:
            yield batch
        else:
            break

        # start_ix = end_ix
        # end_ix += batch_size


def sample_rows(*args: csr_matrix, sample_size: int = 1000) -> List[csr_matrix]:
    """Samples rows from the matrices

    Rows are sampled from the nonzero rows in the first csr_matrix argument.
    The return value will contain a matrix for each of the matrix arguments, with only the sampled rows nonzero.

    :param sample_size: Number of rows to sample, defaults to 1000
    :type sample_size: int, optional
    :return: List of all matrices passed as args
    :rtype: List[csr_matrix]
    """
    nonzero_users = list(set(args[0].nonzero()[0]))
    users = np.random.choice(nonzero_users, size=min(sample_size, len(nonzero_users)), replace=False)

    sampled_matrices = []

    for mat in args:
        sampled_mat = csr_matrix(mat.shape)
        sampled_mat[users, :] = mat[users, :]

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


def create_front_padded_history_tensor(
    X: InteractionMatrix, padding_token: int, max_hist_length: Optional[int] = None
) -> torch.LongTensor:
    """Construct a sorted tensor from an interaction matrix, with the sorted interactions of the users.
    Padding is added to the front of the tensor to make sure all users have the same length vectors.

    :param X: InteracationMatrix to process into a tensor
    :type X: InteractionMatrix
    :param padding_token: The token to use as padding
    :type padding_token: int
    :param max_hist_length: Users with longer histories only retain their `max_hist_length` last ones,
        defaults to None
    :type max_hist_length: Optional[int], optional
    :return: tensor with item_ids and padding tokens for each user.
    :rtype: torch.LongTensor
    """
    if max_hist_length is None:
        max_hist_length = X.binary_values.sum(axis=1).max()
    history_tensor = torch.zeros((X.num_active_users, max_hist_length), dtype=int)
    history_tensor = torch.ones((X.num_active_users, max_hist_length), dtype=int) * padding_token
    for ix, pair in enumerate(X.sorted_item_history):
        _, sorted_history = pair
        hist_length = min(len(sorted_history), max_hist_length)
        history_tensor[ix, -hist_length:] = torch.LongTensor(sorted_history[-hist_length:])

    return history_tensor
