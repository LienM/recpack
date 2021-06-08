
from math import ceil
from typing import Iterator, List, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.sparse import csr_matrix
import torch


from recpack.data.matrix import InteractionMatrix, Matrix


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


def get_users(data: Matrix) -> list:
    return list(set(data.nonzero()[0]))


def get_batches(users: List[int], batch_size=1000) -> Iterator[List[int]]:
    """Get user ids in batches from a list of ids.

    The list of users will be split into batches of batch_size.
    The final batch might contain less users, as it will be the remainder.

    :param users: list of user ids that will be split
    :type users: List[int]
    :param batch_size: Size of each batch, defaults to 1000
    :type batch_size: int, optional
    :yield: Iterator of lists of users
    :rtype: Iterator[List[int]]
    """
    for i in range(ceil(len(users) / batch_size)):
        yield users[i * batch_size: min((i * batch_size) + batch_size, len(users))]


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
    users = np.random.choice(
        nonzero_users, size=min(sample_size, len(nonzero_users)), replace=False
    )

    sampled_matrices = []

    for mat in args:
        sampled_mat = csr_matrix(mat.shape)
        sampled_mat[users, :] = mat[users, :]

        sampled_matrices.append(sampled_mat)

    return sampled_matrices


def matrix_to_tensor(
    X: InteractionMatrix,
    batch_size: int,
    device: str = "cpu",
    shuffle: bool = False,
    include_last: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a user-item interactions matrix to torch tensors.

    Information about the interactions is split over three different tensors
    of identical shape, containing item id, next item id and user id of the
    interaction. Interactions are grouped by user and ordered by time along
    the first dimension of the tensors. The second dimension corresponds to
    the batch size.

    As an example, take following interactions (already grouped by user and
    ordered by time for clarity)::

        time      0 1 2  3 4  5 6 7  8 9 10
        iid       0 1 2  3 4  5 6 7  8 9 10
        uid       0 0 0  1 1  2 2 2  3 3 3

    The last interaction of a user is never used as an input during training
    because we don't know what the next action will be. Removing it, we get::

        time      0 1  3  5 6  8 9
        iid_in    0 1  3  5 6  8 9
        iid_next  1 2  4  6 7  9 10
        uid       0 0  1  2 2  3 3
                        ^       ^
    These interactions are turned into tensors by cutting them in ``batch_size``
    pieces of equal length, which form the columns of the tensors::

        iid_in        iid_next      uid

        [[0, 5],      [[1, 6],      [[0, 2],
         [1, 6],       [2, 7],       [0, 2],
         [3, 8]]   ,   [4, 9]]   ,   [1, 3]]

    Note that these are simply the interactions from above, laid out column by
    column. Each row is a mini-batch, containing interactions that took place
    after the corresponding user's interaction in the previous row. Using this
    format a recurrent network can process many user histories in parallel.

    The last interaction (iid_in: 9, uid: 3) is discarded because the number of
    interactions (7) is not a multiple of the batch size (2). Some users may
    also have their interactions split across different columns. This has neg-
    ligible impact on training when using any moderately large dataset, but may
    be undesirable for evaluation. In that case a batch size of 1 can be used.

    :param data_m: DataM object to be converted to tensors, must have timestamps
    :param batch_size: Number of actions per batch. If the number of actions is
        not divisible by batch_size, up to (batch_size - 1) actions will be
        dropped. In cases where this is unacceptable, use a batch_size of 1.
    :param device: Torch device to store the tensors on.
    :param shuffle: Randomize the position of users/sessions in the tensors. If
        False, actions will be ordered by user id.
    :param include_last: Whether to include the last interaction of each user.
        If true, the value of the next item id is undefined for the last action.
    :return: Three tensors containing input item ids, next item ids and user ids,
        respectively. All of shape (N, B), where B is the batch size and N is
        the number of complete batches that can be created.
    """
    # Convert the item and user ids to 1D tensors

    sorted_item_histories = list(X.sorted_item_history)
    sorted(sorted_item_histories)

    if include_last:
        w = torch.LongTensor([
            [u, i]
            for u, sequence in sorted_item_histories
            for i in sequence
        ])

        uids = w[:, 0]
        actions = w[:, 1]
        # Not used but just for consistency
        targets = w[:, 1]

        batched_actions = batchify(actions, batch_size)
        batched_targets = batchify(targets, batch_size)
        batched_uids = batchify(uids, batch_size)
        is_last_action = is_last_action = batched_uids != batched_uids.roll(
            -1, dims=0)
        is_last_action[-1] = True

    else:
        w = torch.LongTensor([
            [u] + w.tolist()
            for u, sequence in sorted_item_histories
            if len(sequence) >= 2
            for w in sliding_window_view(sequence, 2)
        ])

        uids = w[:, 0]
        actions = w[:, 1]
        targets = w[:, 2]

        batched_actions = batchify(actions, batch_size)
        batched_targets = batchify(targets, batch_size)
        batched_uids = batchify(uids, batch_size)
        is_last_action = is_last_action = batched_uids != batched_uids.roll(
            -1, dims=0)
        is_last_action[-1] = True

    return (
        batched_actions,
        batched_targets,
        batched_uids,
        is_last_action
    )


# TODO Understand how this works
def batchify(data: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Splits a sequence into contiguous batches, indexed along dim 0.

    :param data: One-dimensional tensor to be split into batches
    :param batch_size: How many elements per batch
    :return: A tensor of shape (N, B), where B is the batch size and N is the number
        of complete batches that can be created
    """
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t()
    data = data.contiguous()
    return data
