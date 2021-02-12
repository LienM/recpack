"""
Data format conversion and manipulation
"""
import numpy as np
import pandas as pd
import torch

from recpack.data.data_matrix import DataM, USER_IX, ITEM_IX, TIMESTAMP_IX
from torch import Tensor
from typing import Tuple


def data_m_to_tensor(
    data_m: DataM,
    batch_size: int,
    device: str = "cpu",
    shuffle: bool = False,
    include_last: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Converts a data matrix with interactions to torch tensors.

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
    df = data_m.dataframe
    if shuffle:
        df = shuffle_and_sort(df)
    else:
        df = df.sort_values(by=[USER_IX, TIMESTAMP_IX], ascending=True)
    iids = torch.tensor(df[ITEM_IX].to_numpy(), device=device)
    uids = torch.tensor(df[USER_IX].to_numpy(), device=device)
    # Drop the last action of each user if include_last is false
    if include_last:
        actions = iids
        targets = iids.roll(-1, dims=0)
    else:
        true = torch.tensor([True], device=device)
        is_first = torch.cat((true, uids[1:] != uids[:-1]))
        is_last = torch.cat((uids[:-1] != uids[1:], true))
        actions = iids[~is_last]
        targets = iids[~is_first]
        uids = uids[~is_last]
    # Create user-parallel mini batches
    return (
        batchify(actions, batch_size),
        batchify(targets, batch_size),
        batchify(uids, batch_size),
    )


def shuffle_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shuffle sessions but keep all interactions for a session together, sorted
    by timestamp.
    """
    # Generate a unique random number for each session/user
    uuid = df[USER_IX].unique()
    rand = pd.Series(data=np.random.permutation(len(uuid)), index=uuid, name="rand")
    df = df.join(rand, on=USER_IX)
    # Shuffle sessions by sorting on their random number
    df = df.sort_values(by=["rand", TIMESTAMP_IX], ascending=True, ignore_index=True)
    del df["rand"]
    return df


def batchify(data: Tensor, batch_size: int) -> Tensor:
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
