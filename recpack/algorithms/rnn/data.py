"""
Data conversion and manipulation
"""
import numpy as np
import pandas as pd
import torch

from recpack.data.data_matrix import DataM, USER_IX, ITEM_IX, TIMESTAMP_IX
from torch import Tensor
from typing import Tuple


def data_m_to_tensor(
    dm: DataM,
    batch_size: int,
    device: str = "cpu",
    shuffle: bool = False,
    include_last: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Converts a data matrix with timestamp information to tensor format.

    The 

    As an example, with following interactions (sorted by time):
        uid  0    1      2      3
        iid  0 1  2 3 4  5 6 7  8 9
    The returned tensors for a batch size of two would be
        [[0, 5],      [[1, 6],        [[0, 2],
         [2, 6],       [3, 7],         [1, 2],
         [3, 8]]   ,   [4, 9]]   and   [1, 3]]
    Containing input item ids, next item ids, and user/session ids. Grouped by
    users, ordered by time. Users at boundaries may be split across columns.
    """
    # Convert the item and user ids to 1D tensors
    df = dm.dataframe
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
