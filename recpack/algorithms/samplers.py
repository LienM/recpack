import numpy as np
from scipy.sparse import csr_matrix
import torch


def bootstrap_sample_pairs(
    X: csr_matrix, batch_size=100, sample_size=None
) -> torch.LongTensor:
    """bootstrap sample triples from the data. Each triple contains (user, positive item, negative item).

    :param X: Interaction matrix
    :type X: csr_matrix
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :param sample_size: The number of samples to generate, defaults to None,
                    if it is None, it is set to the number of positive samples in X
    :type sample_size: int, optional
    :yield: tensor of shape (batch_size, 3), with user, positive item, negative item for each row.
    :rtype: torch.LongTensor
    """
    if sample_size is None:
        sample_size = X.nnz
    # Need positive and negative pair. Requires the existence of a positive for this item.
    positives = np.array(X.nonzero()).T  # As a (num_interactions, 2) numpy array
    num_positives = positives.shape[0]
    np.random.shuffle(positives)

    # Pick interactions at random, with replacement
    samples = np.random.choice(num_positives, size=(sample_size,), replace=True)

    # TODO Could be better to only yield this when required, to keep the memory footprint low.
    possible_negatives = np.random.randint(0, X.shape[1], size=(sample_size,))

    for start in range(0, sample_size, batch_size):
        sample_batch = samples[start : start + batch_size]
        positives_batch = positives[sample_batch]
        negatives_batch = possible_negatives[start : start + batch_size]
        while True:
            # Fix the negatives that are equal to the positives, if there are any
            mask = positives_batch[:, 1] == negatives_batch
            num_incorrect = np.sum(mask)

            if num_incorrect > 0:
                new_negatives = np.random.randint(0, X.shape[1], size=(num_incorrect,))
                negatives_batch[mask] = new_negatives
            else:
                # Exit the while loop
                break

        sample_pairs_batch = np.empty((positives_batch.shape[0], 3))
        sample_pairs_batch[:, :2] = positives_batch
        sample_pairs_batch[:, 2] = negatives_batch
        yield torch.LongTensor(sample_pairs_batch)


def warp_sample_pairs(X: csr_matrix, U=10, batch_size=100):
    """
    Sample U negatives for every user-item-pair (positive).

    :param X: Interaction matrix
    :type X: csr_matrix
    :param U: Number of negative samples for each positive, defaults to 10
    :type U: int, optional
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :yield: Iterator of torch.LongTensor of shape (batch_size, U+2). [User, Item, Negative Sample1, Negative Sample2, ...]
    :rtype: Iterator[torch.LongTensor]
    """
    # Need positive and negative pair. Requires the existence of a positive for this item.
    positives = np.array(X.nonzero()).T  # As a (num_interactions, 2) numpy array
    num_positives = positives.shape[0]
    np.random.shuffle(positives)

    for start in range(0, num_positives, batch_size):
        batch = positives[start : start + batch_size]
        users = batch[:, 0]
        positives_batch = batch[:, 1]

        # Important only for final batch, if smaller than batch_size
        true_batch_size = min(batch_size, num_positives - start)

        negatives_batch = np.random.randint(0, X.shape[1], size=(true_batch_size, U))
        while True:
            # Fix the negatives that are equal to the positives, if there are any
            mask = np.apply_along_axis(
                lambda col: col == positives_batch, 0, negatives_batch
            )
            num_incorrect = np.sum(mask)

            if num_incorrect > 0:
                new_negatives = np.random.randint(0, X.shape[1], size=(num_incorrect,))
                negatives_batch[mask] = new_negatives
            else:
                # Exit the while loop
                break

        yield torch.LongTensor(users), torch.LongTensor(
            positives_batch
        ), torch.LongTensor(negatives_batch)
