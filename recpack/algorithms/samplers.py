import numpy as np
from scipy.sparse import csr_matrix
import torch

from recpack.data.matrix import to_binary


def bootstrap_sample_pairs(
    X: csr_matrix, batch_size=100, sample_size=None, exact=False
):
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
    print(len(samples))
    # TODO Could be better to only yield this when required, to keep the memory footprint low.
    possible_negatives = np.random.randint(0, X.shape[1], size=(sample_size,))
    for start in range(0, sample_size, batch_size):
        sample_batch = samples[start : start + batch_size]
        positives_batch = positives[sample_batch]
        negatives_batch = possible_negatives[start : start + batch_size]
        while True:
            # Fix the negatives that are equal to the positives, if there are any
            if not exact:
                # Do so approximately. Rely on sparsity of the positives matrix to ensure collisions are rare.
                negatives_mask = positives_batch[:, 1] == negatives_batch
                num_incorrect = np.sum(negatives_mask)
            else:
                # Eliminate the collisions, exactly.
                # Turn this batch of negatives into a csr_matrix
                negatives_batch_csr = csr_matrix(
                    (
                        np.ones(negatives_batch.shape[0]),
                        (positives_batch[:, 0], negatives_batch),
                    ),
                    X.shape,
                )

                # negatives_batch_csr[negatives_batch_csr > 1] = 1
                # print(negatives_batch_csr.toarray())
                # Detect differences between the batch of negatives and X.
                # Ideally, every negative sample should be different from samples in X (no collisions).
                negative_samples_mask = negatives_batch_csr.astype(bool)
                match_or_mismatch = to_binary(negatives_batch_csr) != X
                # If there are no false negatives, all values in false_negatives should be True.
                false_negatives = np.bitwise_not(
                    match_or_mismatch[negative_samples_mask]
                )
                # print("With binary negatives", false_negatives)
                # Count all locations where false_negatives = True
                # print("Values", negatives_batch_csr[negative_samples_mask])
                num_incorrect = int(
                    np.sum(
                        false_negatives[0]
                        @ negatives_batch_csr[negative_samples_mask][0].T.astype(bool)
                    )
                )
                # print(num_incorrect)
                # Initialize mask to all zeros = all False
                negatives_mask = np.zeros(negatives_batch.shape).astype(bool)
                # Get the indices of the false_negatives
                _, false_negative_indices_csr = false_negatives.nonzero()
                # Get the corresponding false negative pairs
                # Assumes the order of samples in false_negatives is the same as in negative_samples_mask
                false_negative_pairs = list(zip(*negative_samples_mask.nonzero()))
                # Get the originally sampled negative pairs, in the batch order
                negative_pairs = np.vstack([positives_batch[:, 0], negatives_batch]).T
                for i in false_negative_indices_csr:
                    # Find the corresponding row in  negative_pairs
                    a = np.all(negative_pairs == false_negative_pairs[i], axis=1)
                    negative_mask_row_indices = a.nonzero()
                    # Set these rows (most of the time should be one row) to True
                    negatives_mask[negative_mask_row_indices] = True
            if num_incorrect > 0:
                new_negatives = np.random.randint(0, X.shape[1], size=(num_incorrect,))
                negatives_batch[negatives_mask] = new_negatives
            else:
                # Exit the while loop
                break
        sample_pairs_batch = np.empty((positives_batch.shape[0], 3))
        sample_pairs_batch[:, :2] = positives_batch
        sample_pairs_batch[:, 2] = negatives_batch
        yield torch.LongTensor(sample_pairs_batch)


def warp_sample_pairs(X: csr_matrix, U=10, batch_size=100, exact=False):
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

        if exact:
            negatives_batch = np.empty((true_batch_size, U))
            for i, u in enumerate(users):
                # sample a negative item for the user
                user_vector = X[u]
                negative_item_ix = (user_vector == 0).nonzero()[1]

                neg = np.random.choice(negative_item_ix, U, replace=False)
                negatives_batch[i] = neg

        else:
            negatives_batch = np.random.randint(
                0, X.shape[1], size=(true_batch_size, U)
            )
            while True:
                # Fix the negatives that are equal to the positives, if there are any
                mask = np.apply_along_axis(
                    lambda col: col == positives_batch, 0, negatives_batch
                )
                num_incorrect = np.sum(mask)

                if num_incorrect > 0:
                    new_negatives = np.random.randint(
                        0, X.shape[1], size=(num_incorrect,)
                    )
                    negatives_batch[mask] = new_negatives
                else:
                    # Exit the while loop
                    break

        yield torch.LongTensor(users), torch.LongTensor(
            positives_batch
        ), torch.LongTensor(negatives_batch)
