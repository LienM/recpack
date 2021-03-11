from typing import Tuple, Iterator
import numpy as np
from scipy.sparse import csr_matrix
import torch

from recpack.data.matrix import to_binary

# TODO: Add to docs!
# TODO: consider allowing bootstrap to also sample multiple negatives
#  -> Rename some functions, bootstrap indicates resampling possible
#     but the warp method just takes every positive sample,
#     and attaches a negative one


def _spot_collisions(
    users: np.array, negatives_batch: np.array, X: csr_matrix
) -> Tuple[int, np.array]:
    """Spot collisions between the negative samples and the interactions in X.

    :param users: Ordered batch of users
    :type users: np.array
    :param negatives_batch: Ordered batch of negative items
    :type negatives_batch: np.array
    :param X: Entirety of all user interactions
    :type X: csr_matrix
    :return: Tuple containing the number of incorrect negative samples,
        and the locations of these incorrect samples in the batch array
    :rtype: Tuple[int, np.array]
    """
    # Eliminate the collisions, exactly.
    # Turn this batch of negatives into a csr_matrix
    negatives_batch_csr = csr_matrix(
        (
            np.ones(negatives_batch.shape[0]),
            (users, negatives_batch),
        ),
        X.shape,
    )
    # Detect differences between the batch of negatives and X.
    # Ideally, every negative sample should be different from samples in X
    # (no collisions).
    negative_samples_mask = negatives_batch_csr.astype(bool)
    match_or_mismatch = to_binary(negatives_batch_csr) != X
    # If there are no false negatives, all values in false_negatives should be True.
    false_negatives = np.bitwise_not(match_or_mismatch[negative_samples_mask])

    # Initialize mask to all zeros = all False
    negatives_mask = np.zeros(negatives_batch.shape).astype(bool)
    # Get the indices of the false_negatives
    _, false_negative_indices_csr = false_negatives.nonzero()
    # Get the corresponding false negative pairs
    # Assumes the order of samples in false_negatives
    # is the same as in negative_samples_mask
    false_negative_pairs = list(zip(*negative_samples_mask.nonzero()))
    # Get the originally sampled negative pairs, in the batch order
    negative_pairs = np.vstack([users, negatives_batch]).T
    for i in false_negative_indices_csr:
        # Find the corresponding row in negative_pairs
        a = np.all(negative_pairs == false_negative_pairs[i], axis=1)
        negative_mask_row_indices = a.nonzero()
        # Set these rows (most of the time should be one row) to True
        negatives_mask[negative_mask_row_indices] = True

    num_incorrect = negatives_mask.sum()
    return num_incorrect, negatives_mask


def bootstrap_sample_pairs(
    X: csr_matrix, batch_size=100, sample_size=None, exact=False
) -> Iterator[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]:
    """bootstrap sample triples from the data.

    Triples are split over three tensors, a user tensor, a positive item tensor,
    and a negative item tensor.

    :param X: Interaction matrix
    :type X: csr_matrix
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :param sample_size: The number of samples to generate, defaults to None,
                    if it is None, it is set to the number of positive samples in X
    :type sample_size: int, optional
    :param exact: if False (default) negatives are checked agains the corresponding
        positive sample only, allowing for (rare) collisions.
        If collisions should be avoided at all costs, use exact = True,
        but suffer decreased performance.
    :type exact: bool, optional
    :yield: Iterator of (user_batch, positive_samples_batch, negative_samples_batch)
    :rtype: Iterator[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]
    """
    if sample_size is None:
        sample_size = X.nnz
    # Need positive and negative pair.
    #   Requires the existence of a positive for this item.
    # As a (num_interactions, 2) numpy array
    positives = np.array(X.nonzero()).T
    num_positives = positives.shape[0]
    np.random.shuffle(positives)
    # Pick interactions at random, with replacement
    samples = np.random.choice(num_positives, size=(sample_size,), replace=True)

    possible_negatives = np.random.randint(0, X.shape[1], size=(sample_size,))
    for start in range(0, sample_size, batch_size):
        sample_batch = samples[start : start + batch_size]

        batch = positives[sample_batch]
        users = batch[:, 0]
        positives_batch = batch[:, 1]
        negatives_batch = possible_negatives[start : start + batch_size]
        while True:
            # Fix the negatives that are equal to the positives, if there are any
            if not exact:
                # Do so approximately. Rely on sparsity
                # of the positives matrix to ensure collisions are rare.
                negatives_mask = positives_batch == negatives_batch
                num_incorrect = np.sum(negatives_mask)
            else:
                num_incorrect, negatives_mask = _spot_collisions(
                    users, negatives_batch, X
                )

            if num_incorrect > 0:
                new_negatives = np.random.randint(0, X.shape[1], size=(num_incorrect,))
                negatives_batch[negatives_mask] = new_negatives
            else:
                # Exit the while loop
                break

        yield torch.LongTensor(users), torch.LongTensor(
            positives_batch
        ), torch.LongTensor(negatives_batch)


def warp_sample_pairs(
    X: csr_matrix, U=10, batch_size=100, exact=False
) -> Iterator[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]:
    """Sample U negatives for every user-item-pair (positive).

    :param X: Interaction matrix
    :type X: csr_matrix
    :param U: Number of negative samples for each positive, defaults to 10
    :type U: int, optional
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :param exact: if False (default) negatives are checked agains the corresponding
        positive sample only, allowing for (rare) collisions.
        If collisions should be avoided at all costs, use exact = True,
        but suffer decreased performance.
    :type exact: bool, optional
    :yield: Iterator of (user_batch, positive_samples_batch, negative_samples_batch)
    :rtype: Iterator[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]
    """
    # Need positive and negative pair.
    # Requires the existence of a positive for this item.
    # As a (num_interactions, 2) numpy array
    positives = np.array(X.nonzero()).T
    num_positives = positives.shape[0]
    np.random.shuffle(positives)

    for start in range(0, num_positives, batch_size):
        batch = positives[start : start + batch_size]
        users = batch[:, 0]
        positives_batch = batch[:, 1]

        # Important only for final batch, if smaller than batch_size
        true_batch_size = min(batch_size, num_positives - start)

        if not exact:
            negatives_batch = np.random.randint(
                0, X.shape[1], size=(true_batch_size, U)
            )
            while True:
                # Approximately fix the negatives that are equal to the positives, if there are any
                # Assumes collisions are rare
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
        else:
            negatives_batch = np.zeros((true_batch_size, U))
            for i in range(0, U):
                # Construct column i in the negatives matrix

                # 1st try true random
                # We will fix collisions in while loop
                negatives_batch_col_i = np.random.randint(
                    0, X.shape[1], size=(true_batch_size,)
                )
                while True:

                    num_incorrect, negatives_mask = _spot_collisions(
                        users, negatives_batch_col_i, X
                    )

                    # Check column against previous columns
                    additional_mask = np.zeros(true_batch_size, dtype=bool)
                    for j in range(0, i):
                        additional_mask += (
                            negatives_batch_col_i == negatives_batch[:, j]
                        )

                    total_mask = negatives_mask + additional_mask
                    num_incorrect = total_mask.sum()

                    if num_incorrect > 0:
                        new_negatives = np.random.randint(
                            0, X.shape[1], size=(num_incorrect,)
                        )
                        negatives_batch_col_i[total_mask] = new_negatives

                    else:
                        # Exit the while(True) loop
                        break

                negatives_batch[:, i] = negatives_batch_col_i

        yield torch.LongTensor(users), torch.LongTensor(
            positives_batch
        ), torch.LongTensor(negatives_batch)
