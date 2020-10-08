import numpy as np
from scipy.sparse import csr_matrix

import torch
from recpack.algorithms.base import Algorithm


class CML(Algorithm):
    """
    Pytorch Implementation of
    [1] Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    Version without features, referred to as CML in the paper.
    """

    def __init__(
        self,
        embedding_dim,
        margin,
        learning_rate,
        clip_norm,
        use_rank_weight,
        use_cov_loss,
    ):
        pass


class CMLWithFeatures(Algorithm):
    """
    Pytorch Implementation of
    [1] Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    Version with features, referred to as CML+F in the paper.
    """

    def __init__(
        self,
        embedding_dim,
        margin,
        learning_rate,
        clip_norm,
        use_rank_weight,
        use_cov_loss,
        hidden_layer_dim,
        feature_l2_reg,
        feature_proj_scaling_factor,
    ):
        pass


# TODO Integrate sampling methods somewhere more logical
def warp_sample_pairs(X: csr_matrix, U=10, batch_size=100) -> torch.LongTensor:
    """
    Sample U negatives for every user-item-pair (positive).

    :param X: Interaction matrix
    :type X: csr_matrix
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :yield: tensor of shape (batch_size, U+1), with user, positive item, U negative items for each row.
    :rtype: torch.LongTensor
    """
    # Need positive and negative pair. Requires the existence of a positive for this item.
    positives = np.array(X.nonzero()).T  # As a (num_interactions, 2) numpy array
    num_positives = positives.shape[0]
    np.random.shuffle(positives)

    # TODO Could be better to only yield this when required, to keep the memory footprint low.

    for start in range(0, num_positives, batch_size):
        positives_batch = positives[start: start + batch_size]

        # Important only for final batch, if smaller than batch_size
        true_batch_size = min(batch_size, num_positives - start)

        negatives_batch = np.random.randint(0, X.shape[1], size=(true_batch_size, U))
        while True:
            # Fix the negatives that are equal to the positives, if there are any
            mask = np.apply_along_axis(lambda col: col == positives_batch[:, 1], 0, negatives_batch)
            num_incorrect = np.sum(mask)

            if num_incorrect > 0:
                new_negatives = np.random.randint(
                    0, X.shape[1], size=(num_incorrect,)
                )
                negatives_batch[mask] = new_negatives
            else:
                # Exit the while loop
                break

        sample_pairs_batch = np.empty((positives_batch.shape[0], U + 2))
        sample_pairs_batch[:, :2] = positives_batch
        sample_pairs_batch[:, 2:] = negatives_batch
        yield torch.LongTensor(sample_pairs_batch)
