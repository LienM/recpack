from scipy.sparse import csr_matrix
import torch
import torch.nn as nn

from tqdm import tqdm

from recpack.algorithms.samplers import bootstrap_sample_pairs, warp_sample_pairs


def covariance_loss(H: nn.Embedding, W: nn.Embedding) -> torch.Tensor:
    # TODO Refactor so that it's no longer specific to CML
    """
    Implementation of covariance loss as described in
    Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    The loss term is used to penalize covariance between embedding dimensions and
    thus disentangle these embedding dimensions.

    It is assumed H and W are embeddings in the same space.

    :param H: Item embedding
    :type H: nn.Embedding
    :param W: User Embedding
    :type W: nn.Embedding
    :return: Covariance loss term
    :rtype: torch.Tensor
    """
    W_as_tensor = next(W.parameters())
    H_as_tensor = next(H.parameters())

    # Concatenate them together. They live in the same metric space, so share the same dimensions.
    #  X is a matrix of shape (|users| + |items|, num_dimensions)
    X = torch.cat([W_as_tensor, H_as_tensor], dim=0)

    # Zero mean
    X = X - X.mean(dim=0)

    cov = X.matmul(X.T)

    # Per element covariance, excluding the variance of individual random variables.
    return cov.fill_diagonal_(0).sum() / (X.shape[0] * X.shape[1])


def warp_loss(
    dist_pos_interaction: torch.Tensor,
    dist_neg_interaction: torch.Tensor,
    margin: float,
    J: int,
    U: int,
) -> torch.Tensor:
    """
    Implementation of
    WARP loss as described in
    Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf
    based on
    J. Weston, S. Bengio, and N. Usunier. Large scale image annotation:
    learning to rank with joint word-image embeddings. Machine learning, 81(1):21–35, 2010.

    Adds a loss penalty for every negative sample that is not at least
    an amount of margin further away from the reference sample than a positive
    sample. This per sample loss penalty has a weight proportional to the
    amount of samples in the negative sample batch were "misclassified",
    i.e. closer than the positive sample.

    :param dist_pos_interaction: Tensor of distances between positive sample and reference sample.
    :type dist_pos_interaction: torch.Tensor
    :param dist_neg_interaction: Tensor of distances between negatives samples and reference sample.
    :type dist_neg_interaction: torch.Tensor
    :param margin: Required margin between positive and negative sample.
    :type margin: float
    :param J: Total number of items in the dataset.
    :type J: int
    :param U: Number of negative samples used for every positive sample.
    :type U: int
    :return: 0-D Tensor containing WARP loss.
    :rtype: torch.Tensor
    """
    dist_diff_pos_neg_margin = margin + dist_pos_interaction - dist_neg_interaction

    # Largest number is "most wrongly classified", f.e.
    # pos = 0.1, margin = 0.1, neg = 0.15 => 0.1 + 0.1 - 0.15 = 0.05 > 0
    # pos = 0.1, margin = 0.1, neg = 0.08 => 0.1 + 0.1 - 0.08 = 0.12 > 0
    most_wrong_neg_interaction, _ = dist_diff_pos_neg_margin.max(dim=-1)

    most_wrong_neg_interaction[most_wrong_neg_interaction < 0] = 0

    M = (dist_diff_pos_neg_margin > 0).sum(axis=-1).float()
    # M * J / U =~ rank(pos_i)
    w = torch.log((M * J / U) + 1)

    loss = (most_wrong_neg_interaction * w).sum()

    return loss


def bpr_loss(positive_sim, negative_sim):
    distance = positive_sim - negative_sim
    # Probability of ranking given parameters
    elementwise_bpr_loss = torch.log(torch.sigmoid(distance))
    bpr_loss = -elementwise_bpr_loss.sum()

    return bpr_loss


def bpr_loss_metric(X_true: csr_matrix, X_pred: csr_matrix, batch_size=1000):
    """Compute BPR reconstruction loss of the X_true matrix in X_pred

    :param X_true: [description]
    :type X_true: [type]
    :param X_pred: [description]
    :type X_pred: [type]
    :raises an: [description]
    :return: [description]
    :rtype: [type]
    :yield: [description]
    :rtype: [type]
    """
    total_loss = 0

    for d in bootstrap_sample_pairs(
        X_true, batch_size=batch_size, sample_size=X_true.nnz
    ):
        # Needed to do copy, to use as index in the predidction matrix
        users = d[:, 0].numpy().copy()
        target_items = d[:, 1].numpy().copy()
        negative_items = d[:, 2].numpy().copy()

        positive_sim = torch.tensor(X_pred[users, target_items])
        negative_sim = torch.tensor(X_pred[users, negative_items])

        total_loss += bpr_loss(positive_sim, negative_sim).item()

    return total_loss


def warp_loss_metric(X_true: csr_matrix, X_pred: csr_matrix, batch_size: int = 1000, U: int = 20, margin: float = 1.9):
    loss = 0.0
    J = X_true.shape[1]

    for users, positives_batch, negatives_batch in tqdm(
        warp_sample_pairs(X_true, U=U, batch_size=batch_size)
    ):
        current_batch_size = users.shape[0]

        dist_pos_interaction = X_pred[users.numpy(
        ).tolist(), positives_batch.numpy().tolist()]

        dist_neg_interaction = X_pred[users.repeat_interleave(U).numpy().tolist(),
                                      negatives_batch.reshape(
            current_batch_size * U, 1).squeeze(-1).numpy().tolist()]

        dist_pos_interaction = torch.tensor(
            dist_pos_interaction.A[0]).unsqueeze(-1)
        dist_neg_interaction_flat = torch.tensor(
            dist_neg_interaction.A[0])
        dist_neg_interaction = dist_neg_interaction_flat.reshape(
            current_batch_size, -1
        )

        loss += warp_loss(dist_pos_interaction,
                          dist_neg_interaction, margin, J, U)

    return loss
