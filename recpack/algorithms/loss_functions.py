# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from recpack.algorithms.samplers import BootstrapSampler, WarpSampler


def covariance_loss(H: nn.Embedding, W: nn.Embedding) -> torch.Tensor:
    # TODO: Refactor so it's not CML specific
    """Covariance loss.

    Convariance loss as described in
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

    # Concatenate them together. They live in the same metric space,
    # so share the same dimensions.
    #  X is a matrix of shape (|users| + |items|, num_dimensions)
    X = torch.cat([W_as_tensor, H_as_tensor], dim=0)

    # Zero mean
    X = X - X.mean(dim=0)

    cov = X.matmul(X.T)

    # Per element covariance, excluding the variance of individual random variables.
    return cov.fill_diagonal_(0).sum() / (X.shape[0] * X.shape[1])


def vae_loss(reconstructed_X, mu, logvar, X, anneal=1.0):
    """VAE loss function for use with Auto Encoders.

    Loss defined in 'Variational Autoencoders for Collaborative Filtering',
    D. Liang et al. @ KDD2018.
    Uses a combination of Binary Cross Entropy loss and
    Kullback–Leibler divergence (relative entropy).

    :param reconstructed_X: The reconstructed matrix X
    :type reconstructed_X: torch.Tensor
    :param mu: The mean tensor
    :type mu: torch.Tensor
    :param logvar: The variance Tensor.
    :type logvar: torch.Tensor
    :param X: The matrix to reconstruct
    :type X: torch.Tensor
    :param anneal: multiplicative factor for the KLD part of the loss function,
        defaults to 1.0
    :type anneal: float, optional
    :return: The loss as a 0D tensor
    :rtype: torch.Tensor
    """

    BCE = -torch.mean(torch.sum(F.log_softmax(reconstructed_X, 1) * X, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


def warp_loss(
    dist_pos_interaction: torch.Tensor,
    dist_neg_interaction: torch.Tensor,
    margin: float,
    num_items: int,
    num_negatives: int,
) -> torch.Tensor:
    """WARP loss

    WARP loss as described in
    Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf
    based on
    J. Weston, S. Bengio, and N. Usunier. Large scale image annotation:
    learning to rank with joint word-image embeddings. Machine learning, 81(1):21–35,
    2010.

    Adds a loss penalty for every negative sample that is not at least
    an amount of margin further away from the reference sample than a positive
    sample. This per sample loss penalty has a weight proportional to the
    amount of samples in the negative sample batch were "misclassified",
    i.e. closer than the positive sample.

    :param dist_pos_interaction: Tensor of distances between positive sample
        and reference sample.
    :type dist_pos_interaction: torch.Tensor
    :param dist_neg_interaction: Tensor of distances between negatives samples
        and reference sample.
    :type dist_neg_interaction: torch.Tensor
    :param margin: Required margin between positive and negative sample.
    :type margin: float
    :param num_items: Total number of items in the dataset. (J in the paper)
    :type num_items: int
    :param num_items: Number of negative samples used for every positive sample. (U in the paper)
    :type num_items: int
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
    # M * num_items / num_negatives =~ rank(pos_i)
    w = torch.log((M * num_items / num_negatives) + 1)

    loss = (most_wrong_neg_interaction * w).mean()

    return loss


def skipgram_negative_sampling_loss(positive_sim: torch.Tensor, negative_sim: torch.Tensor) -> torch.Tensor:
    """Skipgram Sampling Loss.

    :param positive_sim: Similarity scores between focus and context item.
    :type positive_sim: torch.Tensor
    :param negative_sim: Similarity scores between focus and negative sample.
    :type negative_sim: torch.Tensor
    :return: Loss value.
    :rtype: torch.Tensor
    """
    pos_loss = positive_sim.sigmoid().log()
    neg_loss = negative_sim.neg().sigmoid().log().sum(-1)

    return -(pos_loss + neg_loss).mean()


def bpr_loss(positive_sim: torch.Tensor, negative_sim: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss.

    BPR loss as defined in Rendle, Steffen, et al.
    "BPR: Bayesian personalized ranking from implicit feedback."

    Input are the scores for positive samples, and scores for negative samples.

    :param positive_sim: Tensor with scores of positive samples
        (dimension = num_samples, 1)
    :type positive_sim: torch.Tensor
    :param negative_sim: Tensor with scores of negative samples
        (dimension = num_samples, 1)
    :type negative_sim: torch.Tensor
    :return: The loss value of the bpr criterion
    :rtype: torch.Tensor
    """

    distance = positive_sim - negative_sim
    # Probability of ranking given parameters
    elementwise_bpr_loss = torch.log(torch.sigmoid(distance))

    # The goal is to minimize loss
    # If negative sim > positive sim -> distance is negative,
    # but loss is positive
    bpr_loss = -elementwise_bpr_loss.mean()

    return bpr_loss


def bpr_loss_wrapper(
    X_true: csr_matrix,
    X_pred: csr_matrix,
    batch_size=1000,
    sample_size=None,
    exact=False,
):
    """Wrapper around :func:`bpr_loss` function for use with
    :class:`recpack.algorithms.stopping_criterion.StoppingCriterion`.

    Positive and negative items are sampled using
    :class:`recpack.algorithms.samplers.BootstrapSampler`.
    Scores are then extracted from the X_pred,
    and these positive and negative predictions are passed to the
    :func:`bpr_loss` function.

    :param X_true: The expected interactions for the users
    :type X_true: csr_matrix
    :param X_pred: The predicted scores for users
    :type X_pred: csr_matrix
    :param batch_size: size of the batches to sample, defaults to 1000
    :type batch_size: int, optional
    :param sample_size: How many samples to construct
    :type sample_size: int, optional
    :param exact: If True sampling happens exact,
        otherwise sampling assumes high sparsity of data,
        accepting a minimal amount of false negatives,
        speeding up sampling without loss of quality, defaults to False
    :type exact: bool, optional
    :return: The mean of the losses of sampled pairs
    :rtype: float
    """

    if sample_size is None:
        sample_size = X_true.nnz

    losses = []

    sampler = BootstrapSampler(num_negatives=1, batch_size=batch_size, exact=exact)

    for users, target_items, negative_items in sampler.sample(X_true, sample_size=sample_size):
        # Needed to do copy, to use as index in the predidction matrix
        users = users.numpy().copy()
        target_items = target_items.numpy().copy()
        # need to squeeze from batch_size x 1 matrix into array.
        negative_items = negative_items.squeeze(-1).numpy().copy()

        positive_sim = torch.tensor(X_pred[users, target_items])
        negative_sim = torch.tensor(X_pred[users, negative_items])

        losses.append(bpr_loss(positive_sim, negative_sim).item())

    return np.mean(losses)


def warp_loss_wrapper(
    X_true: csr_matrix,
    X_pred: csr_matrix,
    batch_size: int = 1000,
    num_negatives: int = 20,
    margin: float = 1.9,
    sample_size=None,
    exact=False,
):
    """Metric wrapper around the :func:`warp_loss` function.

    Positives and negatives are sampled from the X_true matrix using
    :class:`recpack.algorithms.samplers.WarpSampler`.
    Their scores are fetched from the X_pred matrix.

    :param X_true: True interactions expected for the users
    :type X_true: csr_matrix
    :param X_pred: Predicted scores.
    :type X_pred: csr_matrix
    :param batch_size: Size of the sample batches, defaults to 1000
    :type batch_size: int, optional
    :param num_negatives: How many negatives to sample for each positive item, defaults to 20
    :type num_negatives: int, optional
    :param margin: required margin between positives and negatives, defaults to 1.9
    :type margin: float, optional
    :param sample_size: How many samples to construct
    :type sample_size: int, optional
    :param exact: If True sampling happens exact,
        otherwise sampling assumes high sparsity of data,
        accepting a minimal amount of false negatives.
        This speeds up sampling without significant loss of quality, defaults to False
    :type exact: bool, optional
    :return: The warp loss
    :rtype: float
    """
    losses = []
    num_items = X_true.shape[1]

    sampler = WarpSampler(num_negatives=num_negatives, batch_size=batch_size, exact=exact)

    for users, positives_batch, negatives_batch in tqdm(sampler.sample(X_true, sample_size=sample_size)):

        current_batch_size = users.shape[0]

        dist_pos_interaction = X_pred[users.numpy().tolist(), positives_batch.numpy().tolist()]

        dist_neg_interaction = X_pred[
            users.repeat_interleave(num_negatives).numpy().tolist(),
            negatives_batch.reshape(current_batch_size * num_negatives, 1).squeeze(-1).numpy().tolist(),
        ]

        dist_pos_interaction = torch.tensor(dist_pos_interaction.A[0]).unsqueeze(-1)
        dist_neg_interaction_flat = torch.tensor(dist_neg_interaction.A[0])
        dist_neg_interaction = dist_neg_interaction_flat.reshape(current_batch_size, -1)

        losses.append(warp_loss(dist_pos_interaction, dist_neg_interaction, margin, num_items, num_negatives))

    return np.mean(losses)


def bpr_max_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor, reg: float = 1.0) -> torch.Tensor:
    """Bayesian Personalized Ranking Max Loss.

    This is a differentiable approximation to the BPR loss between the target item
    and the negative sample with the highest score, with an added regularization
    term. It can be defined as:

    .. math::
        L_{bpr-max} = -\\log \\sum\\limits_{j=1}^{N_S} s_j \\sigma(r_i - r_j) +
        \\lambda \\sum\\limits_{j=1}^{N_S} s_j r_j^2

    where :math:`N_S` is the number of negative samples, :math:`r_i` is the target
    score and :math:`r_j` is the score given to the sampled negative. The BPR loss
    between target score and the maximum sampled score is approximated by computing
    a softmax distribution over the negative samples and using the softmax values
    :math:`s_j` as weights.

    See the 2018 paper "Recurrent Neural Networks with Top-K Gains for Session-based
    Recommendations" by Hidasi et al. for the motivation behind these changes to the
    original BPR loss.

    :param positive_scores: Output values assigned to positive samples
    :type positive_scores: torch.Tensor
    :param negative_scores: Output values assigned to negative samples
    :type negative_scores: torch.Tensor
    :param reg: Regularization weight, defaults to 1.0
    :type reg: float, optional
    :return: Computed BPR Max Loss
    :rtype: torch.Tensor
    """
    # Mean over all samples
    if negative_scores.ndim == 1:
        negative_scores = negative_scores.unsqueeze(-1)

    if positive_scores.ndim == 1:
        positive_scores = positive_scores.unsqueeze(-1)

    size = positive_scores.size()
    if len(size) == 2 and size[0] < size[1]:
        positive_scores = positive_scores.t()
        negative_scores = negative_scores.t()

    weights = torch.softmax(negative_scores, dim=1)
    score_diff = weights * torch.sigmoid(positive_scores - negative_scores)
    norm_penalty = weights * negative_scores ** 2

    return (-torch.log(score_diff.sum(dim=1)) + reg * norm_penalty.sum(dim=1)).mean()


def top1_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    """TOP1 Loss.

    This is a pairwise loss function similar to BPR loss, but with an added score
    regularization term. It was devised specifically for use with the Session RNN.
    It can be defined as:

    .. math::
        L_{top1} = \\frac{1}{N_S} \\sum\\limits_{j=1}^{N_S} \\sigma(r_j - r_i) + \\sigma(r_j^2)

    where :math:`N_S` is the number of negative samples, :math:`r_i` is the target
    score and :math:`r_j` is the score given to the sampled negative item.

    See the 2016 paper "Session-based Recommendations with Recurrent Neural Networks"
    by Hidasi et al. for the motivation behind using it for top-k recommendations.

    :param positive_scores: Output values assigned to positive samples
    :type positive_scores: torch.Tensor
    :param negative_scores: Output values assigned to negative samples
    :type negative_scores: torch.Tensor
    :return: Computed Top-1 Loss
    :rtype: torch.Tensor
    """
    # Mean over all samples
    if negative_scores.ndim == 1:
        negative_scores = negative_scores.unsqueeze(-1)

    if positive_scores.ndim == 1:
        positive_scores = positive_scores.unsqueeze(-1)

    size = positive_scores.size()
    if len(size) == 2 and size[0] < size[1]:
        positive_scores = positive_scores.t()
        negative_scores = negative_scores.t()

    score_diff = torch.sigmoid(negative_scores - positive_scores)
    norm_penalty = torch.sigmoid(negative_scores ** 2)
    return (score_diff + norm_penalty).mean()


def top1_max_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    """TOP1 Max Loss.

    This is a differentiable approximation to the TOP1 loss between the target item
    and the negative sample with the highest score. It can be defined as:

    .. math::
        L_{top1-max} = \\sum\\limits_{j=1}^{N_S} s_j\\left(\\sigma(r_j - r_i) + \\sigma(r_j^2)\\right)

    where :math:`N_S` is the number of negative samples, :math:`r_i` is the target
    score and :math:`r_j` is the score given to the sampled negative. The TOP1 loss
    between target score and the maximum sampled score is approximated by computing
    a softmax distribution over the negative samples and using the softmax values
    :math:`s_j` as weights.

    See the 2018 paper "Recurrent Neural Networks with Top-K Gains for Session-based
    Recommendations" by Hidasi et al. for the motivation behind these changes to the
    original TOP1 loss.

    :param positive_scores: Output values assigned to positive samples
    :type positive_scores: torch.Tensor
    :param negative_scores: Output values assigned to negative samples
    :type negative_scores: torch.Tensor
    :return: Computed Top-1 Max Loss
    :rtype: torch.Tensor
    """
    # Mean over all samples
    if negative_scores.ndim == 1:
        negative_scores = negative_scores.unsqueeze(-1)

    if positive_scores.ndim == 1:
        positive_scores = positive_scores.unsqueeze(-1)

    size = positive_scores.size()
    if len(size) == 2 and size[0] < size[1]:
        positive_scores = positive_scores.t()
        negative_scores = negative_scores.t()

    weights = torch.softmax(negative_scores, dim=1)

    score_diff = torch.sigmoid(negative_scores - positive_scores)

    norm_penalty = torch.sigmoid(negative_scores ** 2)
    loss_terms = (weights * (score_diff + norm_penalty)).sum(dim=1)

    return loss_terms.mean()
