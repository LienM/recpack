import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from recpack.algorithms.samplers import BootstrapSampler, WarpSampler


def covariance_loss(H: nn.Embedding, W: nn.Embedding) -> torch.Tensor:
    # TODO: Refactor so it's not CML specific
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

    loss = (most_wrong_neg_interaction * w).mean()

    return loss


def skipgram_negative_sampling_loss(
    positive_sim: torch.Tensor, negative_sim: torch.Tensor
) -> torch.Tensor:
    pos_loss = positive_sim.sigmoid().log()
    neg_loss = negative_sim.neg().sigmoid().log().sum(-1)

    return -(pos_loss + neg_loss).mean()


def bpr_loss(positive_sim: torch.Tensor, negative_sim: torch.Tensor) -> torch.Tensor:
    """Implementation of the Bayesian Personalized Ranking loss.

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

    sampler = BootstrapSampler(U=1, batch_size=batch_size, exact=exact)

    for users, target_items, negative_items in sampler.sample(
        X_true, sample_size=sample_size
    ):
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
    U: int = 20,
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
    :param U: How many negatives to sample for each positive item, defaults to 20
    :type U: int, optional
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
    J = X_true.shape[1]

    sampler = WarpSampler(U=U, batch_size=batch_size, exact=exact)

    for users, positives_batch, negatives_batch in tqdm(
        sampler.sample(X_true, sample_size=sample_size)
    ):

        current_batch_size = users.shape[0]

        dist_pos_interaction = X_pred[
            users.numpy().tolist(), positives_batch.numpy().tolist()
        ]

        dist_neg_interaction = X_pred[
            users.repeat_interleave(U).numpy().tolist(),
            negatives_batch.reshape(current_batch_size * U, 1)
            .squeeze(-1)
            .numpy()
            .tolist(),
        ]

        dist_pos_interaction = torch.tensor(dist_pos_interaction.A[0]).unsqueeze(-1)
        dist_neg_interaction_flat = torch.tensor(dist_neg_interaction.A[0])
        dist_neg_interaction = dist_neg_interaction_flat.reshape(current_batch_size, -1)

        losses.append(
            warp_loss(dist_pos_interaction, dist_neg_interaction, margin, J, U)
        )

    return np.mean(losses)


"""
Custom loss functions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch import Tensor


Sampler = Callable[[int, Tensor], Tensor]  # (num_samples, targets) -> samples

# TODO Sampler is called in train_epoch 

class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss.

    This is a pairwise loss function, where the target score is compared against
    that of a number of negative samples. It can be defined as:

    .. math::
        L_{bpr} = -\frac{1}{N_S} \sum_{j=1}^{N_S} \log \sigma(r_i - r_j)

    where :math:`N_S` is the number of negative samples, :math:`r_i` is the target
    score and :math:`r_j` is the score given to the sampled item.

    See the 2016 paper "Session-based Recommendations with Recurrent Neural Networks"
    by Hidasi et al. for the motivation behind using it for top-k recommendations.

    :param sampler: Sampler to draw negative samples from
    :param num_samples: Number of samples to use in loss calculation
    """

    def __init__(self, sampler: Sampler, num_samples: int):
        super().__init__()
        self.sampler = sampler
        self.num_samples = num_samples

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Computes the average loss over a mini-batch.

        :param input: The computed scores for each item for every example in the batch,
            as a tensor of shape (B, I)
        :param target: The index of the target item in each batch, shape (B,)
        :return: The average loss as a single value tensor
        """
        samples = self.sampler(self.num_samples, target)
        target_scores = torch.gather(input, 1, target.unsqueeze(1))
        sample_scores = torch.gather(input, 1, samples)
        score_diff = F.logsigmoid(target_scores - sample_scores)
        return -score_diff.mean()


class BPRMaxLoss(nn.Module):
    """
    Bayesian Personalized Ranking Max Loss.

    This is a differentiable approximation to the BPR loss between the target item 
    and the negative sample with the highest score, with an added regularization 
    term. It can be defined as:

    .. math::
        L_{bpr-max} = -\log \sum_{j=1}^{N_S} s_j \sigma(r_i - r_j) + 
                      \lambda \sum_{j=1}^{N_S} s_j r_j^2

    where :math:`N_S` is the number of negative samples, :math:`r_i` is the target
    score and :math:`r_j` is the score given to the sampled item. The BPR loss 
    between target score and the maximum sampled score is approximated by computing 
    a softmax distribution over the negative samples and using the softmax values 
    :math:`s_j` as weights.

    See the 2018 paper "Recurrent Neural Networks with Top-K Gains for Session-based 
    Recommendations" by Hidasi et al. for the motivation behind these changes to the 
    original BPR loss.

    :param sampler: Sampler to draw negative samples from
    :param num_samples: Number of samples to use in loss calculation. BPR-Max loss 
        tends to scale better with sample size than base BPR.
    """

    def __init__(self, sampler: Sampler, num_samples: int):
        super().__init__()
        self.sampler = sampler
        self.num_samples = num_samples

    def forward(self, input: Tensor, target: Tensor, reg: float = 1.0) -> Tensor:
        """
        Computes the average loss over a mini-batch.

        :param input: The computed scores for each item for every example in the batch,
            as a tensor of shape (B, I)
        :param target: The index of the target item in each batch, shape (B,)
        :param reg: The amount of regularization, :math:`lambda` in the formula above
        :return: The average loss as a single value tensor
        """
        samples = self.sampler(self.num_samples, target)
        target_scores = torch.gather(input, 1, target.unsqueeze(1))
        sample_scores = torch.gather(input, 1, samples)
        weights = torch.softmax(sample_scores, dim=1)
        score_diff = weights * torch.sigmoid(target_scores - sample_scores)
        norm_penalty = weights * sample_scores ** 2
        return (
            -torch.log(score_diff.sum(dim=1)) + reg * norm_penalty.sum(dim=1)
        ).mean()


class TOP1Loss(nn.Module):
    """
    TOP1 Loss.

    This is a pairwise loss function similar to BPR loss, but with an added score 
    regularization term. It was devised specifically for use with the Session RNN.
    It can be defined as:

    .. math::
        L_{top1} = \frac{1}{N_S} \sum_{j=1}^{N_S} \sigma(r_j - r_i) + \sigma(r_j^2)

    where :math:`N_S` is the number of negative samples, :math:`r_i` is the target
    score and :math:`r_j` is the score given to the sampled item.

    See the 2016 paper "Session-based Recommendations with Recurrent Neural Networks"
    by Hidasi et al. for the motivation behind using it for top-k recommendations.

    :param sampler: Sampler to draw negative samples from
    :param num_samples: Number of samples to use in loss calculation
    """

    def __init__(self, sampler: Sampler, num_samples: int):
        super().__init__()
        self.sampler = sampler
        self.num_samples = num_samples

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Computes the average loss over a mini-batch.

        :param input: The computed scores for each item for every example in the batch,
            as a tensor of shape (B, I)
        :param target: The index of the target item in each batch, shape (B,)
        :return: The average loss as a single value tensor
        """
        samples = self.sampler(self.num_samples, target)
        target_scores = torch.gather(input, 1, target.unsqueeze(1))
        sample_scores = torch.gather(input, 1, samples)
        score_diff = torch.sigmoid(sample_scores - target_scores)
        norm_penalty = torch.sigmoid(sample_scores ** 2)
        return (score_diff + norm_penalty).mean()


class TOP1MaxLoss(nn.Module):
    """
    TOP1 Max Loss.

    This is a differentiable approximation to the TOP1 loss between the target item 
    and the negative sample with the highest score. It can be defined as:

    .. math::
        L_{top1-max} = \sum_{j=1}^{N_S} s_j\left(\sigma(r_j - r_i) + \sigma(r_j^2)\right)

    where :math:`N_S` is the number of negative samples, :math:`r_i` is the target
    score and :math:`r_j` is the score given to the sampled item. The TOP1 loss 
    between target score and the maximum sampled score is approximated by computing 
    a softmax distribution over the negative samples and using the softmax values 
    :math:`s_j` as weights.

    See the 2018 paper "Recurrent Neural Networks with Top-K Gains for Session-based 
    Recommendations" by Hidasi et al. for the motivation behind these changes to the 
    original TOP1 loss.

    :param sampler: Sampler to draw negative samples from
    :param num_samples: Number of samples to use in loss calculation. TOP1-Max loss 
        tends to scale better with sample size than base TOP1.
    """

    def __init__(self, sampler: Sampler, num_samples: int):
        super().__init__()
        self.sampler = sampler
        self.num_samples = num_samples

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Computes the average loss over a mini-batch.

        :param input: The computed scores for each item for every example in the batch,
            as a tensor of shape (B, I)
        :param target: The index of the target item in each batch, shape (B,)
        :return: The average loss as a single value tensor
        """
        samples = self.sampler(self.num_samples, target)
        target_scores = torch.gather(input, 1, target.unsqueeze(1))
        sample_scores = torch.gather(input, 1, samples)
        weights = torch.softmax(sample_scores, dim=1)
        score_diff = torch.sigmoid(sample_scores - target_scores)
        norm_penalty = torch.sigmoid(sample_scores ** 2)
        loss_terms = weights * (score_diff + norm_penalty)
        return loss_terms.sum(dim=1).mean()


class BatchSampler:
    """
    A sampler that uses the other targets in a minibatch as samples.

    As an example, if the targets in a minibatch of size 5 are [1, 2, 3, 4, 5],
    the resulting samples will be

        [[2, 3, 4, 5]
         [1, 3, 4, 5]
         [1, 2, 4, 5]
         [1, 2, 3, 5]
         [1, 2, 3, 4]]

    If the number of samples needed exceeds (batch size - 1), extra samples across all
    items (not necessarily those in the target set) are added by sampling according
    to the given item weights. The additional samples will be the same for every 
    example in the batch.

    :param weights: Sampling weights for each item as a tensor of shape (I,)
    :param device: The device where generated samples will be stored
    """

    def __init__(self, weights: Tensor, device: str = "cpu"):
        self._wsampler = _WeightedSampler(weights, device=device)
        self.device = device

    def __call__(self, num_samples: int, targets: Tensor) -> Tensor:
        """
        Generate new samples

        :param num_samples: The number of samples to draw, including the samples taken 
            from the targets in the batch.
        :param targets: The index of the target item in each batch, shape (B,)
        :return: Samples as a tensor of shape (B, N), where N is the number of samples
        """
        m = len(targets)
        batch_samples = (targets.flip(0).repeat(m - 1).reshape((m, -1))).to(self.device)
        if num_samples > m - 1:
            r = num_samples - (m - 1)
            extra_samples = self._wsampler((1, r)).repeat((m, 1))
            return torch.cat((batch_samples, extra_samples), dim=1)
        else:
            return batch_samples[:, :num_samples]


class _WeightedSampler:
    def __init__(self, weights, device="cpu"):
        self.weights = torch.as_tensor(weights, dtype=torch.float, device=device)
        self._cache_size = 1_000_000
        self._cache_samples()

    def __call__(self, shape):
        n = np.prod(shape)
        res = self._cache[self._i : self._i + n]
        self._i += n
        while len(res) < n:
            self._cache_samples()
            rem = n - len(res)
            res = torch.cat((res, self._cache[:rem]))
            self._i += rem
        return res.reshape(shape)

    def _cache_samples(self):
        self._i = 0
        self._cache = torch.multinomial(
            self.weights, self._cache_size, replacement=True
        )


