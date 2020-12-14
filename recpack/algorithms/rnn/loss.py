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
