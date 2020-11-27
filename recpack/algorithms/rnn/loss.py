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

    As defined in the 2016 paper "Session-based Recommendations with Recurrent
    Neural Networks" by Hidasi et al.
    """

    def __init__(self, sampler: Sampler, num_samples: int):
        super().__init__()
        self.sampler = sampler
        self.num_samples = num_samples

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        samples = self.sampler(self.num_samples, target)
        target_scores = torch.gather(input, 1, target.unsqueeze(1))
        sample_scores = torch.gather(input, 1, samples)
        score_diff = F.logsigmoid(target_scores - sample_scores)
        return -score_diff.mean()


class BPRMaxLoss(nn.Module):
    """
    Bayesian Personalized Ranking Max Loss.

    As defined in the 2018 paper "Recurrent Neural Networks with Top-K Gains for
    Session-based Recommendations" by Hidasi et al. This is a differentiable
    approximation to the BPR loss between the target item and the negative
    sample with the highest score.
    """

    def __init__(self, sampler: Sampler, num_samples: int):
        super().__init__()
        self.sampler = sampler
        self.num_samples = num_samples

    def forward(self, input: Tensor, target: Tensor, reg: float = 1.0) -> Tensor:
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

    As defined in the 2016 paper "Session-based Recommendations with Recurrent
    Neural Networks" by Hidasi et al.
    """

    def __init__(self, sampler: Sampler, num_samples: int):
        super().__init__()
        self.sampler = sampler
        self.num_samples = num_samples

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        samples = self.sampler(self.num_samples, target)
        target_scores = torch.gather(input, 1, target.unsqueeze(1))
        sample_scores = torch.gather(input, 1, samples)
        score_diff = torch.sigmoid(sample_scores - target_scores)
        norm_penalty = torch.sigmoid(sample_scores ** 2)
        return (score_diff + norm_penalty).mean()


class TOP1MaxLoss(nn.Module):
    """
    TOP1 Max Loss.

    As defined in the 2018 paper "Recurrent Neural Networks with Top-K Gains for
    Session-based Recommendations" by Hidasi et al. This is a differentiable
    approximation to the TOP1 loss between the target item and the negative
    sample with the highest score.
    """

    def __init__(self, sampler: Sampler, num_samples: int):
        super().__init__()
        self.sampler = sampler
        self.num_samples = num_samples

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
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
    the resulting samples would be

        [[2, 3, 4, 5]
         [1, 3, 4, 5]
         [1, 2, 4, 5]
         [1, 2, 3, 5]
         [1, 2, 3, 4]]

    If the number of samples needed exceeds (batch size - 1), extra samples across all 
    items (not necessarily those in the target set) are added by sampling according 
    to the given item weights.

    :param weights: Sampling weight for each item
    :param device: The device where generated samples will be stored
    """

    def __init__(self, weights: Tensor, device: str = "cpu"):
        self._wsampler = _WeightedSampler(weights, device=device)
        self.device = device

    def __call__(self, num_samples: int, targets: Tensor) -> Tensor:
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
