# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from typing import List

from scipy.sparse import csr_matrix, lil_matrix

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.loss_functions import bpr_loss
from recpack.algorithms.samplers import BootstrapSampler

logger = logging.getLogger("recpack")


class BPRMF(TorchMLAlgorithm):
    """Implements Matrix Factorization by using the BPR-OPT objective
    and SGD optimization.

    MF implementation using the BPR criterion as defined in Rendle, Steffen, et al.
    "BPR: Bayesian personalized ranking from implicit feedback."

    The BPR optimization criterion aims to construct a factorization that optimally
    ranks interesting items (interacted with previously)
    above uninteresting or unknown items for all users.

    :param num_components: The size of the latent vectors for both users and items.
                            defaults to 100
    :type num_components: int, optional
    :param lambda_h: The regularization parameter for the item embedding,
        should be a value between 0 and 1.
        Defaults to 0.0
    :type lambda_h: float, optional
    :param lambda_w: The regularization parameter for the user embedding,
        defaults to 0.0
    :type lambda_w: float, optional
    :param batch_size: Size of the batches to use during gradient descent. Defaults to 1000.
    :type batch_size: int, optional
    :param max_epochs: The max amount of epochs to train the model, defaults to 20
    :type max_epochs: int, optional
    :param learning_rate: The learning rate of the optimization procedure,
                            defaults to 0.01
    :type learning_rate: float, optional
    :param seed: Seed to fix random numbers, to make results reproducible,
                    defaults to None
    :type seed: int, optional
    :param stopping_criterion: Which criterion to use optimise the parameters,
        a string which indicates the name of the stopping criterion.
        Which criterions are available can be found at
        :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`.
        Defaults to 'recall'
    :type stopping_criterion: str, optional
    :param stop_early: If True, early stopping is enabled,
        and after ``max_iter_no_change`` iterations where improvement of loss function
        is below ``min_improvement`` the optimisation is stopped,
        even if max_epochs is not reached.
        Defaults to False
    :type stop_early: bool, optional
    :param max_iter_no_change: If early stopping is enabled,
        stop after this amount of iterations without change.
        Defaults to 5
    :type max_iter_no_change: int, optional
    :param min_improvement: If early stopping is enabled, no change is detected,
        if the improvement is below this value.
        Defaults to 0.01
    :type min_improvement: float, optional
    :param save_best_to_file: If True, the best model is saved to disk after fit.
    :type save_best_to_file: bool, optional
    :param sample_size: How many samples to take during training using bootstrap sampling.
        If None a sample is taken for each interaction,
        there is no guarantee that all interactions will be used though,
        since sampling happens with replacement.
        Defaults to None
    :type sample_size: int, optional
    :param keep_last: Retain last model, rather than best
        (according to stopping criterion value on validation data), defaults to False
    :type keep_last: bool, optional
    :param predict_topK: The topK recommendations to keep per row in the matrix.
        Use when the user x item output matrix would become too large for RAM.
        Defaults to None, which results in no filtering.
    :type predict_topK: int, optional
    :param validation_sample_size: Amount of users that will be sampled to calculate
        validation loss and stopping criterion value.
        This reduces computation time during validation, such that training times are strongly reduced.
        If None, all nonzero users are used. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        num_components: int = 100,
        lambda_h: float = 0.0,
        lambda_w: float = 0.0,
        batch_size: int = 1_000,
        max_epochs: int = 20,
        learning_rate: float = 0.01,
        stopping_criterion: str = "bpr",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.01,
        seed: int = None,
        save_best_to_file: bool = False,
        sample_size=None,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
    ):

        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )

        self.num_components = num_components
        self.lambda_h = lambda_h
        self.lambda_w = lambda_w

        self.sample_size = sample_size

        self.sampler = BootstrapSampler(
            num_negatives=1,
            batch_size=self.batch_size,
        )

    def _init_model(self, X: csr_matrix):
        num_users, num_items = X.shape
        self.model_ = MFModule(num_users, num_items, num_components=self.num_components).to(self.device)

        self.optimizer = optim.Adagrad(self.model_.parameters(), lr=self.learning_rate)

    def _batch_predict(self, X: csr_matrix, users: List[int]) -> csr_matrix:
        """Predict scores for matrix X, given the selected users in this batch

        :param X: Matrix of user item interactions,
            expected to only contain interactions for those users that are in `users`
        :type X: csr_matrix
        :param users: users selected for recommendation
        :type users: List[int]
        :return: Sparse matrix of scores per user item pair.
        :rtype: csr_matrix
        """

        user_tensor = torch.LongTensor(users).to(self.device)
        item_tensor = torch.arange(X.shape[1]).to(self.device)

        result = lil_matrix(X.shape)
        result[users] = self.model_(user_tensor, item_tensor).detach().cpu().numpy()

        return result.tocsr()

    def _train_epoch(self, train_data: csr_matrix):
        """train a single epoch. Uses sampler to generate samples,
        and loop through them in batches of self.batch_size.
        After each batch, update the parameters according to gradients.

        :param train_data: interaction matrix.
        :type train_data: csr_matrix
        """
        losses = []

        # For each positive item sample a single negative item.
        for users, target_items, mnar_items in tqdm(
            self.sampler.sample(
                train_data,
                sample_size=self.sample_size,
            ),
            desc="train_epoch BPRMF",
        ):
            users = users.to(self.device)
            # Target items are items the user has interacted with,
            # and we expect to recommend high
            target_items = target_items.to(self.device)
            # Items the user has not seen, and assuming MNAR data
            # Is a batch_size x 1 matrix, squeeze into an array.
            mnar_items = mnar_items.squeeze(-1).to(self.device)

            self.optimizer.zero_grad()

            target_sim = self.model_.forward(users, target_items).diag()
            mnar_sim = self.model_.forward(users, mnar_items).diag()

            # Checks to make sure the shapes are correct.
            if not ((mnar_sim.shape == target_sim.shape) or (target_sim.shape[0] == users.shape[0])):
                raise AssertionError("Shapes should match")

            loss = self._compute_loss(target_sim, mnar_sim)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

        return losses

    def _compute_loss(self, positive_sim, negative_sim):

        loss = bpr_loss(positive_sim, negative_sim)
        loss += (
            self.lambda_h * self.model_.item_embedding_.weight.norm()
            + self.lambda_w * self.model_.user_embedding_.weight.norm()
        )

        return loss


class MFModule(nn.Module):
    """MF torch module, encodes the embeddings and the forward functionality.

    :param num_users: the amount of users
    :type num_users: int
    :param num_items: the amount of items
    :type num_items: int
    :param num_components: The size of the embedding per user and item, defaults to 100
    :type num_components: int, optional
    """

    def __init__(self, num_users: int, num_items: int, num_components: int = 100):
        super().__init__()

        self.num_components = num_components
        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding_ = nn.Embedding(num_users, num_components)  # User embedding
        self.item_embedding_ = nn.Embedding(num_items, num_components)  # Item embedding

        # Keep variance low enough, to alow learning
        self.std = min(1 / num_components ** 0.5, 0.05)
        # Initialise embeddings to a random start
        nn.init.normal_(self.user_embedding_.weight, std=self.std)
        nn.init.normal_(self.item_embedding_.weight, std=self.std)

    def forward(self, user_tensor: torch.Tensor, item_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute dot-product of user embedding (w_u) and item embedding (h_i)
        for every user and item pair in user_tensor and item_tensor.

        :param user_tensor: [description]
        :type user_tensor: [type]
        :param item_tensor: [description]
        :type item_tensor: [type]
        """
        w_u = self.user_embedding_(user_tensor)
        h_i = self.item_embedding_(item_tensor)

        return w_u.matmul(h_i.T)
