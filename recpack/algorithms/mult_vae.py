# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from typing import List, Tuple
from scipy.sparse import lil_matrix

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
import numpy as np

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.loss_functions import vae_loss
from recpack.algorithms.util import naive_sparse2tensor, get_batches

logger = logging.getLogger("recpack")


class MultVAE(TorchMLAlgorithm):
    """MultVAE Algorithm as first discussed in
    'Variational Autoencoders for Collaborative Filtering',
    D. Liang et al. @ KDD2018.

    An Auto Encoder neural network's goal is to reconstruct the original matrix after
    being passed through a bottleneck layer and several hidden layers.
    This method assumes a Multinomial likelihood for the data distribution.
    This rewards the model for putting probability mass on the non-zero entries in x_u.
    But the model has a limited budget of probability mass since π(z_u) must sum to 1;
    the items must compete for a limited budget.

    Default values for parameters were taken from the paper.

    :param batch_size: Batch size for SGD,
                        defaults to 500
    :type batch_size: int, optional
    :param max_epochs: Maximum number of epochs (iterations),
                        defaults to 200
    :type max_epochs: int, optional
    :param learning_rate: Learning rate, defaults to 1e-4
    :type learning_rate: [type], optional
    :param seed: Random seed for Torch, provided for reproducibility,
                    defaults to None.
    :type seed: int, optional
    :param dim_bottleneck_layer: Size of the latent representation,
                                    defaults to 200
    :type dim_bottleneck_layer: int, optional
    :param dim_hidden_layer: Dimension of the hidden layer, defaults to 600
    :type dim_hidden_layer: int, optional
    :param max_beta: Regularization parameter, annealed over ``anneal_steps``
                    until it reaches max_beta, defaults to 0.2
    :type max_beta: float, optional
    :param anneal_steps: Number of steps to anneal beta to ``max_beta``,
                            defaults to 200000
    :type anneal_steps: int, optional
    :param dropout: Dropout rate to apply at the inputs, defaults to 0.5
    :type dropout: float, optional
    :param stopping_criterion: Used to identify the best model computed thus far.
        The string indicates the name of the stopping criterion.
        Which criterions are available can be found at StoppingCriterion.FUNCTIONS
        Defaults to ``'ndcg'``
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
    :param save_best_to_file: If True, the best model is saved to disk after fit.
    :type save_best_to_file: bool, optional
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
        batch_size: int = 500,
        max_epochs: int = 200,
        learning_rate: float = 1e-4,
        seed: int = None,
        dim_bottleneck_layer: int = 200,
        dim_hidden_layer: int = 600,
        max_beta: float = 0.2,
        anneal_steps: int = 200000,
        dropout: float = 0.5,
        stopping_criterion: str = "ndcg",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: int = 0.01,
        save_best_to_file=False,
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

        self.dim_hidden_layer = dim_hidden_layer
        self.dim_bottleneck_layer = dim_bottleneck_layer

        self.max_beta = max_beta
        self.anneal_steps = anneal_steps

        self.steps = 0
        self.dropout = dropout

        self.optimizer = None
        self.loss_function = vae_loss

    @property
    def _beta(self):
        """
        As discussed in the paper, Beta is a regularization parameter
        that controls the importance of the KL-divergence term.
        It is slowly annealed from 0 to self.max_beta over self.anneal_steps.
        """
        return self.max_beta if self.steps >= self.anneal_steps else self.steps / self.anneal_steps

    def _init_model(self, X: csr_matrix):
        """
        Initialize Torch model and optimizer.

        :param dim_input_layer: Dimension of the input layer
                                (corresponds to number of items)
        :type dim_input_layer: int
        """

        dim_input_layer = X.shape[1]

        self.model_ = MultiVAETorch(
            dim_input_layer,
            dim_hidden_layer=self.dim_hidden_layer,
            dim_bottleneck_layer=self.dim_bottleneck_layer,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

    def _train_epoch(self, train_data: csr_matrix):
        """
        Perform one training epoch.
        Data is processed in batches of self.batch_size users.

        :param train_data: Training data (UxI)
        :type train_data: [type]
        """
        losses = []

        users = list(set(train_data.nonzero()[0]))

        np.random.shuffle(users)

        for batch_idx, user_batch in enumerate(get_batches(users, self.batch_size)):
            X = naive_sparse2tensor(train_data[user_batch, :]).to(self.device)

            # Clear gradients
            self.optimizer.zero_grad()
            X_pred, mu, logvar = self.model_(X)
            loss = self._compute_loss(X, X_pred, mu, logvar)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

            self.steps += 1

        return losses

    def _compute_loss(
        self,
        X: torch.Tensor,
        X_pred: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the prediction loss.

        More info on the loss function in the paper

        :param X: input data
        :type X: torch.Tensor
        :param X_pred: output data
        :type X_pred: torch.Tensor
        :param mu: the mean tensor
        :type mu: torch.Tensor
        :param logvar: the variance tensor
        :type logvar: torch.Tensor
        :return: the loss tensor
        :rtype: torch.Tensor
        """
        loss = self.loss_function(X_pred, mu, logvar, X, anneal=self._beta)

        return loss

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
        active_users = X[users]

        in_tensor = naive_sparse2tensor(active_users).to(self.device)

        out_tensor, _, _ = self.model_(in_tensor)

        result = lil_matrix(X.shape)
        result[users] = out_tensor.detach().cpu().numpy()

        return result.tocsr()


class MultiVAETorch(nn.Module):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(
        self,
        dim_input_layer,
        dim_hidden_layer=600,
        dim_bottleneck_layer=200,
        dropout=0.5,
    ):
        super().__init__()

        self.dim_input_layer = dim_input_layer
        self.dim_hidden_layer = dim_hidden_layer
        self.dim_bottleneck_layer = dim_bottleneck_layer

        self.tanh = nn.Tanh()

        self.q_in_hid_layer = nn.Linear(dim_input_layer, dim_hidden_layer)
        # Last dimension of q- network is for mean and variance (*2)
        # Use PyTorch Distributions for this.
        self.q_hid_bn_layer = nn.Linear(dim_hidden_layer, dim_bottleneck_layer * 2)

        self.p_bn_hid_layer = nn.Linear(dim_bottleneck_layer, dim_hidden_layer)
        self.p_hid_out_layer = nn.Linear(dim_hidden_layer, dim_input_layer)

        self.layers = nn.ModuleList(
            [
                self.q_in_hid_layer,
                self.q_hid_bn_layer,
                self.p_bn_hid_layer,
                self.p_hid_out_layer,
            ]
        )

        self.drop = nn.Dropout(p=dropout, inplace=False)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pass the input through the network, and return result.

        :param x: input tensor
        :type x: torch.Tensor
        :return: A tuple with
                (predicted output value, mean values, average values)
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def encode(self, x):
        h = F.normalize(x)

        # Torch will only do the dropout if the model is supposed to be
        # training
        h = self.drop(h)

        h = self.q_in_hid_layer(h)
        h = self.tanh(h)

        h = self.q_hid_bn_layer(h)

        # TODO This is a terrible hack. Do something about it.
        mu = h[:, : self.dim_bottleneck_layer]
        logvar = h[:, self.dim_bottleneck_layer :]
        return mu, logvar

    def decode(self, z):
        h = self.p_bn_hid_layer(z)
        h = self.tanh(h)

        h = self.p_hid_out_layer(h)

        return h

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _init_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias, mean=0, std=0.001)  # TODO This should be truncated normal
