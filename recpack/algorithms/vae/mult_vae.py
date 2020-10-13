import time
from typing import List
import logging

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
import numpy as np

from recpack.algorithms.vae.base import VAE  #, VAETorch
from recpack.algorithms.vae.util import naive_sparse2tensor
from recpack.splitters.splitter_base import batch
from recpack.metrics import NDCGK

logger = logging.getLogger('recpack')


class MultVAE(VAE):
    def __init__(
        self,
        batch_size=500,
        max_epochs=200,
        seed=42,
        learning_rate=1e-4,
        dim_bottleneck_layer=200,
        dim_hidden_layer=600,
        max_beta=0.2,
        anneal_steps=200000,
        dropout=0.5,
    ):
        """
        MultVAE Algorithm as first discussed in
        'Variational Autoencoders for Collaborative Filtering',
        D. Liang et al. @ KDD2018
        Default values were taken from the paper.

        :param batch_size: Batch size for SGD,
                           defaults to 500
        :type batch_size: int, optional
        :param max_epochs: Maximum number of epochs (iterations),
                           defaults to 200
        :type max_epochs: int, optional
        :param seed: Random seed for Torch, provided for reproducibility,
                     defaults to 42.
        :type seed: int, optional
        :param learning_rate: Learning rate, defaults to 1e-4
        :type learning_rate: [type], optional
        :param dim_bottleneck_layer: Size of the latent representation,
                                     defaults to 200
        :type dim_bottleneck_layer: int, optional
        :param dim_hidden_layer: Dimension of the hidden layer, defaults to 600
        :type dim_hidden_layer: int, optional
        :param max_beta: Regularization parameter, annealed over {anneal_steps}
                        until it reaches max_beta, defaults to 0.2
        :type max_beta: float, optional
        :param anneal_steps: Number of steps to anneal beta to {max_beta},
                             defaults to 200000
        :type anneal_steps: int, optional
        :param dropout: Dropout rate to apply at the inputs, defaults to 0.5
        :type dropout: float, optional
        """
        super().__init__(
            batch_size,
            max_epochs,
            seed,
            learning_rate,
            StoppingCriterion(NDCGK, K=100)
        )

        self.dim_hidden_layer = dim_hidden_layer
        self.dim_bottleneck_layer = dim_bottleneck_layer

        self.max_beta = max_beta
        self.anneal_steps = anneal_steps

        self.steps = 0
        self.dropout = dropout

        self.optimizer = None
        self.loss_function = vae_loss_function

    @property
    def _beta(self):
        """
        As discussed in the paper, Beta is a regularization parameter
        that controls the importance of the KL-divergence term.
        It is slowly annealed from 0 to self.max_beta over self.anneal_steps.
        """
        return (
            self.max_beta
            if self.steps >= self.anneal_steps
            else self.steps / self.anneal_steps
        )

    def _init_model(self, dim_input_layer: int):
        """
        Initialize Torch model and optimizer.

        :param dim_input_layer: Dimension of the input layer
                                (corresponds to number of items)
        :type dim_input_layer: int
        """
        self.model_ = MultiVAETorch(
            dim_input_layer,
            dim_hidden_layer=self.dim_hidden_layer,
            dim_bottleneck_layer=self.dim_bottleneck_layer,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model_.parameters(), lr=self.learning_rate)

    def _train_epoch(self, train_data: csr_matrix, users: List[int]):
        """
        Perform one training epoch.
        Data is processed in batches of self.batch_size users.

        :param train_data: Training data (UxI)
        :type train_data: [type]
        :param users: List of all users in the training dataset.
        :type users: List[int]
        """
        start_time = time.time()
        train_loss = 0.0
        # Set to training
        self.model_.train()

        np.random.shuffle(users)

        for batch_idx, user_batch in enumerate(batch(users, self.batch_size)):
            X = naive_sparse2tensor(train_data[user_batch, :]).to(self.device)

            # Clear gradients
            self.optimizer.zero_grad()
            X_pred, mu, logvar = self.model_(X)
            loss = self._compute_loss(X, X_pred, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            self.steps += 1

        end_time = time.time()

        logger.info(
            f"Processed one batch in {end_time-start_time} s."
            f" Training Loss = {train_loss}"
        )

    def _compute_loss(self, X: torch.Tensor, X_pred: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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


class MultiVAETorch(VAETorch):
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

        # TODO Make it possible to go deeper?
        self.q_in_hid_layer = nn.Linear(dim_input_layer, dim_hidden_layer)
        # Last dimension of q- network is for mean and variance (*2)
        # Use PyTorch Distributions for this.
        self.q_hid_bn_layer = nn.Linear(
            dim_hidden_layer, dim_bottleneck_layer * 2)

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

    def forward(self, x):
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
        logvar = h[:, self.dim_bottleneck_layer:]
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
            nn.init.normal_(
                layer.bias, mean=0, std=0.001
            )  # TODO This should be truncated normal


def vae_loss_function(recon_x, mu, logvar, x, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar -
                                      mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD
