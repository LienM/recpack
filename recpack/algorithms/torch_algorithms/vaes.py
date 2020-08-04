import time
from typing import List

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils.validation import check_is_fitted

from recpack.splitters.splitter_base import batch
from recpack.algorithms.algorithm_base import Algorithm

from recpack.metrics.dcg import NDCGK
from recpack.utils import logger


class StoppingCriterion(NDCGK):
    def __init__(self, K):
        """
        Stopping Criterion is (should be) a generic wrapper around a Metric,
        allowing resetting of the metric to zero and keeping track of the
        best value.
        """
        # TODO You can do better than this
        # Maybe something with an "initialize" and "refresh" in the actual metrics?
        super().__init__(K)
        self.best_value = -np.inf

    def reset(self):
        """
        Reset the metric
        """
        # TODO Set these in an initialize method for the metric
        # so that Stopping Criterion can call that to reset the metric
        # Then stopping criterion can inherit from metric. BOOM!
        if self.is_best:
            self.best_value = self.value
        self.value_ = 0
        self.num_users_ = 0

    @property
    def is_best(self):
        """
        Is the current value of the metric the best value?
        """
        return True if self.value > self.best_value else False


# TODO Create a base TorchAlgorithm
class MultVAE(Algorithm):
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
        'Variational Autoencoders for Collaborative Filtering', D. Liang et al. @ KDD2018
        Default values were taken from the paper.

        :param batch_size: Batch size for SGD, defaults to 500
        :type batch_size: int, optional
        :param max_epochs: Maximum number of epochs (iterations), defaults to 200
        :type max_epochs: int, optional
        :param seed: Random seed for Torch, provided for reproducibility, defaults to 42.
        :type seed: int, optional
        :param learning_rate: Learning rate, defaults to 1e-4
        :type learning_rate: [type], optional
        :param dim_bottleneck_layer: Size of the latent representation, defaults to 200
        :type dim_bottleneck_layer: int, optional
        :param dim_hidden_layer: Dimension of the hidden layer, defaults to 600
        :type dim_hidden_layer: int, optional
        :param max_beta: Regularization parameter, annealed over {anneal_steps} until it reaches max_beta, defaults to 0.2
        :type max_beta: float, optional
        :param anneal_steps: Number of steps to anneal beta to {max_beta}, defaults to 200000
        :type anneal_steps: int, optional
        :param dropout: Dropout rate to apply at the inputs, defaults to 0.5
        :type dropout: float, optional
        """
        super().__init__()
        self.max_epochs = max_epochs
        self.seed = seed
        torch.manual_seed(seed)
        self.learning_rate = learning_rate
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.batch_size = (
            batch_size  # TODO * torch.cuda.device_count() if cuda else batch_size
        )
        self.dim_hidden_layer = dim_hidden_layer
        self.dim_bottleneck_layer = dim_bottleneck_layer
        self.max_beta = max_beta
        self.anneal_steps = anneal_steps
        self.steps = 0
        self.dropout = dropout

        self.optimizer = None
        self.criterion = vae_loss_function
        self.stopping_criterion = StoppingCriterion(100)

    @property
    def _beta(self):
        """
        As discussed in the paper, Beta is a regularization parameter that controls
        the importance of the KL-divergence term.
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

        :param dim_input_layer: Dimension of the input layer (corresponds to number of items)
        :type dim_input_layer: int
        """
        self.model_ = MultiVAETorch(
            dim_input_layer,
            dim_hidden_layer=self.dim_hidden_layer,
            dim_bottleneck_layer=self.dim_bottleneck_layer,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

    def fit(self, X: csr_matrix, validation_data):
        """
        Perform self.max_epoch training iterations over the complete training dataset.
        After each epoch, reports value of self.stopping_criterion and stores
        the model if it is better than the last.

        :param train_data: Training data (UxI)
        :type train_data: csr_matrix
        :param val_in: Validation data used as history (UxI)
        :type val_in: csr_matrix
        :param val_out: Validation data we will try to predict (UxI)
        :type val_out: csr_matrix
        """
        self._init_model(X.shape[1])

        # TODO Investigate Multi-GPU
        # multi_gpu = False
        # if torch.cuda.device_count() > 1:
        #     self.model_ = torch.nn.DataParallel(self.model_)
        #     multi_gpu = True

        val_in, val_out = validation_data

        train_users = list(set(X.nonzero()[0]))
        val_users = list(set(val_in.nonzero()[0]))

        for epoch in range(0, self.max_epochs):
            self._train_epoch(X, train_users)
            self._evaluate(val_in, val_out, val_users)

        # Load best model, not necessarily last model
        self.load(self.stopping_criterion.best_value)

        return self

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
            loss = self.criterion(X_pred, mu, logvar, X, anneal=self._beta)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            self.steps += 1

        end_time = time.time()

        logger.info(
            f"Processed one batch in {end_time-start_time} s. Training Loss = {train_loss}"
        )

    def _evaluate(self, val_in: csr_matrix, val_out: csr_matrix, users: List[int]):
        val_loss = 0.0
        # Set to evaluation
        self.model_.eval()

        with torch.no_grad():
            val_X = naive_sparse2tensor(val_in).to(self.device)
            # Get value for parameter Beta

            val_X_pred, mu, logvar = self.model_(val_X)
            loss = self.criterion(val_X_pred, mu, logvar, val_X, anneal=self._beta)
            val_loss += loss.item()

            val_X_pred_cpu = csr_matrix(val_X_pred.cpu())
            val_X_true = val_out

            self.stopping_criterion.calculate(val_X_true, val_X_pred_cpu)

        logger.info(
            f"Evaluation Loss = {val_loss}, NDCG@100 = {self.stopping_criterion.value}"
        )

        if self.stopping_criterion.is_best:
            logger.info("Model improved. Storing better model.")
            self.save()

        self.stopping_criterion.reset()

    def load(self, value):
        # TODO Give better names
        with open(f"{self.name}_ndcg_100_{value}.trch", "rb") as f:
            self.model_ = torch.load(f)

    def save(self):
        with open(
            f"{self.name}_ndcg_100_{self.stopping_criterion.value}.trch", "wb"
        ) as f:
            torch.save(self.model_, f)

    def predict(self, X, user_ids=None):
        check_is_fitted(self)

        users = list(set(X.nonzero()[0]))

        tensorX = naive_sparse2tensor(X[users, :]).to(self.device)

        tensorX_pred, _, _ = self.model_(tensorX)

        # [[0, 1, 2], [3, 4, 5]] -> [0, 1, 2, 3, 4, 5]
        V = tensorX_pred.cpu().flatten().detach().numpy()  # Flattens row-major.
        # -> [1, 2] -> 1, 1, 1, 2, 2, 2 (2 users x 3 items)
        U = np.repeat(users, X.shape[1])
        # -> [1, 2, 3] -> 1, 2, 3, 1, 2, 3 (2 users x 3 items)
        I = list(range(0, X.shape[1])) * len(users)

        X_pred = csr_matrix((V, (U, I)), shape=X.shape)

        return X_pred


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


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

        # TODO Make it possible to go deeper?
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

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def encode(self, x):
        h = F.normalize(x)
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
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD
