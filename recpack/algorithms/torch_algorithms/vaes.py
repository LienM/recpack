import time
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np

from scipy.sparse import csr_matrix
from recpack.splitters.splitter_base import batch
from recpack.algorithms.algorithm_base import Algorithm

from recpack.metrics import NDCGK


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
        self.NDCG = 0
        self.num_users = 0

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

        self.model = None
        self.optimizer = None
        self.criterion = vae_loss_function
        self.stopping_criterion = StoppingCriterion(100)

    @property
    def _beta(self):
        """
        Parameter beta is slowly annealed from 0 to self.max_beta over self.anneal_steps.
        """
        return (
            self.max_beta
            if self.steps > self.anneal_steps
            else self.steps / self.anneal_steps
        )

    def _init_model(self, dim_input_layer: int):
        self.model = MultiVAETorch(dim_input_layer, self.dim_hidden_layer, self.dim_bottleneck_layer).to(
            self.device
        )

    def fit(self, train_data: csr_matrix, val_in: csr_matrix, val_out: csr_matrix):
        """
        Perform self.max_epoch training iterations over the complete training dataset.
        After each epoch, reports value of self.stopping_criterion and stores
        the model if it is better than the last.
        """

        self._init_model(train_data.shape[1])

        # TODO Investigate Multi-GPU
        # multi_gpu = False
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        #     multi_gpu = True

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        train_users = list(set(train_data.nonzero()[0]))
        val_users = list(set(val_in.nonzero()[0]))

        for epoch in range(0, self.max_epochs):
            self._train_epoch(train_data, train_users)
            self._evaluate(val_in, val_out, val_users)

    def _train_epoch(self, train_data, users):
        start_time = time.time()
        train_loss = 0.0
        # Set to training
        self.model.train()

        np.random.shuffle(users)

        for batch_idx, user_batch in enumerate(batch(users, self.batch_size)):
            X = naive_sparse2tensor(train_data[user_batch, :]).to(self.device)

            # Clear gradients
            self.optimizer.zero_grad()
            # Get value for parameter Beta
            beta = self._get_beta(self.steps)

            X_pred, mu, logvar = self.model(X)
            loss = self.criterion(X_pred, mu, logvar, X, anneal=beta)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            self.steps += 1

        end_time = time.time()

        self.logger.info(
            f"Processed one batch in {end_time-start_time} s. Training Loss = {train_loss}"
        )

    def _evaluate(self, val_in, val_out, users):
        val_loss = 0.0
        # Set to evaluation
        self.model.eval()

        with torch.no_grad():
            for batch_idx, user_batch in enumerate(batch(users, self.batch_size)):
                val_X = naive_sparse2tensor(val_in[user_batch, :]).to(self.device)
                # Get value for parameter Beta
                beta = self._get_beta(self.steps)

                val_X_pred, mu, logvar = self.model(val_X)
                loss = self.criterion(val_X_pred, val_X, mu, logvar, beta)
                val_loss += loss.item()
                self.optimizer.step()

                val_X_pred_cpu = csr_matrix(val_X_pred.cpu())
                val_X_true = val_out[user_batch, :]

                self.stopping_criterion.update(val_X_pred_cpu, val_X_true)

        self.logger.info(
            f"Evaluation Loss = {val_loss}, NDCG@100 = {self.stopping_criterion.value}"
        )

        if self.stopping_criterion.is_best:
            self.logger("Model improved. Storing better model.")
            self.save()

        self.stopping_criterion.refresh()

    def load(self, value):
        # TODO Give better names
        with open(f"{self.name}_ndcg_100_{value}.trch", "rb") as f:
            self.model = torch.load(f)

    def save(self):
        with open(
            f"{self.name}_ndcg_100_{self.stopping_criterion.value}.trch", "wb"
        ) as f:
            torch.save(self.model, f)

    def predict(self, X):
        X_pred, _, _ = self.model(X)

        return X_pred

    @property
    def name(self):
        return f"mult_vae"


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i: row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(
        indices, torch.from_numpy(values).float(), [samples, features]
    )
    return t


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

        self.layers = nn.ModuleList([
            self.q_in_hid_layer,
            self.q_hid_bn_layer,
            self.p_bn_hid_layer,
            self.p_hid_out_layer
        ])

        self.drop = nn.Dropout(p=dropout, inplace=False)
        self._init_weights()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, x):
        h = F.normalize(x)
        h = self.drop(h)

        h = self.q_in_hid_layer(h)
        h = self.tanh(h)

        h = self.q_hid_bn_layer(h)

        # TODO This is a terrible hack. Do something about it.
        mu = h[:, :self.dim_bottleneck_layer]
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
