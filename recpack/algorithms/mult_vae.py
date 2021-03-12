import logging
import time
from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
import numpy as np

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.stopping_criterion import StoppingCriterion
from recpack.algorithms.util import naive_sparse2tensor
from recpack.splitters.splitter_base import batch


logger = logging.getLogger("recpack")


class MultVAE(TorchMLAlgorithm):
    """MultVAE Algorithm as first discussed in
        'Variational Autoencoders for Collaborative Filtering',
        D. Liang et al. @ KDD2018.

    Default values were taken from the paper.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import MultVAE

        # Since MultVAE uses iterative optimisation, it needs validation data
        # To decide which of the iterations yielded the best model
        # This validation data should be split into an input and output matrix.
        # In this example the data has been split in a strong generalization fashion
        X = csr_matrix(np.array(
            [[1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]])
        )
        x_val_in = csr_matrix(np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]])
        )
        x_val_out = csr_matrix(np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]])
        )

        algo = MultVAE(num_components=2, batch_size=3, max_epochs=4)
        # Fit algorithm
        algo.fit(X, (x_val_in, x_val_out))

        # Recommend for the validation input data,
        # so we can inspect what the model learned
        # In a realistic setting you would have a test dataset
        # To predict for
        predictions = algo.predict(x_val_in)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    :param batch_size: Batch size for SGD,
                        defaults to 500
    :type batch_size: int, optional
    :param max_epochs: Maximum number of epochs (iterations),
                        defaults to 200
    :type max_epochs: int, optional
    :param learning_rate: Learning rate, defaults to 1e-4
    :type learning_rate: [type], optional
    :param seed: Random seed for Torch, provided for reproducibility,
                    defaults to 42.
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
    :param stop_early: Use early stopping during optimisation,
        defaults to False
    :type stop_early: bool, optional
    :param save_best_to_file: If True, the best model is saved to disk after fit.
    :type save_best_to_file: bool, optional
    """

    def __init__(
        self,
        batch_size=500,
        max_epochs=200,
        learning_rate=1e-4,
        seed=42,
        dim_bottleneck_layer=200,
        dim_hidden_layer=600,
        max_beta=0.2,
        anneal_steps=200000,
        dropout=0.5,
        stopping_criterion="ndcg",
        stop_early: bool = False,
        save_best_to_file=False,
    ):

        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            StoppingCriterion.create(stopping_criterion, stop_early=stop_early),
            seed,
            save_best_to_file=save_best_to_file,
        )

        self.stop_early = stop_early

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
        start_time = time.time()
        losses = []
        # Set to training
        self.model_.train()

        users = list(set(train_data.nonzero()[0]))

        np.random.shuffle(users)

        for batch_idx, user_batch in enumerate(batch(users, self.batch_size)):
            X = naive_sparse2tensor(train_data[user_batch, :]).to(self.device)

            # Clear gradients
            self.optimizer.zero_grad()
            X_pred, mu, logvar = self.model_(X)
            loss = self._compute_loss(X, X_pred, mu, logvar)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

            self.steps += 1

        end_time = time.time()

        logger.info(
            f"Processed one batch in {end_time-start_time} s."
            f" Training Loss = {np.mean(losses)}"
        )

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

    def _predict(self, X: csr_matrix, users: List[int] = None) -> np.ndarray:
        """Predict scores for matrix X, given the selected users.

        If there are no selected users, you can assume X is a full matrix,
        and users can be retrieved as the nonzero indices in the X matrix.

        :param X: Matrix of user item interactions
        :type X: csr_matrix
        :param users: users selected for recommendation
        :type users: List[int]
        :return: dense matrix of scores per user item pair.
        :rtype: np.ndarray
        """

        in_tensor = naive_sparse2tensor(X).to(self.device)

        out_tensor, _, _ = self.model_(in_tensor)
        return out_tensor.detach().cpu().numpy()


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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            nn.init.normal_(
                layer.bias, mean=0, std=0.001
            )  # TODO This should be truncated normal


# TODO: Move out of this file / move the KLD loss and BCE loss out of the file
def vae_loss_function(recon_x, mu, logvar, x, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD
