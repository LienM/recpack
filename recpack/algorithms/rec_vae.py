# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from copy import deepcopy
import logging
from typing import List, Tuple, Optional

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.util import swish, log_norm_pdf, naive_sparse2tensor, get_batches


logger = logging.getLogger("recpack")


class RecVAE(TorchMLAlgorithm):
    """RecVAE Algorithm as first discussed in
    'RecVAE: a New Variational Autoencoder for
    Top-NRecommendations with Implicit Feedback',
    I. Shenbin et al. @ WSDM2020.

    The RecVAE model is an extension of :class:`recpack.algorithms.MultVAE`.
    With the biggest changes being: alternating updates to decoder and encoder during training
    and a novel composite prior distribution.


    Default values for parameters were taken from the paper.

    :param batch_size: Batch size for SGD, defaults to 500
    :type batch_size: int, optional
    :param max_epochs: Maximum number of epochs (iterations),
        defaults to 200
    :type max_epochs: int, optional
    :param n_enc_epochs: The training happens alternating,
        in every epoch this amount of optimizations happen
        for the encoder network
    :type n_enc_epochs: int
    :param n_dec_epochs: The number of times to optimize
        the decoder network each epoch.
    :type n_dec_epochs: int
    :param seed: Random seed for Torch, provided for reproducibility,
        defaults to None.
    :type seed: int, optional
    :param learning_rate: Learning rate, defaults to 1e-4
    :type learning_rate: float, optional
    :param dim_bottleneck_layer: Size of the latent representation,
        defaults to 200
    :type dim_bottleneck_layer: int, optional
    :param dim_hidden_layer: Dimension of the hidden layer, defaults to 600
    :type dim_hidden_layer: int, optional
    :param gamma: Parameter defining regularization of the KL loss
        together with the norm of the output,
        defaults to 1
    :type gamma: float, optional
    :param beta: Regularization parameter of the KL loss,
        only used if gamma = None, defaults to None
    :type beta: float, optional
    :param dropout: Dropout rate to apply at the inputs, defaults to 0.5
    :type dropout: float, optional
    :param stopping_criterion: Used to identify the best model computed thus far.
        The string indicates the name of the stopping criterion.
        Which criterions are available can be found at
        :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`.
        Defaults to 'ndcg'
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
        Defaults to False.
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
        learning_rate: float = 5e-4,
        n_enc_epochs: int = 3,
        n_dec_epochs: int = 1,
        seed: Optional[int] = None,
        dim_bottleneck_layer: int = 200,
        dim_hidden_layer: int = 600,
        gamma: Optional[float] = 0.005,
        beta: Optional[float] = None,
        dropout: float = 0.5,
        stopping_criterion: str = "ndcg",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        save_best_to_file: bool = False,
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

        self.n_enc_epochs = n_enc_epochs
        self.n_dec_epochs = n_dec_epochs

        self.dim_hidden_layer = dim_hidden_layer
        self.dim_bottleneck_layer = dim_bottleneck_layer

        # Either gamma or beta, with gamma having precedence
        self.gamma = gamma
        if not self.gamma:
            self.beta = beta
        else:
            self.beta = None

        self.steps = 0
        self.dropout = dropout

        self.enc_optimizer = None
        self.dec_optimizer = None

    def _init_model(self, X: csr_matrix):
        """
        Initialize Torch model and optimizer.

        :param X: Dimension of the input layer
                                (corresponds to number of items)
        :type X: int
        """

        dim_input_layer = X.shape[1]

        self.model_ = RecVAETorch(
            dim_input_layer=dim_input_layer,
            dim_hidden_layer=self.dim_hidden_layer,
            dim_bottleneck_layer=self.dim_bottleneck_layer,
            dropout_rate=self.dropout,
            gamma=self.gamma,
            beta=self.beta,
        ).to(self.device)

        self.enc_optimizer = optim.Adam(self.model_.encoder.parameters(), lr=self.learning_rate)
        self.dec_optimizer = optim.Adam(self.model_.decoder.parameters(), lr=self.learning_rate)

    def _compute_loss(
        self,
        X: torch.Tensor,
        X_pred: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the prediction loss.

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
        if self.gamma:
            norm = X.sum(dim=-1)
            kl_weight = self.gamma * norm
        elif self.beta:
            kl_weight = self.beta

        mll = (F.log_softmax(X_pred, dim=-1) * X).sum(dim=-1).mean()

        z = self.model_.reparameterize(mu, logvar)
        kld = (log_norm_pdf(z, mu, logvar) - self.model_.prior(X, z)).sum(dim=-1).mul(kl_weight).mean()
        negative_elbo = -(mll - kld)

        return negative_elbo

    def _train_partial(self, train_data, users, optimizer):
        """
        Part of the train mathod,
        optimizes a single part of the combined Neural network.
        The optimizer should be passed in the argument.
        """
        losses = []
        np.random.shuffle(users)

        for batch_idx, user_batch in enumerate(get_batches(users, self.batch_size)):
            X = naive_sparse2tensor(train_data[user_batch, :]).to(self.device)

            # Clear gradients
            optimizer.zero_grad()

            # Optimize
            X_pred, mu, logvar = self.model_(X)
            loss = self._compute_loss(X, X_pred, mu, logvar)
            loss.backward()

            losses.append(loss.item())
            optimizer.step()

            self.steps += 1

        return losses

    def _train_epoch(self, train_data: csr_matrix):
        """
        Perform one training epoch.
        Data is processed in batches of self.batch_size users.

        :param train_data: Training data (UxI)
        :type train_data: [type]
        :param users: List of all users in the training dataset.
        :type users: List[int]
        """

        users = list(set(train_data.nonzero()[0]))

        for _ in range(self.n_enc_epochs):
            losses = self._train_partial(train_data, users, self.enc_optimizer)

        self.model_.update_prior()

        for _ in range(self.n_dec_epochs):
            losses = self._train_partial(train_data, users, self.dec_optimizer)

        return losses

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


class CompositePrior(nn.Module):
    def __init__(
        self,
        dim_hidden_layer: int,
        dim_bottleneck_layer: int,
        dim_input_layer: int,
        mixture_weights=[3 / 20, 3 / 4, 1 / 10],
    ):
        """
        Composite prior, based on a gaussian prior, a uniform prior and
            the posterior of the previously trained model.

        :param dim_hidden_layer: The size of the hidden dimensions
        :type dim_hidden_layer: int
        :param dim_bottleneck_layer: The size of the latent dimension
        :type dim_bottleneck_layer: int
        :param dim_input_layer: The number of features in the input
        :type dim_input_layer: int
        :param mixture_weights: the weights to combine the priors with.
            In order: standard prior, post_prior and uniform prior,
            defaults to [3/20, 3/4, 1/10]
        :type mixture_weights: list, optional
        """
        super(CompositePrior, self).__init__()

        self.mixture_weights = mixture_weights

        self.mu_prior = nn.Parameter(torch.Tensor(1, dim_bottleneck_layer), requires_grad=False)
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(torch.Tensor(1, dim_bottleneck_layer), requires_grad=False)
        self.logvar_prior.data.fill_(0)

        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, dim_bottleneck_layer), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)

        self.encoder_old = Encoder(dim_hidden_layer, dim_bottleneck_layer, dim_input_layer)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)

        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]

        density_per_gaussian = torch.stack(gaussians, dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)


class Encoder(nn.Module):
    def __init__(
        self,
        dim_hidden_layer: int,
        dim_bottleneck_layer: int,
        dim_input_layer: int,
        eps: float = 1e-1,
    ):
        """
        Encode part of the Neural network, takes data from the input,
            passes it through 5 hidden layers to get a latent representation.

        All hidden layers have the same dimension,
        the latent layer and input layer can have any dimensions needed.

        :param dim_hidden_layer: The number of hidden dimensions
        :type dim_hidden_layer: int
        :param dim_bottleneck_layer: The number of latent dimensions
        :type dim_bottleneck_layer: int
        :param dim_input_layer: The number of features in the input
        :type dim_input_layer: int
        :param eps: stabilization value for normalization of the hidden layers,
                    defaults to 1e-1
        :type eps: float, optional
        """
        super(Encoder, self).__init__()

        # Hidden layers
        self.fc1 = nn.Linear(dim_input_layer, dim_hidden_layer)
        self.ln1 = nn.LayerNorm(dim_hidden_layer, eps=eps)
        self.fc2 = nn.Linear(dim_hidden_layer, dim_hidden_layer)
        self.ln2 = nn.LayerNorm(dim_hidden_layer, eps=eps)
        self.fc3 = nn.Linear(dim_hidden_layer, dim_hidden_layer)
        self.ln3 = nn.LayerNorm(dim_hidden_layer, eps=eps)
        self.fc4 = nn.Linear(dim_hidden_layer, dim_hidden_layer)
        self.ln4 = nn.LayerNorm(dim_hidden_layer, eps=eps)
        self.fc5 = nn.Linear(dim_hidden_layer, dim_hidden_layer)
        self.ln5 = nn.LayerNorm(dim_hidden_layer, eps=eps)

        # Latent encoding layers for mean and variance
        self.fc_mu = nn.Linear(dim_hidden_layer, dim_bottleneck_layer)
        self.fc_logvar = nn.Linear(dim_hidden_layer, dim_bottleneck_layer)

    def forward(self, x, dropout_rate):
        """
        Encode the data in x to the latent dimension,
        randomly dropping some values, to add noise,
        which makes the encoder more robust.

        :param x: the input
        :type x: torch.tensor
        :param dropout_rate: the rate with which to drop values from the input
                             in order to add noise.
        :type dropout_rate: float
        :return: tuple of encoded values for x, both in the latent dimension.
                 1st is average, 2nd logarithmic variance?
        :rtype: tuple[torch.tensor, torch.tensor]
        """
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]

        # Drop some of the entries in the interactions to force the network to
        # do denoising.
        x = F.dropout(x, p=dropout_rate, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


class RecVAETorch(nn.Module):
    def __init__(
        self,
        dim_hidden_layer,
        dim_bottleneck_layer,
        dim_input_layer,
        gamma=1,
        beta=None,
        dropout_rate=0.5,
    ):
        """
        RecVAE torch module.

        The recVAE network consists of a separate encoder
        and decoder structure.
        Where the encoder is a deeper neural network, with several layers,
        the decoder is a single layer.

        The prior is a composite prior, used to compute a better loss function.

        :param dim_hidden_layer: The number of hidden dimensions
        :type dim_hidden_layer: int
        :param dim_bottleneck_layer: The number of latent dimensions
        :type dim_bottleneck_layer: int
        :param dim_input_layer: The number of features in the input
        :type dim_input_layer: int
        :param gamma: mutually exclusive with beta.
            If Gamma is used weigh the KL loss based on gamma * norm,
            defaults to 1
        :type gamma: int, optional
        :param beta: Beta weighting parameter in the KL loss, defaults to None.
                     If Gamma is set, this will never be used.
        :type beta: float, optional
        :param dropout_rate: the fraction of interactions to drop
                             when training, defaults to 0.5
        :type dropout_rate: float, optional
        """
        super(RecVAETorch, self).__init__()

        self.encoder = Encoder(dim_hidden_layer, dim_bottleneck_layer, dim_input_layer)
        self.prior = CompositePrior(dim_hidden_layer, dim_bottleneck_layer, dim_input_layer)
        self.decoder = nn.Linear(dim_bottleneck_layer, dim_input_layer)

        self.gamma = gamma
        # This way representation of the parameters will be better.
        if self.gamma is None:
            self.beta = beta
        else:
            self.beta = None

        self.dropout_rate = dropout_rate

    def reparameterize(self, mu, logvar):
        """
        During training we don't use the mean values,
            but we use the variance to sample a score
            based on a normal distribution around the average.

        :param mu: tensor with mean values for each latent feature
        :type mu: torch.tensor
        :param logvar: tensor with the logvariance
                       of the latent feature values.
        :type logvar: torch.tensor
        :return: a single tensor with the latent encoding.
                 If not training this is just the average,
                 when training, sampled around the mean.
        :rtype: [type]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, X) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Use the encoder and decoder to return the predicted autoencoded result.

        :param X: input tensor of the user ratings.
        :type X: torch.Tensor
        :return: A tuple, with predictions, averages and variances.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        mu, logvar = self.encoder(X, dropout_rate=self.dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)

        return x_pred, mu, logvar

    def update_prior(self):
        """
        The encoder in the prior is updated to the current encoder state.
        """
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))
