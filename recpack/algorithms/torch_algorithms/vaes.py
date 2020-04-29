import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


# Implementation taken from: https://github.com/younggyoseo/vae-cf-pytorch

class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super().__init__()
        self.p_dims = p_dims
        if q_dims:
            assert (
                q_dims[0] == p_dims[-1]
            ), "In and Out dimensions must be equal to each other"
            assert (
                q_dims[-1] == p_dims[0]
            ), "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])
            ]
        )
        self.p_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])
            ]
        )

        self.drop = nn.Dropout(dropout, p=dropout, inplace=False)
        self.init_weights()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, x):
        h = F.normalize(x)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, : self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1] :]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        # Done
        for layer in self.q_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias, mean=0, std=0.001)  # TODO This should be truncated normal

        for layer in self.p_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias, mean=0, std=0.001)  # TODO This should be truncated normal


def vae_loss_function(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD
