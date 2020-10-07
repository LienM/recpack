import logging

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as t_data

from recpack.algorithms.base import Algorithm


logger = logging.getLogger("recpack")

# TODO
# Check loss function, minimize
# Implement the different lambda's instead of the one lambda


class BPRMF(Algorithm):
    """Implements Matrix Factorization by using the BPR-OPT objective
    and SGD optimization.

    The BPR optimization aims to construct a factorization that optimally
    ranks interesting items (interacted with previously)
    above uninteresting or unknown items for all users.

    :param num_components: The size of the latent vectors for both users and items.
                            defaults to 100
    :type num_components: int, optional
    :param reg: The regularization, determines with how much to
                regularize the parameter values, defaults to 0.0
    :type reg: float, optional
    :param num_epochs: The max amount of epochs to train the model, defaults to 20
    :type num_epochs: int, optional
    :param learning_rate: The learning rate of the optimization procedure,
                            defaults to 0.01
    :type learning_rate: float, optional
    :param weight_decay: The decay to be used in SGD, defaults to 0.0001
    :type weight_decay: float, optional
    :param seed: seed to fix random numbers, to make results reproducible,
                    defaults to None
    :type seed: [type], optional
    """

    def __init__(
        self,
        num_components=100,
        lambda_h=0.0,
        lambda_w=0.0,
        num_epochs=20,
        learning_rate=0.01,
        sample_size=10_000,
        batch_size=1_000,
        # weight_decay=0.0001, Shouldn't be used, this a L2 penalty, but this is part of the loss function already
        seed=None,
    ):

        self.num_components = num_components
        self.lambda_h = lambda_h
        self.lambda_w = lambda_w
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.seed = seed
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        # placeholders, will get initialized in _init_model
        self.optimizer = None
        self.model_ = None
        self.steps = 0

    def _init_model(self, num_users, num_items):
        self.model_ = MFModule(
            num_users, num_items, num_components=self.num_components
        ).to(self.device)

        self.optimizer = optim.SGD(self.model_.parameters(), lr=self.learning_rate)
        # TODO We initialize this twice, kinda weird
        self.steps = 0

    def fit(self, X: csr_matrix):
        # TODO Store the best model, like in VAE

        self._init_model(X.shape[0], X.shape[1])

        for epoch in range(self.num_epochs):
            self._train_epoch(X)

            # TODO Validation loss/some form of early stopping?

            # TODO: early stopping
            # delta_loss = float(train_loss - last_loss)
            # if (abs(delta_loss) < 1e-5) and self.early_stop:
            #     print('Satisfy early stop mechanism')
            #   break

    def predict(self, X: csr_matrix):
        # TODO We can make it so that we can recommend for unknown users by giving them an embedding equal to the sum of all items viewed previously.
        # TODO Or raise an error
        users = list(set(X.nonzero()[0]))

        U = torch.LongTensor(users)
        I = torch.arange(X.shape[1])

        result = np.zeros(X.shape)
        result[users] = self.model_.forward(U, I).detach().cpu().numpy()
        return csr_matrix(result)

    def _train_epoch(self, train_data: csr_matrix):
        train_loss = 0.0
        self.model_.train()

        for d in tqdm(
            bootstrap_sample_pairs(
                train_data, batch_size=self.batch_size, sample_size=self.sample_size
            )
        ):
            users = d[:, 0]
            target_items = d[:, 1]
            negative_items = d[:, 2]

            self.optimizer.zero_grad()
            # TODO Maybe rename?
            positive_sim = self.model_.forward(users, target_items)
            negative_sim = self.model_.forward(users, negative_items)
            loss = self._compute_loss(positive_sim, negative_sim)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            self.steps += 1

        logger.info(f"training loss = {train_loss}")
        return train_loss

    def _compute_loss(self, positive_sim, negative_sim):
        distance = positive_sim - negative_sim
        # Probability of ranking given parameters
        elementwise_bpr_loss = torch.log(torch.sigmoid(distance))
        bpr_loss = -elementwise_bpr_loss.sum()

        #  Add regularization
        return (
            bpr_loss
            + self.lambda_h * self.model_.H.weight.norm()
            + self.lambda_w * self.model_.W.weight.norm()
        )


class MFModule(nn.Module):
    def __init__(self, num_users, num_items, num_components=100):
        super().__init__()

        self.num_components = num_components
        self.num_users = num_users
        self.num_items = num_items

        self.W = nn.Embedding(num_users, num_components)  # User embedding
        self.H = nn.Embedding(num_items, num_components)  # Item embedding

        # Initialise embeddings to a random start
        nn.init.normal_(self.W.weight, std=0.01)
        nn.init.normal_(self.H.weight, std=0.01)

    def forward(self, U: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        """
        Compute dot-product of user embedding (w_u) and item embedding (h_i)
        for every user and item pair in U and I.

        :param U: [description]
        :type U: [type]
        :param I: [description]
        :type I: [type]
        """
        w_U = self.W(U)
        h_I = self.H(I)

        return w_U.matmul(h_I.T)


def bootstrap_sample_pairs(X, batch_size=100, sample_size=10000):
    # Need positive and negative pair. Requires the existence of a positive for this item.
    positives = np.array(X.nonzero()).T  # As a (num_interactions, 2) numpy array
    num_positives = positives.shape[0]
    np.random.shuffle(positives)

    # Pick interactions at random, with replacement
    samples = np.random.choice(num_positives, size=(sample_size,), replace=True)

    # TODO Could be better to only yield this when required, to keep the memory footprint low.
    possible_negatives = np.random.randint(0, X.shape[1], size=(sample_size,))

    for start in range(0, sample_size, batch_size):
        sample_batch = samples[start : start + batch_size]
        positives_batch = positives[sample_batch]
        negatives_batch = possible_negatives[start : start + batch_size]
        while True:
            # Fix the negatives that are equal to the positives, if there are any
            mask = positives_batch[:, 1] == negatives_batch
            num_incorrect = np.sum(mask)

            if num_incorrect > 0:
                new_negatives = np.random.randint(0, X.shape[1], size=(num_incorrect,))
                negatives_batch[mask] = new_negatives
            else:
                # Exit the while loop
                break

        sample_pairs_batch = np.empty((positives_batch.shape[0], 3))
        sample_pairs_batch[:, :2] = positives_batch
        sample_pairs_batch[:, 2] = negatives_batch
        yield torch.LongTensor(sample_pairs_batch)
