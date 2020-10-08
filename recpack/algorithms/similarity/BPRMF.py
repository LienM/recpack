import logging

import math

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from recpack.algorithms.base import Algorithm


logger = logging.getLogger("recpack")


class StoppingCriterion:
    def __init__(self):
        self.best_value = np.inf
        self.converged = False

    def update(self, validation_loss):
        delta = self.best_value - validation_loss
        EPSILON = 1e-5
        if abs(delta) < EPSILON:
            # The scores are so close to the best solution
            # We assume convergence.
            self.converged = True

        if self.best_value > validation_loss:
            self.best_value = validation_loss


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

        self.stopping_criterion = StoppingCriterion()
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

    def load(self, value):
        # TODO Give better names
        with open(f"{self.name}_loss_{value}.trch", "rb") as f:
            self.model_ = torch.load(f)

    def save(self):
        with open(
            f"{self.name}_loss_{self.stopping_criterion.best_value}.trch", "wb"
        ) as f:
            torch.save(self.model_, f)

    def fit(self, X: csr_matrix, X_validation: csr_matrix):
        """Fit the model on the X dataset, and evaluate model quality on X_validation.

        :param X: The training data matrix.
        :type X: csr_matrix
        :param X_validation: The validation data matrix, should have same dimensions as X
        :type X_validation: csr_matrix
        """
        assert X.shape == X_validation.shape

        self._init_model(X.shape[0], X.shape[1])
        for epoch in range(self.num_epochs):
            self._train_epoch(X)
            self._evaluate(X_validation)

        # Load the best of the models during training.
        self.load(self.stopping_criterion.best_value)
        return

    def predict(self, X: csr_matrix):
        """Predict recommendations for each user with at least a single event in their history.

        :param X: interaction matrix, should have same size as model.
        :type X: csr_matrix
        :raises an: [description]
        :return: csr matrix of same shape, with recommendations.
        :rtype: [type]
        """
        # TODO We can make it so that we can recommend for unknown users by giving them an embedding equal to the sum of all items viewed previously.
        # TODO Or raise an error
        assert X.shape == (self.model_.num_users, self.model_.num_items)
        users = list(set(X.nonzero()[0]))

        U = torch.LongTensor(users)
        I = torch.arange(X.shape[1])

        result = np.zeros(X.shape)
        result[users] = self.model_.forward(U, I).detach().cpu().numpy()
        return csr_matrix(result)

    def _train_epoch(self, train_data: csr_matrix):
        """train a single epoch. Uses sampler to generate self.sample_size samples,
        and loop through them in batches of self.batch_size.
        After each batch, update the parameters according to gradients.

        :param train_data: interaction matrix.
        :type train_data: csr_matrix
        """
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

    def _evaluate(self, validation_data: csr_matrix):
        """Perform evaluation step, sample 20% of self.sample_size
        from the validation data, and compute loss.

        If loss improved over previous epoch, store the model, and update best value.

        :param validation_data: validation data interaction matrix.
        :type validation_data: csr_matrix
        """
        # TODO, quite a bit of code dupe from train epoch, check if we can consolidate.
        val_loss = 0.0
        self.model_.eval()
        with torch.no_grad():
            # Bootstrap 20% of the number of training samples from validation data.
            for d in bootstrap_sample_pairs(
                validation_data,
                batch_size=self.batch_size,
                sample_size=math.floor(self.sample_size * 0.2),
            ):
                users = d[:, 0]
                target_items = d[:, 1]
                negative_items = d[:, 2]

                # TODO Maybe rename?
                positive_sim = self.model_.forward(users, target_items)
                negative_sim = self.model_.forward(users, negative_items)
                loss = self._compute_loss(positive_sim, negative_sim)
                val_loss += loss.item()

            logger.info(f"validation loss = {val_loss}")
            self.stopping_criterion.update(val_loss)
            if val_loss == self.stopping_criterion.best_value:
                self.save()

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
    """MF torch module, encodes the embeddings and the forward functionality.

    :param num_users: the amount of users
    :type num_users: int
    :param num_items: the amount of items
    :type num_items: int
    :param num_components: The size of the embedding per user and item, defaults to 100
    :type num_components: int, optional
    """

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


def bootstrap_sample_pairs(
    X: csr_matrix, batch_size=100, sample_size=10000
) -> torch.LongTensor:
    """bootstrap sample triples from the data. Each triple contains (user, positive item, negative item).

    :param X: Interaction matrix
    :type X: csr_matrix
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :param sample_size: The number of samples to generate, defaults to 10000
    :type sample_size: int, optional
    :yield: tensor of shape (batch_size, 3), with user, positive item, negative item for each row.
    :rtype: torch.LongTensor
    """
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
