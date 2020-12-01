import logging
import tempfile
from typing import Tuple
import warnings

import numpy as np
from scipy.sparse import csr_matrix
import scipy.spatial
from sklearn.utils.validation import check_is_fitted

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from recpack.algorithms.base import Algorithm
from recpack.algorithms.samplers import warp_sample_pairs
from recpack.algorithms.util import (
    StoppingCriterion,
    EarlyStoppingException,
)
from recpack.data.matrix import Matrix, to_csr_matrix


logger = logging.getLogger("recpack")


# TODO: make WARP LOSS usable as stopping criterion
class CML(Algorithm):
    def __init__(
        self,
        num_components: int,
        margin: float,
        learning_rate: float,
        num_epochs: int,
        seed: int = 42,
        batch_size: int = 50000,
        U: int = 20,
        stopping_criterion: str = "recall",
        save_best_to_file=False,
        approximate_user_vectors=False,
        disentangle=False,
    ):
        """
        Pytorch Implementation of
        Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
        http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

        Version without features, referred to as CML in the paper.

        :param num_components: Embedding dimension
        :type num_components: int
        :param margin: Hinge loss margin. Required difference in score, smaller than which we will consider a negative sample item in violation
        :type margin: float
        :param learning_rate: Learning rate for AdaGrad optimization
        :type learning_rate: float
        :param num_epochs: Number of epochs
        :type num_epochs: int
        :param seed: Random seed used for initialization, defaults to 42
        :type seed: int, optional
        :param batch_size: Sample batch size, defaults to 50000
        :type batch_size: int, optional
        :param U: Number of negative samples used in WARP loss function for every positive sample, defaults to 20
        :type U: int, optional
        :param stopping_criterion: Used to identify the best model computed thus far.
            The string indicates the name of the stopping criterion.
            Which criterions are available can be found at StoppingCriterion.FUNCTIONS
            Defaults to 'recall'
        :type stopping_criterion: str, optional
        :param save_best_to_file: If True, the best model is saved to disk after fit.
        :type save_best_to_file: bool
        :param approximate_user_vectors: If True, make an approximation of user vectors for unknown users.
        :type approximate_user_vectors: bool
        :param disentangle: Disentangle embedding dimensions by adding a covariance loss term to regularize.
        :type disentangle: bool
        """
        self.num_components = num_components
        self.margin = margin
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.U = U
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        self.save_best_to_file = save_best_to_file
        self.stopping_criterion = StoppingCriterion.create(stopping_criterion)
        self.approximate_user_vectors = approximate_user_vectors
        self.disentangle = disentangle

    def _init_model(self, X):
        """
        Initialize model.

        :param num_users: Number of users.
        :type num_users: int
        :param num_items: Number of items.
        :type num_items: int
        """
        num_users, num_items = X.shape
        self.model_ = CMLTorch(
            num_users, num_items, num_components=self.num_components
        ).to(self.device)

        self.optimizer = optim.Adagrad(self.model_.parameters(), lr=self.learning_rate)

        self.known_users_ = set(X.nonzero()[0])

        self.best_model_ = tempfile.NamedTemporaryFile()
        torch.save(self.model_, self.best_model_)

    @property
    def filename(self):
        return f"{self.name}_loss_{self.stopping_criterion.best_value}.trch"

    # TODO: loading just the model is not enough to reuse it.
    # We also need known users etc. Pickling seems like a good way to go here.
    def load(self, filename: str):
        """
        Load a previously computed model.

        :param filename: path to file to load
        :type filename: str
        """
        with open(filename, "rb") as f:
            self.model_ = torch.load(f)

    def save(self):
        """Save the current model to disk"""
        with open(self.filename, "wb") as f:
            torch.save(self.model_, f)

    def _save_best(self):
        """Save the best model in a temp file"""

        # First removes the old saved file,
        # and then creates a new one in which the model is saved
        self.best_model_.close()
        self.best_model_ = tempfile.NamedTemporaryFile()
        torch.save(self.model_, self.best_model_)

    def _load_best(self):
        self.best_model_.seek(0)
        self.model_ = torch.load(self.best_model_)

    def fit(self, X: Matrix, validation_data: Tuple[Matrix, Matrix]):
        """
        Fit the model on the X dataset, and evaluate model quality on validation_data.

        :param X: The training data matrix
        :type X: Matrix
        :param validation_data: Validation data, as matrix to be used as input and matrix to be used as output
        :type validation_data: Tuple[csr_matrix, csr_matrix]
        """
        X, validation_data = to_csr_matrix((X, validation_data), binary=True)

        self._init_model(X)
        try:
            for epoch in range(self.num_epochs):
                self._train_epoch(X)
                self._evaluate(validation_data)
        except EarlyStoppingException:
            pass

        # Load the best of the models during training.
        self._load_best()

        # If saving turned on: save to file.
        if self.save_best_to_file:
            self.save()

        return self

    def _batch_predict(
        self, users_to_predict_for: set, W_as_tensor: torch.Tensor, H_as_tensor: torch.Tensor
    ) -> csr_matrix:
        # TODO Update docstring
        """
        Method for internal use only. Users should use `predict`.

        :param X: interaction matrix, should have same dimensions as the matrix the model was fit on.
        :type X: csr_matrix
        :raises an: AssertionError when the input and model's number of items and users are incongruent.
        :return: csr matrix of same shape, with recommendations.
        :rtype: csr_matrix
        """
        U = torch.LongTensor(list(users_to_predict_for)).repeat_interleave(
            self.model_.num_items
        )
        I = torch.arange(self.model_.num_items).repeat(len(users_to_predict_for))

        num_interactions = U.shape[0]

        V = np.array([])

        for batch_ix in range(0, num_interactions, 10000):
            batch_U = U[batch_ix : min(num_interactions, batch_ix + 10000)].to(
                self.device
            )
            batch_I = I[batch_ix : min(num_interactions, batch_ix + 10000)].to(
                self.device
            )

            batch_w_U = W_as_tensor[batch_U]
            batch_h_I = H_as_tensor[batch_I]

            # Score = -distance
            batch_V = - nn.PairwiseDistance(p=2)(
                batch_w_U, batch_h_I
            ).detach().cpu().numpy()

            V = np.append(V, batch_V)

        X_pred = csr_matrix((V, (U.numpy(), I.numpy())), shape=(self.model_.num_users, self.model_.num_items))

        return X_pred

    def predict(self, X: Matrix) -> csr_matrix:
        """
        Predict recommendations for each user with at least a single event in their history.

        :param X: interaction matrix, should have same size as model.
        :type X: Matrix
        :raises an: AssertionError when the input and model's number of items and users are incongruent.
        :return: csr matrix of same shape, with recommendations.
        :rtype: csr_matrix
        """
        check_is_fitted(self)

        X = to_csr_matrix(X, binary=True)

        assert X.shape == (self.model_.num_users, self.model_.num_items)

        W_as_tensor = self.model_.W.state_dict()["weight"]
        H_as_tensor = self.model_.H.state_dict()["weight"]

        users = set(X.nonzero()[0])

        if self.approximate_user_vectors:
            users_to_predict_for = users
        else:
            users_to_predict_for = users.intersection(self.known_users_)
            users_we_cannot_predict_for = users.difference(self.known_users_)

            if users_we_cannot_predict_for:
                warnings.warn(
                    f"Cannot make predictions for users: {users_we_cannot_predict_for}. No embeddings for these users."
                )

            W_as_tensor = self.approximate_W(X, W_as_tensor, H_as_tensor)

        X_pred = self._batch_predict(users_to_predict_for, W_as_tensor, H_as_tensor)

        self._check_prediction(X_pred, X)

        return X_pred

    def _train_epoch(self, train_data: csr_matrix):
        """
        Train model for a single epoch. Uses sampler to generate samples,
        and loop through them in batches of self.batch_size.
        After each batch, update the parameters according to gradients.

        :param train_data: interaction matrix
        :type train_data: csr_matrix
        """
        train_loss = 0.0
        self.model_.train()

        for users, positives_batch, negatives_batch in tqdm(
            warp_sample_pairs(train_data, U=self.U, batch_size=self.batch_size)
        ):
            users = users.to(self.device)
            positives_batch = positives_batch.to(self.device)
            negatives_batch = negatives_batch.to(self.device)

            self.optimizer.zero_grad()

            current_batch_size = users.shape[0]

            dist_pos_interaction = self.model_.forward(users, positives_batch)
            dist_neg_interaction_flat = self.model_.forward(
                users.repeat_interleave(self.U),
                negatives_batch.reshape(current_batch_size * self.U, 1).squeeze(-1),
            )
            dist_neg_interaction = dist_neg_interaction_flat.reshape(
                current_batch_size, -1
            )

            loss = self._compute_loss(
                dist_pos_interaction.unsqueeze(-1), dist_neg_interaction
            )
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        logger.info(f"training loss = {train_loss}")

    def _evaluate(self, validation_data: Tuple[csr_matrix, csr_matrix]):
        """
        Perform evaluation step, samples get drawn
        from the validation data, and compute loss.

        If loss improved over previous epoch, store the model, and update best value.

        :param validation_data: validation data interaction matrix
        :type validation_data: Tuple[csr_matrix, csr_matrix]
        """
        self.model_.eval()
        with torch.no_grad():
            # Need to make a selection, otherwise this step is way too slow.
            val_data_in, val_data_out = validation_data
            nonzero_users = list(set(val_data_in.nonzero()[0]))
            validation_users = np.random.choice(
                nonzero_users, size=min(1000, len(nonzero_users)), replace=False
            )

            val_data_in_selection = csr_matrix(val_data_in.shape)
            val_data_in_selection[validation_users, :] = val_data_in[
                validation_users, :
            ]

            val_data_out_selection = csr_matrix(val_data_in.shape)
            val_data_out_selection[validation_users, :] = val_data_out[
                validation_users, :
            ]

            X_val_pred = self.predict(val_data_in_selection)
            X_val_pred[val_data_in_selection.nonzero()] = 0
            better = self.stopping_criterion.update(val_data_out_selection, X_val_pred)

            if better:
                self._save_best()

    def _compute_loss(
        self, dist_pos_interaction: torch.Tensor, dist_neg_interaction: torch.Tensor
    ) -> torch.Tensor:
        """
        Method for internal use only. Please use `warp_loss` or `covariance_loss`.

        Compute differentiable loss function.

        :param dist_pos_interaction: Tensor containing distances between positive sample pairs.
        :type dist_pos_interaction: torch.Tensor
        :param dist_neg_interaction: Tensor containing distance between negative sample pairs.
        :type dist_neg_interaction: torch.Tensor
        :return: 0-D Tensor containing computed loss value.
        :rtype: torch.Tensor
        """
        loss = warp_loss(
            dist_pos_interaction,
            dist_neg_interaction,
            self.margin,
            self.model_.num_items,
            self.U,
        )

        if self.disentangle:
            loss += covariance_loss(self.model_.H, self.model_.W)

        return loss

    def approximate_W(
        self, X: csr_matrix, W_as_tensor: torch.Tensor, H_as_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Approximate user embeddings of unknown users by setting them to the average of item embeddings the user visited.

        :param X: Interaction matrix of shape (num_users, num_items)
        :type X: csr_matrix
        """
        U = set(X.nonzero()[0])
        users_to_approximate = U.difference(self.known_users_)

        W_as_tensor_approximated = W_as_tensor.clone()

        for user in users_to_approximate:
            item_indices = X[user].nonzero()[1]

            W_as_tensor_approximated[user] = H_as_tensor[
                torch.LongTensor(item_indices).to(self.device)
            ].mean(axis=0)

        return W_as_tensor_approximated


def covariance_loss(H: nn.Embedding, W: nn.Embedding) -> torch.Tensor:
    """
    Implementation of covariance loss as described in
    Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    The loss term is used to penalize covariance between embedding dimensions and
    thus disentangle these embedding dimensions.

    It is assumed H and W are embeddings in the same space.

    :param H: Item embedding
    :type H: nn.Embedding
    :param W: User Embedding
    :type W: nn.Embedding
    :return: Covariance loss term
    :rtype: torch.Tensor
    """
    W_as_tensor = next(W.parameters())
    H_as_tensor = next(H.parameters())

    # Concatenate them together. They live in the same metric space, so share the same dimensions.
    #  X is a matrix of shape (|users| + |items|, num_dimensions)
    X = torch.cat([W_as_tensor, H_as_tensor], dim=0)

    # Zero mean
    X = X - X.mean(dim=0)

    cov = X.matmul(X.T)

    # Per element covariance, excluding the variance of individual random variables.
    return cov.fill_diagonal_(0).sum() / (X.shape[0] * X.shape[1])


def warp_loss(
    dist_pos_interaction: torch.Tensor,
    dist_neg_interaction: torch.Tensor,
    margin: float,
    J: int,
    U: int,
) -> torch.Tensor:
    """
    Implementation of
    WARP loss as described in
    Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf
    based on
    J. Weston, S. Bengio, and N. Usunier. Large scale image annotation:
    learning to rank with joint word-image embeddings. Machine learning, 81(1):21–35, 2010.

    Adds a loss penalty for every negative sample that is not at least
    an amount of margin further away from the reference sample than a positive
    sample. This per sample loss penalty has a weight proportional to the
    amount of samples in the negative sample batch were "misclassified",
    i.e. closer than the positive sample.

    :param dist_pos_interaction: Tensor of distances between positive sample and reference sample.
    :type dist_pos_interaction: torch.Tensor
    :param dist_neg_interaction: Tensor of distances between negatives samples and reference sample.
    :type dist_neg_interaction: torch.Tensor
    :param margin: Required margin between positive and negative sample.
    :type margin: float
    :param J: Total number of items in the dataset.
    :type J: int
    :param U: Number of negative samples used for every positive sample.
    :type U: int
    :return: 0-D Tensor containing WARP loss.
    :rtype: torch.Tensor
    """
    dist_diff_pos_neg_margin = margin + dist_pos_interaction - dist_neg_interaction

    # Largest number is "most wrongly classified", f.e.
    # pos = 0.1, margin = 0.1, neg = 0.15 => 0.1 + 0.1 - 0.15 = 0.05 > 0
    # pos = 0.1, margin = 0.1, neg = 0.08 => 0.1 + 0.1 - 0.08 = 0.12 > 0
    most_wrong_neg_interaction, _ = dist_diff_pos_neg_margin.max(dim=-1)

    most_wrong_neg_interaction[most_wrong_neg_interaction < 0] = 0

    M = (dist_diff_pos_neg_margin > 0).sum(axis=-1).float()
    # M * J / U =~ rank(pos_i)
    w = torch.log((M * J / U) + 1)

    loss = (most_wrong_neg_interaction * w).sum()

    return loss


class CMLTorch(nn.Module):
    """
    Implementation of CML in PyTorch.

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

        self.W = nn.Embedding(num_users, num_components)  # User embedding
        self.H = nn.Embedding(num_items, num_components)  # Item embedding

        self.std = 1 / num_components ** 0.5
        # Initialise embeddings to a random start
        nn.init.normal_(self.W.weight, std=self.std)
        nn.init.normal_(self.H.weight, std=self.std)

        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, U: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidian distance between user embedding (w_u) and item embedding (h_i)
        for every user and item pair in U and I.

        :param U: User identifiers.
        :type U: torch.Tensor
        :param I: Item identifiers.
        :type I: torch.Tensor
        :return: Euclidian distances between user-item pairs.
        :rtype: torch.Tensor
        """
        w_U = self.W(U)
        h_I = self.H(I)

        # U and I are unrolled -> [u=0, i=1, i=2, i=3] -> [0, 1], [0, 2], [0,3]

        return self.pdist(w_U, h_I)
