import logging
from typing import Tuple, List

import numpy as np
from scipy.sparse import csr_matrix

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.utils.validation import check_is_fitted

from recpack.algorithms.base import Algorithm
from recpack.algorithms.util import StoppingCriterion, EarlyStoppingException
from recpack.metrics.recall import recall_k


logger = logging.getLogger("recpack")


class CML(Algorithm):
    """
    Pytorch Implementation of
    [1] Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    Version without features, referred to as CML in the paper.
    """

    def __init__(
        self,
        num_components,
        margin,
        learning_rate,
        clip_norm,
        use_cov_loss,
        num_epochs,
        seed=42,
        batch_size=50000,
        U=20,
    ):
        # TODO Figure out clip_norm?
        self.num_components = num_components
        self.margin = margin
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.use_cov_loss = use_cov_loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.U = U
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        # TODO Make this configurable
        self.stopping_criterion = StoppingCriterion(
            recall_k, minimize=False, stop_early=False
        )

    def _init_model(self, num_users: int, num_items: int):
        """
        Initialize model.

        :param num_users: Number of users.
        :type num_users: int
        :param num_items: Number of items.
        :type num_items: int
        """
        self.model_ = CMLTorch(
            num_users, num_items, num_components=self.num_components
        ).to(self.device)

        self.optimizer = optim.Adagrad(self.model_.parameters(), lr=self.learning_rate)

    def load(self, validation_loss: float):
        with open(f"{self.name}_loss_{validation_loss}.trch", "rb") as f:
            self.model_ = torch.load(f)

    def save(self, validation_loss: float):
        with open(f"{self.name}_loss_{validation_loss}.trch", "wb") as f:
            torch.save(self.model_, f)

    def fit(self, X: csr_matrix, validation_data: Tuple[csr_matrix, csr_matrix]):
        """Fit the model on the X dataset, and evaluate model quality on validation_data.

        :param X: The training data matrix.
        :type X: csr_matrix
        :param validation_data: Validation data, as matrix to be used as input and matrix to be used as output.
        :type validation_data: Tuple[csr_matrix, csr_matrix]
        """

        self._init_model(X.shape[0], X.shape[1])
        try:
            for epoch in range(self.num_epochs):
                self._train_epoch(X)
                self._evaluate(validation_data)
        except EarlyStoppingException:
            pass

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
        check_is_fitted(self)
        # TODO We can make it so that we can recommend for unknown users by giving them an embedding equal to the sum of all items viewed previously.
        # TODO Or raise an error
        assert X.shape == (self.model_.num_users, self.model_.num_items)
        users = list(set(X.nonzero()[0]))

        # TODO
        # Probably need batch_predict as well
        U = (
            torch.LongTensor(users)
            .to(self.device)
            .repeat_interleave(self.model_.num_items)
        )
        I = torch.arange(X.shape[1]).to(self.device).repeat(len(users))

        # Score = -distance
        V = -self.model_.forward(U, I).detach().cpu().numpy()

        return csr_matrix((V, (U, I)), shape=X.shape)

    def _train_epoch(self, train_data: csr_matrix):
        """
        Train model for a single epoch. Uses sampler to generate samples,
        and loop through them in batches of self.batch_size.
        After each batch, update the parameters according to gradients.

        :param train_data: interaction matrix.
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

    def _evaluate(self, validation_data: csr_matrix):
        """
        Perform evaluation step, samples get drawn
        from the validation data, and compute loss.

        If loss improved over previous epoch, store the model, and update best value.

        :param validation_data: validation data interaction matrix.
        :type validation_data: csr_matrix
        """
        self.model_.eval()
        with torch.no_grad():
            X_val_pred = self.predict(validation_data[0])
            X_val_pred[validation_data[0].nonzero()] = 0
            # K = 50 as in the paper
            better = self.stopping_criterion.update(
                validation_data[1], X_val_pred, k=50
            )

            if better:
                self.save(self.stopping_criterion.best_value)

    def _compute_loss(self, dist_pos_interaction, dist_neg_interaction):

        loss = warp_loss(
            dist_pos_interaction,
            dist_neg_interaction,
            self.margin,
            self.model_.num_items,
            self.U,
            self.device
        )

        if self.use_cov_loss:
            loss += covariance_loss()

        return loss


def covariance_loss():
    # TODO Implement
    # Their implementation really confuses me
    # X = tf.concat((self.item_embeddings, self.user_embeddings), 0)
    # n_rows = tf.cast(tf.shape(X)[0], tf.float32)
    # X = X - (tf.reduce_mean(X, axis=0))
    # cov = tf.matmul(X, X, transpose_a=True) / n_rows

    # return tf.reduce_sum(tf.matrix_set_diag(cov, tf.zeros(self.embed_dim, tf.float32))) * self.cov_loss_weight
    return 0


def warp_loss(dist_pos_interaction, dist_neg_interaction, margin, J, U, device):
    dist_diff_pos_neg_margin = margin + dist_pos_interaction - dist_neg_interaction

    # Largest number is "most wrongly classified", f.e.
    # pos = 0.1, margin = 0.1, neg = 0.15 => 0.1 + 0.1 - 0.15 = 0.05 > 0
    # pos = 0.1, margin = 0.1, neg = 0.08 => 0.1 + 0.1 - 0.08 = 0.12 > 0
    most_wrong_neg_interaction, _ = dist_diff_pos_neg_margin.max(dim=-1)

    most_wrong_neg_interaction[most_wrong_neg_interaction < 0] = 0

    # pairwise_hinge_loss = torch.max(
    #     most_wrong_neg_interaction, torch.zeros(dist_pos_interaction.shape).to(device)
    # )

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

    def __init__(self, num_users, num_items, num_components=100):
        super().__init__()

        self.num_components = num_components
        self.num_users = num_users
        self.num_items = num_items

        self.W = nn.Embedding(num_users, num_components)  # User embedding
        self.H = nn.Embedding(num_items, num_components)  # Item embedding

        std = 1 / num_components ** 0.5
        # Initialise embeddings to a random start
        nn.init.normal_(self.W.weight, std=std)
        nn.init.normal_(self.H.weight, std=std)

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


# TODO Integrate sampling methods somewhere more logical
def warp_sample_pairs(X: csr_matrix, U=10, batch_size=100):
    """
    Sample U negatives for every user-item-pair (positive).

    :param X: Interaction matrix
    :type X: csr_matrix
    :param U: Number of negative samples for each positive, defaults to 10
    :type U: int, optional
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :yield: Iterator of torch.LongTensor of shape (batch_size, U+2). [User, Item, Negative Sample1, Negative Sample2, ...]
    :rtype: Iterator[torch.LongTensor]
    """
    # Need positive and negative pair. Requires the existence of a positive for this item.
    positives = np.array(X.nonzero()).T  # As a (num_interactions, 2) numpy array
    num_positives = positives.shape[0]
    np.random.shuffle(positives)

    for start in range(0, num_positives, batch_size):
        batch = positives[start: start + batch_size]
        users = batch[:, 0]
        positives_batch = batch[:, 1]

        # Important only for final batch, if smaller than batch_size
        true_batch_size = min(batch_size, num_positives - start)

        negatives_batch = np.random.randint(0, X.shape[1], size=(true_batch_size, U))
        while True:
            # Fix the negatives that are equal to the positives, if there are any
            mask = np.apply_along_axis(
                lambda col: col == positives_batch, 0, negatives_batch
            )
            num_incorrect = np.sum(mask)

            if num_incorrect > 0:
                new_negatives = np.random.randint(0, X.shape[1], size=(num_incorrect,))
                negatives_batch[mask] = new_negatives
            else:
                # Exit the while loop
                break

        yield torch.LongTensor(users), torch.LongTensor(
            positives_batch
        ), torch.LongTensor(negatives_batch)


class CMLWithFeatures(Algorithm):
    """
    Pytorch Implementation of
    [1] Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    Version with features, referred to as CML+F in the paper.
    """

    def __init__(
        self,
        embedding_dim,
        margin,
        learning_rate,
        clip_norm,
        use_cov_loss,
        hidden_layer_dim,
        feature_l2_reg,
        feature_proj_scaling_factor,
    ):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        pass
