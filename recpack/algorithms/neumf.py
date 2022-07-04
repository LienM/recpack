from typing import Callable, List, Optional, Union

from scipy.sparse import csr_matrix, lil_matrix
import torch
from torch import nn

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.samplers import PositiveNegativeSampler
from recpack.algorithms.util import get_users


class NeuMFMLPOnly(TorchMLAlgorithm):
    """Implementation of Neural Matrix Factoration using only the MLP component.

    Neural Matrix Factorization based on the MLP architecture as presented in Figure 2 of
    He, Xiangnan, et al. "Neural collaborative filtering."
    In Proceedings of the 26th international conference on world wide web. 2017.

    Represents the users and items using an embedding, similarity between the two is modelled using a neural network.

    The network consists of an embedding for both users and items.
    To compute similarity those two embeddings are concatenated and passed through the MLP
    Finally the similarity is transformed to the [0,1] domain using a sigmoid function.

    As in the paper, the sum of square errors is used as loss function.
    Positive items should get a prediction close to 1, while sampled negatives should get a value close to 0.

    The MLP has 3 layers, as suggested in the experiments section.
    Bottom layer has dimension `4 * predictive_factors`, middle layer `2 * predictive_factors`
    and the top layer has `predictive_factors`.

    :param predictive_factors: Size of the final hidden layer in the MLP. Defaults to 16
    :type predictive_factors: int, optional
    :param dropout: Dropout parameter used in MLP, defaults to 0.0
    :type dropout: float, optional
    :param n_negatives_per_positive: Amount of negatives to sample for each positive example, defaults to 4
    :type n_negatives_per_positive: int, optional
    :param exact_sampling: Enable or disable exact checks while sampling.
        With exact sampling the sampled negatives are guaranteed to not have been visited by the user.
        Non exact sampling assumes that the space for item selection is large enough,
        such that most items are likely not seen before.
        Defaults to False,
    :type exact_sampling: bool, optional
    :param batch_size: How many samples to use in each update step.
        Higher batch sizes make each epoch more efficient,
        but increases the amount of epochs needed to converge to the optimum,
        by reducing the amount of updates per epoch.
        Defaults to 512.
    :type batch_size: Optional[int]
    :param max_epochs: The max number of epochs to train.
        If the stopping criterion uses early stopping, less epochs could be used.
        Defaults to 10.
    :type max_epochs: Optional[int]
    :param learning_rate: How much to update the weights at each update. Defaults to 0.01
    :type learning_rate: Optional[float]
    :param stopping_criterion: Name of the stopping criterion to use for training.
        For available values,
        check :meth:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`
        Defaults to 'ndcg'
    :type stopping_criterion: Optional[str]
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
    :type min_improvement: float, optional
    :param seed: Seed to the randomizers, useful for reproducible results,
        defaults to None
    :type seed: int, optional
    :param save_best_to_file: If true, the best model will be saved after training,
        defaults to False
    :type save_best_to_file: bool, optional
    :param keep_last: Retain last model, rather than best
        (according to stopping criterion value on validation data), defaults to False
    :type keep_last: bool, optional
    :param predict_topK: The topK recommendations to keep per row in the matrix.
        Use when the user x item output matrix would become too large for RAM.
        Defaults to None, which results in no filtering.
    :type predict_topK: int, optional
    """

    def __init__(
        self,
        predictive_factors: int = 16,
        dropout: Optional[float] = 0.0,
        n_negatives_per_positive: Optional[int] = 4,
        exact_sampling: Optional[bool] = False,
        batch_size: Optional[int] = 512,
        max_epochs: Optional[int] = 10,
        learning_rate: Optional[float] = 0.001,
        stopping_criterion: Optional[str] = "ndcg",
        stop_early: Optional[bool] = False,
        max_iter_no_change: Optional[int] = 5,
        min_improvement: Optional[float] = 0.0,
        seed: Optional[int] = None,
        save_best_to_file: Optional[bool] = False,
        keep_last: Optional[bool] = False,
        predict_topK: Optional[int] = 100,
    ):
        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            stopping_criterion,
            stop_early,
            max_iter_no_change,
            min_improvement,
            seed,
            save_best_to_file,
            keep_last,
            predict_topK,
        )

        self.predictive_factors = predictive_factors
        self.dropout = dropout
        self.n_negatives_per_positive = n_negatives_per_positive
        self.exact_sampling = exact_sampling

        self.sampler = PositiveNegativeSampler(
            U=self.n_negatives_per_positive, replace=False, batch_size=self.batch_size, exact=exact_sampling
        )

    def _init_model(self, X: csr_matrix):
        num_users, num_items = X.shape
        self.model_ = NeuMFMLPModule(self.predictive_factors, num_users, num_items, self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

    def _train_epoch(self, X: csr_matrix) -> List[int]:
        losses = []
        for users, positives, negatives in self.sampler.sample(X):

            self.optimizer.zero_grad()

            # Predict for the positives
            positive_scores = self.model_.forward(users.to(self.device), positives.to(self.device))
            # Predict for the negatives
            negative_scores = self.model_.forward(
                *self._construct_negative_prediction_input(users.to(self.device), negatives.to(self.device))
            )

            loss = self._compute_loss(positive_scores, negative_scores)

            # Backwards propagation of the loss
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return losses

    def _compute_loss(
        self, positive_scores: torch.FloatTensor, negative_scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute the Square Error loss given recommendations for positive items, and sampled negatives."""
        mse = nn.MSELoss(reduction="sum")
        return mse(positive_scores, torch.ones_like(positive_scores, dtype=torch.float)) + mse(
            negative_scores, torch.zeros_like(negative_scores, dtype=torch.float)
        )

    def _construct_negative_prediction_input(self, users, negatives):
        """Since negatives has shape batch x U, and users is a 1d vector,
        these need to be turned into two 1d vectors of size batch*U

        First the users as a row are stacked U times and transposed,
        so that this is also a batch x U tensor

        then both are reshaped to remove the 2nd dimension, resulting in a single long 1d vector
        """
        return users.repeat(self.n_negatives_per_positive, 1).T.reshape(-1), negatives.reshape(-1)

    def _batch_predict(self, X: csr_matrix, users: List[int]) -> csr_matrix:
        X_pred = lil_matrix(X.shape)
        if users is None:
            users = get_users(X)

        _, n_items = X.shape
        n_users = len(users)

        # Create tensors such that each user, item pair gets a score.
        # The user tensor contains the users in order
        # (eg. [1, 1, 2, 2]),
        # item indices are repeated (eg. [0, 1, 2, 0, 1, 2]).
        user_tensor = torch.LongTensor(users).repeat(n_items, 1).T.reshape(-1).to(self.device)
        item_tensor = torch.arange(n_items).repeat(n_users).to(self.device)

        X_pred[users] = self.model_(user_tensor, item_tensor).detach().cpu().numpy().reshape(n_users, n_items)
        return X_pred.tocsr()


class NeuMFMLPModule(nn.Module):
    """Model that encodes the Neural Matrix Factorization Network.

    Implements the 3 tiered network defined in the He et al. paper.

    :param predictive_factors: size of the last hidden layer in MLP.
        Embedding sizes computed as 2 * predictive_factors powers.
    :type predictive_factors: int
    :param n_users: number of users in the network
    :type n_users: int
    :param n_items: number of items in the network
    :type n_items: int
    :param dropout: Dropout chance between layers of the MLP
    :type dropout: float
    """

    def __init__(self, predictive_factors: int, n_users: int, n_items: int, dropout: float):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, 2 * predictive_factors)
        self.item_embedding = nn.Embedding(n_items, 2 * predictive_factors)

        # we use a three tiered MLP as described in the experiments of the paper.
        hidden_dims = [4 * predictive_factors, 2 * predictive_factors, predictive_factors]

        # Output is always shape 1, since we need a single score for u,i
        self.mlp = MLP(4 * predictive_factors, 1, hidden_dims, dropout=dropout)

        self.final = nn.Sigmoid()

        # weight initialization
        self.user_embedding.weight.data.normal_(0, 1.0 / self.user_embedding.embedding_dim)
        self.item_embedding.weight.data.normal_(0, 1.0 / self.item_embedding.embedding_dim)

    def forward(self, users: torch.LongTensor, items: torch.LongTensor) -> torch.FloatTensor:
        """Predict scores for the user item pairs obtained when zipping together the two 1D tensors

        :param users: 1D tensor with user ids
        :type users: torch.LongTensor
        :param items: 1D tensor with item ids
        :type items: torch.LongTensor
        :return: 1D tensor with on position i the prediction similarity between `users[i]` and `items[i]`
        :rtype: torch.FloatTensor
        """

        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)

        return self.final(self.mlp(torch.hstack([user_emb, item_emb])))


class MLP(nn.Module):
    """A multi-layer perceptron module.
    This module is a sequence of linear layers plus activation functions.
    The user can optionally add normalization and/or dropout to each of the layers.

    Code based on https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py

    :param in_dim: Input dimension.
    :type in_dim: int
    :param out_dim: Output dimension.
    :type out_dim: int
    :param hidden_dims: Output dimension for each hidden layer.
    :type hidden_dims: Optional[Union[int, List[int]]]
    :param dropout: Probability for dropout layers between each hidden layer.
    :type dropout: float
    :param activation: Which activation function to use.
        Supports module type or partial.
    :type activation: Callable[..., nn.Module]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Union[int, List[int]]],
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        layers = nn.ModuleList()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
