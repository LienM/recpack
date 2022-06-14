from typing import Callable, List, Optional, Union

from scipy.sparse import csr_matrix, lil_matrix
import torch
from torch import nn

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.samplers import PositiveNegativeSampler
from recpack.algorithms.util import get_users


class NeuMFMLPOnly(TorchMLAlgorithm):
    """Implementation of Neural Matrix Factoration.

    Neural Matrix Factorization based on MLP architecture
    as presented in Figure 2 in He, Xiangnan, et al. "Neural collaborative filtering."
    Proceedings of the 26th international conference on world wide web. 2017.

    Represents the users and items using an embedding, and models similarity using a neural network.
    An MLP is used with as input the concatenated embeddings of users and items.

    As in the paper, the sum of square error is used as the loss function.
    Positive items should get a prediction close to 1, while sampled negatives should get a value close to 0.
    The MLP has 3 layers, whose dimensions are based on the `num_components` parameter.
    Bottom layer has `num_components * 2`, middle layer `num_components`
    and the top layer has `num_components / 2` dimensions.

    :param num_components: Size of the embeddings, needs to be an even number.
    :type num_components: int
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
    :param U: Amount of negatives to sample for each positive example, defaults to 1
    :type U: int, optional
    :param dropout: Dropout parameter used in MLP, defaults to 0.0
    :type dropout: float, optional

    """

    def __init__(
        self,
        num_components: int,
        batch_size: Optional[int] = 512,
        max_epochs: Optional[int] = 10,
        learning_rate: Optional[float] = 0.01,
        stopping_criterion: Optional[str] = "ndcg",
        stop_early: Optional[bool] = False,
        max_iter_no_change: Optional[int] = 5,
        min_improvement: Optional[float] = 0.0,
        seed: Optional[int] = None,
        save_best_to_file: Optional[bool] = False,
        keep_last: Optional[bool] = False,
        predict_topK: Optional[int] = None,
        U: Optional[int] = 1,
        dropout: Optional[float] = 0.0,
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

        self.num_components = num_components
        if self.num_components % 2 != 0:
            raise ValueError("Please use an even number of components for training the NeuMF model.")

        self.hidden_dims = [self.num_components * 2, self.num_components, self.num_components // 2]
        self.U = U
        self.dropout = dropout

        self.sampler = PositiveNegativeSampler(U=self.U, replace=False, batch_size=self.batch_size)

    def _init_model(self, X: csr_matrix):
        num_users, num_items = X.shape
        self.model_ = NeuMFMLPModule(self.num_components, num_users, num_items, self.hidden_dims, self.dropout).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

    def _compute_loss(
        self, positive_scores: torch.FloatTensor, negative_scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute the Square Error loss given recommendations for positive items, and sampled negatives."""
        mse = nn.MSELoss(reduction="sum")
        return mse(positive_scores, torch.ones_like(positive_scores, dtype=torch.float)) + mse(
            negative_scores, torch.zeros_like(negative_scores, dtype=torch.float)
        )

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

    def _construct_negative_prediction_input(self, users, negatives):
        """Since negatives has shape batch x U, and users is a 1d vector,
        these need to be turned into two 1d vectors of size batch*U

        First the users as a row are stacked U times and transposed,
        so that this is also a batch x U tensor

        then both are reshaped to remove the 2nd dimension, resulting in a single long 1d vector
        """
        return users.repeat(self.U, 1).T.reshape(-1), negatives.reshape(-1)

    def _batch_predict(self, X: csr_matrix, users: List[int]) -> csr_matrix:
        X_pred = lil_matrix(X.shape)
        if users is None:
            users = get_users(X)

        _, n_items = X.shape
        n_users = len(users)

        # Turn the np arrays and lists to torch tensors
        user_tensor = torch.LongTensor(users).repeat(n_items, 1).T.reshape(-1).to(self.device)
        item_tensor = torch.arange(n_items).repeat(n_users).to(self.device)

        X_pred[users] = self.model_(user_tensor, item_tensor).detach().cpu().numpy().reshape(n_users, n_items)
        return X_pred.tocsr()


class NeuMFMLPModule(nn.Module):
    """Model that encodes the Neural Matrix Factorization Network.

    :param num_components: size of the embeddings
    :type num_components: int
    :param n_users: number of users in the network
    :type n_users: int
    :param n_items: number of items in the network
    :type n_items: int
    :param hidden_dims: dimensions of the MLP hidden layers.
    :type hidden_dims: Union[int, List[int]]
    :param dropout: Dropout chance between layers of the MLP
    :type dropout: float
    """

    def __init__(
        self, num_components: int, n_users: int, n_items: int, hidden_dims: Union[int, List[int]], dropout: float
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, num_components)
        self.item_embedding = nn.Embedding(n_items, num_components)

        # 2 x embedding size as input, since the user and item embedding are concatenated.
        # Output is always 1, since we need a single score for u,i
        self.mlp = MLP(2 * num_components, 1, hidden_dims, dropout=dropout)

        # In order to interpret the output as a probability, the score should be between 0 and 1
        # The papers mentions probit / logistic activation,
        # but in pytorch sigmoid seems like the only one to give 0 to 1 values
        self.final = nn.Sigmoid()

        # weight initialization
        self.user_embedding.weight.data.normal_(0, 1.0 / self.user_embedding.embedding_dim)
        self.item_embedding.weight.data.normal_(0, 1.0 / self.item_embedding.embedding_dim)

        # TODO: do I need to randomly initialise the MLP layer weights?

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


# Code used from https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
# Another option is to use torchvision.ops.MLP
# which is nearly identical in implementation, but is in torchvision and not in base torch
# TODO: move to it's own file?
class MLP(nn.Module):
    """A multi-layer perceptron module.
    This module is a sequence of linear layers plus activation functions.
    The user can optionally add normalization and/or dropout to each of the layers.

    Code used from https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dims ([List[int]]): Output dimension for each hidden layer.
        dropout (float): Probability for dropout layer.
        activation (Callable[..., nn.Module]): Which activation
            function to use. Supports module type or partial.
        normalization (Optional[Callable[..., nn.Module]]): Which
            normalization layer to use (None for no normalization).
            Supports module type or partial.
    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Union[int, List[int]]],
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        normalization: Optional[Callable[..., nn.Module]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        layers = nn.ModuleList()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization:
                layers.append(normalization(hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
