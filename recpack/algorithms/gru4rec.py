from __future__ import annotations

import logging

from numpy.lib.arraysetops import isin
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.sparse import csr_matrix
from torch import Tensor
from tqdm import tqdm

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.data.matrix import InteractionMatrix
from typing import Tuple, Optional, List

from recpack.algorithms.loss_functions import (
    bpr_loss,
    bpr_max_loss,
    top1_loss,
    top1_max_loss
)
from recpack.algorithms.samplers import BatchSampler
from recpack.algorithms.util import matrix_to_tensor


logger = logging.getLogger("recpack")
ITEM_IX = InteractionMatrix.ITEM_IX


# TODO Inherit from TorchMLAlgorithm

class GRU4Rec(TorchMLAlgorithm):
    """A recurrent neural network for session-based recommendations.

    The algorithm, also known as GRU4Rec, was introduced in the 2016 and 2018 papers
    "Session-based Recommendations with Recurrent Neural Networks" and
    "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations"

    The algorithm makes recommendations by training a recurrent neural network to
    predict the next action of a user, and using the most likely next actions as
    recommendations. At the heart of it is a Gated Recurrent Unit (GRU), a recurrent
    network architecture that is able to form long-term memories.

    Predictions are made by processing a user's actions so far one by one,
    in chronological order::

                                          iid_3_predictions
                                                  |
                 0 --> [ GRU ] --> [ GRU ] --> [ GRU ]
                          |           |           |
                        iid_0       iid_1       iid_2

    here 'iid' are item ids, which can represent page views, purchases, or some other
    action. The GRU builds up a memory of the actions so far and predicts what the
    next action will be based on what other users with similar histories did next.
    While originally devised to make recommendations based on (often short) user
    sessions, the algorithm can be used with long user histories as well.

    For the mathematical details of GRU see "Empirical Evaluation of Gated Recurrent
    Neural Networks on Sequence Modeling" by Chung et al.

    :param num_layers: Number of hidden layers in the RNN
    :param hidden_size: Number of neurons in the hidden layer(s)
    :param embedding_size: Size of item embeddings. If None, no embeddings are used and
        the input to the network is a one-of-N binary vector.
    :param dropout: Dropout applied to embeddings and hidden layer(s), 0 for no dropout.
    :param activation: Final layer activation function, one of "identity", "tanh",
        "softmax", "relu", "elu-<X>", "leaky-<X>"
    :param loss_fn: Loss function. One of "cross-entropy", "top1", "top1-max", "bpr",
        "bpr-max"
    :param sample_size: Number of negative samples used for bpr, bpr-max, top1, top1-max
        loss calculation. This number includes samples from the same minibatch.
    :param alpha: Sampling weight parameter, 0 is uniform, 1 is popularity-based
    :param optimization_algorithm: Gradient descent optimizer, one of "sgd", "adagrad"
    :param batch_size: Number of examples in a mini-batch
    :param learning_rate: Gradient descent initial learning rate
    :param momentum: Momentum when using the sgd optimizer
    :param clip_norm: Clip the gradient's l2 norm, None for no clipping
    :param seed: Seed for random number generator
    :param bptt: Number of backpropagation through time steps
    :param max_epochs: Max training runs through entire dataset
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        embedding_size: Optional[int] = 250,
        dropout: float = 0.0,
        activation: str = "identity",
        loss_fn: str = "top1",
        # loss_fn: str = "cross-entropy",
        sample_size: int = 5000,
        alpha: float = 0.5,
        optimization_algorithm: str = "adagrad",
        batch_size: int = 512,
        learning_rate: float = 0.03,
        momentum: float = 0.0,
        clip_norm: float = 1.0,
        seed: int = 2,
        bptt: int = 1,
        max_epochs: int = 5,
        save_best_to_file: bool = False,
        keep_last: bool = True,
        stopping_criterion: str = "recall"
    ):
        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            # TODO Figure out early stopping for the RNN
            stopping_criterion=stopping_criterion,
            stop_early=False,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last
        )

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.activation = activation
        self.loss_fn = loss_fn
        self.sample_size = sample_size
        self.alpha = alpha
        self.optimization_algorithm = optimization_algorithm
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.bptt = bptt

        self._criterion = {
            # "cross-entropy": nn.CrossEntropyLoss(),
            "top1": top1_loss,
            "top1-max": top1_max_loss,
            "bpr": bpr_loss,
            "bpr-max": bpr_max_loss,
        }[self.loss_fn]

    def _init_model(self, X: InteractionMatrix) -> None:
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        num_items = X.shape[1]

        self.model_ = GRU4RecTorch(
            num_items=int(num_items),  # PyTorch 1.4 can't handle numpy ints
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device)

        if self.optimization_algorithm == "sgd":
            self.optimizer = optim.SGD(
                self.model_.parameters(), lr=self.learning_rate, momentum=self.momentum
            )
        elif self.optimization_algorithm == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model_.parameters(), lr=self.learning_rate)

    def _transform_fit_input(self, X: InteractionMatrix, validation_data: Tuple[InteractionMatrix, InteractionMatrix]):
        """Transform the input matrices of the training function to the expected types

        All matrices get converted to binary csr matrices

        :param X: The interactions matrix
        :type X: Matrix
        :param validation_data: The tuple with validation_in and validation_out data
        :type validation_data: Tuple[Matrix, Matrix]
        :return: The transformed matrices
        :rtype: Tuple[csr_matrix, Tuple[csr_matrix, csr_matrix]]
        """
        if not isinstance(X, InteractionMatrix):
            raise TypeError(
                "GRU4Rec requires training and validation data to be an instance of InteractionMatrix.")

        return X, validation_data

    def _transform_predict_input(self, X: InteractionMatrix) -> InteractionMatrix:
        return X

    def _predict(self, X: InteractionMatrix) -> csr_matrix:
        """Predict recommendations for each user with at least a single event in their
        history.

        :param X: Data matrix, same shape as training matrix. Timestamps required.
        """

        if X.num_interactions == 0:
            return csr_matrix(X.shape)

        num_users = X.num_active_users
        num_items = self.model_.output_size
        num_scores = num_users * num_items

        data = np.zeros(num_scores, dtype=np.float32)
        row = np.zeros(num_scores, dtype=np.int32)
        col = np.tile(np.arange(num_items), num_users)

        with torch.no_grad():
            actions, _,  uids = matrix_to_tensor(
                X, batch_size=1, device=self.device, shuffle=False, include_last=True
            )

            print(actions.shape, uids.shape)

            is_last_action = uids != uids.roll(-1, dims=0)
            is_last_action[-1] = True

            i = 0  # User/session count
            self.model_.train(False)
            hidden = self.model_.init_hidden(1)
            for action, uid, is_last in tqdm(
                zip(actions, uids, is_last_action), total=len(actions)
            ):
                output, hidden = self.model_(action.unsqueeze(0), hidden)
                if not is_last:
                    continue
                start, end = i * num_items, (i + 1) * num_items

                print(output.shape)
                data[start:end] = F.softmax(
                    output.reshape(-1), dim=0).cpu().numpy()
                row[start:end] = uid.item()
                hidden = hidden * 0
                i = i + 1

        X_pred = csr_matrix((data, (row, col)), shape=X.shape)

        return X_pred

    def _train_epoch(self, X: InteractionMatrix) -> None:
        """Train model for a single epoch."""
        actions, targets, uids = matrix_to_tensor(
            X, batch_size=self.batch_size, device=self.device, shuffle=True
        )
        is_last_action = uids != uids.roll(-1, dims=0)
        is_last_action[-1] = True

        loss, losses = 0.0, []


        hidden = self.model_.init_hidden(self.batch_size)
        for i, (action, target, is_last) in tqdm(
            enumerate(zip(actions, targets, is_last_action)), total=len(actions)
        ):
            output, hidden = self.model_(action.unsqueeze(0), hidden)
            loss += self._compute_loss(X, output, target) / self.bptt
            if i % self.bptt == self.bptt - 1:
                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_norm:
                    nn.utils.clip_grad_norm_(
                        self.model_.parameters(), self.clip_norm)
                self.optimizer.step()
                losses.append(loss.item())
                loss = 0.0
                hidden = hidden.detach()  # Prevent backprop past bptt steps
            hidden = hidden * ~is_last.view(
                1, -1, 1
            )  # Reset hidden state between users

        logger.info("training loss = {}".format(np.mean(losses)))

        return losses

    def _compute_loss(self, X, output, target) -> torch.Tensor:
        num_items = X.shape[1]
        dataframe = X.timestamps.reset_index()

        # TODO Some of these should be moved I think
        item_counts_tr = dataframe[ITEM_IX].value_counts()
        item_counts = pd.Series(np.arange(num_items))
        item_counts = item_counts.map(item_counts_tr).fillna(0)
        item_weights = torch.as_tensor(item_counts.to_numpy()) ** self.alpha

        sampler = BatchSampler(item_weights, device=self.device)

        samples = sampler(self.sample_size, target)



        return self._criterion(output, target, samples)

    # def _sample_session_parallel_mini_batches(self, X: InteractionMatrix):
    #     # TODO Shuffle users first.

    #     from numpy.lib.stride_tricks import sliding_window_view

    #     sorted_item_histories = list(X.sorted_item_history)
    #     sorted_item_histories.sort()   # Shuffle users

    #     w = torch.LongTensor([
    #         [u] + w.tolist()
    #         for u, sequence in sorted_item_histories
    #         if len(sequence) >= 2
    #         for w in sliding_window_view(sequence, 2)
    #     ])

    #     uids = w[:, 0]
    #     actions = w[:, 1]
    #     targets = w[:, 2]

    #     # Create user-parallel mini batches
    #     return (
    #         batchify(actions, batch_size),
    #         batchify(targets, batch_size),
    #         batchify(uids, batch_size),
    #     )

    #     pass


class GRU4RecTorch(nn.Module):
    """
    PyTorch definition of a basic recurrent neural network for session-based
    recommendations.

    :param num_items: Number of items
    :param hidden_size: Number of neurons in the hidden layer(s)
    :param num_layers: Number of hidden layers
    :param embedding_size: Size of the item embeddings, None for no embeddings
    :param dropout: Dropout applied to embeddings and hidden layers
    :param activation: Final layer activation function, one of "identity", "tanh",
        "softmax", "relu", "elu-<X>", "leaky-<X>"
    """

    def __init__(
        self,
        num_items: int,
        hidden_size: int,
        num_layers: int = 1,
        embedding_size: Optional[int] = None,
        dropout: float = 0,
        activation: str = "elu-1",
    ):
        super().__init__()
        self.input_size = num_items
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = num_items
        self.drop = nn.Dropout(dropout)
        self.emb = nn.Embedding(
            num_items, embedding_size) if embedding_size else None
        self.rnn = nn.GRU(
            embedding_size or num_items,
            hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0),
        )
        self.h2o = nn.Linear(hidden_size, num_items)
        self.act = self._create_activation(activation)
        self.init_weights()

    def forward(self, input: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes scores for each item given an action sequence and previous
        hidden state.

        :param input: Action sequences as a Tensor of shape (A, B)
        :param hidden: Previous hidden state, shape (L, B, H)
        :return: The computed scores for all items at each point in the sequence
                 for every sequence in the batch, as well as the next hidden state.
                 As Tensors of shapes (A*B, I) and (L, B, H) respectively.
        """
        input = (
            self.drop(self.emb(input))
            if self.emb
            else F.one_hot(input, self.input_size).float()
        )
        output, hidden = self.rnn(input, hidden)
        output = self.drop(output)
        output = self.act(self.h2o(output))
        output = output.view(-1, self.output_size)
        return output, hidden

    def init_weights(self) -> None:
        """
        Initializes all model parameters uniformly.
        """
        initrange = 0.01
        for param in self.parameters():
            nn.init.uniform_(param, -initrange, initrange)

    def init_hidden(self, batch_size: int) -> Tensor:
        """
        Returns an initial, zero-valued hidden state with shape (L, B, H).
        """
        weights = next(self.rnn.parameters())
        return weights.new_zeros((self.rnn.num_layers, batch_size, self.hidden_size))

    def _create_activation(self, a):
        if a == "identity":
            return nn.Identity()
        elif a == "tanh":
            return nn.Tanh()
        elif a == "softmax":
            return nn.Softmax()
        elif a == "relu":
            return nn.ReLU()
        elif a.startswith("elu-"):
            return nn.ELU(alpha=float(a.split("-")[1]))
        elif a.startswith("leaky-"):
            return nn.LeakyReLU(negative_slope=float(a.split("-")[1]))
        raise NotImplementedError("Unsupported activation function")
