from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.sparse import csr_matrix, lil_matrix
from torch import Tensor
from tqdm import tqdm
from sklearn.utils.validation import check_is_fitted

from recpack.algorithms.base import Algorithm
from recpack.metrics.recall import recall_k
from recpack.data.matrix import InteractionMatrix
from typing import Tuple, Optional, List

from recpack.algorithms.rnn.loss import (
    BatchSampler,
    BPRLoss,
    BPRMaxLoss,
    TOP1Loss,
    TOP1MaxLoss,
)
from recpack.algorithms.rnn.data import matrix_to_tensor
from recpack.algorithms.rnn.model import GRU4RecTorch


logger = logging.getLogger("recpack")
ITEM_IX = InteractionMatrix.ITEM_IX


# TODO Inherit from TorchMLAlgorithm

class GRU4Rec(Algorithm):
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
    :param optimizer: Gradient descent optimizer, one of "sgd", "adagrad"
    :param batch_size: Number of examples in a mini-batch
    :param learning_rate: Gradient descent initial learning rate
    :param momentum: Momentum when using the sgd optimizer
    :param clip_norm: Clip the gradient's l2 norm, None for no clipping
    :param seed: Seed for random number generator
    :param bptt: Number of backpropagation through time steps
    :param num_epochs: Max training runs through entire dataset
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        embedding_size: Optional[int] = 250,
        dropout: float = 0.0,
        activation: str = "identity",
        loss_fn: str = "cross-entropy",
        sample_size: int = 5000,
        alpha: float = 0.5,
        optimizer: str = "adagrad",
        batch_size: int = 512,
        learning_rate: float = 0.03,
        momentum: float = 0.0,
        clip_norm: Optional[float] = 1.0,
        seed: int = 2,
        bptt: int = 1,
        num_epochs: int = 5,
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.activation = activation
        self.loss_fn = loss_fn
        self.sample_size = sample_size
        self.alpha = alpha
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.seed = seed
        self.bptt = bptt
        self.num_epochs = num_epochs
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

    def _init_random_state(self) -> None:
        """
        Resets the random number generator state.
        """
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def _init_model(self, train_data: InteractionMatrix) -> None:
        """
        Initializes the neural network model.
        """
        num_items = train_data.shape[1]

        self.model_ = GRU4RecTorch(
            num_items=int(num_items),  # PyTorch 1.4 can't handle numpy ints
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device)

    def _init_training(self, train_data: InteractionMatrix) -> None:
        """
        Initializes the objects required for network optimization.
        """
        num_items = train_data.shape[1]
        dataframe = train_data.timestamps.reset_index()

        item_counts_tr = dataframe[ITEM_IX].value_counts()
        item_counts = pd.Series(np.arange(num_items))
        item_counts = item_counts.map(item_counts_tr).fillna(0)
        item_weights = torch.as_tensor(item_counts.to_numpy()) ** self.alpha

        sampler = BatchSampler(item_weights, device=self.device)
        self._criterion = {
            "cross-entropy": nn.CrossEntropyLoss(),
            "top1": TOP1Loss(sampler, self.sample_size),
            "top1-max": TOP1MaxLoss(sampler, self.sample_size),
            "bpr": BPRLoss(sampler, self.sample_size),
            "bpr-max": BPRMaxLoss(sampler, self.sample_size),
        }[self.loss_fn]

        self._optimizer = {
            "sgd": optim.SGD(
                self.model_.parameters(), lr=self.learning_rate, momentum=self.momentum
            ),
            "adagrad": optim.Adagrad(self.model_.parameters(), lr=self.learning_rate),
        }[self.optimizer]

    def save(self, file) -> None:
        """Saves the algorithm, all its learned parameters and optimization state.

        :param file: A file-like or string containing a filepath.
        """
        torch.save(self, file)

    @staticmethod
    def load(file) -> GRU4Rec:
        """Loads an algorithm previously stored with save().

        :param file: A file-like or string with the filepath of the algorithm
        """
        return torch.load(file)

    def fit(
        self,
        X: InteractionMatrix,
        validation_data: Tuple[InteractionMatrix, InteractionMatrix] = None,
    ) -> GRU4Rec:
        """Fit the model on the X dataset.

        Model quality is evaluated after every epoch on validation_data using the
        algorithm's stopping criterion. If no validation data is provided or early
        stopping is disabled, training will continue for exactly num_epochs.

        :param train_data: The training data matrix. Timestamps required.
        :param validation_data: Validation in and out matrices. None for no validation.
            Timestamps required.
        """
        self._init_random_state()
        self._init_model(X)
        self._init_training(X)

        # try:
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch}")
            self._train_epoch(X)
            # if validation_data:
            #    self._evaluate(validation_data)
        # except EarlyStoppingException:
        #    pass

        return self

    def predict(self, X: InteractionMatrix) -> csr_matrix:
        """Predict recommendations for each user with at least a single event in their
        history.

        :param X: Data matrix, same shape as training matrix. Timestamps required.
        """
        check_is_fitted(self)

        if X.num_interactions == 0:
            return csr_matrix(X.shape)

        num_users = X.num_active_users
        num_items = self.model_.output_size
        num_scores = num_users * num_items

        data = np.zeros(num_scores, dtype=np.float32)
        row = np.zeros(num_scores, dtype=np.int32)
        col = np.tile(np.arange(num_items), num_users)

        with torch.no_grad():
            actions, _, uids = matrix_to_tensor(
                X, batch_size=1, device=self.device, shuffle=False, include_last=True
            )
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
                data[start:end] = F.softmax(output.reshape(-1), dim=0).cpu().numpy()
                row[start:end] = uid.item()
                hidden = hidden * 0
                i = i + 1

        X_pred = csr_matrix((data, (row, col)), shape=X.shape)

        self._check_prediction(X_pred, X.values)

        return X_pred

    def _train_epoch(self, X: InteractionMatrix) -> None:
        """Train model for a single epoch."""
        actions, targets, uids = matrix_to_tensor(
            X, batch_size=self.batch_size, device=self.device, shuffle=True
        )
        is_last_action = uids != uids.roll(-1, dims=0)
        is_last_action[-1] = True

        loss, losses = 0.0, []
        self.model_.train(True)
        hidden = self.model_.init_hidden(self.batch_size)
        for i, (action, target, is_last) in tqdm(
            enumerate(zip(actions, targets, is_last_action)), total=len(actions)
        ):
            output, hidden = self.model_(action.unsqueeze(0), hidden)
            loss += self._criterion(output, target) / self.bptt
            if i % self.bptt == self.bptt - 1:
                self._optimizer.zero_grad()
                loss.backward()
                if self.clip_norm:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.clip_norm)
                self._optimizer.step()
                losses.append(loss.item())
                loss = 0.0
                hidden = hidden.detach()  # Prevent backprop past bptt steps
            hidden = hidden * ~is_last.view(
                1, -1, 1
            )  # Reset hidden state between users

        logger.info("training loss = {}".format(np.mean(losses)))

    def _evaluate(
        self, validation_data: Tuple[InteractionMatrix, InteractionMatrix]
    ) -> None:
        """Evaluate the current model on the validation data.

        If performance improved over previous epoch, store the model and update
        best value. If performance stagnates, stop training.

        :param validation_data: Validation data interaction matrices.
        """
        self.model_.train(False)
        X_true = validation_data[1].binary_values
        X_val_pred = self.predict(validation_data[0])
        # TODO: Maybe StoppingCriterion here? Tho unsure non-loss criteria make sense..
