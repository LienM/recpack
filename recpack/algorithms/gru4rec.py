import logging
from math import ceil
from typing import Tuple, List

import numpy as np
from scipy.sparse.lil import lil_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.loss_functions import (bpr_loss, bpr_max_loss,
                                               top1_loss, top1_max_loss)
from recpack.algorithms.samplers import SequenceMiniBatchSampler
from recpack.data.matrix import InteractionMatrix


logger = logging.getLogger("recpack")
ITEM_IX = InteractionMatrix.ITEM_IX


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
        embedding_size: int = 250,
        dropout: float = 0.0,
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

        # Invalid item ID. Used to mask inputs to the RNN
        self.num_items = X.shape[1]
        self.pad_token = self.num_items

        self.model_ = GRU4RecTorch(
            self.num_items,
            self.hidden_size,
            self.embedding_size,
            self.pad_token,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        if self.optimization_algorithm == "sgd":
            self.optimizer = optim.SGD(
                self.model_.parameters(), lr=self.learning_rate, momentum=self.momentum
            )
        elif self.optimization_algorithm == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model_.parameters(), lr=self.learning_rate)

        # TODO Make exact configurable
        self.sampler = SequenceMiniBatchSampler(
            self.sample_size,
            self.pad_token,
            self.batch_size,
            exact=False)

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

    def _train_epoch(self, X: InteractionMatrix) -> List[float]:

        losses = []

        for _, positives_batch, negatives_batch in self.sampler.sample(X):
            loss = self._train_batch(positives_batch, negatives_batch)
            losses.append(loss)

        return losses

    def _train_batch(self, positives_batch: torch.LongTensor, negatives_batch: torch.LongTensor):
        batch_loss = 0
        true_batch_size, max_hist_len = positives_batch.shape
        # Want to reuse this between chunks of the same batch of sequences
        hidden = self.model_.init_hidden(true_batch_size)

        # Feed the sequence into the network in chunks of size bptt.
        # We do this because we want to propagate after every bptt elements, but keep hidden states.
        for input_chunk, negatives_chunk in zip(
            positives_batch.tensor_split(
                ceil(max_hist_len / self.bptt), axis=1),
            negatives_batch.tensor_split(
                ceil(max_hist_len / self.bptt), axis=1)
        ):
            # Remove rows with only pad tokens from chunk and from hidden. We can do this because the array is sorted.
            true_rows = (input_chunk != self.pad_token).any(axis=1)
            true_input_chunk = input_chunk[true_rows]
            true_negative_chunk = negatives_chunk[true_rows]
            # This will not work because it makes a copy of the hidden state.
            true_hidden = hidden[:, true_rows, :]

            # If there are any values remaining
            if true_input_chunk.any():
                self.optimizer.zero_grad()
                output, hidden[:, true_rows, :] = self.model_(
                    true_input_chunk, true_hidden)

                # Select score for positive and negative sample for all tokens
                positive_scores = torch.gather(
                    output, 2, true_input_chunk.unsqueeze(-1))
                negative_scores = torch.gather(
                    output, 2, true_negative_chunk)

                # TODO Remove pad tokens
                loss = self._compute_loss(positive_scores, negative_scores)

                batch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                hidden = hidden.detach()

        return batch_loss

    def _predict(self, X: InteractionMatrix):
        X_pred = lil_matrix(X.shape)

        for uid_batch, positives_batch, _ in self.sampler.sample(X):
            batch_size = positives_batch.shape[0]
            hidden = self.model_.init_hidden(batch_size)
            output, hidden = self.model_(positives_batch, hidden)
            # Use only the final prediction
            # Not so easy of course because of padding...
            last_item_in_hist = (
                positives_batch != self.pad_token).sum(axis=1) - 1

            item_scores = output[torch.arange(
                0, batch_size, dtype=int), last_item_in_hist]

            X_pred[uid_batch.detach().cpu().numpy()] = item_scores.detach().cpu().numpy()

        return X_pred.tocsr()

    def _compute_loss(self, positive_scores, negative_scores) -> torch.Tensor:
        return self._criterion(positive_scores, negative_scores)


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
        embedding_size: int,
        pad_token: int,
        num_layers: int = 1,
        dropout: float = 0,
    ):
        # TODO I made it impossible to not use an embedding. Is this OK?
        super().__init__()
        self.input_size = num_items
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = num_items
        self.pad_token = pad_token
        self.drop = nn.Dropout(dropout, inplace=True)

        # Passing pad_token will make sure these embeddings are always zero-valued.
        # TODO Make this pad_token thing clearer
        self.emb = nn.Embedding(
            num_items + 1, embedding_size, padding_idx=pad_token)
        self.rnn = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0),
            batch_first=True
        )
        self.lin = nn.Linear(hidden_size, num_items)
        self.act = nn.Softmax(dim=2)
        self.init_weights()

    def forward(self, x: torch.LongTensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes scores for each item given an action sequence and previous
        hidden state.

        :param input: Action sequences as a Tensor of shape (A, B)
        :param hidden: Previous hidden state, shape (L, B, H)
        :return: The computed scores for all items at each point in the sequence
                 for every sequence in the batch, as well as the next hidden state.
                 As Tensors of shapes (A*B, I) and (L, B, H) respectively.
        """
        emb_x = self.emb(x)
        self.drop(emb_x)

        # Check if it needs padding
        any_row_requires_padding = (x == self.pad_token).any()

        if any_row_requires_padding:
            seq_lengths = (x != self.pad_token).sum(axis=1)

            padded_emb_x = nn.utils.rnn.pack_padded_sequence(
                emb_x, seq_lengths, batch_first=True)

            padded_rnn_x, hidden = self.rnn(padded_emb_x, hidden)

            rnn_x, _ = nn.utils.rnn.pad_packed_sequence(
                padded_rnn_x, batch_first=True)

        else:
            rnn_x, hidden = self.rnn(emb_x, hidden)

        self.drop(rnn_x)

        out = self.act(self.lin(rnn_x))

        return out, hidden

    def init_weights(self) -> None:
        """
        Initializes all model parameters uniformly.
        """
        initrange = 0.01
        for param in self.parameters():
            nn.init.uniform_(param, -initrange, initrange)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Returns an initial, zero-valued hidden state with shape (B, L, H).
        """
        return torch.zeros((batch_size, self.rnn.num_layers, self.hidden_size))
