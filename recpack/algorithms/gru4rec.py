import logging
from math import ceil
import time
from typing import Tuple, List, Iterator

import numpy as np
from scipy.sparse.lil import lil_matrix
import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRU
import torch.optim as optim
from tqdm import tqdm

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.loss_functions import (
    bpr_loss,
    bpr_max_loss,
    top1_loss,
    top1_max_loss,
)
from recpack.algorithms.samplers import (
    SequenceMiniBatchPositivesTargetsNegativesSampler,
    SequenceMiniBatchSampler
)
from recpack.data.matrix import InteractionMatrix


logger = logging.getLogger("recpack")


class GRU4Rec(TorchMLAlgorithm):
    # TODO Docs
    """Base class for GRU4Rec.

    :param num_layers: Number of hidden layers in the RNN. Defaults to 1
    :type num_layers: int, optional
    :param hidden_size: Number of neurons in the hidden layer(s). Defaults to 100
    :type hidden_size: int, optional
    :param embedding_size: Size of item embeddings. Defaults to 250
    :type embedding_size: int, optional
    :param dropout: Dropout applied to embeddings and hidden layer(s), 0 for no dropout
    :type dropout: float
    :param optimization_algorithm: Gradient descent optimizer, one of "sgd", "adagrad".
        Defaults to adagrad.
    :type optimization_algorithm: str, optional
    :param batch_size: Number of examples in a mini-batch
        Defaults to 512.
    :type batch_size: int, optional
    :param learning_rate: Gradient descent initial learning rate
        Defaults to 0.03
    :type learning_rate: float, optional
    :param momentum: Momentum when using the sgd optimizer
        Defaults to 0.0
    :type momentum: float, optional
    :param clipnorm: Clip the gradient's l2 norm, None for no clipping
        Defaults to 1.0
    :type clipnorm: float, optional
    :param seed: Seed for random number generator
        Defaults to 2
    :type seed: int, optional
    :param bptt: Number of backpropagation through time steps
        Defaults to 1
    :type bptt: int, optional
    :param max_epochs: Max training runs through entire dataset
        Defaults to 5
    :type max_epochs: int, optional
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        embedding_size: int = 250,
        dropout: float = 0.0,
        optimization_algorithm: str = "adagrad",
        batch_size: int = 512,
        learning_rate: float = 0.03,
        momentum: float = 0.0,
        clipnorm: float = 1.0,
        seed: int = 2,
        bptt: int = 1,
        max_epochs: int = 5,
        save_best_to_file: bool = False,
        keep_last: bool = True,
        stopping_criterion: str = "recall",
        U: int = 0,
    ):
        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            stopping_criterion=stopping_criterion,
            stop_early=False,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
        )

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.optimization_algorithm = optimization_algorithm
        self.momentum = momentum
        self.clipnorm = clipnorm
        self.bptt = bptt
        self.U = U

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
            dropout=self.dropout,
        ).to(self.device)

        if self.optimization_algorithm == "sgd":
            self.optimizer = optim.SGD(
                self.model_.parameters(), lr=self.learning_rate, momentum=self.momentum
            )
        elif self.optimization_algorithm == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model_.parameters(), lr=self.learning_rate
            )

        self.predict_sampler = SequenceMiniBatchSampler(
            self.pad_token, batch_size=self.batch_size
        )

        self.fit_sampler = SequenceMiniBatchPositivesTargetsNegativesSampler(
            self.U, self.pad_token, batch_size=self.batch_size
        )

    def _transform_fit_input(
        self,
        X: InteractionMatrix,
        validation_data: Tuple[InteractionMatrix, InteractionMatrix],
    ):
        """Transform the input matrices of the training function to the expected types

        :param X: The interactions matrix
        :type X: Matrix
        :param validation_data: The tuple with validation_in and validation_out data
        :type validation_data: Tuple[Matrix, Matrix]
        :return: The transformed matrices
        :rtype: Tuple[csr_matrix, Tuple[csr_matrix, csr_matrix]]
        """
        if not isinstance(X, InteractionMatrix):
            raise TypeError(
                "GRU4Rec requires training and validation data to be an instance of InteractionMatrix."
            )

        return X, validation_data

    def _transform_predict_input(self, X: InteractionMatrix) -> InteractionMatrix:
        return X

    def _train_epoch(self, X: InteractionMatrix) -> List[float]:

        losses = []

        for (
            _,
            positives_batch,
            targets_batch,
            negatives_batch,
        ) in self.fit_sampler.sample(X):
            # st = time.time()
            # positives shape = (batch_size x |max_hist_length|)
            # targets shape = (batch_size x |max_hist_length|)
            # negatives shape = (batch_size x |max_hist_length| x self.U)
            positives_batch = positives_batch.to(self.device)
            targets_batch = targets_batch.to(self.device)
            negatives_batch = negatives_batch.to(self.device)

            # print(f"Takes {time.time() - st} seconds to convert to GPU")

            batch_loss = 0
            true_batch_size = positives_batch.shape[0]
            # Want to reuse this between chunks of the same batch of sequences
            hidden = self.model_.init_hidden(true_batch_size).to(self.device)

            # Generate vertical chunks of BPTT width
            for input_chunk, target_chunk, neg_chunk in self._chunk(
                positives_batch, targets_batch, negatives_batch
            ):
                input_mask = input_chunk != self.pad_token
                # Remove rows with only pad tokens from chunk and from hidden.
                # We can do this because the array is sorted.
                true_rows = input_mask.any(axis=1)
                true_input_chunk = input_chunk[true_rows]
                true_target_chunk = target_chunk[true_rows]
                true_neg_chunk = neg_chunk[true_rows]
                true_hidden = hidden[:, true_rows, :]
                true_input_mask = input_mask[true_rows]

                # If there are any values remaining
                if true_input_chunk.any():
                    self.optimizer.zero_grad()
                    output, hidden[:, true_rows, :] = self.model_(
                        true_input_chunk, true_hidden
                    )

                    loss = self._compute_loss(
                        output, true_target_chunk, true_neg_chunk, true_input_mask)

                    batch_loss += loss.item()
                    loss.backward()
                    if self.clipnorm:
                        nn.utils.clip_grad_norm_(
                            self.model_.parameters(), self.clipnorm)

                    self.optimizer.step()

                hidden = hidden.detach()
            # print(f"Takes {time.time() - st} seconds to process batch")
            losses.append(batch_loss)

        return losses

    def _compute_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def _chunk(self, *tensors: torch.LongTensor) -> Iterator[Tuple[torch.LongTensor, ...]]:
        """Split tensors into chunks of self.bptt width, or max hist len width.

        The input tensors of  shape (batch_size, max_hist_length, 1 or U)
        are sliced on axis 1 into tensors of shape (batch_size, chunk_size, 1 or U)

        # TODO Improve docs
        :return: Zipped split tensors.
        :rtype: list of tuples, with 1 entry in the tuple per input tensor.
        """
        max_hist_len = tensors[0].shape[1]

        chunk_size = ceil(max_hist_len / self.bptt)

        split_tensors = [t.tensor_split(chunk_size, axis=1) for t in tensors]

        return zip(*split_tensors)

    def _predict(self, X: InteractionMatrix):
        X_pred = lil_matrix(X.shape)
        for uid_batch, positives_batch in self.predict_sampler.sample(X):
            batch_size = positives_batch.shape[0]
            hidden = self.model_.init_hidden(batch_size)
            output, hidden = self.model_(positives_batch.to(
                self.device), hidden.to(self.device))
            # Use only the final prediction
            # Last item is the last non padding token per row.
            last_item_in_hist = (
                positives_batch != self.pad_token).sum(axis=1) - 1

            # Item scores is a matrix with the scores for each item
            # based on the last item in the sequence
            item_scores = output[
                torch.arange(0, batch_size, dtype=int), last_item_in_hist
            ]

            X_pred[uid_batch.detach().cpu().numpy()] = (
                item_scores.detach().cpu().numpy()[:, :-1]
            )

        return X_pred.tocsr()


class GRU4RecCrossEntropy(GRU4Rec):
    # TODO Docs
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

    Note: The paper mentions both a GRU trained to optimise cross-entropy loss, as
    methods that employ negative sampling. This is an implementation of the GRU
    using cross-entropy loss.

    :param num_layers: Number of hidden layers in the RNN. Defaults to 1
    :type num_layers: int, optional
    :param hidden_size: Number of neurons in the hidden layer(s). Defaults to 100
    :type hidden_size: int, optional
    :param embedding_size: Size of item embeddings. Defaults to 250
    :type embedding_size: int, optional
    :param dropout: Dropout applied to embeddings and hidden layer(s), 0 for no dropout
    :type dropout: float
    :param optimization_algorithm: Gradient descent optimizer, one of "sgd", "adagrad".
        Defaults to adagrad.
    :type optimization_algorithm: str, optional
    :param batch_size: Number of examples in a mini-batch
        Defaults to 512.
    :type batch_size: int, optional
    :param learning_rate: Gradient descent initial learning rate
        Defaults to 0.03
    :type learning_rate: float, optional
    :param momentum: Momentum when using the sgd optimizer
        Defaults to 0.0
    :type momentum: float, optional
    :param clipnorm: Clip the gradient's l2 norm, None for no clipping
        Defaults to 1.0
    :type clipnorm: float, optional
    :param seed: Seed for random number generator
        Defaults to 2
    :type seed: int, optional
    :param bptt: Number of backpropagation through time steps
        Defaults to 1
    :type bptt: int, optional
    :param max_epochs: Max training runs through entire dataset
        Defaults to 5
    :type max_epochs: int, optional
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        embedding_size: int = 250,
        dropout: float = 0.0,
        optimization_algorithm: str = "adagrad",
        batch_size: int = 512,
        learning_rate: float = 0.03,
        momentum: float = 0.0,
        clipnorm: float = 1.0,
        seed: int = 2,
        bptt: int = 1,
        max_epochs: int = 5,
        save_best_to_file: bool = False,
        keep_last: bool = True,
        stopping_criterion: str = "recall",
    ):
        super().__init__(
            num_layers=num_layers,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            dropout=dropout,
            optimization_algorithm=optimization_algorithm,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            clipnorm=clipnorm,
            seed=seed,
            bptt=bptt,
            max_epochs=max_epochs,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            stopping_criterion=stopping_criterion,
            U=0
        )

        self._criterion = nn.CrossEntropyLoss()

    def _compute_loss(
        self,
        output: torch.FloatTensor,
        targets_chunk: torch.LongTensor,
        negatives_chunk: torch.LongTensor,
        true_input_mask: torch.BoolTensor
    ) -> torch.Tensor:

        return self._criterion(output[true_input_mask], targets_chunk[true_input_mask])


class GRU4RecNegSampling(GRU4Rec):
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

    Note: Cross-Entropy loss was mentioned in the paper, but omitted for implementation reasons.

    :param num_layers: Number of hidden layers in the RNN. Defaults to 1
    :type num_layers: int, optional
    :param hidden_size: Number of neurons in the hidden layer(s). Defaults to 100
    :type hidden_size: int, optional
    :param embedding_size: Size of item embeddings. Defaults to 250
    :type embedding_size: int, optional
    :param dropout: Dropout applied to embeddings and hidden layer(s), 0 for no dropout
    :type dropout: float
    :param loss_fn: Loss function. One of "top1", "top1-max", "bpr",
        "bpr-max". Defaults to "bpr"
    :type loss_fn: str, optional
    :param U: Number of negative samples used for bpr, bpr-max, top1, top1-max
        loss calculation. Sampled uniformly random
    :type U: int, optional
    :param optimization_algorithm: Gradient descent optimizer, one of "sgd", "adagrad".
        Defaults to adagrad.
    :type optimization_algorithm: str, optional
    :param batch_size: Number of examples in a mini-batch
        Defaults to 512.
    :type batch_size: int, optional
    :param learning_rate: Gradient descent initial learning rate
        Defaults to 0.03
    :type learning_rate: float, optional
    :param momentum: Momentum when using the sgd optimizer
        Defaults to 0.0
    :type momentum: float, optional
    :param clipnorm: Clip the gradient's l2 norm, None for no clipping
        Defaults to 1.0
    :type clipnorm: float, optional
    :param seed: Seed for random number generator
        Defaults to 2
    :type seed: int, optional
    :param bptt: Number of backpropagation through time steps
        Defaults to 1
    :type bptt: int, optional
    :param max_epochs: Max training runs through entire dataset
        Defaults to 5
    :type max_epochs: int, optional
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        embedding_size: int = 250,
        dropout: float = 0.0,
        loss_fn: str = "bpr",
        U: int = 50,
        optimization_algorithm: str = "adagrad",
        batch_size: int = 512,
        learning_rate: float = 0.03,
        momentum: float = 0.0,
        clipnorm: float = 1.0,
        seed: int = 2,
        bptt: int = 1,
        max_epochs: int = 5,
        save_best_to_file: bool = False,
        keep_last: bool = True,
        stopping_criterion: str = "recall",
    ):
        super().__init__(
            num_layers=num_layers,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            dropout=dropout,
            optimization_algorithm=optimization_algorithm,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            clipnorm=clipnorm,
            seed=seed,
            bptt=bptt,
            max_epochs=max_epochs,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            stopping_criterion=stopping_criterion,
            U=U
        )

        self.loss_fn = loss_fn

        self._criterion = {
            "top1": top1_loss,
            "top1-max": top1_max_loss,
            "bpr": bpr_loss,
            "bpr-max": bpr_max_loss,
        }[self.loss_fn]

    def _compute_loss(
        self,
        output: torch.FloatTensor,
        targets_chunk: torch.LongTensor,
        negatives_chunk: torch.LongTensor,
        true_input_mask: torch.BoolTensor
    ) -> torch.Tensor:

        # Select score for positive and negative sample for all tokens
        # For each target, gather the score predicted in output.
        # positive_scores has shape (batch_size x bptt x 1)
        positive_scores = torch.gather(
            output, 2, targets_chunk.unsqueeze(-1)
        ).squeeze(-1)
        # negative scores has shape (batch_size x bptt x U)
        negative_scores = torch.gather(output, 2, negatives_chunk)

        assert true_input_mask.shape == positive_scores.shape

        true_batch_size, max_hist_len = positive_scores.shape

        # Check if I need to do all this flattening
        true_input_mask_flat = true_input_mask.view(
            true_batch_size * max_hist_len)

        positive_scores_flat = positive_scores.view(
            true_batch_size * max_hist_len, 1)
        negative_scores_flat = negative_scores.view(
            true_batch_size * max_hist_len, negative_scores.shape[2]
        )

        return self._criterion(positive_scores_flat[true_input_mask_flat], negative_scores_flat[true_input_mask_flat])


class GRU4RecTorch(nn.Module):
    """PyTorch definition of a basic recurrent neural network for session-based
    recommendations.

    :param num_items: Number of items
    :type num_items: int
    :param hidden_size: Number of neurons in the hidden layer(s)
    :type hidden_size: int
    :param embedding_size: Size of the item embeddings, None for no embeddings
    :type embedding_size: int
    :param pad_token: Index of the padding_token
    :type pad_token: int
    :param num_layers: Number of hidden layers, defaults to 1
    :type num_layers: int, optional
    :param dropout: Dropout applied to embeddings and hidden layers, defaults to 0
    :type dropout: float, optional
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
        super().__init__()
        self.input_size = num_items
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = num_items
        self.pad_token = pad_token
        self.drop = nn.Dropout(dropout, inplace=True)

        # Passing pad_token will make sure these embeddings are always zero-valued.
        self.emb = nn.Embedding(
            num_items + 1, embedding_size, padding_idx=pad_token)
        self.rnn = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0),
            batch_first=True,
        )
        # Also return the padding token
        self.lin = nn.Linear(hidden_size, num_items + 1)
        self.act = nn.Softmax(dim=2)
        self.init_weights()

    def forward(
        self, x: torch.LongTensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes scores for each item given an action sequence and previous
        hidden state.

        :param input: Action sequences as a Tensor of shape (A, B)
        :param hidden: Previous hidden state, shape (sequence length, batch size, hidden layer size)
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
                emb_x, seq_lengths.cpu(), batch_first=True
            )

            padded_rnn_x, hidden = self.rnn(padded_emb_x, hidden)

            rnn_x, _ = nn.utils.rnn.pad_packed_sequence(
                padded_rnn_x, batch_first=True)

        else:
            rnn_x, hidden = self.rnn(emb_x, hidden)

        self.drop(rnn_x)

        out = self.act(self.lin(rnn_x))

        return out, hidden

    def init_weights(self) -> None:
        """Initializes all model parameters uniformly.
        """
        initrange = 0.01
        for param in self.parameters():
            nn.init.uniform_(param, -initrange, initrange)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Returns an initial, zero-valued hidden state with shape (L B, H).
        """
        return torch.zeros((self.rnn.num_layers, batch_size, self.hidden_size))
