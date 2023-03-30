# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from math import ceil
import time
from typing import Tuple, List, Iterator, Optional

from scipy.sparse import csr_matrix, lil_matrix

import torch
import torch.nn as nn
import torch.optim as optim

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.loss_functions import (
    bpr_loss,
    bpr_max_loss,
    top1_loss,
    top1_max_loss,
)
from recpack.algorithms.samplers import (
    SequenceMiniBatchPositivesTargetsNegativesSampler,
    SequenceMiniBatchSampler,
)
from recpack.matrix import InteractionMatrix


logger = logging.getLogger("recpack")


class GRU4Rec(TorchMLAlgorithm):
    """Base class for recurrent neural networks for session-based recommendations.

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

    :param num_layers: Number of hidden layers in the RNN. Defaults to 1
    :type num_layers: int, optional
    :param hidden_size: Number of neurons in the hidden layer(s). Defaults to 100
    :type hidden_size: int, optional
    :param num_components: Size of item embeddings. Defaults to 250
    :type num_components: int, optional
    :param dropout: Dropout applied to embeddings and hidden layer(s).
        Defauls to 0 (no dropout)
    :type dropout: float
    :param optimization_algorithm: Gradient descent optimizer, one of "sgd", "adagrad".
        Defaults to "adagrad"
    :type optimization_algorithm: str, optional
    :param momentum: Momentum when using the sgd optimizer.
        Defaults to 0.0
    :type momentum: float, optional
    :param clipnorm: Clip the gradient's l2 norm, None for no clipping.
        Defaults to 1.0
    :type clipnorm: float, optional
    :param bptt: Number of backpropagation through time steps.
        Defaults to 1
    :type bptt: int, optional
    :param num_negatives: Number of negatives to sample for every positive.
        Defaults to 0
    :type num_negatives: int, optional
    :param batch_size: Number of examples in a mini-batch.
        Defaults to 512.
    :type batch_size: int, optional
    :param max_epochs: Max training runs through entire dataset.
        Defaults to 5
    :type max_epochs: int, optional
    :param learning_rate: Gradient descent initial learning rate
        Defaults to 0.03
    :type learning_rate: float, optional
    :param stopping_criterion: Name of the stopping criterion to use for training.
        For available values,
        check :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`
    :type stopping_criterion: str, optional
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
        Defaults to 0.0
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
    :param validation_sample_size: Amount of users that will be sampled to calculate
        validation loss and stopping criterion value.
        This reduces computation time during validation, such that training times are strongly reduced.
        If None, all nonzero users are used. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        num_components: int = 250,
        dropout: float = 0.0,
        optimization_algorithm: str = "adagrad",
        momentum: float = 0.0,
        clipnorm: float = 1.0,
        bptt: int = 1,
        num_negatives: int = 0,
        batch_size: int = 512,
        max_epochs: int = 5,
        learning_rate: float = 0.03,
        stopping_criterion: str = "recall",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        seed: Optional[int] = None,
        save_best_to_file: bool = False,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
    ):
        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_components = num_components
        self.dropout = dropout
        self.optimization_algorithm = optimization_algorithm
        self.momentum = momentum
        self.clipnorm = clipnorm
        self.bptt = bptt
        self.num_negatives = num_negatives

    def _init_model(self, X: InteractionMatrix) -> None:
        # Invalid item ID. Used to mask inputs to the RNN
        self.num_items = X.shape[1]
        self.pad_token = self.num_items

        self.model_ = GRU4RecTorch(
            self.num_items,
            self.hidden_size,
            self.num_components,
            self.pad_token,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        if self.optimization_algorithm == "sgd":
            self.optimizer = optim.SGD(self.model_.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif self.optimization_algorithm == "adagrad":
            self.optimizer = optim.Adagrad(self.model_.parameters(), lr=self.learning_rate)

        self.predict_sampler = SequenceMiniBatchSampler(self.pad_token, batch_size=self.batch_size)

        self.fit_sampler = SequenceMiniBatchPositivesTargetsNegativesSampler(
            self.num_negatives, self.pad_token, batch_size=self.batch_size
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
        self._assert_is_interaction_matrix(X, *validation_data)
        self._assert_has_timestamps(X, *validation_data)

        return X, validation_data

    def _transform_predict_input(self, X: InteractionMatrix) -> InteractionMatrix:
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _train_epoch(self, X: InteractionMatrix) -> List[float]:

        losses = []

        for (
            _,
            positives_batch,
            targets_batch,
            negatives_batch,
        ) in self.fit_sampler.sample(X):
            st = time.time()
            # positives shape = (batch_size x |max_hist_length|)
            # targets shape = (batch_size x |max_hist_length|)
            # negatives shape = (batch_size x |max_hist_length| x self.num_negatives)
            positives_batch = positives_batch.to(self.device)
            targets_batch = targets_batch.to(self.device)
            negatives_batch = negatives_batch.to(self.device)

            logger.debug(f"Takes {time.time() - st} seconds to convert to GPU")

            batch_loss = 0
            true_batch_size = positives_batch.shape[0]
            # Want to reuse this between chunks of the same batch of sequences
            hidden = self.model_.init_hidden(true_batch_size).to(self.device)

            # Generate vertical chunks of BPTT width
            for input_chunk, target_chunk, neg_chunk in self._chunk(
                self.bptt, positives_batch, targets_batch, negatives_batch
            ):
                input_mask = target_chunk != self.pad_token
                # Remove rows with only pad tokens from chunk and from hidden.
                # We can do this because the array is sorted.
                true_rows = input_mask.any(axis=1)
                true_input_chunk = input_chunk[true_rows]
                true_target_chunk = target_chunk[true_rows]
                true_neg_chunk = neg_chunk[true_rows]
                true_hidden = hidden[:, true_rows, :]
                true_input_mask = input_mask[true_rows]

                if true_input_chunk.shape[0] != 0:

                    self.optimizer.zero_grad()
                    output, hidden[:, true_rows, :] = self.model_(true_input_chunk, true_hidden)

                    loss = self._compute_loss(output, true_target_chunk, true_neg_chunk, true_input_mask)

                    batch_loss += loss.item()
                    loss.backward()

                    if self.clipnorm:
                        nn.utils.clip_grad_norm_(self.model_.parameters(), self.clipnorm)
                    
                    self.optimizer.step()
                    
                    hidden = hidden.detach()
            logger.debug(f"Takes {time.time() - st} seconds to process batch")
            losses.append(batch_loss)

        return losses

    def _compute_loss(
        self,
        output: torch.FloatTensor,
        targets_chunk: torch.LongTensor,
        negatives_chunk: torch.LongTensor,
        true_input_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _chunk(self, chunk_size: int, *tensors: torch.LongTensor) -> Iterator[Tuple[torch.LongTensor, ...]]:
        """Split tensors into chunks of self.bptt width.
        Chunks can be of width self.bptt - 1 if max_hist_len % self. bptt != 0.

        :param tensors: Tensors to be chunked.
        :type tensors: torch.LongTensor
        :yield: Tuple of chunked tensors, one chunk of each of the input tensors.
        :rtype: Iterator[Tuple[torch.LongTensor, ...]]
        """
        max_hist_len = tensors[0].shape[1]

        num_chunks = ceil(max_hist_len / chunk_size)
        split_tensors = [t.tensor_split(num_chunks, axis=1) for t in tensors]

        return zip(*split_tensors)

    def _predict(self, X: InteractionMatrix):
        X_pred = lil_matrix(X.shape)
        self.model_.eval()
        with torch.no_grad():
            # Loop through users in batches
            for uid_batch, positives_batch in self.predict_sampler.sample(X):
                batch_size = positives_batch.shape[0]
                hidden = self.model_.init_hidden(batch_size).to(self.device)

                # Process the history in chunks, otherwise the linear layer goes OOM.
                # 3M entries seemed a reasonable max
                chunk_size = int((10 ** 9) / (self.batch_size * self.num_items))
                for (input_chunk,) in self._chunk(chunk_size, positives_batch.to(self.device)):
                    input_mask = input_chunk != self.pad_token
                    # Remove rows with only pad tokens from chunk and from hidden.
                    # We can do this because the array is sorted.
                    true_rows = input_mask.any(axis=1)
                    true_input_chunk = input_chunk[true_rows]
                    true_hidden = hidden[:, true_rows, :]

                    if true_input_chunk.shape[0] != 0:
                        output_chunk, hidden[:, true_rows, :] = self.model_(true_input_chunk, true_hidden)
                        # Use only the prediction for the final item, i.e. last non-padding ID.
                        last_item_ix_in_chunk = (input_chunk != self.pad_token).sum(axis=1) - 1

                        is_last_item_in_chunk = (last_item_ix_in_chunk >= 0).detach().cpu().numpy()
                        # Item scores is a matrix with the scores for each item
                        # based on the last item in the sequence
                        last_item_ix = last_item_ix_in_chunk[is_last_item_in_chunk]
                        uid_batch_w_last_item = uid_batch[is_last_item_in_chunk].detach().cpu().numpy()
                        item_scores = (
                            output_chunk[
                                torch.arange(0, last_item_ix.shape[0], dtype=int),
                                last_item_ix,
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        X_pred[uid_batch_w_last_item] = self._get_top_k_recommendations(
                            csr_matrix(item_scores[:, :-1])
                        )

        return X_pred.tocsr()


class GRU4RecCrossEntropy(GRU4Rec):
    """A recurrent neural network for session-based recommendations.

    The algorithm, also known as GRU4Rec, was introduced in the 2016 and 2018 papers
    "Session-based Recommendations with Recurrent Neural Networks" and
    "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations".

    This version implements the CrossEntropy variant of the algorithm. For negative sampling,
    see :class:`GRU4RecNegSampling`.

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

    :param num_layers: Number of hidden layers in the RNN. Defaults to 1
    :type num_layers: int, optional
    :param hidden_size: Number of neurons in the hidden layer(s). Defaults to 100
    :type hidden_size: int, optional
    :param num_components: Size of item embeddings. Defaults to 250
    :type num_components: int, optional
    :param dropout: Dropout applied to embeddings and hidden layer(s), 0 for no dropout
    :type dropout: float
    :param optimization_algorithm: Gradient descent optimizer, one of "sgd", "adagrad".
        Defaults to adagrad.
    :type optimization_algorithm: str, optional
    :param momentum: Momentum when using the sgd optimizer
        Defaults to 0.0
    :type momentum: float, optional
    :param clipnorm: Clip the gradient's l2 norm, None for no clipping
        Defaults to 1.0
    :type clipnorm: float, optional
    :param bptt: Number of backpropagation through time steps.
        Defaults to 1
    :type bptt: int, optional
    :param batch_size: Number of examples in a mini-batch.
        Defaults to 512.
    :type batch_size: int, optional
    :param max_epochs: Max training runs through entire dataset.
        Defaults to 5
    :type max_epochs: int, optional
    :param learning_rate: Gradient descent initial learning rate
        Defaults to 0.03
    :type learning_rate: float, optional
    :param stopping_criterion: Name of the stopping criterion to use for training.
        For available values,
        check :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`
    :type stopping_criterion: str, optional
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
        Defaults to 0.0
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
    :param validation_sample_size: Amount of users that will be sampled to calculate
        validation loss and stopping criterion value.
        This reduces computation time during validation, such that training times are strongly reduced.
        If None, all nonzero users are used. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        num_components: int = 250,
        dropout: float = 0.0,
        optimization_algorithm: str = "adagrad",
        momentum: float = 0.0,
        clipnorm: float = 1.0,
        bptt: int = 1,
        batch_size: int = 512,
        max_epochs: int = 5,
        learning_rate: float = 0.03,
        stopping_criterion: str = "recall",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        seed: int = None,
        save_best_to_file: bool = False,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
    ):
        super().__init__(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_components=num_components,
            dropout=dropout,
            optimization_algorithm=optimization_algorithm,
            momentum=momentum,
            clipnorm=clipnorm,
            bptt=bptt,
            num_negatives=0,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            stopping_criterion=stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )

        self._criterion = nn.CrossEntropyLoss()

    def _compute_loss(
        self,
        output: torch.FloatTensor,
        targets_chunk: torch.LongTensor,
        negatives_chunk: torch.LongTensor,
        true_input_mask: torch.BoolTensor,
    ) -> torch.Tensor:

        return self._criterion(output[true_input_mask], targets_chunk[true_input_mask])


class GRU4RecNegSampling(GRU4Rec):
    """A recurrent neural network for session-based recommendations.

    The algorithm, also known as GRU4Rec, was introduced in the 2016 and 2018 papers
    "Session-based Recommendations with Recurrent Neural Networks" and
    "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations"

    This version implements the Negative Sampling variant of the algorithm. For cross-entropy,
    see :class:`GRU4RecCrossEntropy`.

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
    :param num_components: Size of item embeddings. Defaults to 250
    :type num_components: int, optional
    :param dropout: Dropout applied to embeddings and hidden layer(s).
        Defauls to 0 (no dropout)
    :type dropout: float
    :param loss_fn: Loss function. One of "top1", "top1-max", "bpr",
        "bpr-max". Defaults to "bpr"
    :type loss_fn: str, optional
    :param optimization_algorithm: Gradient descent optimizer, one of "sgd", "adagrad".
        Defaults to "adagrad"
    :type optimization_algorithm: str, optional
    :param momentum: Momentum when using the sgd optimizer.
        Defaults to 0.0
    :type momentum: float, optional
    :param clipnorm: Clip the gradient's l2 norm, None for no clipping.
        Defaults to 1.0
    :type clipnorm: float, optional
    :param bptt: Number of backpropagation through time steps.
        Defaults to 1
    :type bptt: int, optional
    :param num_negatives: Number of negatives to sample for every positive.
        Defaults to 50
    :type num_negatives: int, optional
    :param batch_size: Number of examples in a mini-batch.
        Defaults to 512.
    :type batch_size: int, optional
    :param max_epochs: Max training runs through entire dataset.
        Defaults to 5
    :type max_epochs: int, optional
    :param learning_rate: Gradient descent initial learning rate
        Defaults to 0.03
    :type learning_rate: float, optional
    :param stopping_criterion: Name of the stopping criterion to use for training.
        For available values,
        check :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`
    :type stopping_criterion: str, optional
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
        Defaults to 0.0
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
    :param validation_sample_size: Amount of users that will be sampled to calculate
        validation loss and stopping criterion value.
        This reduces computation time during validation, such that training times are strongly reduced.
        If None, all nonzero users are used. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        num_components: int = 250,
        dropout: float = 0.0,
        loss_fn: str = "bpr",
        optimization_algorithm: str = "adagrad",
        momentum: float = 0.0,
        clipnorm: float = 1.0,
        bptt: int = 1,
        num_negatives: int = 50,
        batch_size: int = 512,
        max_epochs: int = 5,
        learning_rate: float = 0.03,
        stopping_criterion: str = "recall",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        seed: int = None,
        save_best_to_file: bool = False,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
    ):
        super().__init__(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_components=num_components,
            dropout=dropout,
            optimization_algorithm=optimization_algorithm,
            momentum=momentum,
            clipnorm=clipnorm,
            bptt=bptt,
            num_negatives=num_negatives,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            stopping_criterion=stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
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
        true_input_mask: torch.BoolTensor,
    ) -> torch.Tensor:

        # Select score for positive and negative sample for all tokens
        # For each target, gather the score predicted in output.
        # positive_scores has shape (batch_size x bptt x 1)
        positive_scores = torch.gather(output, 2, targets_chunk.unsqueeze(-1)).squeeze(-1)
        # negative scores has shape (batch_size x bptt x U)
        negative_scores = torch.gather(output, 2, negatives_chunk)

        assert true_input_mask.shape == positive_scores.shape

        true_batch_size, max_hist_len = positive_scores.shape

        # Check if I need to do all this flattening
        true_input_mask_flat = true_input_mask.view(true_batch_size * max_hist_len)

        positive_scores_flat = positive_scores.view(true_batch_size * max_hist_len, 1)
        negative_scores_flat = negative_scores.view(true_batch_size * max_hist_len, negative_scores.shape[2])

        return self._criterion(
            positive_scores_flat[true_input_mask_flat],
            negative_scores_flat[true_input_mask_flat],
        )


class GRU4RecTorch(nn.Module):
    """PyTorch definition of a basic recurrent neural network for session-based
    recommendations.

    :param num_items: Number of items
    :type num_items: int
    :param hidden_size: Number of neurons in the hidden layer(s)
    :type hidden_size: int
    :param num_components: Size of the item embeddings, None for no embeddings
    :type num_components: int
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
        num_components: int,
        pad_token: int,
        num_layers: int = 1,
        dropout: float = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.num_components = num_components
        self.hidden_size = hidden_size
        self.output_size = num_items
        self.pad_token = pad_token
        self.drop = nn.Dropout(dropout, inplace=True)

        # Passing pad_token will make sure these embeddings are always zero-valued.
        self.emb = nn.Embedding(num_items + 1, num_components, padding_idx=pad_token)
        self.rnn = nn.GRU(
            num_components,
            hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0),
            batch_first=True,
        )
        # Also return the padding token
        self.lin = nn.Linear(hidden_size, num_items + 1)
        self.init_weights()

        # set the embedding for the padding token to 0
        with torch.no_grad():
            self.emb.weight[pad_token] = torch.zeros(num_components)

    def forward(self, x: torch.LongTensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

            padded_emb_x = nn.utils.rnn.pack_padded_sequence(emb_x, seq_lengths.cpu(), batch_first=True)

            padded_rnn_x, hidden = self.rnn(padded_emb_x, hidden)

            rnn_x, _ = nn.utils.rnn.pad_packed_sequence(padded_rnn_x, batch_first=True)

        else:
            rnn_x, hidden = self.rnn(emb_x, hidden)

        self.drop(rnn_x)

        out = self.lin(rnn_x)

        return out, hidden

    def init_weights(self) -> None:
        """Initializes all model parameters uniformly."""
        initrange = 0.01
        for param in self.parameters():
            nn.init.uniform_(param, -initrange, initrange)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Returns an initial, zero-valued hidden state with shape (L B, H)."""
        return torch.zeros((self.rnn.num_layers, batch_size, self.hidden_size))
