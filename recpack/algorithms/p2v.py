# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from typing import Tuple, Iterator, List, Optional
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.samplers import PositiveNegativeSampler
from recpack.algorithms.loss_functions import skipgram_negative_sampling_loss
from recpack.algorithms.util import get_batches
from recpack.matrix import InteractionMatrix, Matrix, to_csr_matrix
from recpack.util import get_top_K_values


logger = logging.getLogger("recpack")


class Prod2Vec(TorchMLAlgorithm):
    """
    Prod2Vec algorithm from the paper:
    "E-commerce in Your Inbox: Product Recommendations at Scale".
    (https://arxiv.org/abs/1606.07154)

    Applies SkipGram Negative Sampling to sequences
    of user interactions to learn an input (target) and output (context) embedding for every item.

    Recommendations are made by computing the similarity between input embeddings
    of different items and recommending those items most similar.

    Where possible, defaults were taken from the paper.

    :param num_components: The size of the embedding vectors for both input and output embeddings, defaults to 300
    :type num_components: int, optional
    :param num_negatives: Number of negative samples for every positive sample, defaults to 10
    :type num_negatives: int, optional
    :param window_size: Size of the context window to the left and to the right of the target item
         used in skipgram negative sampling. Defaults to 2
    :type window_size: int, optional
    :param stopping_criterion: Used to identify the best model computed thus far.
        The string indicates the name of the stopping criterion.
        Which criterions are available can be found at StoppingCriterion.FUNCTIONS.
        Defaults to 'precision'
    :type stopping_criterion: str, optional
    :param K: Used to set a maximum number of neighbors, i.e., similar items,
        for every item. Useful when the number of items in the dataset is large.
        Defaults to 200
    :type K: int, optional
    :param batch_size: Batch size for Adam optimizer. Higher batch sizes make each epoch more efficient,
        but increases the amount of epochs needed to converge to the optimum,
        by reducing the amount of updates per epoch. Defaults to 1000
    :type batch_size: int, optional
    :param learning_rate: Learning rate, defaults to 0.01
    :type learning_rate: float, optional
    :param clipnorm: Clips gradient norm.
        The norm is computed over all gradients together,
        as if they were concatenated into a single vector. Defaults to 1.0
    :type clipnorm: float, optional
    :param max_epochs: Maximum number of epochs (iterations), defaults to 10
    :type max_epochs: int, optional
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
    :param seed: Seed for random sampling. Useful for reproducible results,
        defaults to None
    :type seed: int, optional
    :param save_best_to_file: If true, the best model will be saved after training.
        Defaults to False
    :type save_best_to_file: bool, optional
    :param replace: Sample with or without replacement
        (see :class:`recpack.algorithms.samplers.PositiveNegativeSampler` ), defaults to False
    :type replace: bool, optional
    :param exact: If False (default) negatives are checked against the corresponding positive sample only,
        allowing for (rare) collisions. If collisions should be avoided at all costs,
        use exact = True, but suffer decreased performance. Defaults to False
    :type exact: bool, optional
    :param keep_last: Retain last model,
        rather than best (according to stopping criterion value on validation data), defaults to False
    :type keep_last: bool, optional
    :param distribution: Which distribution to use to sample negatives. Options are `["uniform", "unigram"]`.
        Uniform distribution will sample all items equally likely.
        Unigram distribution puts more weight on popular items. Defaults to "uniform"
    :type distribution: str, optional
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
        num_components: int = 300,
        num_negatives: int = 10,
        window_size: int = 2,
        stopping_criterion: str = "precision",
        K: int = 200,
        batch_size: int = 1000,
        learning_rate: float = 0.01,
        clipnorm: float = 1.0,
        max_epochs: int = 10,
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        seed: Optional[int] = None,
        save_best_to_file: bool = False,
        replace: bool = False,
        exact: bool = False,
        keep_last: bool = False,
        distribution="uniform",
        predict_topK: Optional[int] = None,
        validation_sample_size: Optional[int] = None,
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

        self.num_components = num_components
        self.num_negatives = num_negatives
        self.window_size = window_size

        self.K = K
        self.replace = replace
        self.exact = exact
        self.clipnorm = clipnorm

        self.distribution = distribution

    def _init_model(self, X: Matrix) -> None:
        self.model_ = SkipGram(X.shape[1], self.num_components).to(self.device)
        self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        # Initialise sampler
        self.sampler = PositiveNegativeSampler(
            num_negatives=self.num_negatives,
            batch_size=self.batch_size,
            replace=self.replace,
            exact=self.exact,
            distribution=self.distribution,
        )

    def _train_epoch(self, X: InteractionMatrix) -> list:
        assert self.model_ is not None
        losses = []
        # generator will just be restarted for each epoch.
        for (
            focus_batch,
            positives_batch,
            negatives_batch,
        ) in self._skipgram_sample_pairs(X):
            positives_batch = positives_batch.to(self.device)
            focus_batch = focus_batch.to(self.device)
            negatives_batch = negatives_batch.to(self.device)

            self.optimizer.zero_grad()

            positive_sim = self.model_(focus_batch.unsqueeze(-1), positives_batch.unsqueeze(-1))
            negative_sim = self.model_(focus_batch.unsqueeze(-1), negatives_batch)

            loss = self._compute_loss(positive_sim, negative_sim)
            loss.backward()
            losses.append(loss.item())
            nn.utils.clip_grad_norm_(self.model_.parameters(), self.clipnorm)
            self.optimizer.step()

        # Construct the similarity matrix based on the updated model.
        # While the base class does not use the X interaction matrix,
        # Other subclasses do, and it reduces code duplication to pass it
        self._create_similarity_matrix(X)

        return losses

    def _compute_loss(self, positive_sim: torch.Tensor, negative_sim: torch.Tensor) -> torch.Tensor:
        return skipgram_negative_sampling_loss(positive_sim, negative_sim)

    def _create_similarity_matrix(self, X: InteractionMatrix) -> None:
        # K similar items + self-similarity
        K = self.K + 1
        batch_size = 1000

        embedding = self.model_.input_embeddings.weight.cpu().detach().numpy()
        num_items = embedding.shape[0]
        if K > num_items:
            K = num_items
            warnings.warn(
                """K was set to a value larger than the number of items.
                 As a result, the number of neighbors for each item will not be restricted""",
                UserWarning,
            )

        # Set embedding of inactive items to 0
        active_items = X.active_items
        inactive_items = list(set(range(num_items)).difference(active_items))
        embedding[inactive_items] = 0

        item_cosine_similarity = lil_matrix((num_items, num_items))

        for batch in get_batches(list(active_items), batch_size=batch_size):
            Y = embedding[batch]
            item_cosine_similarity_batch = csr_matrix(cosine_similarity(Y, embedding))

            item_cosine_similarity[batch] = get_top_K_values(item_cosine_similarity_batch, K)
        # no self similarity, set diagonal to zero
        item_cosine_similarity.setdiag(0)
        self.similarity_matrix_ = csr_matrix(item_cosine_similarity)

    def _batch_predict(self, X: csr_matrix, users: List[int]) -> csr_matrix:
        """Predict scores for matrix X, given the selected users in this batch

        :param X: Matrix of user item interactions,
            expected to only contain interactions for those users that are in `users`
        :type X: csr_matrix
        :param users: users selected for recommendation
        :type users: List[int]
        :return: Sparse matrix of scores per user item pair.
        :rtype: csr_matrix
        """
        scores = X @ self.similarity_matrix_
        return scores

    def _skipgram_sample_pairs(
        self, X: InteractionMatrix
    ) -> Iterator[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]:
        """Creates a training dataset using the skipgrams and negative sampling method.

        First, the sequences of items (iid) are grouped per user (uid).
        Next, a windowing operation is applied over each seperate item sequence.
        These windows are sliced and re-stacked in order to create skipgrams of two items.

        :param X: InteractionMatrix
        :yield: focus_batch, positive_samples_batch, negative_samples_batch
        :rtype: Iterator[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]
        """
        # Window, then extract focus (middle element) and context (all other elements).
        windowed_sequences = window(X.sorted_item_history, self.window_size)
        context = np.hstack(
            (
                windowed_sequences[:, : self.window_size],
                windowed_sequences[:, self.window_size + 1 :],
            )
        )
        focus = windowed_sequences[:, self.window_size]
        # Apply np.repeat to focus to turn [0,1,2,3] into [0, 0, 1, 1, ...]
        # where number of repetitions is self.window_size
        # Squeeze final dimension from context so they line up neatly.
        positives = np.column_stack([focus.repeat(self.window_size * 2), context.reshape(-1)])
        # Remove any NaN valued rows (consequence of windowing)
        positives = positives[~np.isnan(positives).any(axis=1)].astype(int)

        coocc = lil_matrix((X.shape[1], X.shape[1]), dtype=int)
        coocc[positives[:, 0], positives[:, 1]] = 1
        coocc.setdiag(1)
        coocc = coocc.tocsr()

        yield from self.sampler.sample(
            X=coocc,
            positives=positives,
        )

    def _transform_fit_input(
        self, X: Matrix, validation_data: Tuple[Matrix, Matrix]
    ) -> Tuple[InteractionMatrix, Tuple[csr_matrix, csr_matrix]]:
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X, to_csr_matrix(validation_data, binary=True)


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, num_components: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_components = num_components
        self.input_embeddings = nn.Embedding(vocab_size, num_components)
        self.output_embeddings = nn.Embedding(vocab_size, num_components)

        self.std = 1 / num_components ** 0.5
        # Initialise embeddings to a random start
        nn.init.normal_(self.input_embeddings.weight, std=self.std)
        nn.init.normal_(self.output_embeddings.weight, std=self.std)

    def forward(self, focus_item_batch: torch.LongTensor, context_items_batch: torch.LongTensor) -> torch.Tensor:
        # Create a (batch_size, embedding_dim, 1) tensor
        focus_vector = torch.movedim(self.input_embeddings(focus_item_batch), 1, 2)
        # Expected of size (batch_size, 1, embedding_dim)
        context_vectors = self.output_embeddings(context_items_batch)
        return torch.bmm(context_vectors, focus_vector).squeeze(-1)


def window(sequences: Iterator[Tuple[int, list]], window_size: int) -> np.ndarray:
    """
    Will apply a windowing operation to a sequence of item sequences.
    Note: pads the sequences to make sure edge sequences are included.

    :param sequences: Iterator yielding sorted user histories as a uid, list of items tuple
    :type sequences: Iterator[Tuple[int, list]]
    :param window_size: Size of the window on both the left and right of the focus item
    :type window_size: int
    :return: Windowed sequences of items
    :rtype: np.ndarray
    """
    padded_sequences = [[np.NAN] * window_size + list(s) + [np.NAN] * window_size for uid, s in sequences]
    w = [
        w.tolist()
        for sequence in padded_sequences
        if len(sequence) >= window_size
        for w in sliding_window_view(sequence, 2 * window_size + 1)
    ]
    return np.array(w)
