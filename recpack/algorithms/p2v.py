import logging
from typing import Tuple
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.sparse import csr_matrix, lil_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.data.matrix import InteractionMatrix, Matrix, to_csr_matrix
from recpack.algorithms.samplers import PositiveNegativeSampler
from recpack.algorithms.loss_functions import skipgram_negative_sampling_loss
from recpack.algorithms.util import sample_rows
from recpack.util import get_top_K_values

logger = logging.getLogger("recpack")


class Prod2Vec(TorchMLAlgorithm):
    """
    Prod2Vec algorithm from the paper: "E-commerce in Your Inbox: Product Recommendations at Scale". (https://arxiv.org/abs/1606.07154)

    Extends the TorchMLAlgorithm class.

    :param embedding_size: size of the embedding vectors
    :type embedding_size: int
    :param num_neg_samples: number of negative samples per positive sample
    :rtype num_neg_samples: int
    :param window_size: number of elements to the left and right of the target
    :rtype: int
    :param batch_size: How many samples to use in each update step.
    Higher batch sizes make each epoch more efficient,
    but increases the amount of epochs needed to converge to the optimum,
    by reducing the amount of updates per epoch.
    :type batch_size: int
    :param max_epochs: The max number of epochs to train.
        If the stopping criterion uses early stopping, less epochs could be used.
    :type max_epochs: int
    :param learning_rate: How much to update the weights at each update.
    :type learning_rate: float
    :param clipnorm: Clips gradient norm. The norm is computed over all gradients together, as if they were concatenated into a single vector.
    :type clipnorm: int, optional
    :param stopping_criterion: Name of the stopping criterion to use for training.
        For available values,
        check :meth:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`
    :type stopping_criterion: str
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
    :param K: for top-K most similar items
    :rtype k: int
    :param replace: sample with or without replacement (see PositiveNegativeSampler)
    :rtype replace: bool
    :param exact: If False (default) negatives are checked against the corresponding positive sample only, allowing for (rare) collisions.
    If collisions should be avoided at all costs, use exact = True, but suffer decreased performance. (see PositiveNegativeSampler)
    :rtype exact: bool
    :param keep_last: Retain last model,
        rather than best (according to stopping criterion value on validation data), defaults to False
    :type keep_last: bool, optional
    :param distribution: Which distribution to use to sample negatives. Options are `["uniform", "unigram"]`.
        defaults to `'uniform'`. Uniform distribution will sample all items equally likely.
        The unigram distribution puts more weight on popular items.
    :type distribution: string, optional
    """

    def __init__(
        self,
        embedding_size: int,
        num_neg_samples: int,
        window_size: int,
        stopping_criterion: str,
        K=10,
        batch_size=1000,
        learning_rate=0.01,
        clipnorm=1,
        max_epochs=10,
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        seed=None,
        save_best_to_file=False,
        replace=False,
        exact=False,
        keep_last=False,
        distribution="uniform",
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
        )

        self.embedding_size = embedding_size
        self.num_neg_samples = num_neg_samples
        self.window_size = window_size
        self.similarity_matrix_ = None
        self.K = K
        self.replace = replace
        self.exact = exact
        self.clipnorm = clipnorm

        self.distribution = distribution

        # Initialise sampler
        self.sampler = PositiveNegativeSampler(
            U=self.num_neg_samples,
            batch_size=self.batch_size,
            replace=self.replace,
            exact=self.exact,
            distribution=self.distribution,
        )

    def _init_model(self, X: Matrix) -> None:
        self.model_ = SkipGram(X.shape[1], self.embedding_size)
        self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

    def _evaluate(self, val_in: csr_matrix, val_out: csr_matrix) -> None:
        if self.similarity_matrix_ is None:
            raise RuntimeError(
                "Expected similarity matrix to be computed before _evaluate"
            )
        val_in_selection, val_out_selection = sample_rows(
            val_in, val_out, sample_size=1000
        )
        predictions = self._batch_predict(val_in_selection)
        better = self.stopping_criterion.update(val_out_selection, predictions)

        if better:
            logger.info("Model improved. Storing better model.")
            self._save_best()

    def _train_epoch(self, X: InteractionMatrix) -> list:
        assert self.model_ is not None
        losses = []
        # generator will just be restarted for each epoch.
        for (
            focus_batch,
            positives_batch,
            negatives_batch,
        ) in self._skipgram_sample_pairs(X):
            self.optimizer.zero_grad()

            positive_sim = self.model_(
                focus_batch.unsqueeze(-1), positives_batch.unsqueeze(-1)
            )
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

    def _compute_loss(self, positive_sim, negative_sim):
        return skipgram_negative_sampling_loss(positive_sim, negative_sim)

    def _create_similarity_matrix(self, X: InteractionMatrix):
        # K similar items + self-similarity
        K = self.K + 1
        batch_size = 1000

        embedding = self.model_.input_embeddings.weight.detach().numpy()
        num_items = embedding.shape[0]
        if K > num_items:
            K = num_items
            warnings.warn("K is larger than the number of items.", UserWarning)

        item_cosine_similarity_ = lil_matrix((num_items, num_items))

        for batch in range(0, num_items, batch_size):
            Y = embedding[batch : batch + batch_size]
            item_cosine_similarity_batch = csr_matrix(cosine_similarity(Y, embedding))

            item_cosine_similarity_[batch : batch + batch_size] = get_top_K_values(
                item_cosine_similarity_batch, K
            )
        # no self similarity, set diagonal to zero
        item_cosine_similarity_.setdiag(0)
        self.similarity_matrix_ = csr_matrix(item_cosine_similarity_)

    def _batch_predict(self, X: csr_matrix):
        # TODO Do we even need a separate batch_predict method?
        scores = X @ self.similarity_matrix_
        if not isinstance(scores, csr_matrix):
            scores = csr_matrix(scores)
        return scores

    def _predict(self, X: csr_matrix) -> csr_matrix:
        results = self._batch_predict(X)
        return results.tocsr()

    def _skipgram_sample_pairs(self, X: InteractionMatrix):
        """Creates a training dataset using the skipgrams and negative sampling method.

        First, the sequences of items (iid) are grouped per user (uid).
        Next, a windowing operation is applied over each seperate item sequence.
        These windows are sliced and re-stacked in order to create skipgrams of two items (pos_training_set).
        Next, a unigram distribution of all the items in the dataset is created.
        This unigram distribution is used for creating the negative samples.

        :param X: InteractionMatrix
        :yield: focus_batch, positive_samples_batch, negative_samples_batch
        :rtype: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
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
        positives = np.column_stack(
            [focus.repeat(self.window_size * 2), context.reshape(-1)]
        )
        # Remove any NaN valued rows (consequence of windowing)
        positives = positives[~np.isnan(positives).any(axis=1)].astype(int)

        coocc = lil_matrix((X.shape[1], X.shape[1]), dtype=int)
        coocc[positives[:, 0], positives[:, 1]] = 1
        # TODO I don't think this is necessary anymore, it's a result of windowing.
        # coocc = to_binary(coocc + coocc.T)
        coocc.setdiag(1)
        coocc = coocc.tocsr()

        yield from self.sampler.sample(
            X=coocc,
            positives=positives,
        )

    def _transform_fit_input(self, X: Matrix, validation_data: Tuple[Matrix, Matrix]):
        return X, to_csr_matrix(validation_data, binary=True)


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.input_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_size)

        self.std = 1 / embedding_size ** 0.5
        # Initialise embeddings to a random start
        nn.init.normal_(self.input_embeddings.weight, std=self.std)
        nn.init.normal_(self.output_embeddings.weight, std=self.std)

    def forward(self, focus_item_batch, context_items_batch):
        # Create a (batch_size, embedding_dim, 1) tensor
        focus_vector = torch.movedim(self.input_embeddings(focus_item_batch), 1, 2)
        # Expected of size (batch_size, 1, embedding_dim)
        context_vectors = self.output_embeddings(context_items_batch)
        return torch.bmm(context_vectors, focus_vector).squeeze(-1)


def window(sequences, window_size):
    """
    Will apply a windowing operation to a sequence of item sequences.

    Note: pads the sequences to make sure edge cases are also accounted for.

    :param sequences: iterable of iterables
    :param window_size: the size of the window
    :returns: windowed sequences
    :rtype: numpy.ndarray
    """
    padded_sequences = [
        [np.NAN] * window_size + list(s) + [np.NAN] * window_size
        for uid, s in sequences
    ]
    w = [
        w.tolist()
        for sequence in padded_sequences
        if len(sequence) >= window_size
        for w in sliding_window_view(sequence, 2 * window_size + 1)
    ]
    return np.array(w)
