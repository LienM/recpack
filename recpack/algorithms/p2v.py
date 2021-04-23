import logging
import time
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
from recpack.data.matrix import InteractionMatrix, Matrix, to_binary, to_csr_matrix
from recpack.algorithms.samplers import sample_positives_and_negatives
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
    :param negative_samples: number of negative samples per positive sample
    :rtype negative_samples: int
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
        Defaults to 0.01
    :type min_improvement: float, optional
    :param seed: Seed to the randomizers, useful for reproducible results,
        defaults to None
    :type seed: int, optional
    :param save_best_to_file: If true, the best model will be saved after training,
        defaults to False
    :type save_best_to_file: bool, optional
    :param K: for top-K most similar items
    :rtype k: int
    :param replace: sample with or without replacement (see sample_positives_and_negatives)
    :rtype replace: bool
    :param exact: If False (default) negatives are checked against the corresponding positive sample only, allowing for (rare) collisions.
    If collisions should be avoided at all costs, use exact = True, but suffer decreased performance. (see sample_positives_and_negatives)
    :rtype exact: bool
    """

    def __init__(self, embedding_size: int, negative_samples: int,
                 window_size: int, stopping_criterion: str, K=10, batch_size=1000, learning_rate=0.01, max_epochs=10,
                 stop_early: bool = False, max_iter_no_change: int = 5, min_improvement: float = 0.01, seed=None,
                 save_best_to_file=False, replace=False, exact=False):

        super().__init__(batch_size, max_epochs, learning_rate, stopping_criterion, stop_early,
                         max_iter_no_change, min_improvement, seed, save_best_to_file)
        self.embedding_size = embedding_size
        self.negative_samples = negative_samples
        self.window_size = window_size
        self.similarity_matrix_ = None
        self.K = K
        self.replace = replace
        self.exact = exact

    def _init_model(self, X: Matrix) -> None:
        self.model_ = SkipGram(X.shape[1], self.embedding_size)
        self.optimizer = optim.Adam(
            self.model_.parameters(), lr=self.learning_rate)

    def _evaluate(self, val_in: csr_matrix, val_out: csr_matrix) -> None:
        val_in_selection, val_out_selection = sample_rows(
            val_in, val_out, sample_size=1000)
        self._create_similarity_matrix()
        predictions = self._batch_predict(val_in_selection)
        better = self.stopping_criterion.update(val_out_selection, predictions)

        if better:
            logger.info("Model improved. Storing better model.")
            self._save_best()

    def _train_epoch(self, X: csr_matrix) -> list:
        assert self.model_ is not None
        losses = []
        # generator will just be restarted for each epoch.
        generator = self._training_generator(
            X, self.negative_samples, self.window_size, batch=self.batch_size)

        for focus_batch, positives_batch, negatives_batch in generator:
            self.optimizer.zero_grad()

            positive_sim = self.model_(
                focus_batch.unsqueeze(-1), positives_batch.unsqueeze(-1))
            negative_sim = self.model_(
                focus_batch.unsqueeze(-1), negatives_batch)

            loss = self._compute_loss(positive_sim, negative_sim)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

        return losses

    def fit(self, X: InteractionMatrix, validation_data: Tuple[InteractionMatrix, InteractionMatrix]) -> "TorchMLAlgorithm":
        super().fit(X, validation_data)

        self._create_similarity_matrix()

        return self

    def _compute_loss(self, positive_sim, negative_sim):
        return skipgram_negative_sampling_loss(positive_sim, negative_sim)

    def _create_similarity_matrix(self):
        # K similar items + self-similarity
        K = self.K + 1
        batch_size = 1000

        embedding = self.model_.embeddings.weight.detach().numpy()
        num_items = embedding.shape[0]
        if K > num_items:
            K = num_items
            warnings.warn("K is larger than the number of items.", UserWarning)

        item_cosine_similarity_ = lil_matrix(
            (num_items, num_items))

        for batch in range(0, num_items, batch_size):
            Y = embedding[batch:batch + batch_size]

            item_cosine_similarity_batch = csr_matrix(cosine_similarity(
                Y, embedding))

            item_cosine_similarity_[
                batch:batch + batch_size] = get_top_K_values(item_cosine_similarity_batch, K)
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

    def _training_generator(self, X: InteractionMatrix, negative_samples: int, window_size: int, batch=1000):
        ''' Creates a training dataset using the skipgrams and negative sampling method.

        First, the sequences of items (iid) are grouped per user (uid).
        Next, a windowing operation is applied over each seperate item sequence.
        These windows are sliced and re-stacked in order to create skipgrams of two items (pos_training_set).
        Next, a unigram distribution of all the items in the dataset is created.
        This unigram distribution is used for creating the negative samples.

        :param X: InteractionMatrix
        :param negative_samples: number of negative samples per positive sample
        :param window_size: the window size
        :param batch: the batch size
        :yield: focus_batch, positive_samples_batch, negative_samples_batch
        :rtype: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
        '''
        # group and window
        windowed_sequences = window(X.sorted_item_history, window_size)
        context = np.hstack(
            (windowed_sequences[:, :window_size], windowed_sequences[:, window_size + 1:]))
        focus = windowed_sequences[:, window_size]
        # stack
        positives = np.empty((0, 2), dtype=int)
        for col in range(context.shape[1]):
            stacked = np.column_stack((focus, context[:, col]))
            positives = np.vstack([positives, stacked])
        # remove any NaN valued rows (consequence of windowing)
        positives = positives[~np.isnan(positives).any(axis=1)].astype(int)
        # create csr matrix from numpy array
        focus_items = positives[:, 0]
        pos_items = positives[:, 1]
        values = [1] * len(focus_items)
        positives_csr = csr_matrix(
            (values, (focus_items, pos_items)), shape=(X.shape[1], X.shape[1]))
        co_occurrence = to_binary(positives_csr + positives_csr.T)
        co_occurrence = lil_matrix(co_occurrence)
        co_occurrence.setdiag(1)
        co_occurrence = co_occurrence.tocsr()
        yield from sample_positives_and_negatives(X=co_occurrence, U=negative_samples, batch_size=batch, replace=self.replace,
                                                  exact=self.exact, positives=positives)

    def _transform_fit_input(self, X: Matrix, validation_data: Tuple[Matrix, Matrix]):
        return X, to_csr_matrix(validation_data, binary=True)


class SkipGram(nn.Module):
    def __init__(self, vocab_size, vector_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, vector_size)

    def forward(self, focus_item_batch, context_items_batch):
        # Create a (batch_size, embedding_dim, 1) tensor
        focus_vector = torch.movedim(self.embeddings(focus_item_batch), 1, 2)
        # Expected of size (batch_size, 1, embedding_dim)
        context_vectors = self.embeddings(context_items_batch)
        return torch.bmm(context_vectors, focus_vector).squeeze(-1)


def window(sequences, window_size):
    '''
    Will apply a windowing operation to a sequence of item sequences.

    Note: pads the sequences to make sure edge cases are also accounted for.

    :param sequences: iterable of iterables
    :param window_size: the size of the window
    :returns: windowed sequences
    :rtype: numpy.ndarray
    '''
    padded_sequences = [[np.NAN] * window_size +
                        list(s) + [np.NAN] * window_size for uid, s in sequences]
    w = [w.tolist() for sequence in padded_sequences if len(sequence) >= window_size for w in
         sliding_window_view(sequence, 2 * window_size + 1)]
    return np.array(w)
