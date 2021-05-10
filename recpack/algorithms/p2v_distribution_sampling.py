import logging
from typing import Tuple

import numpy as np
from scipy.sparse import lil_matrix
import torch

from recpack.algorithms.p2v import Prod2Vec, window
from recpack.data.matrix import InteractionMatrix
from recpack.algorithms.samplers import _sample_positives_and_negatives_from_distribution

logger = logging.getLogger("recpack")


class Prod2VecDistributionSampling(Prod2Vec):
    '''
    Implements the sampling method outlined in: Distributed Representations of Words and Phrases and their Compositionality (https://arxiv.org/abs/1310.4546).

    A unigram noise distribution (to the power 3/4) is used for sampling.
    '''
    def __init__(self, embedding_size: int, num_neg_samples: int,
                 window_size: int, stopping_criterion: str, K=10, batch_size=1000, learning_rate=0.01,
                 max_epochs=10,
                 stop_early: bool = False, max_iter_no_change: int = 5, min_improvement: float = 0.01, seed=None,
                 save_best_to_file=False, replace=False, exact=False, keep_last=False):
        super().__init__(embedding_size, num_neg_samples,
                         window_size, stopping_criterion, K=K, batch_size=batch_size, learning_rate=learning_rate,
                         max_epochs=max_epochs,
                         stop_early=stop_early, max_iter_no_change=max_iter_no_change, min_improvement=min_improvement,
                         seed=seed,
                         save_best_to_file=save_best_to_file, replace=replace, exact=exact, keep_last=keep_last)

    def _skipgram_sample_pairs(self, X: InteractionMatrix):
        """ Creates a training dataset using the skipgrams and negative sampling method.

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
            (windowed_sequences[:, :self.window_size], windowed_sequences[:, self.window_size + 1:]))
        focus = windowed_sequences[:, self.window_size]
        # Apply np.repeat to focus to turn [0,1,2,3] into [0, 0, 1, 1, ...]
        # where number of repetitions is self.window_size
        # Squeeze final dimension from context so they line up neatly.
        positives = np.column_stack(
            [focus.repeat(self.window_size * 2), context.reshape(-1)])
        # Remove any NaN valued rows (consequence of windowing)
        positives = positives[~np.isnan(positives).any(axis=1)].astype(int)
        coocc = lil_matrix((X.shape[1], X.shape[1]), dtype=int)
        coocc[positives[:, 0], positives[:, 1]] = 1
        coocc.setdiag(1)
        coocc = coocc.tocsr()

        distribution = self._unigram_distribution(X)
        yield from _sample_positives_and_negatives_from_distribution(X=coocc, distribution=distribution,
                                                                     U=self.num_neg_samples,
                                                                     batch_size=self.batch_size, replace=self.replace,
                                                                     exact=self.exact, positives=positives)

    def _unigram_distribution(self, X: InteractionMatrix):
        ''' Creates a unigram distribution based on the item frequency.

        Follows the advice outlined in https://arxiv.org/abs/1310.4546 to create this noise distribution:
        the noise distribution is taken to be the unigram distribution to the power (3/4).
        Note: this is a heuristic based on the original Word2Vec paper.
        Returns a dict with key: iid and value: probability(iid).

        :return: unigram distribution as a dict of items (iids)
        '''
        values = X.values
        # csr_matrix returns np.matrix, use .A1 to get np.ndarray
        item_freq = values.sum(axis=0).A1
        item_freq = item_freq ** (3 / 4)
        total_items = item_freq.sum()
        unigram_values = item_freq * (1 / total_items)
        unigram = {i: val for (i, val) in enumerate(unigram_values)}
        return unigram