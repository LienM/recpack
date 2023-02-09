# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from typing import Tuple, Iterator, Union
import numpy as np
from scipy.sparse import csr_matrix
import torch

from recpack.matrix import InteractionMatrix, to_binary
from recpack.algorithms.util import get_batches


def unigram_distribution(X: csr_matrix) -> np.ndarray:
    """Creates a unigram distribution based on the item frequency.

    Follows the advice outlined in https://arxiv.org/abs/1310.4546 to create this noise distribution:
    the noise distribution is taken to be the unigram distribution to the power (3/4).
    Note: this is a heuristic based on the original Word2Vec paper.
    """
    item_counts_powered = np.power(X.sum(axis=0).A[0], 3 / 4)
    return item_counts_powered / item_counts_powered.sum()


class Sampler:
    pass


class PositiveNegativeSampler(Sampler):
    """Samples linked positive and negative interactions for users.

    Provides a :meth:`sample` method that samples positives and negatives.
    Positives are sampled uniformly from all positive interactions.
    Negative samples are sampled either based on a uniform distribution
    or a unigram distribution.

    The uniform distrbution makes it so each item has the same probability to
    be selected as negative.
    With the unigram distribution, items are sampled according to their weighted
    popularity.

    .. math::

        P(w_i) = \\frac{  {f(w_i)}^{3/4}  }{\\sum_{j=0}^{n}\\left(  {f(w_j)}^{3/4} \\right) }

    :param num_negatives: Number of negative samples for each positive, defaults to 1
    :type num_negatives: int, optional
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :param replace: Sample positives with or without replacement. Defaults to True
    :type replace: bool, optional
    :param exact: If False (default) negatives are checked agains the corresponding
        positive sample only, allowing for (rare) collisions.
        If collisions should be avoided at all costs, use exact = True,
        but suffer decreased performance.
    :type exact: bool, optional
    :param distribution: The distribution used to sample negative items,
        defaults to uniform. Options are `'uniform'` and `'unigram'`
    :type distribution: string, optional
    """

    ALLOWED_DISTRIBUTIONS = ["uniform", "unigram"]

    def __init__(
        self,
        num_negatives=1,
        batch_size=100,
        replace=True,
        exact=False,
        distribution="uniform",
    ):

        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.replace = replace
        self.exact = exact

        if distribution not in self.ALLOWED_DISTRIBUTIONS:
            raise ValueError(f"unknown distribution, use one of {self.ALLOWED_DISTRIBUTIONS}")

        self.distribution = distribution  # TODO: Enum style value, to avoid mismatches?

    def _get_distribution(self, X: csr_matrix) -> Union[None, np.array]:
        if self.distribution == "uniform":
            # passing None as probabilities is the default for np.random.choice
            return None
        elif self.distribution == "unigram":
            return unigram_distribution(X)

        raise ValueError("The requested distribution is unknown")

    def _sample_negatives(self, X: csr_matrix, size, probabilities):
        candidates = np.arange(X.shape[1])
        return np.random.choice(candidates, size=size, p=probabilities)

    def sample(
        self, X: csr_matrix, sample_size=None, positives=None
    ) -> Iterator[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]:
        """Sample num_negatives negatives for each sampled user-item-pair (positive).

        When sampling without replacement,
        ``sample_size`` cannot exceed the number of positives in X.

        :param X: Matrix with interactions to sample from.
        :type X: csr_matrix
        :param sample_size: The number of samples to create,
            if None, the number of positives entries in X will be used.
            Defaults to None.
        :type sample_size: int, optional
        :param positives: Restrict positives samples to only samples
            in this np.array of dimension (num_samples, 2).
        :type positives: np.array, optional
        :raises ValueError: [description]
        :yield: Iterator of (user_batch, positive_samples_batch, negative_samples_batch)
        :rtype: Iterator[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]
        """
        # Need positive and negative pair.
        # Requires the existence of a positive for this item.
        # As a (num_interactions, 2) numpy array

        if positives is None:
            positives = np.array(X.nonzero()).T

        num_positives = positives.shape[0]

        if sample_size is None:
            sample_size = num_positives

        X = to_binary(X)

        # Make sure we can actually sample the requested sample_size
        # without replacement samplesize should <= number of positives to choose from.
        if not self.replace and sample_size > num_positives:
            raise RuntimeError("Can't sample more samples than positive entries without replacement")

        # Pick interactions at random, with replacement
        samples = np.random.choice(num_positives, size=(sample_size,), replace=self.replace)

        negative_sample_probabilities = self._get_distribution(X)

        for start in range(0, sample_size, self.batch_size):
            sample_batch = samples[start : start + self.batch_size]

            batch = positives[sample_batch]
            users = batch[:, 0]
            positives_batch = batch[:, 1]

            # Important only for final batch, if smaller than batch_size
            true_batch_size = min(self.batch_size, sample_size - start)

            if not self.exact:
                negatives_batch = self._sample_negatives(
                    X,
                    size=(true_batch_size, self.num_negatives),
                    probabilities=negative_sample_probabilities,
                )
                while True:
                    # Approximately fix the negatives that are equal to the positives,
                    # if there are any, assumes collisions are rare
                    mask = np.apply_along_axis(lambda col: col == positives_batch, 0, negatives_batch)
                    num_incorrect = np.sum(mask)

                    if num_incorrect > 0:
                        new_negatives = self._sample_negatives(
                            X,
                            size=(num_incorrect,),
                            probabilities=negative_sample_probabilities,
                        )

                        negatives_batch[mask] = new_negatives
                    else:
                        # Exit the while loop
                        break
            else:
                num_nonzeros = X.shape[1] - X.sum(axis=1)
                if (num_nonzeros < self.num_negatives).any():
                    raise ValueError("Cannot request more negative samples than are possible.")

                negatives_batch = np.zeros((true_batch_size, self.num_negatives))
                for i in range(0, self.num_negatives):
                    # Construct column i in the negatives matrix

                    # 1st try true random
                    # We will fix collisions in while loop
                    negatives_batch_col_i = self._sample_negatives(
                        X,
                        size=(true_batch_size,),
                        probabilities=negative_sample_probabilities,
                    )

                    while True:

                        num_incorrect, negatives_mask = _spot_collisions(users, negatives_batch_col_i, X)

                        # Check column against previous columns
                        additional_mask = np.zeros(true_batch_size, dtype=bool)
                        for j in range(0, i):
                            additional_mask += negatives_batch_col_i == negatives_batch[:, j]

                        total_mask = negatives_mask + additional_mask
                        num_incorrect = total_mask.sum()

                        if num_incorrect > 0:
                            new_negatives = self._sample_negatives(
                                X,
                                size=(num_incorrect,),
                                probabilities=negative_sample_probabilities,
                            )
                            negatives_batch_col_i[total_mask] = new_negatives

                        else:
                            # Exit the while(True) loop
                            break

                    negatives_batch[:, i] = negatives_batch_col_i

            yield torch.LongTensor(users), torch.LongTensor(positives_batch), torch.LongTensor(negatives_batch)


class BootstrapSampler(PositiveNegativeSampler):
    """Sampler that samples positives with replacement.

    This approach allows to learn multiple times from the same positive interactions.
    For more information on implementation see :class:`PositiveNegativeSampler`

    :param num_negatives: Number of negative samples for each positive, defaults to 1
    :type num_negatives: int, optional
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :param exact: If False (default) negatives are checked agains the corresponding
        positive sample only, allowing for (rare) collisions.
        If collisions should be avoided at all costs, use exact = True,
        but suffer decreased performance.
    :type exact: bool, optional
    """

    def __init__(self, num_negatives=1, batch_size=100, exact=False):
        #  Bootstrap sampling is samping with replacement.
        super().__init__(num_negatives=num_negatives, batch_size=batch_size, replace=True, exact=exact)


class WarpSampler(PositiveNegativeSampler):
    """Samples `num_negatives` negatives for each positive.

    This approach allows to learn multiple times from the same positive interactions.
    For more information on implementation see :class:`PositiveNegativeSampler`

    :param num_negatives: Number of negative samples for each positive, defaults to 1
    :type num_negatives: int, optional
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :param exact: If False (default) negatives are checked agains the corresponding
        positive sample only, allowing for (rare) collisions.
        If collisions should be avoided at all costs, use exact = True,
        but suffer decreased performance.
    :type exact: bool, optional
    """

    def __init__(self, num_negatives=10, batch_size=100, exact=False) -> None:
        super().__init__(num_negatives=num_negatives, batch_size=batch_size, replace=False, exact=exact)


class SequenceMiniBatchSampler(Sampler):
    """Samples batches of user, input sequences.

    Handles sequences of unequal length by padding them with `pad_token`.

    :param pad_token: Token used to indicate that this location in the sequence
        contains a padding element.
    :type pad_token: int
    :param batch_size: The number of sequences returned per batch, defaults to 100
    :type batch_size: int, optional
    """

    def __init__(self, pad_token: int, batch_size: int = 100) -> None:
        super().__init__()
        self.pad_token = pad_token
        self.batch_size = batch_size

    def sample(self, X: InteractionMatrix) -> Iterator[Tuple[torch.LongTensor, torch.LongTensor]]:
        # item_histories = list(X.sorted_item_history)
        # Do I introduce bias if I sort them by length?
        # item_histories.sort(key=lambda x: len(x[1]), reverse=True)

        # Generate batches of users. Take maximum len of history in batch
        for batch in get_batches(X.sorted_item_history, self.batch_size):
            # Because they were sorted in reverse order the first element contains the max len in this batch,
            # the last the min len.
            batch.sort(key=lambda x: len(x[1]), reverse=True)
            max_hist_len = len(batch[0][1])
            batch_size = len(batch)

            uid_batch = np.zeros((batch_size,), dtype=int)

            # Initialize seq_batch with self.pad_token
            positives_batch = np.ones((batch_size, max_hist_len), dtype=int) * self.pad_token

            # Add sequences in batch
            for batch_ix, (uid, hist) in enumerate(batch):
                hist_len = hist.shape[0]
                positives_batch[batch_ix, :hist_len] = hist
                uid_batch[batch_ix] = uid

            yield (torch.LongTensor(uid_batch), torch.LongTensor(positives_batch))


class SequenceMiniBatchPositivesTargetsNegativesSampler(SequenceMiniBatchSampler):
    """Samples `num_negatives` negatives for every positive in a sequence.

    This approach allows to learn multiple times from the same positive interactions.
    Because the sequence-aspect is important here, we only eliminate collisions
    in the exact same location in the sequence.
    As a result, a sample that occurs at a later or earlier time in the sequence
    may be sampled as a negative for all other locations in the sequence.

    Handles sequences of unequal length by padding them with `pad_token`.

    :param num_negatives: Number of negative samples for each positive
    :type num_negatives: int
    :param pad_token: Token used to indicate that this location in the sequence
        contains a padding element.
    :type pad_token: int
    :param batch_size: The number of sequences returned per batch, defaults to 100
    :type batch_size: int, optional
    """

    def __init__(self, num_negatives: int, pad_token: int, batch_size: int = 100) -> None:
        super().__init__(pad_token, batch_size)
        self.num_negatives = num_negatives

    def sample(
        self, X: InteractionMatrix
    ) -> Iterator[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]]:
        """Sample positives, targets and negatives from the input matrix.

        Yields tuples of:

        - uids: 1D tensor with the user ids in this batch.
          Shape = (batch_size,)
        - positives: 2D tensor with row per user, and history item_ids in order on each row.
          Rows are sorted, such that longest histories are higher in the tensor.
          Histories shorter than the width of the tensor are filled up with padding tokens.
          Shape = (batch_size, max_hist_len(batch))
        - targets: 2D tensor with targets to predict for each user.
          This is the positives, but rolled 1 position to the left.
          Such that the target of the first positive is the second positive in the sequence.
          Each sequence ends with a padding token as target,
          since there is no knowledge of the next item at the end of the sequence.
          Shape = (batch size, max_hist_len(batch))
        - negatives: 3D tensor, with negative examples for each positive.
          For each positive self.num_negatives negatives are sampled,
          these negatives are checked against only the target item.
          Shape = (batch_size, max_hist_len(batch), self.num_negatives)


        :param X: Interaction matrix to generate samples from.
        :type X: InteractionMatrix
        :yield: tuples of (uids, positives, targets, negatives)
        :rtype: Iterator[ Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor] ]
        """

        num_items = X.shape[1]

        # Generate batches of users. Take maximum len of history in batch
        for uid_batch, positives_batch in super().sample(X):
            negatives_batch = np.random.randint(0, num_items, (*positives_batch.shape, self.num_negatives))

            targets_batch = np.roll(positives_batch, -1, axis=1)
            # set last item to padding, otherwise 1st item is rolled till here
            targets_batch[:, -1] = self.pad_token

            while True:
                mask = np.equal(negatives_batch, targets_batch[:, :, None])

                num_incorrect = np.sum(mask)

                if num_incorrect:
                    new_negatives = np.random.randint(0, num_items, size=(num_incorrect,))

                    negatives_batch[mask] = new_negatives
                else:
                    break

            yield (
                uid_batch,
                positives_batch,
                torch.LongTensor(targets_batch),
                torch.LongTensor(negatives_batch),
            )


def _spot_collisions(users: np.ndarray, negatives_batch: np.ndarray, X: csr_matrix) -> Tuple[int, np.ndarray]:
    """Spot collisions between the negative samples and the interactions in X.

    :param users: Ordered batch of users
    :type users: np.ndarray
    :param negatives_batch: Ordered batch of negative items
    :type negatives_batch: np.ndarray
    :param X: Entirety of all user interactions
    :type X: csr_matrix
    :return: Tuple containing the number of incorrect negative samples,
        and the locations of these incorrect samples in the batch array
    :rtype: Tuple[int, np.ndarray]
    """
    # Eliminate the collisions, exactly.
    # Turn this batch of negatives into a csr_matrix
    negatives_batch_csr = csr_matrix(
        (
            np.ones(negatives_batch.shape[0]),
            (users, negatives_batch),
        ),
        X.shape,
    )
    # Detect differences between the batch of negatives and X.
    # Ideally, every negative sample should be different from samples in X
    # (no collisions).
    negative_samples_mask = negatives_batch_csr.astype(bool)
    match_or_mismatch = to_binary(negatives_batch_csr) != X
    # If there are no false negatives, all values in false_negatives should be True.
    false_negatives = np.bitwise_not(match_or_mismatch[negative_samples_mask])

    # Initialize mask to all zeros = all False
    negatives_mask = np.zeros(negatives_batch.shape).astype(bool)
    # Get the indices of the false_negatives
    _, false_negative_indices_csr = false_negatives.nonzero()
    # Get the corresponding false negative pairs
    # Assumes the order of samples in false_negatives
    # is the same as in negative_samples_mask
    false_negative_pairs = list(zip(*negative_samples_mask.nonzero()))
    # Get the originally sampled negative pairs, in the batch order
    negative_pairs = np.vstack([users, negatives_batch]).T
    for i in false_negative_indices_csr:
        # Find the corresponding row in negative_pairs
        a = np.all(negative_pairs == false_negative_pairs[i], axis=1)
        negative_mask_row_indices = a.nonzero()
        # Set these rows (most of the time should be one row) to True
        negatives_mask[negative_mask_row_indices] = True

    num_incorrect = negatives_mask.sum()
    return num_incorrect, negatives_mask
