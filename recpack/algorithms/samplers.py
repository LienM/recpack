from typing import Tuple, Iterator, Union
import numpy as np
from scipy.sparse import csr_matrix
import torch

from recpack.data.matrix import to_binary


def unigram_distribution(X: csr_matrix) -> np.array:
    """Creates a unigram distribution based on the item frequency.

    Follows the advice outlined in https://arxiv.org/abs/1310.4546 to create this noise distribution:
    the noise distribution is taken to be the unigram distribution to the power (3/4).
    Note: this is a heuristic based on the original Word2Vec paper.
    """
    item_counts_powered = np.power(X.sum(axis=0).A[0], 3 / 4)
    return item_counts_powered / item_counts_powered.sum()


class PositiveNegativeSampler:
    def __init__(
        self,
        U=1,
        batch_size=100,
        replace=True,
        exact=False,
        distribution="uniform",
    ):
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

        :param U: Number of negative samples for each positive, defaults to 1
        :type U: int, optional
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
        self.U = U
        self.batch_size = batch_size
        self.replace = replace
        self.exact = exact

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
        """Sample U negatives for each sampled user-item-pair (positive).

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
            raise RuntimeError(
                "Can't sample more samples than positive entries without replacement"
            )

        # Pick interactions at random, with replacement
        samples = np.random.choice(
            num_positives, size=(sample_size,), replace=self.replace
        )

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
                    size=(true_batch_size, self.U),
                    probabilities=negative_sample_probabilities,
                )
                while True:
                    # Approximately fix the negatives that are equal to the positives,
                    # if there are any, assumes collisions are rare
                    mask = np.apply_along_axis(
                        lambda col: col == positives_batch, 0, negatives_batch
                    )
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
                if (num_nonzeros < self.U).any():
                    raise ValueError(
                        "Cannot request more negative samples than are possible."
                    )

                negatives_batch = np.zeros((true_batch_size, self.U))
                for i in range(0, self.U):
                    # Construct column i in the negatives matrix

                    # 1st try true random
                    # We will fix collisions in while loop
                    negatives_batch_col_i = self._sample_negatives(
                        X,
                        size=(true_batch_size,),
                        probabilities=negative_sample_probabilities,
                    )

                    # np.random.randint(
                    #     0, X.shape[1], size=(true_batch_size,)
                    # )
                    while True:

                        num_incorrect, negatives_mask = _spot_collisions(
                            users, negatives_batch_col_i, X
                        )

                        # Check column against previous columns
                        additional_mask = np.zeros(true_batch_size, dtype=bool)
                        for j in range(0, i):
                            additional_mask += (
                                negatives_batch_col_i == negatives_batch[:, j]
                            )

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

            yield torch.LongTensor(users), torch.LongTensor(
                positives_batch
            ), torch.LongTensor(negatives_batch)


class BootstrapSampler(PositiveNegativeSampler):
    """Sampler that samples positives with replacement.

    This approach allows to learn multiple times from the same positive interactions.
    For more information on implementation see :class:`PositiveNegativeSampler`

    :param U: Number of negative samples for each positive, defaults to 1
    :type U: int, optional
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :param exact: If False (default) negatives are checked agains the corresponding
        positive sample only, allowing for (rare) collisions.
        If collisions should be avoided at all costs, use exact = True,
        but suffer decreased performance.
    :type exact: bool, optional
    """

    def __init__(self, U=1, batch_size=100, exact=False):
        #  Bootstrap sampling is samping with replacement.
        super().__init__(U=U, batch_size=batch_size, replace=True, exact=exact)


class WarpSampler(PositiveNegativeSampler):
    """Samples `U` negatives for each positive.

    This approach allows to learn multiple times from the same positive interactions.
    For more information on implementation see :class:`PositiveNegativeSampler`

    :param U: Number of negative samples for each positive, defaults to 1
    :type U: int, optional
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :param exact: If False (default) negatives are checked agains the corresponding
        positive sample only, allowing for (rare) collisions.
        If collisions should be avoided at all costs, use exact = True,
        but suffer decreased performance.
    :type exact: bool, optional
    """

    def __init__(self, U=10, batch_size=100, exact=False):
        super().__init__(U=U, batch_size=batch_size, replace=False, exact=exact)


def _spot_collisions(
    users: np.array, negatives_batch: np.array, X: csr_matrix
) -> Tuple[int, np.array]:
    """Spot collisions between the negative samples and the interactions in X.

    :param users: Ordered batch of users
    :type users: np.array
    :param negatives_batch: Ordered batch of negative items
    :type negatives_batch: np.array
    :param X: Entirety of all user interactions
    :type X: csr_matrix
    :return: Tuple containing the number of incorrect negative samples,
        and the locations of these incorrect samples in the batch array
    :rtype: Tuple[int, np.array]
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
