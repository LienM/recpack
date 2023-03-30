# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from typing import Tuple, Optional
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from recpack.matrix import InteractionMatrix
from recpack.util import get_top_K_values
from recpack.algorithms.p2v import Prod2Vec


logger = logging.getLogger("recpack")


class Prod2VecClustered(Prod2Vec):
    """
    Clustered Prod2Vec implementation outlined in:
    E-commerce in Your Inbox: Product Recommendations at Scale
    (https://arxiv.org/abs/1606.07154)

    Products with similar embeddings are grouped into clusters
    using Kmeans clustering.
    Product recommendations are made only from the top-Kcl related clusters.
    A cluster is considered related if users often consume an item
    from this cluster after the item.
    Clusters are ranked based on the probability
    that an interaction with an item from cluster ci is followed by an
    interaction with an item from cluster cj.
    Products from these top clusters are sorted by their cosine similarity.

    Where possible, defaults were taken from the paper.

    :param num_components: The size of the embedding vectors for both input and output embeddings, defaults to 300
    :type num_components: int, optional
    :param num_negatives: Number of negative samples for every positive sample, defaults to 10
    :type num_negatives: int, optional
    :param window_size: Size of the context window to the left and to the right of the target item
         used in skipgram negative sampling, defaults to 2
    :type window_size: int, optional
    :param stopping_criterion: Used to identify the best model computed thus far.
        The string indicates the name of the stopping criterion.
        Which criterions are available can be found at StoppingCriterion.FUNCTIONS
        Defaults to 'precision'
    :type stopping_criterion: str, optional
    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param num_clusters: Number of clusters for Kmeans clustering, defaults to 5
    :type num_clusters: int, optional
    :param Kcl: Maximum number of top-K clusters recommendations can be made from, defaults to 2
    :type Kcl: int, optional
    :param batch_size: Batch size for Adam optimizer. Higher batch sizes make each epoch more efficient,
        but increases the amount of epochs needed to converge to the optimum,
        by reducing the amount of updates per epoch. Defaults to 1000
    :type batch_size: int, optional
    :param learning_rate: Learning rate, defaults to 0.01
    :type learning_rate: float, optional
    :param clipnorm: Clips gradient norm.
        The norm is computed over all gradients together,
        as if they were concatenated into a single vector, defaults to 1.0
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
        num_clusters: int = 5,
        Kcl: int = 2,
        batch_size: int = 1000,
        learning_rate: float = 0.01,
        clipnorm: float = 1.0,
        max_epochs: int = 10,
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.01,
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
            num_components,
            num_negatives,
            window_size,
            stopping_criterion,
            K=K,
            batch_size=batch_size,
            learning_rate=learning_rate,
            clipnorm=clipnorm,
            max_epochs=max_epochs,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            replace=replace,
            exact=exact,
            keep_last=keep_last,
            distribution=distribution,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )
        self.num_clusters = num_clusters
        self.Kcl = Kcl

    def _create_similarity_matrix(self, X: InteractionMatrix):
        # K similar items + self-similarity
        # While self similarity is not guaranteed in this case,
        # this is not enough of a problem to add more complex solutions.
        K = self.K + 1

        embedding = self.model_.input_embeddings.weight.cpu().detach().numpy()
        num_items = embedding.shape[0]
        if K > num_items:
            K = num_items
            warnings.warn("K is larger than the number of items.", UserWarning)

        # Set embedding of inactive items to 0
        active_items = X.active_items
        inactive_items = list(set(range(num_items)).difference(active_items))
        embedding[inactive_items] = 0

        # empty easily updated sparse matrix
        # Will be filled in per row
        item_cosine_similarity_ = lil_matrix((num_items, num_items))

        # Cluster the items in the embedding space:
        cluster_assignments = self._cluster(embedding)

        # Compute cluster to cluster similarities
        cluster_to_cluster_neighbours = self._get_top_K_clusters(X, cluster_assignments)

        cluster_neighbour_cnt = cluster_to_cluster_neighbours.sum(axis=1)

        if (cluster_neighbour_cnt == 0).any():
            warnings.warn("There are clusters without neighbours", UserWarning)

        # Compute similarities per cluster
        for cluster in np.arange(self.num_clusters):
            # Get clusters that occur after `cluster` often.
            cluster_neighbours = cluster_to_cluster_neighbours[cluster, :].nonzero()[1]

            if not cluster_neighbours.any():
                continue

            cluster_items = (cluster_assignments == cluster).nonzero()[0]

            context = embedding[cluster_items, :]

            adjacent_cluster_items = np.asarray(np.isin(cluster_assignments, cluster_neighbours)).nonzero()[0]

            target = embedding[adjacent_cluster_items, :]

            local_sims = lil_matrix((cluster_items.shape[0], num_items))

            local_sims[:, adjacent_cluster_items] = cosine_similarity(context, target)

            item_cosine_similarity_[cluster_items] = get_top_K_values(local_sims.tocsr(), K)

        # no self similarity, set diagonal to zero
        item_cosine_similarity_.setdiag(0)
        # Remove all similarities for inactive items
        item_cosine_similarity_[inactive_items] = 0
        item_cosine_similarity_[:, inactive_items] = 0

        self.similarity_matrix_ = csr_matrix(item_cosine_similarity_)

    def _cluster(self, embedding: np.ndarray) -> np.ndarray:
        """Use Kmeans to assign a cluster label to each item.

        :return: array with a cluster label for every item. Shape = (|I|,)
        :rtype: np.array
        """
        kmeans = KMeans(self.num_clusters)
        cluster_assignments = kmeans.fit_predict(embedding)
        return cluster_assignments

    def _get_top_K_clusters(self, X: InteractionMatrix, item_to_cluster: np.ndarray) -> csr_matrix:
        """Compute the clusters that should be considered neighbours.

        Similarity between two clusters i, j is computed by the number of times
        an item in cluster i is interacted with before an interaction
        with an item from cluster j.

        :param X: Interactions to use for similarity computation.
        :type X: InteractionMatrix
        :param item_to_cluster: Item to cluster assignment vector
        :type item_to_cluster: np.array
        :return: Sparse matrix with top Kcl neighbours for each cluster.
            Shape = (|C| x |C|)
        :rtype: csr_matrix
        """
        # do a singular window operation to get an item and the next interacted item.
        context_items, focus_items = self._create_pairs(X)
        # replace all iids by cluster labels
        from_clusters = [item_to_cluster[i] for i in context_items]
        to_clusters = [item_to_cluster[i] for i in focus_items]

        # create a cluster to cluster matrix
        # cheap trick: csr matrix automatically adds up duplicate entries
        values = np.ones(len(from_clusters))
        cluster_to_cluster_csr = csr_matrix(
            (values, (from_clusters, to_clusters)),
            shape=(self.num_clusters, self.num_clusters),
        )

        # Cut to topK most similar neighborhoods.
        cluster_neighbourhood = get_top_K_values(cluster_to_cluster_csr, self.Kcl)

        return cluster_neighbourhood

    def _create_pairs(self, X: InteractionMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create pairs of positive samples.
        """
        windowed_sequences = np.array(
            [
                w.tolist()
                for _, sequence in X.sorted_item_history
                if len(sequence) >= 2
                for w in sliding_window_view(sequence, 2)
            ]
        )
        context = windowed_sequences[:, 0]
        focus = windowed_sequences[:, 1]

        return context, focus
