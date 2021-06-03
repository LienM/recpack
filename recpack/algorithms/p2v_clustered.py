import logging
from typing import Tuple
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from recpack.data.matrix import InteractionMatrix
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
    that a purchase from cluster ci is followed by a purchase from cluster cj.
    Products from these top clusters are sorted by their cosine similarity.

    :param num_clusters: number of clusters for Kmeans clustering
    :type num_clusters: int
    :param Kcl: top-K clusters
    :rtype Kcl: int
    """

    def __init__(
        self,
        embedding_size: int,
        num_neg_samples: int,
        window_size: int,
        stopping_criterion: str,
        K=200,
        num_clusters=5,
        Kcl=2,
        batch_size=1000,
        learning_rate=0.01,
        max_epochs=10,
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.01,
        seed=None,
        save_best_to_file=False,
        replace=False,
        exact=False,
        keep_last=False,
    ):
        super().__init__(
            embedding_size,
            num_neg_samples,
            window_size,
            stopping_criterion,
            K=K,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            replace=replace,
            exact=exact,
            keep_last=keep_last,
        )
        self.num_clusters = num_clusters
        self.Kcl = Kcl

    def _create_similarity_matrix(self, X: InteractionMatrix):
        # K similar items + self-similarity
        # While self similarity is not guaranteed in this case,
        # this is not enough of a problem to add more complex solutions.
        K = self.K + 1

        embedding = self.model_.input_embeddings.weight.detach().numpy()
        num_items = embedding.shape[0]
        if K > num_items:
            K = num_items
            warnings.warn("K is larger than the number of items.", UserWarning)

        # empty easily updated sparse matrix
        # Will be filled in per row
        item_cosine_similarity_ = lil_matrix((num_items, num_items))

        # Cluster the items in the embedding space:
        cluster_assignments = self._cluster()

        # Compute cluster to cluster similarities
        cluster_to_cluster_neighbours = self._get_top_K_clusters(
            X, cluster_assignments
        )

        cluster_neighbour_cnt = cluster_to_cluster_neighbours.sum(axis=1)

        if (cluster_neighbour_cnt == 0).any():
            warnings.warn("There are clusters without neighbours", UserWarning)

        # Compute similarities per cluster
        for cluster in np.arange(self.num_clusters):
            # Get clusters that occur after `cluster` often.
            cluster_neighbours = cluster_to_cluster_neighbours[cluster, :].nonzero()[
                1]

            if not cluster_neighbours.any():
                continue

            cluster_items = (cluster_assignments == cluster).nonzero()[0]

            context = embedding[cluster_items, :]

            adjacent_cluster_items = np.asarray(np.isin(
                cluster_assignments, cluster_neighbours)).nonzero()[0]

            target = embedding[adjacent_cluster_items, :]

            local_sims = lil_matrix((cluster_items.shape[0], num_items))

            local_sims[:, adjacent_cluster_items] = cosine_similarity(
                context, target)

            item_cosine_similarity_[cluster_items] = get_top_K_values(
                local_sims.tocsr(), K)

        # no self similarity, set diagonal to zero
        item_cosine_similarity_.setdiag(0)
        self.similarity_matrix_ = csr_matrix(item_cosine_similarity_)

    def _cluster(self) -> np.ndarray:
        """Use Kmeans to assign a cluster label to each item.

        :return: array with a cluster label for every item. Shape = (|I|,)
        :rtype: np.array
        """
        embedding = self.model_.input_embeddings.weight.detach().numpy()
        kmeans = KMeans(self.num_clusters)
        cluster_assignments = kmeans.fit_predict(embedding)
        return cluster_assignments

    def _get_top_K_clusters(
        self, X: InteractionMatrix, item_to_cluster: np.ndarray
    ) -> csr_matrix:
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
        cluster_neighbourhood = get_top_K_values(
            cluster_to_cluster_csr, self.Kcl)

        return cluster_neighbourhood

    def _create_pairs(self, X: InteractionMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create pairs of positive samples.
        """
        windowed_sequences = np.array([w.tolist(
        ) for _, sequence in X.sorted_item_history if len(sequence) >= 2 for w in sliding_window_view(sequence, 2)])
        context = windowed_sequences[:, 0]
        focus = windowed_sequences[:, 1]

        return context, focus
