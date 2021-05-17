import logging
import sys
from typing import Tuple
import warnings

import numpy as np
import torch.nn as nn
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from recpack.data.matrix import InteractionMatrix
from recpack.util import get_top_K_values
from recpack.algorithms.p2v import Prod2Vec
from recpack.algorithms.p2v import window
from recpack.algorithms.util import sample_rows

logger = logging.getLogger("recpack")


class Prod2VecClustered(Prod2Vec):
    '''
    Clustered Prod2Vec implementation outlined in: E-commerce in Your Inbox: Product Recommendations at Scale (https://arxiv.org/abs/1606.07154)

    Similar products are grouped into clusters, products are recommended based on the top related clusters.
    Uses a Kmeans clustering algorithm. Clusters are ranked based on the probability that a purchase from cluster ci is followed
    by a purchase from cluster cj. Products from the top clusters are sorted by their cosine similarity, top-K are returned as recommendations.


    :param num_clusters: number of clusters for Kmeans clustering
    :type num_clusters: int
    :param Kcl: top-K clusters
    :rtype Kcl: int
    '''
    def __init__(self, embedding_size: int, num_neg_samples: int,
                 window_size: int, stopping_criterion: str, K=10, num_clusters=5, Kcl=2, batch_size=1000, learning_rate=0.01,
                 max_epochs=10,
                 stop_early: bool = False, max_iter_no_change: int = 5, min_improvement: float = 0.01, seed=None,
                 save_best_to_file=False, replace=False, exact=False, keep_last=False):
        super().__init__(embedding_size, num_neg_samples,
                         window_size, stopping_criterion, K=K, batch_size=batch_size, learning_rate=learning_rate,
                         max_epochs=max_epochs,
                         stop_early=stop_early, max_iter_no_change=max_iter_no_change, min_improvement=min_improvement,
                         seed=seed,
                         save_best_to_file=save_best_to_file, replace=replace, exact=exact, keep_last=keep_last)
        self.num_clusters = num_clusters
        self.Kcl = Kcl

    def _train_epoch(self, X: InteractionMatrix) -> list:
        assert self.model_ is not None
        losses = []
        # generator will just be restarted for each epoch.
        for focus_batch, positives_batch, negatives_batch in self._skipgram_sample_pairs(X):
            self.optimizer.zero_grad()

            positive_sim = self.model_(
                focus_batch.unsqueeze(-1), positives_batch.unsqueeze(-1))
            negative_sim = self.model_(
                focus_batch.unsqueeze(-1), negatives_batch)

            loss = self._compute_loss(positive_sim, negative_sim)
            loss.backward()
            losses.append(loss.item())
            nn.utils.clip_grad_norm_(self.model_.parameters(), self.clipnorm)
            self.optimizer.step()

        # Note: had to be moved here in order to work. Reason is embedding + X are needed to create a similarity matrix -> _evaluate doesn't have access to X so can't create it there
        self._create_similarity_matrix(X)
        return losses

    def fit(self, X: InteractionMatrix,
            validation_data: Tuple[InteractionMatrix, InteractionMatrix]) -> "TorchMLAlgorithm":
        super(Prod2Vec, self).fit(X, validation_data)
        return self

    def _evaluate(self, val_in: csr_matrix, val_out: csr_matrix) -> None:
        assert self.similarity_matrix_ is not None
        val_in_selection, val_out_selection = sample_rows(val_in, val_out, sample_size=1000)
        predictions = self._batch_predict(val_in_selection)
        better = self.stopping_criterion.update(val_out_selection, predictions)
        if better:
            logger.info("Model improved. Storing better model.")
            self._save_best()

    def _create_similarity_matrix(self, X: InteractionMatrix):
        # K similar items + self-similarity
        K = self.K + 1
        batch_size = 1000

        embedding = self.model_.input_embeddings.weight.detach().numpy()
        num_items = embedding.shape[0]
        if K > num_items:
            K = num_items
            warnings.warn("K is larger than the number of items.", UserWarning)
        # create a ranking of top clusters
        self._create_clustered_ranking(X)
        # create a mask for the similarity matrix
        mask = self._create_cluster_mask(num_items)
        item_cosine_similarity_ = lil_matrix((num_items, num_items))

        for batch in range(0, num_items, batch_size):
            Y = embedding[batch:batch + batch_size]
            item_cosine_similarity_batch = csr_matrix(cosine_similarity(
                Y, embedding))
            item_cosine_similarity_batch = item_cosine_similarity_batch.multiply(mask[batch:batch + batch_size]).tocsr()
            item_cosine_similarity_batch.eliminate_zeros()
            item_cosine_similarity_[batch:batch + batch_size] = get_top_K_values(item_cosine_similarity_batch, K)
        # no self similarity, set diagonal to zero
        item_cosine_similarity_.setdiag(0)
        self.similarity_matrix_ = csr_matrix(item_cosine_similarity_)

    def _create_cluster_mask(self, num_items):
        '''
        Creates a mask for the similarity matrix based on cluster ranking.
        '''
        items = range(num_items)
        items_as_clusters = self._map_to_cluster(items, self.item_to_cluster_mapping)
        mask = csr_matrix((num_items, num_items))
        for item in items:
            item_cluster = self.item_to_cluster_mapping[item]
            mask_item = np.zeros(items_as_clusters.shape)
            for cluster in self.cluster_ranking[item_cluster]:
                try:
                    indexes = np.where(items_as_clusters == cluster)
                    mask_item[indexes] = 1
                except ValueError:
                    pass
            mask[item] = csr_matrix(mask_item)
        # check the minimum amount of items: we create a mask over the similarity matrix, it's possible that this mask leads to less than K items in the similarity matrix (for a particular item)
        min_K = mask.sum(axis=1).min()
        if self.K > min_K:
            warnings.warn("An item mask has less values than K.", UserWarning)
        # need to fill the diagonal with 1 values here to be able to remove them efficiently later on
        mask.setdiag(1)
        return mask

    def _create_clustered_ranking(self, X: InteractionMatrix):
        embedding = self.model_.input_embeddings.weight.detach().numpy()
        kmeans = KMeans(self.num_clusters)
        clusters = kmeans.fit_predict(embedding)
        self.item_to_cluster_mapping = dict(enumerate(clusters))
        # do a singular window operation
        positives = self._singular_window(X)
        # replace all iids by cluster labels
        positives_as_clusters = self._map_to_cluster(positives, self.item_to_cluster_mapping)
        # create a cluster to cluster matrix
        # cheap trick: csr matrix automatically adds up duplicate entries
        from_clusters = positives_as_clusters[:, 0]
        to_clusters = positives_as_clusters[:, 1]
        values = [1] * len(from_clusters)
        cluster_to_cluster_csr = csr_matrix(
            (values, (from_clusters, to_clusters)), shape=(self.num_clusters, self.num_clusters))
        self.cluster_to_cluster = cluster_to_cluster_csr.toarray()
        # get indices of the top ranking clusters according to top Kc
        self.cluster_ranking = self.cluster_to_cluster.argpartition(-self.Kcl)[:, -self.Kcl:]

    def _map_to_cluster(self, items: np.ndarray, item_to_cluster_mapping: dict):
        k = np.array(list(item_to_cluster_mapping.keys()))
        v = np.array(list(item_to_cluster_mapping.values()))
        mapping_ar = np.zeros(k.max() + 1, dtype=v.dtype)
        mapping_ar[k] = v
        clusters = mapping_ar[items]
        return clusters

    # todo not efficient since half of the windows and half of the window are useless
    def _singular_window(self, X: InteractionMatrix):
        '''
        Create pairs of positive samples.
        '''
        window_size = 1
        windowed_sequences = window(X.sorted_item_history, window_size)
        context = windowed_sequences[:, :window_size].flatten()
        focus = windowed_sequences[:, window_size]
        positives = np.vstack([context, focus]).T
        # remove any NaN valued rows (consequence of windowing)
        positives = positives[~np.isnan(positives).any(axis=1)].astype(int)
        return positives
