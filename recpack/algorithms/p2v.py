from recpack.algorithms.base import ItemSimilarityMatrixAlgorithm
from recpack.data.matrix import InteractionMatrix
from recpack.data.matrix import to_csr_matrix
from recpack.data.matrix import to_binary
from recpack.metrics.precision import PrecisionK
from recpack.algorithms.samplers import sample_positives_and_negatives
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time


class Prod2Vec(ItemSimilarityMatrixAlgorithm):
    def __init__(self, vector_size: int, item_vocabulary: int):
        '''
        Initialize the skipgram model.

        :param vector_size: size of the embedding vectors (embedding cols)
        :param item_vocabulary: number of items in the entire dataset (embedding rows)
        '''
        ItemSimilarityMatrixAlgorithm.__init__(self)
        self.vector_size = vector_size
        self.item_vocabulary = item_vocabulary
        self.model = SkipGram(self.item_vocabulary, self.vector_size)
        self.similarity_matrix_ = None
        print(self.model)

    def __str__(self):
        return self.name

    def fit(self, X: InteractionMatrix, negative_samples: int, window: int, validation_in=None, validation_out=None,
            learning_rate=0.01, num_epochs=10, prints_every_epoch=1, batch=100, K=10):
        '''
        Fit the model on a specified training set.

        Note that the Precision@K value is averaged over all users, where the value is 1/K if the test product appears in the top K list of recommended products.

        :param X: training set
        :param negative_samples: number of negative samples per positive sample
        :param window: the window size
        :param validation_in: validation data used for creating predictions
        :param validation_out: validation data used as target
        :param learning_rate: the learning rate of the optimizer
        :param num_epochs: number of epochs to train model
        :param prints_every_epoch: print result very x epochs
        :param batch: the batch size
        :param K: Precision@K for validation data
        :return: embedding: the trained embedding vectors as a numpy array
        :rtype: numpy.ndarray
        '''
        assert self.model is not None
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            start = time.time()
            epoch_loss = 0.0
            n_batches = 0
            # generator will just be restarted for each epoch.
            generator = self._training_generator(X, negative_samples, window, batch=batch)
            for focus_batch, positives_batch, negatives_batch in generator:
                focus_vectors = self.model.get_embeddings(focus_batch)
                positives_vectors = self.model.get_embeddings(positives_batch)
                negatives_vectors = self.model.get_embeddings(negatives_batch)
                loss = self.model(focus_vectors, positives_vectors, negatives_vectors)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # the epoch loss with averaging over number of batches
                epoch_loss += loss.item()
                n_batches += 1
            epoch_loss = epoch_loss/n_batches
            end = time.time()
            if epoch % prints_every_epoch == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}] (in {int((end-start))}s) Epoch Loss: {epoch_loss:.4f}')
            if validation_in is not None and validation_out is not None:
                validation_in = to_csr_matrix(validation_in)
                validation_out = to_csr_matrix(validation_out)
                embedding = self.model.embeddings.weight
                embedding = embedding.detach().numpy()
                predictions = self.predict(K, validation_in, embedding)
                preck = PrecisionK(K)
                preck.calculate(validation_out, predictions)
                sum = preck.scores_.sum()
                score = sum / validation_in.shape[0]
                # Reset similarity matrix
                self.similarity_matrix_ = None
                if epoch % prints_every_epoch == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}] Validation Precision@{K}: {score:.4f}')
        embedding = self.model.embeddings.weight
        embedding = embedding.detach().numpy()
        return embedding

    def predict(self, K, X: csr_matrix, embedding: np.ndarray, batch_size=1000):
        '''
        Batched predict method.

        Only the top-K most similar items are kept in the similarity matrix.
        Will call SimilarityMatrixAlgorithm predict.

        :param K: only top-K most similar items are kept
        :param X: predict dataset
        :param embedding: the trained embedding vectors
        :param batch_size: batch size
        :returns predictions (scores)
        :rtype: scipy.sparse.csr_matrix
        '''
        # need K+1 since we set diagonal to zero after calculating top-K
        K=K+1
        num_items = embedding.shape[0]
        item_cosine_similarity_ = scipy.sparse.lil_matrix((num_items, num_items))
        for batch in range(0, num_items, batch_size):
            Y = embedding[batch:batch+batch_size]
            item_cosine_similarity_batch = csr_matrix(cosine_similarity(Y, embedding))

            indices = [(i, j) for i, best_items_row in enumerate(np.argpartition(item_cosine_similarity_batch.toarray(), -K)) for j in best_items_row[-K:]]

            mask = scipy.sparse.csr_matrix(([1 for i in range(len(indices))], (list(zip(*indices)))), shape=(Y.shape[0], num_items))

            item_cosine_similarity_batch = item_cosine_similarity_batch.multiply(mask)
            item_cosine_similarity_[batch:batch+batch_size] = item_cosine_similarity_batch
        # no self similarity, set diagonal to zero
        item_cosine_similarity_.setdiag(0)
        self.similarity_matrix_ = csr_matrix(item_cosine_similarity_)
        predictions = super().predict(X)
        return predictions


    def save(self, model_path=None, similarity_matrix_path=None):
        '''
        Save the entire SkipGram model and similarity matrix to a file.

        Note: Pytorch convention is to use a ".pt" extension for models.
        Note: Numpy uses a ".npy" extension.

        :param model_path: file path for skipgram model
        :param similarity_matrix_path: file path for similarity matrix
        '''
        assert self.model is not None
        if model_path is None:
            model_path = "./model_"+time.strftime("%Y%m%d-%H%M%S")+".pt"
        if similarity_matrix_path is None:
            similarity_matrix_path = "./similarities_"+time.strftime("%Y%m%d-%H%M%S")+".npy"

        torch.save(self.model, model_path)
        if self.similarity_matrix_ is not None:
            np.save(similarity_matrix_path, self.similarity_matrix_)

    def load(self, model_path, similarity_matrix_path=None):
        '''
        Load the entire SkipGram model from a file.

        Note: Pytorch convention is to use a ".pt" extension for models.
        Note: Numpy uses a ".npy" extension.

        :param model_path: file path of skipgram model
        :param similarity_matrix_path: file path of similarity matrix
        '''
        self.model = torch.load(model_path)
        if similarity_matrix_path is not None:
            self.similarity_matrix_ = np.load(similarity_matrix_path)


    def _sorted_item_history(self, X: InteractionMatrix) -> list:
        # TODO don't think there is a sorted_item_history in recpack
        # Note: this is not a generator as similar methods in InteractionMatrix
        """
        Group sessions (products) by user.

        Returns a list of lists.
        Each list represents a user's sequence of purchased products.
        Each product is designated by it's iid.

        :param: X: InteractionMatrix
        :returns: list of lists
        """
        df = X._df[['iid', 'uid']]
        grouped_df = df.groupby(['uid'])['iid'].apply(list)
        return grouped_df.values.tolist()


    def _window(self, sequences, window):
        '''
        Will apply a windowing operation to a sequence of item sequences.

        :param sequences: iterable of iterables
        :param window: the size of the window
        :returns: windowed sequences
        :rtype: numpy.ndarray
        '''
        w = [w.tolist() for sequence in sequences if len(sequence) >= window for w in
             sliding_window_view(sequence, window_shape=window)]
        return np.array(w)


    def _training_generator(self, X: InteractionMatrix, negative_samples: int, window: int, batch=1000):
        ''' Creates a training dataset using the skipgrams and negative sampling method.

        First, the sequences of items (iid) are grouped per user (uid).
        Next, a windowing operation is applied over each seperate item sequence.
        These windows are sliced and re-stacked in order to create skipgrams of two items (pos_training_set).
        Next, a unigram distribution of all the items in the dataset is created.
        This unigram distribution is used for creating the negative samples.

        :param X: InteractionMatrix
        :param negative_samples: number of negative samples per positive sample
        :param window: the window size
        :param batch: the batch size
        :yield: focus_batch, positive_samples_batch, negative_samples_batch
        :rtype: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
        '''
        # Group and Window
        sequences = self._sorted_item_history(X)
        elements_per_window = (window * 2 + 1)
        windowed_sequences = self._window(sequences, elements_per_window)
        context = np.hstack((windowed_sequences[:, :window], windowed_sequences[:, window + 1:]))
        focus = windowed_sequences[:, window]
        # Stack and Shuffle
        positives = np.empty((0, 2), dtype=int)
        for col in range(context.shape[1]):
            stacked = np.column_stack((focus, context[:, col]))
            positives = np.vstack([positives, stacked])

        focus_items = positives[:, 0]
        pos_items = positives[:, 1]
        values = [1] * len(focus_items)
        positives_csr = scipy.sparse.csr_matrix((values, (focus_items, pos_items)), shape=(X.shape[1], X.shape[1]))
        co_occurrence = to_binary(positives_csr + positives_csr.T)
        co_occurrence = scipy.sparse.lil_matrix(co_occurrence)
        co_occurrence.setdiag(1)
        co_occurrence = co_occurrence.tocsr()
        yield from sample_positives_and_negatives(X=co_occurrence, U=negative_samples, batch_size=batch, replace=True, exact=True, positives=positives)

class SkipGram(nn.Module):
    def __init__(self, vocab_size, vector_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, vector_size)

    def get_embeddings(self, inputs):
        vectors = self.embeddings(inputs)
        return vectors

    def forward(self, focus_vectors, positive_vectors, negative_vectors):
        embedding_dim = self.embeddings.embedding_dim
        focus_vectors = focus_vectors.reshape(-1, embedding_dim, 1)
        positive_vectors = positive_vectors.reshape(-1, 1, embedding_dim)
        # Focus vector and positive vector calculation:
        loss = torch.bmm(positive_vectors, focus_vectors).sigmoid().log()
        # Focus vector and negative vectors calculation:
        neg_loss = torch.bmm(negative_vectors.neg(), focus_vectors).sigmoid().log()
        neg_loss = neg_loss.squeeze().sum(1)
        return -(loss + neg_loss).mean()
