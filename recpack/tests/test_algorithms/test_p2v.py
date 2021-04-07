from recpack.data.datasets import ThirtyMusicSessionsSmall
from recpack.algorithms.p2v import Prod2Vec
from recpack.data.matrix import InteractionMatrix
from recpack.metrics.precision import PrecisionK
import pytest
import numpy as np
import scipy
import os


class TestProd2Vec():
    @pytest.fixture()
    def thirty_music_small(self):
        dataset = ThirtyMusicSessionsSmall("")
        df = dataset.load_dataframe()
        im = dataset.load_interaction_matrix()
        prod2vec = Prod2Vec(vector_size=50, item_vocabulary=im.shape[1])
        return df, im, prod2vec

    def test_predict(self):
        # the target values for our predictions
        # we have five users, the target is the last item the user bought
        values = [1] * 5
        users = [0, 1, 2, 3, 4]
        items = [0, 1, 2, 3, 4]
        target = scipy.sparse.csr_matrix((values, (users, items)))
        target = InteractionMatrix.from_csr_matrix(target)

        # embedding vectors for each item
        embedding = [[0.5, 0.5, 0.0, 0.0, 0.0],
                     [0.4, 0.4, 0.1, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.5, 0.5],
                     [0.0, 0.0, 0.5, 0.5, 0.0],
                     [1.0, 0.0, 0.0, 0.0, 0.0]]
        embedding = np.array(embedding)

        prod2vec = Prod2Vec(vector_size=5, item_vocabulary=5)
        # get the most similar item to each item
        predictions = prod2vec.predict(1, target, embedding, batch_size=2)
        similarity_matrix = prod2vec.similarity_matrix_.toarray()
        # get the most similar item for each item
        max_similarity_items = list(np.argmax(similarity_matrix, axis=1))
        assert max_similarity_items == [1, 0, 3, 2, 0]
        # cosine similarities can be calculated by hand easily
        # only get the most similar item using max
        assert max(similarity_matrix[0]) == pytest.approx(0.98473, 0.00005)
        assert max(similarity_matrix[2]) == pytest.approx(0.5, 0.00005)
        assert max(similarity_matrix[4]) == pytest.approx(0.70711, 0.00005)

        # let's create some truth values:
        # the truth values are equal to the most similar product according to the similarity matrix
        # this means the following should hold:
        # - each prediction should be present in the top 2 (k=2)
        # - a value of 1/2 (1/k) is added when we have a prediction is present
        # - the final score is the sum of all values divided by the number of predictions
        # i.e. 0.5*5/5 = 0.5

        values = [1] * 5
        users = [0, 1, 2, 3, 4]
        items = [1, 0, 3, 2, 0]
        truth = scipy.sparse.csr_matrix((values, (users, items)), shape=(5, 5))

        preck = PrecisionK(2)
        preck.calculate(truth, predictions)
        sum = preck.scores_.sum()
        score = sum / 5
        assert score == 0.5

        # what if we miss a value?
        # i.e. 0.5 * 4 / 5 = 0.4

        values = [1] * 5
        users = [0, 1, 2, 3, 4]
        items = [1, 0, 3, 2, 3]
        truth = scipy.sparse.csr_matrix((values, (users, items)), shape=(5, 5))

        preck = PrecisionK(2)
        preck.calculate(truth, predictions)
        sum = preck.scores_.sum()
        score = sum / 5
        assert score == 0.4


    def test_train_predict(self, thirty_music_small):
        # create some training data
        users = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        items = [0, 1, 2, 0, 1, 2, 3, 4, 5]
        values = [1] * len(users)
        train = scipy.sparse.csr_matrix((values, (users, items)))
        train = InteractionMatrix.from_csr_matrix(train)

        # predict the closest item to each of the items
        values = [1] * 6
        users = [0, 1, 2, 3, 4, 5]
        items = [0, 1, 2, 3, 4, 5]
        target = scipy.sparse.csr_matrix((values, (users, items)))
        target = InteractionMatrix.from_csr_matrix(target)

        prod2vec = Prod2Vec(vector_size=5, item_vocabulary=6)
        # Note: overfitting to make sure we get "deterministic" results
        embedding = prod2vec.fit(train, 2, 1, num_epochs=200, batch=20)
        # get the most similar item to each item
        predictions = prod2vec.predict(1, target, embedding, batch_size=2)
        similarity_matrix = prod2vec.similarity_matrix_.toarray()
        # get the most similar item for each item
        max_similarity_items = list(np.argmax(similarity_matrix, axis=1))
        # Note: 3,4,5 should be close together in the vectors space -> 0,1,2 shouldn't be the most similar to either 3,4,5
        assert 0 not in max_similarity_items[3:]
        assert 1 not in max_similarity_items[3:]
        assert 2 not in max_similarity_items[3:]
        # Note: 0,1,2 should be close together in the vectors space -> 3,4,5 shouldn't be the most similar to either 0,1,2
        assert 3 not in max_similarity_items[:3]
        assert 4 not in max_similarity_items[:3]
        assert 5 not in max_similarity_items[:3]

    def test__group_by_user(self, thirty_music_small):
        df, im, prod2vec = thirty_music_small
        unique_users = df[['user_id']].nunique()[0]
        user_list = prod2vec._sorted_item_history(im)
        assert len(user_list) == unique_users
        for index, user in enumerate(user_list):
            len_user_sequence_df = len(im._df[im._df['uid'] == index])
            len_user_sequence_dict = len(user)
            assert len_user_sequence_dict == len_user_sequence_df

    def test__window(self, thirty_music_small):
        prod2vec = Prod2Vec(vector_size=50, item_vocabulary=5)
        sequence = [
            ['computer', 'artificial', 'intelligence', 'dog', 'trees'],
            ['human', 'intelligence', 'cpu', 'graph'],
            ['intelligence'],
            ['artificial', 'intelligence', 'system']
        ]
        # create a window of size 3: sequence 1: 3 windows, sequence 2: 2 windows, sequence 4: 1 window => 6 windows
        windowed_seq = prod2vec._window(sequence, 3)
        row, col = windowed_seq.shape
        assert row == 6
        assert col == 3

    def test_save_load(self, thirty_music_small):
        df, im, prod2vec = thirty_music_small
        model_path = './save_model_test.pt'
        matrix_path = './save_matrix_test.npy'
        prod2vec.save(model_path, matrix_path)
        prod2vec.load(model_path)
        os.remove(model_path)