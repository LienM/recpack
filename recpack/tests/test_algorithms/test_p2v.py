import os
import tempfile
from unittest.mock import MagicMock

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recpack.algorithms.p2v import Prod2Vec, window
from recpack.data.matrix import InteractionMatrix
from recpack.metrics.precision import PrecisionK
from recpack.splitters.scenarios import NextItemPrediction
from recpack.data.matrix import to_csr_matrix

from recpack.tests.test_algorithms.util import assert_changed, assert_same


@pytest.fixture(scope="function")
def prod2vec(p2v_embedding, mat):
    prod = Prod2Vec(
        embedding_size=50,
        negative_samples=5,
        window_size=2,
        stopping_criterion="precision",
        K=5,
        batch_size=500,
        learning_rate=0.01,
        max_epochs=10,
        stop_early=False,
        max_iter_no_change=5,
        min_improvement=0.0001,
        save_best_to_file=False,
        replace=False,
        exact=False,
    )
    prod._init_model(mat)
    prod.model_.embeddings = p2v_embedding

    prod.save = MagicMock(return_value=True)
    return prod


def test_window():
    # todo what about the error for small values?

    sequence = [
        (123, ['computer', 'artificial', 'intelligence', 'dog', 'trees']),
        (145, ['human', 'intelligence', 'cpu', 'graph']),
        (1, ['intelligence']),
        (3, ['artificial', 'intelligence', 'system'])
    ]
    # Create a window of size 3:
    # sequence 1: 5 windows
    # sequence 2: 4 windows
    # sequence 3: 1 window
    # sequence 4: 3 window
    # => 13 windows
    windowed_seq = window(sequence, 1)

    row, col = windowed_seq.shape
    assert row == 13
    assert col == 3


def test_training_epoch(prod2vec, mat):

    prod2vec._init_model(mat)

    params = [o for o in prod2vec.model_.named_parameters()
              if o[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    prod2vec._train_epoch(mat)

    device = prod2vec.device

    assert_changed(params_before, params, device)


def test_evaluation_epoch(prod2vec, mat):
    prod2vec._init_model(mat)

    params = [o for o in prod2vec.model_.named_parameters()
              if o[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    prod2vec.best_model = tempfile.NamedTemporaryFile()
    # run a training step
    prod2vec._evaluate(to_csr_matrix(mat), to_csr_matrix(mat))

    device = prod2vec.device

    assert_same(params_before, params, device)
    prod2vec.best_model.close()


def test_predict_warning(prod2vec, p2v_embedding, mat):

    with pytest.warns(UserWarning, match='K is larger than the number of items.'):
        prod2vec._create_similarity_matrix()


def test_create_similarity_matrix(prod2vec):

    prod2vec._create_similarity_matrix()
    similarity_matrix = prod2vec.similarity_matrix_.toarray()

    # Get the most similar item for each item
    np.testing.assert_array_equal(
        np.argmax(similarity_matrix, axis=1), [1, 0, 3, 2, 0])

    # Cosine similarities can be calculated by hand easily
    # Only get the most similar item using max
    assert max(similarity_matrix[0]) == pytest.approx(0.98473, 0.00005)
    assert max(similarity_matrix[2]) == pytest.approx(0.5, 0.00005)
    assert max(similarity_matrix[4]) == pytest.approx(0.70711, 0.00005)


def test_batch_predict(prod2vec, mat):
    prod2vec._create_similarity_matrix()
    predictions = prod2vec._batch_predict(to_csr_matrix(mat))

    np.testing.assert_array_almost_equal(
        predictions.toarray(),
        np.array([[0.98473193, 1.10782342, 0.5, 0.62309149, 1.40341741],
                  [0.98473193, 0.98473193, 0., 0.62309149, 1.40341741],
                  [1.69183871, 1.68104255, 0., 0.62309149, 1.40341741]]))


#     # let's create some truth values:
#     # the truth values are equal to the most similar product according to the similarity matrix
#     # this means the following should hold:
#     # - each prediction should be present in the top 2 (k=2)
#     # - a value of 1/2 (1/k) is added when we a hit
#     # - the final score is the sum of all values divided by the number of predictions
#     # i.e. 0.5*5/5 = 0.5

#     values = [1] * 5
#     users = [0, 1, 2, 3, 4]
#     items = [1, 0, 3, 2, 0]
#     truth = csr_matrix((values, (users, items)), shape=(5, 5))

#     preck = PrecisionK(2)
#     preck.calculate(truth, predictions)
#     assert preck.value == 0.5

#     # what if we miss a value?
#     # i.e. 0.5 * 4 / 5 = 0.4

#     values = [1] * 5
#     users = [0, 1, 2, 3, 4]
#     items = [1, 0, 3, 2, 3]
#     truth = csr_matrix((values, (users, items)), shape=(5, 5))

#     preck = PrecisionK(2)
#     preck.calculate(truth, predictions)
#     assert preck.value == 0.4


# def test_train_predict():
#     data = {
#         "user": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
#         "item": [0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 3, 4, 5, 3, 4],
#         "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#     }
#     df = pd.DataFrame.from_dict(data)
#     im = InteractionMatrix(df, "item", "user", timestamp_ix="timestamp")
#     scenario = NextItemPrediction(validation=True)
#     scenario.split(im)
#     train = scenario.train_X
#     val_data_in = scenario._validation_data_in
#     val_data_out = scenario._validation_data_out

#     # overfitting to make sure we get "deterministic" results
#     prod2vec = Prod2Vec(embedding_size=5, negative_samples=2, window_size=2, stopping_criterion="precision",
#                         batch_size=2, max_epochs=50, K=2)
#     prod2vec.fit(train, (val_data_in, val_data_out))
#     similarity_matrix = prod2vec.similarity_matrix_.toarray()
#     # get the most similar item for each item
#     max_similarity_items = list(np.argmax(similarity_matrix, axis=1))
#     # 3,4,5 should be close together in the vectors space -> 0,1,2 shouldn't be the most similar to either 3,4,5
#     assert 0 not in max_similarity_items[3:]
#     assert 1 not in max_similarity_items[3:]
#     assert 2 not in max_similarity_items[3:]
#     # 0,1,2 should be close together in the vectors space -> 3,4,5 shouldn't be the most similar to either 0,1,2
#     assert 3 not in max_similarity_items[:3]
#     assert 4 not in max_similarity_items[:3]
#     assert 5 not in max_similarity_items[:3]


# def test_save_load(p2v_embedding):
#     prod2vec = Prod2Vec(embedding_size=5, negative_samples=2, window_size=2, stopping_criterion="precision",
#                         K=2)
#     target, embedding = p2v_embedding
#     prod2vec._init_model(target)
#     prod2vec.model_.embeddings = embedding
#     # saving and loading should work
#     prod2vec.save()
#     prod2vec.load(prod2vec.filename)
#     os.remove(prod2vec.filename)
