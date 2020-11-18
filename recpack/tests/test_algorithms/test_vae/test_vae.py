import os.path

import numpy as np

from recpack.algorithms import MultVAE


def test_batched_prediction(larger_matrix):

    mult_vae_1 = MultVAE(seed=1)

    mult_vae_1._init_model(larger_matrix.shape[1])

    X_pred_batched = mult_vae_1._batch_predict(larger_matrix)

    input_users = set(larger_matrix.nonzero()[0])
    batched_users = set(X_pred_batched.nonzero()[0])

    # We want to make sure all users got recommendations
    np.testing.assert_array_equal(list(input_users), list(batched_users))


def test_vae_save_load(larger_matrix):

    mult_vae_1 = MultVAE(seed=1, save_best_to_file=True, max_epochs=10)

    mult_vae_1.fit(larger_matrix, (larger_matrix, larger_matrix))
    assert os.path.isfile(mult_vae_1.file_name)

    os.remove(mult_vae_1.file_name)


def test_cleanup():
    def inner():
        a = MultVAE(seed=1)
        assert os.path.isfile(a.best_model.name)
        return a.best_model.name

    n = inner()
    assert not os.path.isfile(n)
