import numpy as np

from recpack.algorithms import MultVAE
from recpack.algorithms.vae.util import naive_sparse2tensor, naive_tensor2sparse

def test_batched_prediction(larger_matrix, mult_vae):


    mult_vae_1 = MultVAE(seed=1)
    
    mult_vae_1._init_model(larger_matrix.shape[1])

    tensor_in = naive_sparse2tensor(larger_matrix)

    X_pred_batched = mult_vae_1._batch_predict(larger_matrix)

    print(len(set(larger_matrix.nonzero()[0])))

    input_users = set(larger_matrix.nonzero()[0])
    batched_users = set(X_pred_batched.nonzero()[0])

    # We want to make sure all users got recommendations
    np.testing.assert_array_equal(list(input_users), list(batched_users))
