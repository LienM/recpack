import math

import numpy as np
from scipy.sparse import rand, csr_matrix

from recpack.algorithms.loss_functions import warp_loss_metric, warp_loss


def test_warp_loss_metric():

    X_true = csr_matrix(
        [
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0]
        ]
    )

    X_pred = csr_matrix(
        [
            [0.2, 0.2, 0],
            [0.8, 1, 0.9],
            [1, 1, 0.95]
        ]
    )

    margin = 0.1
    U = 2

    loss = warp_loss_metric(X_true, X_pred, batch_size=1, U=U, margin=margin)

    expected_loss = (  
        # First term, correctly classified (distance to pos < distance to neg)
        math.log((0 * 3 / U) + 1) * (0 - 0.2 + margin)
        # 2 items were wrongly classified
        + math.log((2 * 3 / U) + 1) * (1 - 0.8 + margin)
        # Wrong classification was within margin 
        + math.log((2 * 3 / U) + 1) * (1 - 0.95 + margin)
    ) / 3 # Take the mean loss per item

    np.testing.assert_almost_equal(loss, expected_loss)
