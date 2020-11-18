from scipy.sparse import csr_matrix
import torch

from recpack.algorithms.samplers import bootstrap_sample_pairs


def bpr_loss(positive_sim, negative_sim):
    distance = positive_sim - negative_sim
    # Probability of ranking given parameters
    elementwise_bpr_loss = torch.log(torch.sigmoid(distance))
    bpr_loss = -elementwise_bpr_loss.sum()

    return bpr_loss


def bpr_loss_metric(X_true: csr_matrix, X_pred: csr_matrix, batch_size=1000):
    """Compute BPR reconstruction loss of the X_true matrix in X_pred

    :param X_true: [description]
    :type X_true: [type]
    :param X_pred: [description]
    :type X_pred: [type]
    :raises an: [description]
    :return: [description]
    :rtype: [type]
    :yield: [description]
    :rtype: [type]
    """
    # This is kinda bad because it's duplication of data
    total_loss = 0

    for d in bootstrap_sample_pairs(
        X_true, batch_size=batch_size, sample_size=X_true.nnz
    ):
        # Needed to do copy, to use as index in the predidction matrix
        users = d[:, 0].numpy().copy()
        target_items = d[:, 1].numpy().copy()
        negative_items = d[:, 2].numpy().copy()

        print(X_pred[users, target_items])
        positive_sim = torch.tensor(X_pred[users, target_items])
        negative_sim = torch.tensor(X_pred[users, negative_items])

        total_loss += bpr_loss(positive_sim, negative_sim).item()

    return total_loss
