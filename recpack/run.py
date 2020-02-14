import os
import gc
import logging
from collections import defaultdict
from itertools import groupby

import pandas as pd
import numpy as np
import scipy.sparse

import scipy.sparse.linalg

from .helpers import load_data_from_gcs, create_csr_matrix_from_pandas_df, get_algorithm, \
    upload_data_to_gcs

from .env import TrainingConfiguration

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('train')


def main():

    config = TrainingConfiguration()

    training_df = load_data_from_gcs(config.BUCKET_NAME, config.DATASET_NAME, config.EXPERIMENT_NAME, config.TRAINING_DATA_FILENAME)
    training_csr_mat = create_csr_matrix_from_pandas_df(training_df, 'iid', 'uid')

    # Free up some unused memory
    del training_df

    gc.collect()

    algo = get_algorithm(config.ALGORITHM_NAME)

    B = algo.fit(training_csr_mat)

    filename = algo.save(B, experiment_name=config.EXPERIMENT_NAME)

    upload_data_to_gcs(config.BUCKET_NAME, config.DATASET_NAME, config.EXPERIMENT_NAME, filename)

    def predict_and_evaluate(X_test, func, k_values, fold_in=0.8):

    #     X_pred = func(test_in)
    # num_users = 0
    # For every user (if needed - this can be parallellised)
    for ix, u in enumerate(test_in):
        num_interactions = u.nnz
        if num_interactions == 0:
            continue

        scores = func(u)[0]


if __name__ == 'main':
    main()
