import logging

import pandas as pd
import numpy as np
import scipy.sparse

from google.cloud import storage


logger = logging.getLogger('train')


def create_csr_matrix_from_pandas_df(df, item_ix, user_ix):
    column_ix = item_ix
    row_ix = user_ix

    num_entries = df.shape[0]
    values = np.ones(num_entries)
    indices = zip(*df.loc[:, [row_ix, column_ix]].values)
    shape = df[row_ix].max() + 1, df[column_ix].max() + 1
    sparse_matrix = scipy.sparse.csr_matrix((values, indices), shape=shape, dtype=np.int32)
    sparsity = sparse_matrix.getnnz() / (sparse_matrix.shape[0] * sparse_matrix.shape[1]) * 100
    logger.info(f"Sparsity is {sparsity}")
    return sparse_matrix


def load_data_from_gcs(bucket_name, dataset_name, experiment_name, filename):

    data_path = f"gs://{bucket_name}/{dataset_name}/{experiment_name}/{filename}"

    df = pd.read_csv(data_path)

    return df


def upload_data_to_gcs(bucket_name, dataset_name, experiment_name, filename, client=None):

    if not client:
        client = storage.Client()

    bucket = client.get_bucket(bucket_name)

    data_path = f"{dataset_name}/{experiment_name}/{filename}"

    blob = bucket.blob(data_path)
    blob.upload_from_filename(filename)

    return


def rescale_id_space(ids):
    org_id_to_new_id = {val: ix for ix, val in enumerate(ids)}
    return org_id_to_new_id
