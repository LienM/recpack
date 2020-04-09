import logging

import pandas as pd
import numpy as np
import recpack
import scipy.sparse

from google.cloud import storage


logger = logging.getLogger('train')


def create_csr_matrix_from_pandas_df(df, item_ix, user_ix, shape=None):
    column_ix = item_ix
    row_ix = user_ix

    num_entries = df.shape[0]
    values = np.ones(num_entries)
    indices = zip(*df.loc[:, [row_ix, column_ix]].values)
    if shape is None:
        shape = df[row_ix].max() + 1, df[column_ix].max() + 1
    sparse_matrix = scipy.sparse.csr_matrix((values, indices), shape=shape, dtype=np.int32)
    sparsity = sparse_matrix.getnnz() / (sparse_matrix.shape[0] * sparse_matrix.shape[1]) * 100
    logger.info(f"Sparsity is {sparsity}")
    return sparse_matrix


def create_temporal_csr_matrix_from_pandas_df(df, item_ix, user_ix, timestamp_ix, shape=None):
    column_ix = item_ix
    row_ix = user_ix

    num_entries = df.shape[0]
    values = df[timestamp_ix].values
    indices = zip(*df.loc[:, [row_ix, column_ix]].values)
    if shape is None:
        shape = df[row_ix].max() + 1, df[column_ix].max() + 1
    sparse_matrix = scipy.sparse.csr_matrix((values, indices), shape=shape, dtype=np.int32)
    return sparse_matrix


def create_data_M_from_pandas_df(df, item_ix, user_ix, timestamp_ix=None, shape=None):
    """
    Convert a pandas dataframe into the recpack internal representation of data.
    If the timestamp_ix is not None the representation will have both values matrix and timestamps matrix
    And will be usable in pipelines using temporal splitters or evaluators.
    """
    values_sp_m = create_csr_matrix_from_pandas_df(df, item_ix, user_ix, shape=shape)
    timestamps_sp_m = None
    if timestamp_ix is not None:
        timestamps_sp_m = create_temporal_csr_matrix_from_pandas_df(df, item_ix, user_ix, timestamp_ix, shape=shape)

    return recpack.DataM(values_sp_m, timestamps_sp_m)


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


def rescale_id_space(ids, id_mapping=None):
    """
    Map the given ids to indices,
    if id_mapping is not None, use that as start, and add new values
    """
    counter = 0

    if id_mapping is not None and len(id_mapping) > 0:
        counter = max(id_mapping.values()) + 1
    else:
        id_mapping = {}
    for val in ids:
        if val not in id_mapping:
            id_mapping[val] = counter
            counter += 1

    return id_mapping


class DataPrep:
    """
    Class to take csv input file, and turn it into a DataM object.
    With storage of mappings so multiple different sources can be combined.
    """
    def __init__(self):
        self.item_id_mapping = {}
        self.user_id_mapping = {}

    def get_dataframe(self, input_file, item_column_name, user_column_name, deduplicate=True):
        """
        Take the raw input file, and turn it into a pandas dataframe
        """
        # Load file data
        loaded_dataframe = pd.read_csv(input_file)

        # Cleanup duplicates
        loaded_dataframe.reset_index(inplace=True)
        if deduplicate:
            data_dd = loaded_dataframe[[item_column_name, user_column_name]].drop_duplicates()
            data_dd.reset_index(inplace=True)
            dataframe = pd.merge(
                data_dd, loaded_dataframe,
                how='inner', on=['index', item_column_name, user_column_name]
            )

            del data_dd
            return dataframe
        return loaded_dataframe

    def update_id_mappings(self, dataframe, item_column_name, user_column_name):
        """
        Update the id mapping so we can combine multiple files
        """

        # Convert user and item ids into a continuous sequence to make
        # training faster and use much less memory.
        item_ids = list(dataframe[item_column_name].unique())
        user_ids = list(dataframe[user_column_name].unique())

        self.user_id_mapping = recpack.helpers.rescale_id_space(
            user_ids, id_mapping=self.user_id_mapping
        )
        self.item_id_mapping = recpack.helpers.rescale_id_space(
            item_ids, id_mapping=self.item_id_mapping
        )

    def get_data(self, dataframe, item_column_name, user_column_name, timestamp_column_name):

        cleaned_item_column_name = 'iid'
        cleaned_user_column_name = 'uid'

        dataframe[cleaned_item_column_name] = dataframe[item_column_name].map(
            lambda x: self.item_id_mapping[x]
        )
        dataframe[cleaned_user_column_name] = dataframe[user_column_name].map(
            lambda x: self.user_id_mapping[x]
        )
        # To avoid confusion, and free up some memory delete the raw fields.
        df = dataframe.drop([user_column_name, item_column_name], axis=1)

        # Convert input data into internal data objects
        data = recpack.helpers.create_data_M_from_pandas_df(
            df,
            cleaned_item_column_name, cleaned_user_column_name, timestamp_column_name,
            shape=(
                max(self.user_id_mapping.values()) + 1,
                max(self.item_id_mapping.values()) + 1
            )
        )

        return data
