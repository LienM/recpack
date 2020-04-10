import recpack
import recpack.preprocessing.helpers as helpers
import recpack.algorithms
import recpack.evaluate
import recpack.splits
import recpack.pipelines
from recpack.config import ParameterGeneratorPipelineConfig, PipelineConfig

import click
import json
import pandas as pd

# TODO Refactor module based on preprocessing

def prep_data(input_file, input_file_2, item_column_name, user_column_name, timestamp_column_name):
    """
    Take the raw input files, and turn them into DataM objects.
    """
    # Load file data
    loaded_dataframe = pd.read_csv(input_file)

    # Cleanup duplicates
    loaded_dataframe.reset_index(inplace=True)
    data_dd = loaded_dataframe[[item_column_name, user_column_name]].drop_duplicates()
    data_dd.reset_index(inplace=True)
    dataframe = pd.merge(
        data_dd, loaded_dataframe,
        how='inner', on=['index', item_column_name, user_column_name]
    )

    del data_dd

    dataframe_2 = None
    if input_file_2 is not None:
        loaded_dataframe_2 = pd.read_csv(input_file_2)
        loaded_dataframe_2.reset_index(inplace=True)
        data_dd = loaded_dataframe_2[[item_column_name, user_column_name]].drop_duplicates()
        data_dd.reset_index(inplace=True)
        dataframe_2 = pd.merge(
            data_dd, loaded_dataframe_2,
            how='inner', on=['index', item_column_name, user_column_name]
        )
        del data_dd

    # Convert user and item ids into a continuous sequence to make
    # training faster and use much less memory.
    if dataframe_2 is None:
        item_ids = list(dataframe[item_column_name].unique())
        user_ids = list(dataframe[user_column_name].unique())
    else:
        item_ids = set(list(dataframe[item_column_name].unique()) + list(dataframe_2[item_column_name].unique()))
        user_ids = set(list(dataframe[user_column_name].unique()) + list(dataframe_2[user_column_name].unique()))

    item_id_mapping = helpers.rescale_id_space(item_ids)
    user_id_mapping = helpers.rescale_id_space(user_ids)

    cleaned_item_column_name = 'iid'
    cleaned_user_column_name = 'uid'

    dataframe[cleaned_item_column_name] = dataframe[item_column_name].map(lambda x: item_id_mapping[x])
    dataframe[cleaned_user_column_name] = dataframe[user_column_name].map(lambda x: user_id_mapping[x])
    # To avoid confusion, and free up some memory delete the raw fields.
    df = dataframe.drop([user_column_name, item_column_name], axis=1)

    df_2 = None
    if dataframe_2 is not None:
        dataframe_2[cleaned_item_column_name] = dataframe_2[item_column_name].map(
            lambda x: item_id_mapping[x])
        dataframe_2[cleaned_user_column_name] = dataframe_2[user_column_name].map(
            lambda x: user_id_mapping[x])
        # To avoid confusion, and free up some memory delete the raw fields.
        df_2 = dataframe_2.drop([user_column_name, item_column_name], axis=1)

    # Convert input data into internal data objects
    data = helpers.create_data_M_from_pandas_df(
        df, cleaned_item_column_name, cleaned_user_column_name, timestamp_column_name,
        shape=(len(user_ids), len(item_ids))
    )
    data_2 = None
    if df_2 is not None:
        data_2 = helpers.create_data_M_from_pandas_df(
            df_2, cleaned_item_column_name, cleaned_user_column_name, timestamp_column_name,
            shape=(len(user_ids), len(item_ids))
        )

    return data, data_2


@click.command()
@click.option('-c', '--config', type=click.File('r'), required=True)
@click.option('--input_file', type=click.Path(readable=True), required=True)
@click.option('--input_file_2', type=click.Path(readable=True), required=False)
@click.option('--output_file', type=click.File('w'), default='results.json', show_default=True)
@click.option('--user_column_name', type=str, default='userId', show_default=True)
@click.option('--item_column_name', type=str, default='itemId', show_default=True)
@click.option('--timestamp_column_name', type=str, default=None, show_default=True)
def run_pipeline(
    config,
    input_file,
    input_file_2,
    output_file,
    user_column_name,
    item_column_name,
    timestamp_column_name
):

    # Construct config obj, will also validate the config.
    config_obj = PipelineConfig(config)

    # Transform data from raw input files to DataM objects.
    data, data_2 = prep_data(input_file, input_file_2, item_column_name, user_column_name, timestamp_column_name)

    # resolve algorithms and their parameters from config, and create the objects
    algorithms = [recpack.algorithms.algorithm_registry.get(type)(**params) for type, params in config_obj.get_algorithms()]

    # Create splitter class
    s_type, s_params = config_obj.get_splitter()
    splitter = recpack.splits.get_splitter(s_type)(**s_params)

    # Create evaluator class
    e_type, e_params = config_obj.get_evaluator()
    evaluator = recpack.evaluate.get_evaluator(e_type)(**e_params)

    # construct pipeline
    pipeline = recpack.pipelines.Pipeline(
        splitter,
        algorithms,
        evaluator,
        config_obj.get_metrics(),
        config_obj.get_K_values()
    )

    # run pipeline with data
    pipeline.run(data, data_2)

    # Get metrics
    metrics = pipeline.get()

    # Write metrics to json file.
    json.dump(metrics, output_file, indent=4)


@click.command()
@click.option('-c', '--config', type=click.File('r'), required=True)
@click.option('--input_file', type=click.Path(readable=True), required=True)
@click.option('--input_file_2', type=click.Path(readable=True), required=False)
@click.option('--output_file', type=click.File('w'), default='results.json', show_default=True)
@click.option('--user_column_name', type=str, default='userId', show_default=True)
@click.option('--item_column_name', type=str, default='itemId', show_default=True)
@click.option('--timestamp_column_name', type=str, default=None, show_default=True)
def run_parameter_generator_pipeline(
    config,
    input_file,
    input_file_2,
    output_file,
    user_column_name,
    item_column_name,
    timestamp_column_name
):

    # Construct config obj, will also validate the config.
    config_obj = ParameterGeneratorPipelineConfig(config)

    # Transform data from raw input files to DataM objects.
    data, data_2 = prep_data(input_file, input_file_2, item_column_name, user_column_name, timestamp_column_name)

    # resolve algorithms and their parameters from config, and create the objects
    algorithms = [recpack.algorithms.algorithm_registry.get(type)(**params) for type, params in config_obj.get_algorithms()]

    # Create the parameter generator instance.
    p_type, p_params = config_obj.get_parameter_generator()
    parameter_generator = recpack.pipelines.get_parameter_generator(p_type)(**p_params)

    # Create splitter class object
    s_type = config_obj.get_splitter()
    splitter = recpack.splits.get_splitter(s_type)

    # Create evaluator class object
    e_type = config_obj.get_evaluator()
    evaluator = recpack.evaluate.get_evaluator(e_type)

    # construct pipeline
    pipeline = recpack.pipelines.ParameterGeneratorPipeline(
        parameter_generator,
        splitter,
        algorithms,
        evaluator,
        config_obj.get_metrics(),
        config_obj.get_K_values()
    )

    # run pipeline with data
    pipeline.run(data, data_2)

    # Get metrics
    metrics = pipeline.get()

    # Write metrics to json file.
    json.dump(metrics, output_file, indent=4)
