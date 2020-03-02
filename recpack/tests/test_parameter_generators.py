import recpack.pipelines
import numpy
import pytest


@pytest.mark.parametrize(
    "t_0, interval_between_t, nr_t, t_delta, t_alpha, expected_params",
    [
        (
            0, 5, 3, None, None,
            [
                {"t": 5, "t_delta": None, "t_alpha": None},
                {"t": 10, "t_delta": None, "t_alpha": None},
                {"t": 15, "t_delta": None, "t_alpha": None},
            ]
        ),
        (
            0, 5, 1, 2, None,
            [
                {"t": 5, "t_delta": 2, "t_alpha": None},
            ]
        ),
        (
            0, 5, 1, None, 2,
            [
                {"t": 5, "t_delta": None, "t_alpha": 2},
            ]
        ),
        (
            0, 5, 1, 3, 2,
            [
                {"t": 5, "t_delta": 3, "t_alpha": 2}
            ]
        )
    ]
)
def test_temporal_parameter_generator(t_0, interval_between_t, nr_t, t_delta, t_alpha, expected_params):

    generator = recpack.pipelines.TemporalSWParameterGenerator(
        t_0, interval_between_t, nr_t, t_delta=t_delta, t_alpha=t_alpha
    )

    assert len(generator) == nr_t
    parameters = [x for x in generator]

    result = []
    for i, params in enumerate(generator):
        result.append(params.splitter_params)
        # assert params.splitter_params[0] == t_0 + ((i+1)*interval_between_t)
        # assert params.splitter_params[1] == t_delta
        # assert params.splitter_params[2] == t_alpha
    assert result == expected_params


@pytest.mark.parametrize(
    "grid_parameters, expected_output",
    [
        (
            {"param_1": [1, 2], "param_2": [11, 22], "param_3": [111, 222]},
            [
                {"param_1": 1, "param_2": 11, "param_3": 111}, {"param_1": 1, "param_2": 11, "param_3": 222}, 
                {"param_1": 1, "param_2": 22, "param_3": 111}, {"param_1": 1, "param_2": 22, "param_3": 222},
                {"param_1": 2, "param_2": 11, "param_3": 111}, {"param_1": 2, "param_2": 11, "param_3": 222},
                {"param_1": 2, "param_2": 22, "param_3": 111}, {"param_1": 2, "param_2": 22, "param_3": 222}
            ]
        )
    ]
)
def test_splitter_grid_search_generator(grid_parameters, expected_output):
    generator = recpack.pipelines.SplitterGridSearchGenerator(grid_parameters)

    result = []
    for params in generator:
        result.append(params.splitter_params)
        # assert params.splitter_params in expected_output
    assert result == expected_output
