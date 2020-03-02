import recpack.pipelines
import pytest


@pytest.mark.parametrize(
    "t_0, interval_between_t, nr_t, t_delta, t_alpha, expected_params",
    [
        (
            0, 5, 3, None, None,
            [
                (5, None, None),
                (10, None, None),
                (15, None, None)
            ]
        ),
        (
            0, 5, 1, 2, None,
            [
                (5, 2, None),
            ]
        ),
        (
            0, 5, 1, None, 2,
            [
                (5, None, 2),
            ]
        ),
        (
            0, 5, 1, 3, 2,
            [
                (5, 3, 2),
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

    for i, params in enumerate(generator):
        assert params.splitter_params[0] == t_0 + ((i+1)*interval_between_t)
        assert params.splitter_params[1] == t_delta
        assert params.splitter_params[2] == t_alpha
