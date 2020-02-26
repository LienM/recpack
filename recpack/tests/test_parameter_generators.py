import recpack.pipelines


def test_temporal_parameter_generator():
    t_0 = 0
    nr_t = 3
    time_between_t = 5
    t_delta = 5

    generator = recpack.pipelines.TemporalSWParameterGenerator(t_0, t_delta, time_between_t, nr_t)

    assert len(generator) == 3
    parameters = [x for x in generator]

    assert parameters[0].splitter_params[0] == 5
    assert parameters[1].splitter_params[0] == 10
    assert parameters[2].splitter_params[0] == 15

    assert parameters[0].splitter_params[1] == parameters[1].splitter_params[1] == parameters[2].splitter_params[1]
