import recpack.metrics as metrics
import recpack.splitters.splitter_base as splitter_base


def test_fold_iterator_and_metrics(data_m):
    K = 2
    metric = metrics.RecallK(K)

    fold_iterator = splitter_base.FoldIterator(data_m, data_m, batch_size=10)

    nonzero_users = set(data_m.indices[0])

    for fold_in, fold_out in fold_iterator:
        # Score should be 1 because fold_in == fold_out
        # metric.update(fold_in, fold_out)
        pass

    # assert metric.num_users == len(nonzero_users)
    # assert metric.value == 1


def test_fold_iterator_and_predict(data_m):
    pass
