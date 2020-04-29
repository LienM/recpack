import recpack.algorithms.item_metadata_algorithms.hashing_algorithms as ha
import pytest
import scipy.sparse

@pytest.mark.parametrize(
    "min_jaccard, n_gram",
    [
        (0.3, 3),
        (0.9, 3),
        (0, 4)
    ]
)
def test_lsh_model(metadata, min_jaccard, n_gram):
    algo = ha.LSHModel(
        metadata, min_jaccard=min_jaccard, n_gram=n_gram,
        content_key='title', item_key='item_id'
    )

    algo.fit(None)

    test_mat = scipy.sparse.csr_matrix(
        (
            [1 for i in range(metadata.shape[0])],
            (
                [i for i in metadata['item_id'].unique()],
                [i for i in metadata['item_id'].unique()]
            )
        ),shape=(metadata.shape[0], metadata.shape[0])
    )

    prediction = algo.predict(test_mat)
    assert prediction[0,1] == prediction[1,0]

