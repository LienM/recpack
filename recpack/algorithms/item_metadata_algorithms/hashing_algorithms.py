import scipy.sparse
import numpy
import pandas
from snapy import MinHash, LSH

# TODO Implement BaseEstimator interface.

class LSHModel:
    """
    Model based on metadata only.
    Item similarity is defined by the jaccard index of the strings,
    this efficiently computed using local sensitivity hasing.

    Local Sensitivity Hashing implemented in the SnaPy library.

    metadata is required to be passed, and will get used to fit a model at runtime.
    TODO: The fit method takes a pandas dataframe as input, which could be the standard for metadata based model
          Not sure though.

    If you want to work on a combination of data fields: combine the fields in a string field to be used by this class.
    """
    def __init__(self, data, shape=None, min_jaccard=0.3, n_gram=3, content_key='title', item_key='itemId'):
        self.lsh_model = None
        self.data = data

        self.n_gram = n_gram
        self.min_jaccard = min_jaccard
        
        self.shape = shape
        if shape is None:
            num_items = data[item_key].nunique()
            self.shape = (num_items, num_items)

        self.content_key = content_key
        self.item_key = item_key

        self.trained = False

    def _train(self, df: pandas.DataFrame):
        if self.trained:
            return

        df_filter_empty = df[df[self.content_key].str.len() >= self.n_gram]
        content = list(df_filter_empty[self.content_key])
        labels = list(df_filter_empty[self.item_key])

        minhash = MinHash(content, n_gram=self.n_gram, n_gram_type='char', permutations=100, hash_bits=64)
        lsh = LSH(minhash, labels, no_of_bands=50)

        # Construct a scipy sparse Mutex prediction model
        x_ids = []
        y_ids = []
        values = []
        tuples = [i for i in zip(*lsh.edge_list(min_jaccard=self.min_jaccard, jaccard_weighted=True))]
        if len(tuples) != 0:
            x_ids, y_ids, values = tuples
        # This will contain half of the similarity matrix
        # So we need to mirror it
        total_x_ids = x_ids + y_ids
        total_y_ids = y_ids + x_ids
        values += values

        self.lsh_model = scipy.sparse.csr_matrix((values, (total_x_ids, total_y_ids)), shape=self.shape)

        self.trained = True

    def fit(self, X: scipy.sparse.csr):
        """
        calls the train method with the stored data
        Does not use X at all!
        """
        self._train(self.data)

    def predict(self, X: scipy.sparse.csr, user_ids=None):
        if self.lsh_model is None:
            raise RuntimeError("mux model has not been trained yet.")
        values = self.lsh_model @ X.T
        return values.T

    @property
    def name(self):
        return f"LSH_n_grams_{self.n_gram}_min_jacc_{self.min_jaccard}"
