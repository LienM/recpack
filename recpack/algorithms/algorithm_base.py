import scipy.sparse
from recpack.utils import get_logger


class Algorithm:

    def __init__(self):
        super().__init__()
        self.logger = get_logger()
        self._is_fit = False

    @property
    def name(self):
        return None

    def fit(self, X):
        pass

    def predict(self, X: scipy.sparse.csr_matrix):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    @property
    def is_fit(self):
        return self._is_fit
