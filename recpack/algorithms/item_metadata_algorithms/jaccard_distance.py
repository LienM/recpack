"""
In order to compute diversity, we need a prediction for distance between items.
"""
from scipy.spatial import distance
import scipy.sparse
class JaccardDictance:
    """
    Computes the Jaccard-Needham dissimilarity between 1-D boolean arrays u and v
    based on scipy.spatial.distance.jaccard
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html
    """

    def fit(self, X):
        self.X = X


        row_indices = []
        col_indices = []
        values = []
        for i in range(X.shape[0]):
            for j in range(i, X.shape[0]):
                row_indices.append(i)
                col_indices.append(j)
                dist = self.get_distance(i, j)
                values.append(dist)
                # Also fill the lower triangle
                row_indices.append(j)
                col_indices.append(i)
                values.append(dist)
        
        self._distances = scipy.sparse.csr_matrix((values, (row_indices, col_indices)), shape=(self.X.shape[0], self.X.shape[0]))



    def get_distance(self, ind_a, ind_b):
        return distance.jaccard(self.X[ind_a, :].toarray()[0], self.X[ind_b, :].toarray()[0])

    def predict(self, X):
        return X @ self._distances
