from recpack.algorithms.time_aware_item_knn.base import TARSItemKNN


class TARSItemKNNDing(TARSItemKNN):
    def __init__(self, K: int, predict_decay: float, similarity: str = "cosine"):
        super().__init__(K=K, fit_decay=0, predict_decay=predict_decay, similarity=similarity)
