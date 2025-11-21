import numpy as np

class IVF:
    def __init__(self, nlist, max_iter=30):
        self.nlist = nlist
        self.max_iter = max_iter
        self.centroids = None

    def train(self, X: np.ndarray):
        # TODO: Run k-means to compute centroids shape (nlist, D)
        pass

    def assign(self, x: np.ndarray) -> int:
        nearest_centroid_index = 0
        # TODO:  return nearest centroid index for vector x
        return nearest_centroid_index

    def search(self, q: np.ndarray, nprobe: int) -> np.ndarray:
        top_nprobe_indices = np.zeros(nprobe, dtype=int)
        # TODO: compute distances q --> all centroids --> return top nprobe
        return top_nprobe_indices

    def save(self, path: str):
        np.save(path, self.centroids)

    def load(self, path: str):
        self.centroids = np.load(path)