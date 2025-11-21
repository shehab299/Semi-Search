import numpy as np

class IVF:
    def __init__(self, nlist, max_iter=30)-> None:
        self.nlist = nlist
        self.max_iter = max_iter
        self.centroids = None

    def train(self, X: np.ndarray) -> None:
        new_centroids = None
        # TODO: Run k-means to compute centroids with shape (nlist, D)
        self.centroids = new_centroids.astype(np.float32)

    def assign(self, x: np.ndarray) -> int:
        nearest_centroid_index = 0
        # TODO: return nearest centroid index using L2 distance
        return nearest_centroid_index

    def search(self, q: np.ndarray, nprobe: int) -> np.ndarray:
        top_nprobe_indices = np.zeros(nprobe, dtype=int)
        # TODO: return top closest nprobe centroids using L2 distance
        return top_nprobe_indices

    def save(self, path: str) -> None:
        np.save(path, self.centroids)

    def load(self, path: str) -> None:
        self.centroids = np.load(path)