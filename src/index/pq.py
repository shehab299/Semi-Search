import numpy as np

class PQ:
    def __init__(self, M, Ks, max_iter=20):
        self.M = M
        self.Ks = Ks
        self.max_iter = max_iter
        self.codebooks = None

    def train(self, residuals: np.ndarray):
        new_codebooks = np.zeros((self.M, self.Ks, residuals.shape[1] // self.M))
        # TODO: Split into M subspaces of size D/M ->kmeans for each subspace --> store centroids in self.codebooks in shape (M, Ks, D/M)
        self.codebooks = new_codebooks


    def encode(self, residuals: np.ndarray) -> np.ndarray:
        codes = np.zeros((residuals.shape[0], self.M), dtype=np.uint8)
        # TODO:  For each subvector --> find nearest centroid --> store its index as uint8
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        reconstructed_codes = np.zeros((codes.shape[0], self.M * (self.codebooks.shape[2])))
        # TODO: Use codebooks to reconstruct vectors (for optional reranking)
        return reconstructed_codes

    def save(self, path: str):
        np.save(path, self.codebooks)

    def load(self, path: str):
        self.codebooks = np.load(path)