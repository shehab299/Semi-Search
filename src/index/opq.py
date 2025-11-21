import numpy as np

class OPQ:
    def __init__(self, M, max_iter=20):
        self.M = M
        self.max_iter = max_iter
        self.R = None

    def train(self, X: np.ndarray):
        new_R = np.eye(X.shape[1])
        # TODO:Compute iterative OPQ rotation matrix (D, D) and save it in self.R
        self.R = new_R
        pass

    def apply(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R

    def save(self, path: str):
        np.save(path, self.R)

    def load(self, path: str):
        self.R = np.load(path)
