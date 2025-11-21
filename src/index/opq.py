import numpy as np

class OPQ:
    def __init__(self, M, max_iter=20):
        self.M = M
        self.max_iter = max_iter
        self.R = None

    def train(self, X: np.ndarray):
        # Simple iterative OPQ:
        # 1. rotate X -> Xr
        # 2. compute PQ codebooks on Xr residuals (we will use PQ on Xr directly)
        # 3. reconstruct Xr_hat
        # 4. solve Procrustes to get R_new that maps X -> Xr_hat
        # Note: For simplicity and to keep runtime reasonable, this implementation uses full PQ on Xr.
        
        new_R = np.eye(X.shape[1])
        # TODO:Compute iterative OPQ rotation matrix (D, D) and save it in self.R
        self.R = new_R
        pass

    def apply(self, X: np.ndarray) -> np.ndarray:
        is_vector = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            is_vector = True
        X_rotated = X @ self.R
        return X_rotated.flatten() if is_vector else X_rotated

    def save(self, path: str):
        np.save(path, self.R)

    def load(self, path: str):
        self.R = np.load(path)
