import numpy as np
from .params import  OPQ_ITER, KS as PQ_KS, M as PQ_M
from .pq import PQ
class OPQ:
    def __init__(self, M: int = PQ_M, max_iter: int = OPQ_ITER):
        self.M = M
        self.max_iter = max_iter
        self.R = None

    def train(self, X: np.ndarray):
        # Simple iterative OPQ:
        # 1. rotate X -> Xr
        # 2. compute PQ codebooks on Xr residuals
        # 3. reconstruct Xr_hat
        # 4. solve Procrustes to get orthogonal R' minimizing || X @ R - X_recon || Solve using SVD on X^T Xr_hat
        N, D = X.shape
        new_R = np.eye(D, dtype=np.float32)
        for _ in range(self.max_iter):
            X_rotated = X @ new_R
            pq = PQ(M=PQ_M, Ks=PQ_KS, max_iter=PQ_M)
            pq.train(X_rotated)
            new_codes = pq.encode(X_rotated)
            reconstracted_codes = pq.decode(new_codes)
            A = X.T @ reconstracted_codes
            try:
                U, _, Vt = np.linalg.svd(A, full_matrices=False)
                R_new = U.dot(Vt)
            except np.linalg.LinAlgError:
                R_new = new_R
            new_R = R_new.astype(np.float32)

        self.R = new_R

    def apply(self, X: np.ndarray) -> np.ndarray:
        is_vector = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            is_vector = True
        X_rotated = X @ self.R
        return X_rotated[0] if is_vector else X_rotated

    def save(self, path: str):
        np.save(path, self.R)

    def load(self, path: str):
        self.R = np.load(path)
