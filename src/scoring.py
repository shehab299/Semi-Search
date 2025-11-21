######## Builtin Modules #######
from abc import ABC, abstractmethod
##### Third-Party Modules #######
import numpy as np
##### Project Modules #######
################################

class ScoringFunction(ABC):

    @abstractmethod
    def __call__(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        pass


class CosineSimilarity(ScoringFunction):

    def __call__(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

class L2Distance(ScoringFunction):

    def __call__(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.sum((vec1 - vec2) ** 2))