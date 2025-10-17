######## Builtin Modules #######
from abc import ABC, abstractmethod
##### Third-Party Modules #######
import numpy as np
##### Project Modules #######
from .config import DB_SEED_NUMBER, DIMENSION
###################################

class DatabaseGenerator:

    def __init__(self, db_file_path = "saved_db.bat", db_size: int = 100_000) -> None:
        self.db_file_path = db_file_path
        self.db_size = db_size

    def generate(self) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random(size=(self.db_size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
    
    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(filename=self.db_file_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()




