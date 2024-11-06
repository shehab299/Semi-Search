from typing import Dict, List, Annotated
import numpy as np
import os

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDBWorst:
    def __init__(self, file_path = "saved_db.csv", new_db = True) -> None:
        self.file_path = file_path
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass
    
    def generate_database(self, size):
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)

        mmap_vectors = np.memmap(self.file_path, dtype=np.float32, mode='w+', shape=vectors.shape)

        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()
        self._build_index()

    def insert_records(self, rows: List[Annotated[List[float], 70]]):
        num_old_ecords = os.path.getsize(self.file_path) // DIMENSION // ELEMENT_SIZE
        num_new_records = len(rows)
        mmap_vectors = np.memmap(self.file_path, dtype=np.float32, mode='w+', shape=(num_old_ecords + num_new_records, DIMENSION))
        
        mmap_vectors[num_old_ecords:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index?
        self._build_index()

    def get_one_row(self, row_num):
        try:
            mmap_vector = np.memmap(self.file_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=row_num * DIMENSION * ELEMENT_SIZE)
            return mmap_vector[0]
        except Exception as e:
            return f"An error occurred: {e}"
        
    def get_all_rows(self):
        num_records = os.path.getsize(self.file_path) // DIMENSION // ELEMENT_SIZE
        vectors = np.memmap(self.file_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return vectors
    
    def retrieve(self, query: Annotated[List[float], DIMENSION], top_k = 5):
        scores = []
        num_records = os.path.getsize(self.file_path) // DIMENSION // ELEMENT_SIZE
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            score = self._cal_score(query, vector)
            scores.append((score, row_num))
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        pass


