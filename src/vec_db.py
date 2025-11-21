####### Builtin Modules #######
from typing import Dict, List, Annotated
import os
from abc import ABC, abstractmethod
import sys
######## Third-Party Modules #######
import numpy as np
######## Project Modules #######
from .scoring import CosineSimilarity
from .index.opq import OPQ
from .index.pq import PQ
from .index.ivf import IVF
from .index.disk import DiskIndex
from .index.retrieval_utils import Retriever
from .index.params import PARAMS
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64
########################################

class IDB(ABC):

    @abstractmethod
    def __init__(self, database_file_path, index_file_path, new_db, db_size) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        pass

class VecDB(IDB):
    
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db and db_size is not None:
            empty = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=(db_size, DIMENSION))
            empty.flush()

        if new_db:
            self._build_index()
        self.disk = DiskIndex(self.index_path)
    
    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)])-> None:
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)
            return np.array([])
        
    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def knn_retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5, score_fn = CosineSimilarity())-> List[int]:
        scores = []
        num_records = self._get_num_records()
        # here we assume that the row number is the ID of each vector
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            score = score_fn(query, vector)
            scores.append((score, row_num))
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]


    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5, score_fn = CosineSimilarity())-> List[int]:
        # Full pipeline: Normalize query -> Load OPQ rotation -> rotate query -> Load centroids -> compute nearest nprobe centroids -> Load PQ codebooks -> Build LUT -> Scan selected lists -> Select top_k by distance
        top_k_results = self.knn_retrieve(query, top_k, score_fn)
        return top_k_results

    def _build_index(self)-> None:
        print("Building index...")
        X = self.get_all_rows()
        # Full pipeline: Train OPQ -> rotate -> Train IVF -> assign -> compute residuals -> Train PQ -> encode residuals -> write disk index -> save metadata
        print("Index build completed")