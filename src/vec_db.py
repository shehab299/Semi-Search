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
from .index.retriever import Retriever
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
    
    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
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
    
    def knn_retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5, score_fn = CosineSimilarity()):
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


    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5, score_fn = CosineSimilarity()):
        retriever = Retriever(
            ivf=IVF.load(self.index_path + "_ivf.npy"),
            opq=OPQ(M=PARAMS.M).load(self.index_path + "_opq.npy"),
            pq=PQ(M=PARAMS.M, Ks=PARAMS.KS).load(self.index_path + "_pq.npy"),
            disk=DiskIndex(base_dir=self.index_path + "_disk"),
            params=PARAMS
        )
        return retriever.retrieve(query, top_k)

    def _build_index(self) -> None:
        # Placeholder for index building logic
        X = self.get_all_rows()
        # 1- train opq -> save it -> apply it
        # 2- train ivf --> save centroids
        # 3- train pq on residuals --> save it
        # 4- encode all rsiduals --> write invrted lists to disk
        pass