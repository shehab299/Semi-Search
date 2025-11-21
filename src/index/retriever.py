import numpy as np

class Retriever:
    def __init__(self, ivf, opq, pq, disk, params):
        self.ivf = ivf
        self.opq = opq
        self.pq = pq
        self.disk = disk
        self.params = params

    def _build_LUT(self, q_rot):
        LUT = np.zeros((self.pq.M, self.pq.Ks))
        # Build the distance (or similarity) lookup table
        return LUT
    
    def _scan_lists(self, centroid_ids, LUT):
        candidates = []
        # for each centroid id in centroid_ids:
        #     read the corresponding inverted list from disk
        #     for each pq_code in the inverted list:
        #         compute approximate distance using LUT
        #        append (distance, vector_id) to candidates
        return candidates
    
    def rflat_rerank(self, query_vector_rotated: np.ndarray, top_l_indices: np.ndarray, db_reader_func: callable, cosine_similarity_func: callable, k: int):        
        final_scores = []
        final_indices = []
        return final_indices, final_scores
    
    def retrieve(self, query, top_k):
        q_rot = self.opq.apply(query)
        centroid_ids = self.ivf.search(q_rot, self.params.NPROBE)
        LUT = self._build_LUT(q_rot)
        candidates = self._scan_lists(centroid_ids, LUT)
        top = sorted(candidates, key=lambda x: x[0])[:top_k]
        # optional rflat reranking 
        return [t[1] for t in top]