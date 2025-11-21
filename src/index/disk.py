import numpy as np

class DiskIndex:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def write(self, cluster_ids: list, pq_codes: np.ndarray, M: int):
        """
            - Write invlists.bin as raw bytes
            - Write offsets.npy
            - Write ids.bin with vector IDs
        """
        pass

    def read_invlists(self):
        invlists = np.memmap(f"{self.base_dir}/invlists.bin", dtype=np.uint8, mode='r')
        return invlists

    def read_offsets(self):
        offsets = np.load(f"{self.base_dir}/offsets.npy", mmap_mode='r')
        return offsets

    def read_ids(self):
        ids = np.memmap(f"{self.base_dir}/ids.bin", dtype=np.int64, mode='r')
        return ids
