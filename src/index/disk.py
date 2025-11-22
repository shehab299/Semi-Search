import numpy as np

class DiskIndex:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def write(self, clusters: list[list[int]], pq_codes: np.ndarray, M: int):
        inv_path = f"{self.base_dir}/invlists.bin"
        ids_path = f"{self.base_dir}/ids.bin"
        offsets_path = f"{self.base_dir}/offsets.npy"

        with open(inv_path, "wb") as finv, open(ids_path, "wb") as fids:
            offsets = []
            offset = 0

            for vectors in clusters:
                offsets.append(offset)

                for vid in vectors:
                    finv.write(bytes(pq_codes[vid].tolist()))
                    fids.write(np.int32(vid).tobytes())
                    offset += M
                    
        np.save(offsets_path, np.array(offsets, dtype=np.int64))


    def read_invlists(self):
        invlists = np.memmap(f"{self.base_dir}/invlists.bin", dtype=np.uint8, mode='r')
        return invlists

    def read_offsets(self):
        offsets = np.load(f"{self.base_dir}/offsets.npy", mmap_mode='r')
        return offsets

    def read_ids(self):
        ids = np.memmap(f"{self.base_dir}/ids.bin", dtype=np.int64, mode='r')
        return ids