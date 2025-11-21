import numpy as np

def build_LUT(q_rot: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    D = q_rot.shape[0]
    M = codebooks.shape[0]
    subdim = codebooks.shape[2]
    q_split = q_rot.reshape(M, subdim)
    LUT = np.empty((M, codebooks.shape[1]), dtype=np.float32)
    for m in range(M):
        qsub = q_split[m]
        cb = codebooks[m]
        dists = np.sum((cb - qsub[None, :]) ** 2, axis=1)
        LUT[m] = dists.astype(np.float32)
    return LUT

def pq_distance(code: np.ndarray, LUT: np.ndarray) -> float:
    return float(np.sum(LUT[np.arange(LUT.shape[0]), code.astype(np.int32)]))

def scan_lists(disk, centroid_ids, LUT, M):
    inv = disk.read_invlists()
    offsets = disk.read_offsets()
    ids = disk.read_ids()
    candidates = []
    total_bytes = inv.shape[0]
    nlist = offsets.shape[0]
    for idx, cid in enumerate(centroid_ids):
        start = int(offsets[cid])
        if cid == nlist - 1:
            end = total_bytes
        else:
            end = int(offsets[cid+1]) if (cid+1 < len(offsets)) else total_bytes
        if end <= start:
            continue
        list_size = (end - start) // M
        id_base_pos = start // M
        for i in range(list_size):
            code_pos = start + i * M
            code = np.array(inv[code_pos: code_pos + M], dtype=np.uint8)
            dist = pq_distance(code, LUT)
            vec_id = int(ids[id_base_pos + i])
            candidates.append((dist, vec_id))
    return candidates