PARAMS = {   
# IVF
"KMEANS_ITER" : 20,
"NLIST" :2048,
"NPROBE" : 8,

# PQ
"M" : 8,
"KS" : 256,  # 256 (2^8) clusters -> store each subvector in 1 byte

# OPQ
"OPQ_ITER" : 6,

# Filenames
"OPQ_PATH" : "opq_matrix.npy",
"CENTROIDS_PATH" :"ivf_centroids.npy",
"PQ_PATH" : "pq_codebooks.npy",
"INVLIST_PATH" : "invlists.bin",
"OFFSETS_PATH" : "offsets.npy",
"IDS_PATH" : "ids.bin",
"METADATA_FILE" : "metadata.npz",

# optional: RFlat rerank (IsA) 
"RERANK" : False,
"RERANK_L" : 2000
}