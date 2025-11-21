PARAMS = {   
# IVF
"NLIST" :2048,
"NPROBE" : 8,

# PQ
"M" : 8,
"KS" : 256,  # 256 (2^8) clusters -> store each subvector in 1 byte

# Filenames
"OPQ_PATH" : "opq.npy",
"CENTROIDS_PATH" :"centroids.npy",
"PQ_PATH" : "pq.npy",
"INVLIST_PATH" : "invlists.bin",
"OFFSETS_PATH" : "offsets.npy",
"IDS_PATH" : "ids.bin",

# optional: RFlat rerank (IsA) 
"RERANK" : False,
"RERANK_L" : 2000
}