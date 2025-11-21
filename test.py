import numpy as np
from src.vec_db import VecDB
# Create and build index for 1M vectors
db = VecDB(database_file_path="OpenSubtitles_en_1M_emb_64.dat", new_db=True, db_size=1_000_000)

# Query
query = np.random.rand(70).astype(np.float32)
results = db.retrieve(query, top_k=5)
print(f"Top 5 results: {results}")