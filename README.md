# Semantic Search Engine with Vectorized Databases
This repository contains the code and documentation for a simple semantic search engine with vectorized databases and the evaluation of its performance. The project focuses on building an efficient indexing system to retrieve information based on vector space embeddings.

## Project Overview

The key components of the project include:
- `VecDB`: A class representing the vectorized database, responsible for storing and retrieving vectors.
- `generate_database()`: A method to generate a random database.
- `get_one_row()`: A method to get one row from the database given its index.
- `insert_records()`: A method to insert multiple records into the database. It then rebuilds the index.
- `retrieve()`: A method to retrieve the top-k most similar based on a given query vector.
- `_cal_score()`: A helper method to calculate the cosine similarity between two vectors.
- `_build_index()`: A placeholder method for implementing an indexing mechanism.



