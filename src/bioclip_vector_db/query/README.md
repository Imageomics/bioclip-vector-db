# Neighborhood Server

This server provides an API to search for similar vectors in a FAISS index.

## Running the server

There are two ways to run the server:

1.  **Directly with Python (for development)**
2.  **With Gunicorn (for production)**

### 1. Running with Python

You can run the server directly using the `neighborhood_server.py` script. You must provide the configuration as command-line arguments.

**Usage:**

```bash
python src/bioclip_vector_db/query/neighborhood_server.py --index_dir <path_to_index_dir> --index_file_prefix <prefix> --leader_index <leader_index_file> --partitions <partitions> [options]
```

**Example:**

```bash
python src/bioclip_vector_db/query/neighborhood_server.py \
    --index_dir /path/to/faiss_index \
    --index_file_prefix local_ \
    --leader_index leader.index \
    --partitions "1,2,5-10" \
    --nprobe 10 \
    --port 5001
```

### 2. Running with Gunicorn

For production, it is recommended to use a WSGI server like Gunicorn.

**Prerequisites:**

*   Install Gunicorn: `pip install gunicorn`
*   Ensure all dependencies from `requirements.txt` are installed.

**Configuration:**

The Gunicorn server is configured using environment variables.

**Required:**

*   `INDEX_DIR`: Directory where the index files are stored.
*   `INDEX_FILE_PREFIX`: The prefix of the index files (e.g., `local_`).
*   `LEADER_INDEX`: The leader index file, which contains all the centroids.
*   `PARTITIONS`: List of partition numbers to load (e.g., `"1,2,5-10"`).

**Optional:**

*   `NPROBE`: Number of inverted list probes (default: `1`).
*   `USE_CACHE`: Enable lazy loading cache (default: `False`).
*   `PORT`: Port to run the server on (default: `5001`).
*   `WORKERS`: Number of Gunicorn worker processes (default: `4`).

**Running the command:**

From the `bioclip-vector-db` directory, run the following command:

```bash
export INDEX_DIR=/path/to/faiss_index
export INDEX_FILE_PREFIX=local_
export LEADER_INDEX=leader.index
export PARTITIONS="1,2,5-10"
export NPROBE=10
export PORT=5001

gunicorn --workers ${WORKERS:-4} --bind 0.0.0.0:${PORT} --chdir src bioclip_vector_db.query.wsgi:app
```