import os
from bioclip_vector_db.query.neighborhood_server import create_app

# Required environment variables
INDEX_DIR = os.environ.get("INDEX_DIR")
INDEX_FILE_PREFIX = os.environ.get("INDEX_FILE_PREFIX")
LEADER_INDEX = os.environ.get("LEADER_INDEX")
PARTITIONS = os.environ.get("PARTITIONS")

# Optional environment variables
NPROBE = int(os.environ.get("NPROBE", 1))
USE_CACHE = os.environ.get("USE_CACHE", "False").lower() in ("true", "1", "t")
PORT = int(os.environ.get("PORT", 5001))

if not all([INDEX_DIR, INDEX_FILE_PREFIX, LEADER_INDEX, PARTITIONS]):
    raise ValueError("Missing one or more required environment variables: INDEX_DIR, INDEX_FILE_PREFIX, LEADER_INDEX, PARTITIONS")

app = create_app(
    index_dir=INDEX_DIR,
    index_file_prefix=INDEX_FILE_PREFIX,
    leader_index=LEADER_INDEX,
    nprobe=NPROBE,
    partitions_str=PARTITIONS,
    use_cache=USE_CACHE,
)
