"""
Usage:
  python neighborhood_server.py --index_dir <path_to_index_dir> --index_file_prefix local_ --partitions 1,2,5-10 --nprobe 10
"""

import numpy as np
import faiss
import os
import json
import logging
import sys
import argparse
import time
import functools
import random
from collections import OrderedDict


from typing import List, Dict
from flask import Flask, request, jsonify

from ..storage.metadata_storage import MetadataDatabase

# Maximum number of neighborhoods to load in memory.
MAX_CACHE_SIZE = 10
_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger()


def timer(func):
    """A decorator that prints the time a function takes to run."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Record the start time
        start_time = time.perf_counter()

        # Call the original function and store its result
        value = func(*args, **kwargs)

        # Record the end time and calculate the duration
        end_time = time.perf_counter()
        run_time = end_time - start_time

        # Print the duration
        logger.info(f"Finished '{func.__name__}' in {run_time:.4f} secs")

        # Return the original function's result
        return value

    return wrapper


class FaissIndexService:
    def __init__(
        self,
        index_path_pattern: str,
        neighborhood_ids: List[int],
        leader_index_path: str,
        nprobe: int = 1,
        metadata_db=None,
        use_cache=False,
    ):
        """
        Initializes the FaissIndexService.

        Args:
            index_path_pattern (str): A string pattern for the local index files,
                e.g., 'index_{}.faiss'. The '{}' is a placeholder for the neighborhood_id.
            neighborhood_ids (List[int]): A list of neighborhood IDs to load.
            leader_index_path (str): The path to the leader index file.
            nprobe (int, optional): The number of partitions to search. Defaults to 1.
            metadata_db (MetadataDatabase, optional): An instance of MetadataDatabase.
                Defaults to None.
            use_cache (bool, optional): Whether to use caching. Defaults to False. If enabled, 
                lazy loading of the local neighborhoods will happen.
        """
        self._index_path_pattern = index_path_pattern
        self._indices = OrderedDict()
        self._nprobe = nprobe
        self._use_cache = use_cache

        if self._use_cache and self._nprobe > MAX_CACHE_SIZE:
            raise ValueError(f"nprobe cannot be greater than MAX_CACHE_SIZE: {MAX_CACHE_SIZE}")
            

        if leader_index_path is None or not os.path.exists(leader_index_path):
            logger.error(f"Loading leader index from: {leader_index_path}")
            raise ValueError(
                f"Leader index not found or is not set: {leader_index_path}"
            )

        self._leader_index = faiss.read_index(leader_index_path)
        if not hasattr(self._leader_index, "quantizer"):
            raise ValueError(
                "Loaded leader index does not have the quantizer attribute."
            )

        if metadata_db is None:
            raise ValueError("metadata_db cannot be None")
        self._metadata_db = metadata_db

        if not use_cache:
            print(neighborhood_ids)
            self._load(neighborhood_ids)
        else:
            self._cache_miss = 0
            self._cache_hits = 0
            self._cache_evictions = 0

    def _load(self, neighborhood_ids: List[int]):
        """Loads the FAISS index from the specified path."""

        assert self._index_path_pattern is not None, "Index path pattern cannot be None"
        assert neighborhood_ids is not None, "Neighborhood IDs cannot be None"
        assert len(neighborhood_ids) > 0, "Neighborhood IDs cannot be empty"

        try:
            for i in range(len(neighborhood_ids)):
                file = self._index_path_pattern.format(neighborhood_ids[i])
                logger.info(f"Loading index file: {file}")
                index = faiss.read_index(file)
                index.nprobe = self._nprobe
                self._indices[neighborhood_ids[i]] = index
            self.dimensions()
        except Exception as e:
            logger.error(
                f"FATAL: Error loading index file. Ensure it is a valid FAISS index. Details: {e}"
            )
            sys.exit(1)

    def _load_with_cache(self, neighborhood_ids: List[int]):
        """
        Loads neighborhood indices into an LRU cache.
        If the cache is full, it evicts the least recently used item.
        """
        for neighborhood_id in neighborhood_ids:
            if neighborhood_id in self._indices:
                self._cache_hits += 1
                self._indices.move_to_end(neighborhood_id)
                logger.info(f"Neighborhood {neighborhood_id} already in cache. Moved to most recently used.")
                continue
            
            self._cache_miss += 1
            if len(self._indices) >= MAX_CACHE_SIZE:
                self._cache_evictions += 1
                evicted_id, _ = self._indices.popitem(last=False)
                logger.info(f"Cache full. Evicting neighborhood: {evicted_id}")

            file_path = self._index_path_pattern.format(neighborhood_id)
            logger.info(f"Loading index file into cache: {file_path}")
            try:
                index = faiss.read_index(file_path)
                index.nprobe = self._nprobe
                self._indices[neighborhood_id] = index
            except Exception as e:
                logger.error(f"Error loading index file {file_path}: {e}")

    def _search(
        self, query_vector: list, top_n: int, neighborhood_id: int, nprobe: int = 1
    ):
        """Performs a search on the loaded FAISS local index."""
        query_np = np.array([query_vector]).astype("float32")
        index = self._indices[neighborhood_id]
        index.nprobe = nprobe
        return index.search(query_np, top_n)

    def _map_to_original_ids(self, neighborhood_id: int, local_indices: Dict) -> Dict:
        """Maps local indices to original IDs."""
        return list(
            map(
                lambda id: self._metadata_db.get_original_id(neighborhood_id, id),
                local_indices[0],
            )
        )

    def _search_leader(self, query_vector: list, top_n: int) -> np.ndarray:
        """Performs a search on the loaded FAISS leader index."""
        query_np = np.array([query_vector]).astype("float32")
        _, partition_ids = self._leader_index.quantizer.search(query_np, top_n)
        return partition_ids[0]

    @timer
    def search(
        self, query_vector: list, top_n: int, nprobe: int = 1
    ) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Performs a search on the loaded FAISS index.
        This method first queries the leader index to identify the most relevant partitions,
        and then searches within those partitions if they are loaded.
        """
        assert all(
            [idx is not None for idx in self._indices.values()]
        ), "Index not loaded"

        # Query the leader index to find the most relevant partitions.
        # The `nprobe` parameter determines how many partitions to check.
        partition_ids_to_search = self._search_leader(query_vector, nprobe)
        logger.info(f"Leader search returned partitions: {partition_ids_to_search}")

        if self._use_cache:
            self._load_with_cache(partition_ids_to_search)

        results = {}
        for partition_id in partition_ids_to_search:
            # Check if the partition's index is loaded in this server.
            if partition_id in self._indices:
                logger.info(f"Searching in loaded partition: {partition_id}")

                distances, indices = self._search(
                    query_vector, top_n, partition_id, nprobe
                )
                results[partition_id] = (
                    distances,
                    self._map_to_original_ids(partition_id, indices),
                )
            else:
                logger.warning(
                    f"Partition {partition_id} not loaded in this server, skipping."
                )

        return results

    def is_trained(self) -> bool:
        return all([idx.is_trained for idx in self._indices.values()])

    def total(self) -> int:
        if self._indices.values():
            return sum([idx.ntotal for idx in self._indices.values()])
        return 0

    def dimensions(self) -> int:
        if self._indices.values():
            all_dims = [idx.d for idx in self._indices.values()]
            assert len(set(all_dims)) == 1, "All indices must have the same dimension"
            return all_dims[0]
        return 0

    def get_nprobe(self) -> int:
        return self._nprobe


class LocalIndexServer:
    """A Flask server class to handle search and health check requests."""

    def __init__(self, service: FaissIndexService):
        self._app = Flask(__name__)
        self._service = service
        self._register_routes()

    def _register_routes(self):
        """Registers the URL routes for the server."""
        self._app.add_url_rule(
            "/search", "search", self.handle_search, methods=["POST"]
        )
        self._app.add_url_rule("/health", "health", self.handle_health, methods=["GET"])

    def _success_response(self, data, status_code=200):
        """Generates a structured success JSON response."""
        return jsonify({"status": "success", "data": data}), status_code

    def _error_response(self, message, status_code=400):
        """Generates a structured error JSON response."""
        return (
            jsonify(
                {"status": "error", "error": {"code": status_code, "message": message}}
            ),
            status_code,
        )

    def handle_health(self):
        """
        Handler for the /health endpoint.
        Returns the status of the index service in a structured format.
        """
        if self._service.is_trained():
            health_data = {
                "status": "ready",
                "vectors": self._service.total(),
                "dimensions": self._service.dimensions(),
            }
            if self._service._use_cache:
                health_data["cache"] = {
                    "hits": self._service._cache_hits,
                    "misses": self._service._cache_miss,
                    "evictions": self._service._cache_evictions,
                }
            return self._success_response(health_data)

        # Use 503 Service Unavailable when the service is not ready
        return self._error_response("Index not loaded or trained", 503)

    def _handle_merging(self, results):
        all_matches = [
            match for matches_dict in results for match in matches_dict["matches"]
        ]
        return sorted(all_matches, key=lambda item: item["distance"])

    def handle_search(self):
        """Handler for the /search endpoint."""
        data = request.get_json()

        if not data or "query_vector" not in data:
            return self._error_response("Missing 'query_vector' in JSON body", 400)

        query_vector = data["query_vector"]
        # number of neighbors to return from each local neighborhood.
        top_n = data.get("top_n", 10)
        nprobe = data.get("nprobe", self._service.get_nprobe())
        is_verbose = data.get("verbose", False)

        if "nprobe" in data:
            logger.info(f"Using nprobe override: {nprobe}")

        try:
            results = self._service.search(query_vector, top_n, nprobe)

            # Format the raw FAISS results into a more descriptive list of objects
            formatted_results = []
            for partition_id, (distances, indices) in results.items():
                matches = [
                    {"id": idx, "distance": float(dist)}
                    for dist, idx in zip(distances[0], indices)
                ]
                formatted_results.append(
                    {"partition_id": partition_id, "matches": matches}
                )

            merged_neighbors = self._handle_merging(formatted_results)

            if is_verbose:
                return self._success_response(
                    {"results": formatted_results, "merged_neighbors": merged_neighbors}
                )
            else:
                return self._success_response({"merged_neighbors": merged_neighbors})

        except Exception as e:
            logger.error(f"An error occurred during search: {e}", exc_info=True)
            return self._error_response(
                "An internal server error occurred during search", 500
            )

    def run(self, host: str, port: int):
        """Starts the Flask server."""
        self._app.run(host=host, port=port)


def parse_partitions(partition_str: str) -> List[int]:
    partitions = set()
    for part in partition_str.split(","):
        if "-" in part:
            start, end = part.split("-")
            partitions.update(range(int(start), int(end) + 1))
        else:
            partitions.add(int(part))
    if len(partitions) == 0:
        raise ValueError(
            "Invalid arguments:No partitions specified or partitions specified could not be parsed."
        )
    return sorted(list(partitions))


def create_app(
    index_dir: str,
    index_file_prefix: str,
    leader_index: str,
    nprobe: int,
    partitions_str: str,
    use_cache: bool,
):
    """Creates and configures the Flask application."""
    index_path_pattern = f"{index_dir}/{index_file_prefix}{{}}.index"
    leader_index_path = f"{index_dir}/{leader_index}"
    partitions = parse_partitions(partitions_str)

    metadata_db = MetadataDatabase(index_dir)

    svc = FaissIndexService(
        index_path_pattern,
        partitions,
        leader_index_path,
        nprobe=nprobe,
        metadata_db=metadata_db,
        use_cache=use_cache,
    )

    server = LocalIndexServer(service=svc)
    return server._app


def __main__():
    parser = argparse.ArgumentParser(description="FAISS Neighborhood Server")
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory where the index files are stored",
    )
    parser.add_argument(
        "--index_file_prefix",
        type=str,
        required=True,
        help="The prefix of the index files (e.g., 'local_')",
    )
    parser.add_argument(
        "--leader_index",
        type=str,
        required=True,
        help="The leader index file, which contains all the centroids",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=1,
        help="Number of inverted list probes to use for the FAISS search. A higher value increases search accuracy at the cost of slower query time",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        required=True,
        help="List of partition numbers to load (e.g., '1,2,5-10')",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=False,
        help="Flag to enable the cache, will use lazy loading.",
    )
    args = parser.parse_args()

    app = create_app(
        index_dir=args.index_dir,
        index_file_prefix=args.index_file_prefix,
        leader_index=args.leader_index,
        nprobe=args.nprobe,
        partitions_str=args.partitions,
        use_cache=args.use_cache,
    )

    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = args.port

    print(f"Starting server at http://{SERVER_HOST}:{SERVER_PORT}")
    app.run(host=SERVER_HOST, port=SERVER_PORT)


if __name__ == "__main__":
    __main__()