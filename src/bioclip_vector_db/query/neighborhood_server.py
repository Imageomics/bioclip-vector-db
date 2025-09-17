"""
Usage:
  python neighborhood_server.py --index_dir <path_to_index_dir> --index_file_prefix local_ --partitions 1,2,5-10 --nprobe 10
"""

import numpy as np
import faiss
import json
import logging
import sys
import argparse
import time
import functools


from typing import List, Dict
from flask import Flask, request, jsonify

from ..storage.metadata_storage import MetadataDatabase

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
        nprobe: int = 1,
        metadata_db=None,
    ):
        self._index_path_pattern = index_path_pattern
        self._indices = {}
        self._nprobe = nprobe

        if metadata_db is None:
            raise ValueError("metadata_db cannot be None")
        self._metadata_db = metadata_db

        self._load(neighborhood_ids)

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
    
    @timer
    def search(
        self, query_vector: list, top_n: int, nprobe: int = 1
    ) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
        """Performs a search on the loaded FAISS index."""
        assert all(
            [idx is not None for idx in self._indices.values()]
        ), "Index not loaded"

        results = {}
        for id in self._indices.keys():
            distances, indices = self._search(query_vector, top_n, id, nprobe)
            results[id] = (distances, self._map_to_original_ids(id, indices))

        return results

    def is_trained(self) -> bool:
        return all([idx.is_trained for idx in self._indices.values()])

    def total(self) -> int:
        return sum([idx.ntotal for idx in self._indices.values()])

    def dimensions(self) -> int:
        all_dims = [idx.d for idx in self._indices.values()]
        assert len(set(all_dims)) == 1, "All indices must have the same dimension"
        return all_dims[0]

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
        top_n = data.get("top_n", 10)
        nprobe = data.get("nprobe", self._service.get_nprobe())
        is_verbose = data.get("verbose", False)

        if "nprobe" in data:
            logger.info(f"Using nprobe override: {nprobe}")

        # Validate vector dimensions
        if len(query_vector) != self._service.dimensions():
            msg = f"Query vector has incorrect dimensions. Expected {self._service.dimensions()}, got {len(query_vector)}"
            return self._error_response(msg, 400)

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
    args = parser.parse_args()

    index_path_pattern = f"{args.index_dir}/{args.index_file_prefix}{{0}}.index"
    partitions = parse_partitions(args.partitions)

    metadata_db = MetadataDatabase(args.index_dir)

    svc = FaissIndexService(
        index_path_pattern, partitions, nprobe=args.nprobe, metadata_db=metadata_db
    )

    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 5001

    # 2. Initialize the server with the index service
    server = LocalIndexServer(service=svc)

    # 3. Run the server
    print(f"Starting server at http://{SERVER_HOST}:{SERVER_PORT}")
    server.run(host=SERVER_HOST, port=SERVER_PORT)


if __name__ == "__main__":
    __main__()
