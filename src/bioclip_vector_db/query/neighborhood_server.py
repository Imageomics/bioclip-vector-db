import numpy as np
import faiss
import json
import logging
import sys

from typing import List, Dict
from flask import Flask, request, jsonify

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger()


class FaissIndexService:
    def __init__(self, index_path_pattern: str, neighborhood_ids: List[int]):
        self._index_path_pattern = index_path_pattern
        self._indices = {}

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
                self._indices[neighborhood_ids[i]] = faiss.read_index(file)
            self.dimensions()
        except Exception as e:
            logger.error(
                f"FATAL: Error loading index file. Ensure it is a valid FAISS index. Details: {e}"
            )
            sys.exit(1)

    def _search(self, query_vector: list, top_n: int, neighborhood_id: int):
        """Performs a search on the loaded FAISS local index."""
        query_np = np.array([query_vector]).astype("float32")
        return self._indices[neighborhood_id].search(query_np, top_n)

    def search(
        self, query_vector: list, top_n: int
    ) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
        """Performs a search on the loaded FAISS index."""
        assert all(
            [idx is not None for idx in self._indices.values()]
        ), "Index not loaded"

        results = {}
        for id in self._indices.keys():
            distances, indices = self._search(query_vector, top_n, id)
            results[id] = (distances, indices)
        return results

    def is_trained(self) -> bool:
        return all([idx.is_trained for idx in self._indices.values()])

    def total(self) -> int:
        return sum([idx.ntotal for idx in self._indices.values()])

    def dimensions(self) -> int:
        all_dims = [idx.d for idx in self._indices.values()]
        assert len(set(all_dims)) == 1, "All indices must have the same dimension"
        return all_dims[0]


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

    def handle_search(self):
        """Handler for the /search endpoint."""
        data = request.get_json()

        if not data or "query_vector" not in data:
            return self._error_response("Missing 'query_vector' in JSON body", 400)

        query_vector = data["query_vector"]
        top_n = data.get("top_n", 10)

        # Validate vector dimensions
        if len(query_vector) != self._service.dimensions():
            msg = f"Query vector has incorrect dimensions. Expected {self._service.dimensions()}, got {len(query_vector)}"
            return self._error_response(msg, 400)

        try:
            results = self._service.search(query_vector, top_n)

            # Format the raw FAISS results into a more descriptive list of objects
            formatted_results = []
            for index_id, (distances, indices) in results.items():
                matches = [
                    {"id": int(idx), "distance": float(dist)}
                    for dist, idx in zip(distances[0], indices[0])
                ]
                formatted_results.append({"index_id": index_id, "matches": matches})

            return self._success_response({"results": formatted_results})

        except Exception as e:
            logger.error(f"An error occurred during search: {e}", exc_info=True)
            return self._error_response(
                "An internal server error occurred during search", 500
            )

    def run(self, host: str, port: int):
        """Starts the Flask server."""
        self._app.run(host=host, port=port)


def __main__():
    # 1. Initialize the index service
    svc = FaissIndexService(
        "/Users/sreejithnoopur/codebase/faiss_index/local_{0}.index", [0, 1, 2, 3]
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
