

import numpy as np
import faiss
import json
import logging
import sys

from flask import Flask, request, jsonify

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger()

class FaissIndexService:
    def __init__(self, index_path: str):
        self._index_path = index_path
        self._load()

    def _load(self):
        """Loads the FAISS index from the specified path."""
        try:
            logging.info(f"Loading FAISS index from {self._index_path}...")
            self._index = faiss.read_index(self._index_path)
            logger.info(f"Index loaded successfully. Vectors: {self._index.ntotal}, Dimensions: {self._index.d}")
        except Exception as e:
            logger.error(f"FATAL: Error loading index file. Ensure it is a valid FAISS index. Details: {e}")
            sys.exit(1)
            
    def search(self, query_vector: list, top_n: int) -> tuple[np.ndarray, np.ndarray]:
        """Performs a search on the loaded FAISS index."""
        if self._index is None:
            raise RuntimeError("Index is not loaded.")
        
        query_np = np.array([query_vector]).astype('float32')
        return self._index.search(query_np, top_n)
    
    def is_trained(self) -> bool:
        return self._index.is_trained
    
    def total(self) -> int:
        return self._index.ntotal
    
    def dimensions(self) -> int:
        return self._index.d
    
class LocalIndexServer:
    """A Flask server class to handle search and health check requests."""

    def __init__(self, service: FaissIndexService):
        self._app = Flask(__name__)
        self._service = service
        self._register_routes()

    def _register_routes(self):
        """Registers the URL routes for the server."""
        self._app.add_url_rule('/search', 'search', self.handle_search, methods=['POST'])
        self._app.add_url_rule('/health', 'health', self.handle_health, methods=['GET'])

    def handle_health(self):
        """Handler for the /health endpoint."""
        if self._service.is_trained():
            return jsonify({"status": "ok", "vectors": self._service.total(), "dimensions": self._service.dimensions()}), 200
        return jsonify({"status": "error", "message": "Index not loaded or trained"}), 500

    def handle_search(self):
        """Handler for the /search endpoint."""
        data = request.get_json()
        
        if not data or 'query_vector' not in data:
            return jsonify({"error": "Missing 'query_vector' in JSON body"}), 400
        
        query_vector = data['query_vector']
        top_n = data.get('top_n', 10)

        # Validate vector dimensions
        if len(query_vector) != self._service.dimensions():
            msg = f"Query vector has incorrect dimensions. Expected {self._service.dimensions}, got {len(query_vector)}"
            return jsonify({"error": msg}), 400
        
        try:
            distances, indices = self._service.search(query_vector, top_n)
            results = {
                "indices": indices[0].tolist(),
                "distances": distances[0].tolist()
            }
            return jsonify(results)
        except Exception as e:
            print(f"An error occurred during search: {e}")
            return jsonify({"error": "An internal server error occurred"}), 500

    def run(self, host: str, port: int):
        """Starts the Flask server."""
        self._app.run(host=host, port=port)

def __main__():
    # 1. Initialize the index service
    svc = FaissIndexService("/Users/sreejithnoopur/codebase/faiss_index/local_12.index")

    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 5001
    
    # 2. Initialize the server with the index service
    server = LocalIndexServer(service=svc)

    # 3. Run the server
    print(f"Starting server at http://{SERVER_HOST}:{SERVER_PORT}")
    server.run(host=SERVER_HOST, port=SERVER_PORT)


if __name__ == "__main__":
    __main__()
