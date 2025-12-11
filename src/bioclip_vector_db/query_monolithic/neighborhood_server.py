"""
A Flask server to handle FAISS index neighborhood search requests (SQLite Backend).

Usage:
    python src/bioclip_vector_db/query_monolithic/neighborhood_server.py \
        --index-path /fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/index_200M_normalized_stratified_merged.index \
        --id-lookup-path /fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/flight_plan/lookup.db \
        --nprobes 12 \
        --port 5001
"""

import numpy as np
import faiss
import logging
import argparse
import time
import functools
import os
import sqlite3
from typing import List, Dict, Any
from flask import Flask, request, jsonify


_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger()

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished '{func.__name__}' in {run_time:.4f} secs")
        return value
    return wrapper

class FaissIndexService:

    def __init__(
        self,
        index_path: str,
        id_lookup_path: str,
        use_gpu: bool = False,
        nprobes: int = 10
    ):
        self.index_path = index_path
        self.id_lookup_path = id_lookup_path # Now path to .db
        self.use_gpu = use_gpu
        self.nprobes = nprobes

        self.index = self._load_faiss_index()
        self._validate_lookup_db()

        if self.use_gpu:
            self._move_index_to_gpu()
    
    def _validate_lookup_db(self):
        if not os.path.exists(self.id_lookup_path):
            raise FileNotFoundError(f"SQLite database not found at {self.id_lookup_path}")
        logger.info(f"Lookup database validated at {self.id_lookup_path}")

    def _load_faiss_index(self):
        logger.info(f"Loading FAISS index from {self.index_path}")
        index = faiss.read_index(self.index_path)
        index.nprobe = self.nprobes
        logger.info("FAISS index loaded successfully")
        return index
    
    def reset_nprobes(self, nprobes: int):
        self.nprobes = nprobes
        self.index.nprobe = nprobes
        logger.info(f"Set nprobes to {nprobes}")
    
    def _move_index_to_gpu(self):
        GPU_ID = 0
        logger.info(f"Moving FAISS index to GPU device {GPU_ID}")
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, GPU_ID, self.index)
        self.gpu_res = res
        logger.info(f"FAISS index moved to GPU device {GPU_ID} successfully")

    @timer
    def query_metadata(self, ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Retrieves metadata from SQLite for a list of IDs.
        Returns a dictionary mapping {id: {col: val, ...}}
        """
        if not ids:
            return {}

        results = {}
        
        # Connect to SQLite in read-only mode
        uri = f"file:{self.id_lookup_path}?mode=ro"
        
        try:
            with sqlite3.connect(uri, uri=True) as conn:
                # Row factory allows accessing columns by name
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Chunking to prevent SQLite variable limit errors
                chunk_size = 900
                for i in range(0, len(ids), chunk_size):
                    chunk = ids[i:i + chunk_size]
                    placeholders = ','.join(['?'] * len(chunk))
                    
                    query = f"SELECT * FROM metadata WHERE id IN ({placeholders})"
                    
                    cursor.execute(query, chunk)
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        # Convert Row object to dict
                        row_dict = dict(row)
                        # Use the 'id' field as key for the results map
                        row_id = row_dict['id'] 
                        results[row_id] = row_dict

        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            
        logger.info(f"Retrieved {len(results)} metadata entries for {len(ids)} ids")
        return results
    
    def _format_search_results(
        self, 
        distances: np.ndarray, 
        indices: np.ndarray, 
        metadata_map: Dict[int, Dict[str, Any]]
    ) -> Dict[int, List[Dict]]:
        """
        Helper to merge FAISS results with metadata dictionary.
        """
        formatted_results = {}
        num_queries = indices.shape[0]
        num_neighbors = indices.shape[1]

        for q_idx in range(num_queries):
            query_results = []
            for n_idx in range(num_neighbors):
                faiss_id = int(indices[q_idx, n_idx])
                distance = float(distances[q_idx, n_idx])

                if faiss_id == -1:
                    continue

                # Retrieve metadata for this ID (default to empty dict if missing)
                meta = metadata_map.get(faiss_id, {})
                
                # Construct the result item
                # We prioritize the 'id' from FAISS, but merge in the metadata
                item = {
                    "id": faiss_id,
                    "distance": distance,
                    **meta
                }
                query_results.append(item)
            
            formatted_results[q_idx] = query_results
            
        return formatted_results

    @timer
    def search(
        self,
        query_vector: np.ndarray | List[np.ndarray],
        nprobes: int = 10,
        top_n: int = 10
    ):
        if nprobes != self.nprobes:
            self.reset_nprobes(nprobes)
        
        # Input Handling
        if not isinstance(query_vector, np.ndarray):
            queries = np.array(query_vector, dtype='float32')
        else:
            queries = query_vector.astype('float32')

        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
            
        n_query = queries.shape[0]
        
        logging.info(f"Searching for top {top_n} neighbors for {n_query} query vectors")
        faiss.normalize_L2(queries)

        # 1. FAISS Search
        distances, indices = self.index.search(queries, top_n)

        # 2. Metadata Lookup
        flat_indices = [idx for idx in indices.flatten().tolist() if idx != -1]
        unique_indices = list(set(flat_indices))
        
        metadata_map = self.query_metadata(unique_indices)
        
        # 3. Format Results
        logger.info(f"Search completed")
        return self._format_search_results(distances, indices, metadata_map)

    def is_trained(self) -> bool:
        return self.index.is_trained

    def total(self) -> int:
        return self.index.ntotal

    def dimensions(self) -> int:
        return self.index.d


class LocalIndexServer:
    """"A Flask server class to handle search and health check requests."""

    def __init__(self, service: FaissIndexService):
        self._app = Flask(__name__)
        self._service = service
        self._register_routes()

    def _register_routes(self):
        self._app.add_url_rule(
            "/search", "search", self.handle_search, methods=["POST"]
        )
        self._app.add_url_rule("/health", "health", self.handle_health, methods=["GET"])
    
    def _success_response(self, data, status_code=200):
        return jsonify({"status": "success", "data": data}), status_code

    def _error_response(self, message, status_code=400):
        return (
            jsonify(
                {"status": "error", "error": {"code": status_code, "message": message}}
            ),
            status_code,
        )   
    
    def handle_health(self):
        if self._service.is_trained():
            health_data = {
                "status": "ready",
                "vectors": self._service.total(),
                "dimensions": self._service.dimensions(),
            }
            return self._success_response(health_data)
        return self._error_response("Index not loaded or trained", 503)
    
    def handle_search(self):
        data = request.get_json()

        if not data or "query_vector" not in data:
            return self._error_response("Missing 'query_vector' in JSON body", 400)
        
        try:
            query_vector = data["query_vector"]
            nprobes = data.get("nprobes", 10)
            top_n = data.get("top_n", 10)

            results = self._service.search(
                query_vector=query_vector,
                nprobes=nprobes,
                top_n=top_n
            )

            return self._success_response(results)
        except Exception as e:
            logger.exception("Error during search")
            return self._error_response(f"Search error: {str(e)}", 500)
    
    def run(self, host: str, port: int):
        self._app.run(host=host, port=port)


def create_app(
    index_path: str,
    id_lookup_path: str,
    use_gpu: bool = False,
    nprobes: int = 12
) -> Flask:
    service = FaissIndexService(
        index_path=index_path,
        id_lookup_path=id_lookup_path,
        use_gpu=use_gpu,
        nprobes=nprobes
    )
    server = LocalIndexServer(service)
    return server._app

def __main__():
    parser = argparse.ArgumentParser(description="FAISS Neighborhood Server")
    parser.add_argument("--index-path", type=str, required=True, help="Path to the FAISS index file")
    parser.add_argument("--id-lookup-path", type=str, required=True, help="Path to the SQLite lookup DB")
    parser.add_argument("--nprobes", type=int, default=12, help="Number of clusters for FAISS search")
    parser.add_argument("--use-gpu", action="store_true", help="Whether to use GPU for FAISS index")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the server on")

    args = parser.parse_args()
    app = create_app(
        index_path=args.index_path,
        id_lookup_path=args.id_lookup_path,
        use_gpu=args.use_gpu,
        nprobes=args.nprobes
    )

    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = args.port

    logger.info(f"Starting FAISS Neighborhood Server on {SERVER_HOST}:{SERVER_PORT}")
    app.run(host=SERVER_HOST, port=SERVER_PORT)

if __name__ == "__main__":
    __main__()