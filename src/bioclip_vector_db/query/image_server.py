"""
Image Retrieval Server (SQLite Backend)
Serves images stored in HDF5 files based on UUID lookups from a SQLite database.
Supports multiprocessing for efficient retrieval.

Usage:
    python src/bioclip_vector_db/query/image_server.py \
        --lookup_path /fs/scratch/PAS2136/TreeOfLife/image_lookup/2024-05-01/hdf5/200M/lookup_temp.db \
        --port 5002 \
        --workers 16 \
        --h5_group images
"""
import base64
import logging
import argparse
import functools
import time
import os
import sqlite3
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

def timer(func):
    """A decorator that prints the time a function takes to run."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished '{func.__name__}' in {run_time:.4f} secs")
        return value
    return wrapper

class ImageRetrievalService:
    def __init__(self, lookup_path: str, h5_group: str = "images", max_workers: int = 4):
        """
        Initializes the ImageRetrievalService.

        Args:
            lookup_path: Path to the SQLite database (.db) mapping UUIDs to H5 paths.
            h5_group: The group name within H5 files where images are stored.
            max_workers: Maximum number of threads to use for parallel H5 file reading.
        """
        self.lookup_path = lookup_path
        self.h5_group = h5_group
        self.max_workers = max_workers
        
        if not os.path.exists(self.lookup_path):
            raise FileNotFoundError(f"SQLite database not found at: {self.lookup_path}")
        
        # Test connection
        try:
            with sqlite3.connect(f"file:{self.lookup_path}?mode=ro", uri=True) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
            logger.info(f"Successfully connected to SQLite DB at {self.lookup_path}")
        except Exception as e:
            logger.error(f"Failed to connect to SQLite DB: {e}")
            raise

    @timer
    def _query_lookup(self, uuids: List[str]) -> Dict[str, str]:
        """
        Queries the SQLite DB for the given UUIDs.
        Returns a Dict mapping uuid -> h5_path.
        """
        if not uuids:
            return {}

        results = {}
        # SQLite limits the number of host parameters (variables in IN clause).
        # We process in chunks (e.g., 900) to be safe across all SQLite versions.
        chunk_size = 900 
        
        uri = f"file:{self.lookup_path}?mode=ro"
        
        try:
            with sqlite3.connect(uri, uri=True) as conn:
                cursor = conn.cursor()
                
                # Iterate through chunks of UUIDs
                for i in range(0, len(uuids), chunk_size):
                    chunk = uuids[i:i + chunk_size]
                    placeholders = ','.join(['?'] * len(chunk))
                    query = f"SELECT uuid, h5_path FROM lookup WHERE uuid IN ({placeholders})"
                    
                    cursor.execute(query, chunk)
                    rows = cursor.fetchall()
                    
                    for r in rows:
                        results[r[0]] = r[1]
                        
        except sqlite3.Error as e:
            logger.error(f"SQLite error during query: {e}")
            
        return results
    
    @timer
    def _read_h5_file(self, h5_path: str, uuids: List[str]) -> Dict[str, Optional[str]]:
        """
        Reads images for a list of UUIDs from a single H5 file.
        Returns a dictionary mapping UUID -> Base64 encoded image string (or None).
        """
        results = {}
        try:
            with h5py.File(h5_path, "r") as f:
                if self.h5_group not in f:
                    logger.warning(f"Group '{self.h5_group}' not found in {h5_path}")
                    for uuid in uuids:
                        results[uuid] = None
                    return results

                group = f[self.h5_group]
                for uuid in uuids:
                    if uuid in group:
                        try:
                            # Read raw bytes
                            raw_bytes = group[uuid][()]
                            # Convert to base64 string
                            b64_str = base64.b64encode(raw_bytes).decode('utf-8')
                            results[uuid] = b64_str
                        except Exception as e:
                            logger.error(f"Error reading image {uuid} from {h5_path}: {e}")
                            results[uuid] = None
                    else:
                        results[uuid] = None
        except Exception as e:
            logger.error(f"Failed to open H5 file {h5_path}: {e}")
            for uuid in uuids:
                results[uuid] = None
        
        return results
    
    @timer
    def retrieve_images(self, uuids: List[str]) -> Dict[str, Optional[str]]:
        """
        Main method to retrieve images for a list of UUIDs.
        """
        # 1. Query SQLite to find where the images are
        uuid_path_map = self._query_lookup(uuids)
        
        if not uuid_path_map:
            logger.info("No UUIDs found in lookup table.")
            return {uuid: None for uuid in uuids}

        # Group UUIDs by H5 file
        file_groups = {}
        for uuid, h5_path in uuid_path_map.items():
            if h5_path not in file_groups:
                file_groups[h5_path] = []
            file_groups[h5_path].append(uuid)

        # Initialize results with None (handles missing UUIDs)
        results = {uuid: None for uuid in uuids} 
        
        # 2. Read from H5 files in parallel
        # Only process files that were actually found in the lookup
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._read_h5_file, h5_path, group_uuids): h5_path
                for h5_path, group_uuids in file_groups.items()
            }
            
            for future in as_completed(future_to_file):
                h5_path = future_to_file[future]
                try:
                    file_results = future.result()
                    results.update(file_results)
                except Exception as e:
                    logger.error(f"Worker failed for file {h5_path}: {e}")

        return results

class ImageServer:
    def __init__(self, service: ImageRetrievalService):
        self.app = Flask(__name__)
        self.service = service
        self._register_routes()

    def _register_routes(self):
        self.app.add_url_rule('/images', 'get_images', self.handle_get_images, methods=['POST'])
        self.app.add_url_rule('/health', 'health', self.handle_health, methods=['GET'])

    def handle_health(self):
        return jsonify({"status": "ready"}), 200

    def handle_get_images(self):
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400
        
        uuids = data.get("uuids")
        if not uuids:
            # Support single uuid input
            uuid = data.get("uuid")
            if uuid:
                uuids = [uuid]
            else:
                return jsonify({"error": "Missing 'uuids' or 'uuid' field"}), 400
        
        if not isinstance(uuids, list):
             return jsonify({"error": "'uuids' must be a list"}), 400

        try:
            images = self.service.retrieve_images(uuids)
            return jsonify({"images": images}), 200
        except Exception as e:
            logger.exception("Error retrieving images")
            return jsonify({"error": str(e)}), 500

    def run(self, host: str, port: int):
        self.app.run(host=host, port=port)

def main():
    parser = argparse.ArgumentParser(description="Image Retrieval Server")
    parser.add_argument("--lookup_path", type=str, required=True, help="Path to SQLite lookup file (.db)")
    parser.add_argument("--port", type=int, default=5002, help="Port to run the server on")
    parser.add_argument("--workers", type=int, default=4, help="Number of threads for H5 I/O")
    parser.add_argument("--h5_group", type=str, default="images", help="H5 group name for images")
    
    args = parser.parse_args()
    
    service = ImageRetrievalService(
        lookup_path=args.lookup_path,
        h5_group=args.h5_group,
        max_workers=args.workers
    )
    
    server = ImageServer(service)
    
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = args.port
    logger.info(f"Starting Image Server on {SERVER_HOST}:{SERVER_PORT}...")
    server.run(host=SERVER_HOST, port=SERVER_PORT)

if __name__ == "__main__":
    main()