import requests
import logging
import json
import random
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class NearestNeighborClient:
    """A client for querying multiple LocalIndexServer instances."""

    def __init__(self, server_urls: List[str]):
        """
        Initializes the client with a list of server URLs.

        :param server_urls: A list of URLs for the LocalIndexServer instances.
        """
        if not server_urls:
            raise ValueError("server_urls cannot be empty.")
        self._server_urls = server_urls

    def _post_request(self, url: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sends a POST request to a given URL and returns the JSON response."""
        try:
            response = requests.post(url, json=json_data)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            return {"status": "error", "error": {"message": str(e)}}

    def search(self, query_vector: List[float], top_n: int = 10, nprobe: int = 1) -> List[Dict[str, Any]]:
        """
        Queries all configured servers for the nearest neighbors.

        :param query_vector: The vector to search for.
        :param top_n: The number of nearest neighbors to return.
        :param nprobe: The number of inverted list probes to use.
        :return: A list of responses from each server.
        """
        results = []
        search_payload = {
            "query_vector": query_vector,
            "top_n": top_n,
            "nprobe": nprobe,
        }
        for url in self._server_urls:
            search_url = f"{url}/search"
            result = self._post_request(search_url, search_payload)
            results.append({"server": url, "response": result})
        return results

    def health(self) -> List[Dict[str, Any]]:
        """
        Checks the health of all configured servers.

        :return: A list of health status responses from each server.
        """
        health_statuses = []
        for url in self._server_urls:
            health_url = f"{url}/health"
            try:
                response = requests.get(health_url)
                response.raise_for_status()
                health_statuses.append({"server": url, "response": response.json()})
            except requests.exceptions.RequestException as e:
                logger.error(f"Health check for {url} failed: {e}")
                health_statuses.append({"server": url, "response": {"status": "error", "error": {"message": str(e)}}})
        return health_statuses

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

    # List of server URLs to query
    SERVER_URLS = [f"http://0.0.0.0:{port}" for port in range(5001, 5007)]

    # Initialize the client
    client = NearestNeighborClient(SERVER_URLS)

    # 1. Check the health of the servers
    print("--- Checking server health ---")
    health_results = client.health()
    print(json.dumps(health_results, indent=2))

    # 2. Perform a search
    print("\n--- Performing search ---")
    
    # Create a dummy query vector.
    # IMPORTANT: The dimension of this vector must match the dimension of the vectors in the FAISS index.
    # For BioCLIP models, this is often 512 or 768. We'll use 512 as an example.
    DUMMY_VECTOR_DIM = 512
    dummy_query_vector = [random.random() for _ in range(DUMMY_VECTOR_DIM)]

    # Perform the search
    search_results = client.search(query_vector=dummy_query_vector, top_n=5, nprobe=10)

    # Print the results
    print(json.dumps(search_results, indent=2))