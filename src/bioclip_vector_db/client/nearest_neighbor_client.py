import requests
import logging
import json
import random
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class NearestNeighborClient:
    """A client for querying multiple LocalIndexServer instances."""

    def __init__(self, server_urls: List[str], global_max_neighbors=100):
        """
        Initializes the client with a list of server URLs.

        :param server_urls: A list of URLs for the LocalIndexServer instances.
        """
        if not server_urls:
            raise ValueError("server_urls cannot be empty.")
        self._server_urls = server_urls
        self._global_max_neighbors = global_max_neighbors

    def _post_request(self, url: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sends a POST request to a given URL and returns the JSON response."""
        try:
            response = requests.post(url, json=json_data)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            return {"status": "error", "error": {"message": str(e)}}

    def _get_request(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Sends a GET request to a given URL and returns the JSON response."""
        try:
            response = requests.get(url, json=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            return {"status": "error", "error": {"message": str(e)}}

    def search(
        self, query_vector: List[float], top_n: int = 10, nprobe: int = 1, fetch_metadata: bool = True
    ) -> List[Dict[str, Any]]:
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
            "verbose": False,
        }
        for url in self._server_urls:
            search_url = f"{url}/search"
            try:
                result = self._post_request(search_url, search_payload)
                results.append({"server": url, "response": result})
            except Exception as e:
                logger.error(f"Search for {url} failed: {e}")
            
        merged_results = self._merge_results(results)

        if fetch_metadata:
            for result in merged_results:
                image_id = result.get("id")
                if image_id:
                    metadata_response = self.get_metadata(image_id)
                    if metadata_response and metadata_response.get("status") == "success":
                        result["metadata"] = metadata_response.get("data")

        return merged_results

    def get_metadata(self, image_id: str, server_url: str = None) -> Dict[str, Any]:
        """
        Retrieves metadata for a given image_id from one of the servers.

        :param image_id: The image_id to retrieve metadata for.
        :param server_url: The specific server to query. If None, a random server is chosen.
        :return: The metadata response from the server.
        """
        if server_url and server_url not in self._server_urls:
            raise ValueError(f"Provided server_url '{server_url}' is not in the configured list of servers.")

        get_payload = {"image_id": image_id}
        target_server = server_url if server_url else random.choice(self._server_urls)
        get_url = f"{target_server}/get"
        
        try:
            return self._get_request(get_url, params=get_payload)
        except Exception as e:
            logger.error(f"Get metadata for {image_id} from {get_url} failed: {e}")
            return {"status": "error", "error": {"message": str(e)}}

    def _merge_results(self, results):
        """
        Produces a global merged result list.
        """
        global_merged_results = []
        for result in results:
            global_merged_results.extend(result["response"]["data"]["merged_neighbors"])

        return sorted(global_merged_results, key=lambda x: x["distance"])[
            : self._global_max_neighbors
        ]

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
                health_statuses.append(
                    {
                        "server": url,
                        "response": {"status": "error", "error": {"message": str(e)}},
                    }
                )
        return health_statuses