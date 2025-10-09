import requests
import json
import pandas as pd
from typing import List, Optional

def check_server_health(server_endpoint: str = "http://localhost:5001") -> bool:
    """
    Simple boolean check for server health status.
    
    Args:
        server_endpoint: Base URL of the server (default: http://localhost:5001)
        
    Returns:
        True if server is healthy and ready, False otherwise
    """
    health_check_endpoint = f"{server_endpoint}/health"
    try:
        response = requests.get(
            health_check_endpoint,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Parse the JSON response
        health_data = response.json()
        
        # Check if the server reports as healthy
        is_healthy = (
            health_data.get("status") == "success" and
            health_data.get("data", {}).get("status") == "ready"
        )
        
        return is_healthy
        
    except requests.exceptions.RequestException:
        # Connection error, timeout, or HTTP error
        print("Failed to connect to the server")
        return False
    except json.JSONDecodeError:
        # Invalid JSON response
        print("Invalid JSON response")
        return False

def search(query_vector: List[float], top_n: int = 10, nprobe: int = 1, 
           server_endpoint: str = "http://localhost:5001") -> Optional[pd.DataFrame]:
    """
    Search for similar vectors in the BioCLIP vector database.
    
    Args:
        query_vector: The input vector to search for (required)
        top_n: Number of nearest neighbors to return (default: 10)
        nprobe: Number of inverted list probes for search (default: 1)
        server_endpoint: Base URL of the server (default: http://localhost:5001)
        
    Returns:
        pandas.DataFrame with columns ['id', 'distance'] sorted by distance (ascending),
        or None if health check fails or search error occurs
    """
    # Health check before searching
    if not check_server_health(server_endpoint):
        print("Health check failed - server not ready")
        return None
    
    search_endpoint = f"{server_endpoint}/search"
    
    # Prepare request payload
    payload = {
        "query_vector": query_vector,
        "top_n": top_n,
        "nprobe": nprobe,
        "verbose": False
    }
    
    try:
        response = requests.post(
            search_endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        if result.get("status") != "success":
            print(f"Search failed: {result}")
            return None
        
        # Extract neighbors and convert to DataFrame
        neighbors = result.get("data", {}).get("merged_neighbors", [])
        
        if not neighbors:
            print("No neighbors found")
            return pd.DataFrame(columns=['uuid', 'distance'])

        # Create DataFrame with uuid and distance columns
        df = pd.DataFrame(neighbors)
        
        # Ensure columns are in the right order and types
        df = df[['uuid', 'distance']].copy()
        df['distance'] = df['distance'].astype(float)
        
        # Sort by distance (ascending - closer matches first)
        df = df.sort_values('distance').reset_index(drop=True)
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None