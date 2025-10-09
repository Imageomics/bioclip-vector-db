
import requests
from PIL import Image
from typing import Dict, Tuple, Optional, List, Union
import pyarrow as pa
import pyarrow.compute as pc
import io

def parse_uploaded_image(uploaded_file: Union[bytes, io.BytesIO, str]) -> Optional[Image.Image]:
    """
    Parse uploaded file into PIL Image.
    
    Args:
        uploaded_file: Uploaded file as bytes, BytesIO stream, or file path string
        
    Returns:
        PIL Image object or None if parsing fails
    """
    try:
        if isinstance(uploaded_file, bytes):
            # Parse from raw bytes
            return Image.open(io.BytesIO(uploaded_file))
        elif isinstance(uploaded_file, io.BytesIO):
            # Parse from BytesIO stream
            uploaded_file.seek(0)  # Reset stream position
            return Image.open(uploaded_file)
        elif isinstance(uploaded_file, str):
            # Parse from file path
            return Image.open(uploaded_file)
        else:
            # Try direct PIL opening (handles file-like objects)
            return Image.open(uploaded_file)
    except Exception as e:
        print(f"Failed to parse uploaded image: {e}")
        return None

def embed_image(image: Image.Image, model, preprocess) -> List[float]:
    """
    Placeholder function to embed an image into a vector.
    Replace with actual embedding logic as needed.
    
    Args:
        image: PIL Image to be embedded
        
    Returns:
        List of floats representing the image embedding
    """
    # Dummy implementation - replace with actual model inference
    return [0.0] * 512  # Example: 512-dimensional zero vector

def retrieve_images(uuid_list: List[str], method: str, lookup_tbl: pa.Table) -> Tuple[Dict[str, Image.Image], Dict[str, str]]:
    """
    Retrieve images by UUID list with fallback handling.
    
    Args:
        uuid_list: List of unique identifiers for the images
        method: "remote" or "local"
        lookup_tbl: Arrow table with columns ['uuid', 'local_path', 'uri'] or similar
        
    Returns:
        Tuple of (Dict of {uuid: PIL Image}, Dict of {uuid: failure_type})
    """
    # lookup tbl: Arrow table that can be queried
    # If lookup_tbl is None, raise error
    if lookup_tbl is None:
        raise ValueError("lookup_tbl cannot be None")
    
    # Initialize result dictionaries
    images_dict = {}
    failed_dict = {}
    
    # First obtain the subset UUID -> (local_path, uri) table
    try:
        # Filter table to get only rows matching our UUID list
        mask = pc.is_in(lookup_tbl['uuid'], pa.array(uuid_list))
        filtered_table = lookup_tbl.filter(mask)
        
        # Convert to dict for efficient lookup: {uuid: (local_path, uri)}
        uuid_mappings = {}
        if len(filtered_table) > 0:
            table_dict = filtered_table.to_pydict()
            for i in range(len(table_dict['uuid'])):
                uuid = table_dict['uuid'][i]
                local_path = table_dict.get('local_path', [None] * len(table_dict['uuid']))[i]
                uri = table_dict.get('uri', [None] * len(table_dict['uuid']))[i]
                uuid_mappings[uuid] = (local_path, uri)
        
        # Check for missing UUIDs
        for uuid in uuid_list:
            if uuid not in uuid_mappings:
                failed_dict[uuid] = "not_found_in_lookup"
                
    except Exception as e:
        failed_dict.update({uuid: f"query_error: {str(e)}" for uuid in uuid_list})
        return images_dict, failed_dict
    
    # Then perform I/O or request for each valid UUID
    for uuid, (local_path, uri) in uuid_mappings.items():
        if method == "remote":
            # Fetch image from URL Object Storage
            if uri:
                try:
                    response = requests.get(uri, timeout=10)
                    response.raise_for_status()
                    image = Image.open(requests.get(uri, stream=True).raw)
                    images_dict[uuid] = image
                    continue
                except Exception:
                    pass  # Fall through to local attempt
            
            # If failed, try local
            if local_path:
                try:
                    image = Image.open(local_path)
                    images_dict[uuid] = image
                    continue
                except Exception:
                    pass
            
            # If both failed
            failed_dict[uuid] = "remote_and_local_failed"
                    
        elif method == "local":
            # Load images from local
            if local_path:
                try:
                    image = Image.open(local_path)
                    images_dict[uuid] = image
                except Exception:
                    failed_dict[uuid] = "local_failed"
            else:
                failed_dict[uuid] = "no_local_path"
        else:
            raise ValueError(f"Method {method} not supported.")
    
    return images_dict, failed_dict


