
from PIL import Image
from typing import Dict, Optional, List, Union
import pyarrow as pa
import pyarrow.compute as pc
import io
import torch
import h5py


def search_hdf5(h5_path, uuids: List[str], group="images") -> Dict[str, Image.Image]:
    """
    Search and retrieve images from HDF5 file by UUIDs.
    
    Args:
        h5_path: Path to the HDF5 file
        uuids: List of UUID strings to search for
        group: HDF5 group name where images are stored
    Returns:
        Dictionary mapping UUIDs to PIL Image objects
    """
    images_dict = {}

    with h5py.File(h5_path, "r") as f:
        imgs = f[group]

        for uuid in uuids:
            if uuid in imgs:
                raw_bytes = imgs[uuid][()]  # read uint8 array
                try:
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    images_dict[uuid] = img
                except Exception as e:
                    print(f"Failed to parse image bytes for UUID {uuid}: {e}")
                    continue              

    return images_dict
    

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

@torch.no_grad()
def embed_image(image: Image.Image, model, preprocess) -> List[float]:
    """
    Placeholder function to embed an image into a vector.
    Replace with actual embedding logic as needed.
    
    Args:
        image: PIL Image to be embedded
        
    Returns:
        List of floats representing the image embedding
    """
    
    image_preprocessed = preprocess(image).unsqueeze(0)
    image_features = model.encode_image(image_preprocessed)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    return image_features.flatten().tolist()

def retrieve_images_hdf5(uuid_list: List[str], lookup_tbl: pa.Table):
    images_dict = {}
    failed_dict = {}
    
    mask = pc.is_in(lookup_tbl['uuid'], pa.array(uuid_list))
    matched_tbl = lookup_tbl.filter(mask)
    
    matched_uuids = set(matched_tbl["uuid"].to_pylist())
    unmatched_uuids = list(set(uuid_list) - matched_uuids)
    matched_uuids = list(matched_uuids)
    
    for uuid in unmatched_uuids:
        failed_dict[uuid] = "not_found_in_lookup"
    
    if len(matched_uuids) == 0:
        return images_dict, failed_dict
    
    
    h5_paths = sorted(set(matched_tbl["file_path"].to_pylist()))
    # Build {h5_path: [uuids,...]} so each file is opened once
    uuids_by_file = {}
    uuids_col = matched_tbl["uuid"].to_pylist()
    paths_col  = matched_tbl["file_path"].to_pylist()
    for u, p in zip(uuids_col, paths_col):
        uuids_by_file.setdefault(p, []).append(u)
    
    for h5_path in h5_paths:
        try:
            imgs_dict = search_hdf5(h5_path, uuids_by_file[h5_path], group="images")
            images_dict.update(imgs_dict)
            
            # Mark any missing UUIDs as failed
            for uuid in uuids_by_file[h5_path]:
                if uuid not in imgs_dict:
                    failed_dict[uuid] = "hdf5_image_not_found"
        except Exception as e:
            # If file open fails, mark all its UUIDs as failed
            for uuid in uuids_by_file[h5_path]:
                failed_dict[uuid] = f"hdf5_open_failed: {str(e)}"

    return images_dict, failed_dict

