
from PIL import Image
from typing import Dict, Optional, List, Union, Tuple
import pyarrow as pa
import pyarrow.compute as pc
import io
import torch
import h5py
import logging
import time
from multiprocessing import Pool
from functools import partial

logger = logging.getLogger(__name__)


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
    
    device = next(model.parameters()).device
    image_preprocessed = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image_preprocessed)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    return image_features.flatten().tolist()


def _process_hdf5_file(h5_path: str, uuids: List[str], group: str = "images") -> Tuple[Dict[str, Image.Image], Dict[str, str]]:
    """
    Worker function to process a single HDF5 file.
    
    Args:
        h5_path: Path to the HDF5 file
        uuids: List of UUIDs to retrieve from this file
        group: HDF5 group name where images are stored
        
    Returns:
        Tuple of (images_dict, failed_dict)
    """
    images_dict = {}
    failed_dict = {}
    
    try:
        imgs_dict = search_hdf5(h5_path, uuids, group=group)
        images_dict.update(imgs_dict)
        
        # Mark any missing UUIDs as failed
        for uuid in uuids:
            if uuid not in imgs_dict:
                failed_dict[uuid] = "hdf5_image_not_found"
    except Exception as e:
        # If file open fails, mark all its UUIDs as failed
        for uuid in uuids:
            failed_dict[uuid] = f"hdf5_open_failed: {str(e)}"
    
    return images_dict, failed_dict


def retrieve_images_hdf5(uuid_list: List[str], lookup_tbl: pa.Table, num_workers: int = 1, group: str = "images"):
    """
    Retrieve images from HDF5 files with optional multiprocessing support.
    
    Args:
        uuid_list: List of UUIDs to retrieve
        lookup_tbl: PyArrow table mapping UUIDs to HDF5 file paths
        num_workers: Number of worker processes (default=1, no multiprocessing)
        group: HDF5 group name where images are stored
        
    Returns:
        Tuple of (images_dict, failed_dict)
    """
    start_time = time.time()
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
        elapsed = time.time() - start_time
        logger.info(f"Retrieved 0 images in {elapsed:.3f}s (all UUIDs not found in lookup)")
        return images_dict, failed_dict
    
    # Build a mapping of HDF5 files to the UUIDs they contain
    # Data structure: uuids_by_file = {
    #     "/path/to/file1.h5": ["uuid1", "uuid2", "uuid3"],
    #     "/path/to/file2.h5": ["uuid4", "uuid5"],
    #     ...
    # }
    # This grouping ensures each HDF5 file is opened only once, even if it contains
    # multiple requested UUIDs. Opening HDF5 files is expensive, so this optimization
    # significantly reduces I/O overhead.
    uuids_by_file = {}
    uuids_col = matched_tbl["uuid"].to_pylist()
    paths_col = matched_tbl["h5_file"].to_pylist()
    for u, p in zip(uuids_col, paths_col):
        uuids_by_file.setdefault(p, []).append(u)
    
    h5_paths = list(uuids_by_file.keys())
    num_files = len(h5_paths)
    
    logger.info(f"Starting image retrieval: {len(matched_uuids)} UUIDs across {num_files} HDF5 files with {num_workers} worker(s)")
    
    if num_workers <= 1:
        # Sequential processing (default)
        # Process each HDF5 file one at a time in the main process
        for h5_path in h5_paths:
            imgs_dict, failed = _process_hdf5_file(h5_path, uuids_by_file[h5_path], group=group)
            images_dict.update(imgs_dict)
            failed_dict.update(failed)
    else:
        # Parallel processing with multiprocessing
        # Task division strategy: Each HDF5 file becomes one task unit
        # 
        # Example with 3 workers and 8 HDF5 files:
        #   Worker 1: processes file1.h5, file4.h5, file7.h5
        #   Worker 2: processes file2.h5, file5.h5, file8.h5
        #   Worker 3: processes file3.h5, file6.h5
        #
        # The scheduler (pool.starmap) automatically distributes tasks to available
        # workers as they become free. Files with more UUIDs may take longer, but
        # the pool ensures balanced workload distribution across workers.
        #
        # Note: Each worker process opens and closes HDF5 files independently,
        # ensuring thread-safe file access without locking issues.
        
        worker_func = partial(_process_hdf5_file, group=group)
        # Create task list: [(file_path1, [uuid1, uuid2]), (file_path2, [uuid3]), ...]
        tasks = [(h5_path, uuids_by_file[h5_path]) for h5_path in h5_paths]
        
        with Pool(processes=num_workers) as pool:
            # starmap distributes tasks across workers and collects results
            results = pool.starmap(worker_func, tasks)
        
        # Merge results from all workers into final dictionaries
        for imgs_dict, failed in results:
            images_dict.update(imgs_dict)
            failed_dict.update(failed)
    
    elapsed = time.time() - start_time
    num_retrieved = len(images_dict)
    num_failed = len(failed_dict)
    logger.info(f"Image retrieval completed in {elapsed:.3f}s: {num_retrieved} retrieved, {num_failed} failed")

    return images_dict, failed_dict

