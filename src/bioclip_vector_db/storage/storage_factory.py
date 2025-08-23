"""
Author: Sreejith Menon

Contains factory methods that abstracts the internal implementation details of various storage implementations.
"""

import enum

from .storage_interface import StorageInterface
from . import storage_impl

class HfDatasetType(enum.Enum):
    BIRD = "Somnath01/Birds_Species"
    TREE_OF_LIFE = "imageomics/TreeOfLife-10M"
    TREE_OF_LIFE_LOCAL = "local_tree_of_life"

class StorageEnum(enum.Enum):
    CHROMADB = 1
    FAISS_IVF = 2


def get_storage(storage_type: StorageEnum, dataset_type: HfDatasetType, **kwargs) -> StorageInterface:
    """
    Returns an instance of the StorageInterface.
    """
    if storage_type == StorageEnum.CHROMADB:
        if "collection_dir" not in kwargs:
            raise ValueError("Chromadb cannot be initialized without collection_dir.")
        chroma = storage_impl.Chroma()
        return chroma.init(dataset_type.name, 
                           collection_dir=kwargs["collection_dir"],
                           metadata={"hnsw:space": "ip", "hnsw:search_ef": 10})
    
    if storage_type == StorageEnum.FAISS_IVF:
        if "collection_dir" not in kwargs:
            raise ValueError("Faiss cannot be initialized without collection_dir.")
        faiss_ivf = storage_impl.FaissIvf()
        return faiss_ivf.init(dataset_type.name, 
                              collection_dir=kwargs["collection_dir"],
                              dimensions=512, 
                              nlist=10)

    raise ValueError(f"Invalid storage type: {storage_type}")
