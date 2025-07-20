"""
Author: Sreejith Menon=

Contains factory methods that abstracts the internal implementation details of various storage implementations.
"""

import enum

from storage.storage_interface import StorageInterface
import storage.storage_impl as storage_impl

class HfDatasetType(enum.Enum):
    BIRD = "Somnath01/Birds_Species"
    TREE_OF_LIFE = "imageomics/TreeOfLife-10M"
    TREE_OF_LIFE_LOCAL = "local_tree_of_life"

class StorageEnum(enum.Enum):
    CHROMADB = 1

def get_storage(storage_type: StorageEnum, dataset_type: HfDatasetType, **kwargs) -> StorageInterface:
    """
    Returns an instance of the StorageInterface.
    """
    if storage_type == StorageEnum.CHROMADB:
        if "collection_dir" not in kwargs:
            raise "Chromadb cannot be initialized without collection_dir."
        chroma = storage_impl.Chroma()
        return chroma.init(dataset_type.name, 
                           collection_dir=kwargs["collection_dir"],
                           metadata={"hnsw:space": "ip", "hnsw:search_ef": 10})

    raise ValueError(f"Invalid storage type: {storage_type}")
