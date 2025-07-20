"""
Author: Sreejith Menon
Virtual class for setting up the vector database.

Provides the ability to abstract away the implementation details 
of the actual vector DB implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class StorageInterface(ABC):
    """
    An abstract base class defining the interface for a vector database storage.

    This class provides a contract for different vector database implementations,
    ensuring they all expose a consistent API for initialization and data insertion.
    This allows for easy swapping of the underlying vector DB technology without
    changing the application logic that uses it.
    """
    def __init__(self):
        # Number of records added.
        self._count = 0

        # Path corresponding to local storage
        self._collection_dir = None
        self._collection_name = None

    @abstractmethod
    def init(self, name: str, **kwargs):
        """
        Initializes the vector database.

        This method should handle the setup of the database, which could involve
        creating a new database instance/collection or connecting to an existing one.

        Args:
            name: The name of the vector database.
            **kwargs: Additional keyword arguments specific to the implementation.
        """
        pass

    @abstractmethod
    def add_embedding(self, id: str, embedding: List[float], metadata: Dict[str, str]):
        """
        Adds a single embedding and its associated metadata to the database.

        Args:
            id: A unique identifier for the embedding.
            embedding: A list of floats representing the vector embedding.
            metadata: A dictionary of metadata associated with the embedding.
        """
        pass

    @abstractmethod
    def batch_add_embeddings(self, ids: List[str], embeddings: List[List[float]], 
                             metadatas: List[Dict[str, str]]):
        """
        Adds a batch of embeddings and their associated metadata to the database.

        This method is designed for efficient bulk insertion of data.

        Args:
            ids: A list of unique identifiers for the embeddings.
            embeddings: A list of embeddings, where each embedding is a list of floats.
            metadatas: A list of metadata dictionaries, corresponding to each embedding.
        """
        pass

    @abstractmethod
    def query(self, id: str):
        """
        Queries the database for the embedding associated with the given id.

        Args:
            id: The unique identifier for the embedding.
        """
        pass

    @abstractmethod
    def reset(self, force=False):
        """ 
        Resets the database to its initial state.
        """
        pass