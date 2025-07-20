import chromadb
import logging

from storage.storage_interface import StorageInterface
from typing import List, Dict

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger()


def get_storage_interface() -> StorageInterface:
    """
    Returns an instance of the StorageInterface.
    """
    return Chroma()

class Chroma(StorageInterface):
    def init(self, name: str, **kwargs):
        if "metadata" not in kwargs:
            raise "Chromadb cannot be initialized without metadata."
        
        if "collection_dir" not in kwargs:
            raise "Chromadb cannot be initialized without collection_dir."
        
        self._metadata = kwargs["metadata"]
        self._collection_dir = kwargs["collection_dir"]

        logger.info("Initializing ChromaDb client.")
        self._chroma_client = chromadb.PersistentClient(path=self._collection_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name=name,
            metadata=self._metadata
        )
        logger.info(f"Created/Initialized collection: {self._collection.name}")

        return self


    def add_embedding(self, id: str, embedding: List[float], metadata: Dict[str, str]):
        self._collection.add(embeddings=[embedding], ids=[id])
        self._count += 1


    def batch_add_embeddings(self, ids: List[str], embeddings: List[List[float]], 
                             metadatas: List[Dict[str, str]]):
        assert len(ids) == len(embeddings) == len(metadatas), "Lengths of ids, embeddings, and metadatas must be the same."
        self._collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)
        self._count += len(ids)

    def query(self, id: str):
        return self._collection.get(id)["documents"]
    
    def reset(self, force=False):
        if force:
            self._chroma_client.delete_collection(self._collection_name)

        return self.init(name=self._collection.name,
                         metadata=self._metadata,
                         collection_dir=self._collection_dir)
