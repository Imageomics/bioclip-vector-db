import chromadb
import logging
import faiss

from .storage_interface import StorageInterface
from typing import List, Dict

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger()


class Chroma(StorageInterface):
    def init(self, name: str, **kwargs):
        if "metadata" not in kwargs:
            raise ValueError("Chromadb cannot be initialized without metadata.")

        if "collection_dir" not in kwargs:
            raise ValueError("Chromadb cannot be initialized without collection_dir.")

        self._metadata = kwargs["metadata"]
        self._collection_dir = kwargs["collection_dir"]

        logger.info("Initializing ChromaDb client.")
        self._chroma_client = chromadb.PersistentClient(path=self._collection_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name=name, metadata=self._metadata
        )
        logger.info(f"Created/Initialized collection: {self._collection.name}")

        return self

    def add_embedding(self, id: str, embedding: List[float], metadata: Dict[str, str]):
        self._collection.add(embeddings=[embedding], ids=[id], metadatas=[metadata])
        self._count += 1

    def batch_add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, str]],
    ):
        assert (
            len(ids) == len(embeddings) == len(metadatas)
        ), "Lengths of ids, embeddings, and metadatas must be the same."
        self._collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)
        self._count += len(ids)

    def query(self, id: str):
        return self._collection.get(id)["documents"]

    def reset(self, force=False):
        if force:
            self._chroma_client.delete_collection(self._collection.name)

        return self.init(
            name=self._collection.name,
            metadata=self._metadata,
            collection_dir=self._collection_dir,
        )


class FaissIvf(StorageInterface):
    """Faiss index with inverted file index. Requires training to use."""

    def init(self, name: str, **kwargs):
        if "collection_dir" not in kwargs:
            raise ValueError("Faiss cannot be initialized without collection_dir.")
        if "dimensions" not in kwargs:
            raise ValueError("Faiss cannot be initialized without dimensions.")
        if "factory_string" not in kwargs:
            raise ValueError("Faiss cannot be initialized without factory_string.")

        self._collection_dir = kwargs["collection_dir"]
        self._dimensions = kwargs["dimensions"]
        self._factory_string = kwargs["factory_string"]

        self._index = faiss.index_factory(self._dimensions, self._factory_string)

        logger.info(
            f"Initializing Faiss client with the factory string: {self._factory_string}."
        )

    def add_embedding(self, id: str, embedding: List[float], metadata: Dict[str, str]):
        pass

    def batch_add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, str]],
    ):
        pass

    def query(self, id: str):
        pass

    def reset(self, force=False):
        pass
